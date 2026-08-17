import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
import zipfile

import numpy as np
import pytest


EXPECTED_STAGES = (
    "preprocessed_image",
    "conditioning_512",
    "sparse_flow",
    "sparse_support",
    "shape_flow",
    "shape_slat",
    "texture_flow",
    "decoder_raw_mesh",
    "texture_voxels",
    "pipeline_filled_mesh",
    "postprocess_stage11_pre_orientation",
    "postprocess_stage12_post_orientation",
    "consumer_glb",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_args(tmp_path: Path, image: Path) -> list[str]:
    return [
        "--image",
        str(image),
        "--expected-image-sha256",
        _sha256(image),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--work-dir",
        str(tmp_path / "runtime"),
    ]


def _write_clean_git_checkout(root: Path) -> None:
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=root,
        check=True,
    )
    (root / "setup.py").write_text("# pinned source\n")
    subprocess.run(["git", "add", "setup.py"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "source"], cwd=root, check=True)


def _write_minimal_wheel(
    path: Path,
    *,
    distribution: str,
    version: str,
    package_files: dict[str, str],
    requires: tuple[str, ...] = (),
) -> None:
    normalized = distribution.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {distribution}",
        f"Version: {version}",
        *(f"Requires-Dist: {requirement}" for requirement in requires),
        "",
        "",
    ]
    files = {
        **package_files,
        f"{dist_info}/METADATA": "\n".join(metadata),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: trellis2mlx-conformance-test\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
    }
    record_path = f"{dist_info}/RECORD"
    files[record_path] = "".join(f"{name},,\n" for name in (*files, record_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def _create_test_venv(root: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(root)],
        check=True,
    )
    return root / "bin" / "python"


def _pip_install(python: Path, wheel: Path, *, force: bool = False) -> None:
    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    if force:
        command.append("--force-reinstall")
    command.extend(["--no-deps", str(wheel)])
    subprocess.run(command, check=True, capture_output=True, text=True)


def _venv_import_record(python: Path, repo_root: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata as m, json, xformers, xformers.ops as ops; "
        "print(json.dumps({'marker': xformers.MARKER, 'ops_marker': ops.MARKER, "
        "'dependency_version': m.version('xformers-conformance-dependency')}))"
    )
    completed = subprocess.run(
        [str(python), "-I", "-c", code],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_expected_post_build_products_are_recorded_removed_and_leave_clean_source(
    tmp_path,
):
    from scripts.source_cuda_native_image_to_glb_witness import (
        _remove_expected_build_products,
    )

    root = tmp_path / "nvdiffrast"
    _write_clean_git_checkout(root)
    (root / "build" / "temp.linux-test").mkdir(parents=True)
    (root / "build" / "temp.linux-test" / "extension.o").write_bytes(b"object")
    (root / "nvdiffrast.egg-info").mkdir()
    (root / "nvdiffrast.egg-info" / "PKG-INFO").write_text("Name: nvdiffrast\n")

    removed = _remove_expected_build_products(
        root,
        allowed_roots=("build", "nvdiffrast.egg-info"),
    )

    assert removed == ["build", "nvdiffrast.egg-info"]
    assert not (root / "build").exists()
    assert not (root / "nvdiffrast.egg-info").exists()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""


def test_pinned_xformers_install_forces_same_version_after_digest_verification(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    payload = b"exact-xformers-wheel"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    commands = []

    def download(url, destination):
        assert url == "https://example.invalid/xformers.whl"
        Path(destination).write_bytes(payload)

    def run(command, cwd=None):
        stdout = "pip 25.1 from /test/site-packages/pip (python 3.12)\n" if command[-1] == "--version" else ""
        receipt = {
            "command": command,
            "cwd": cwd,
            "exit_code": 0,
            "stdout": stdout,
            "stderr": "",
        }
        commands.append(receipt)
        return receipt

    report = {"setup_commands": []}
    record = witness._install_pinned_xformers(
        python="/usr/bin/python3",
        work_dir=tmp_path,
        report=report,
        wheel_url="https://example.invalid/xformers.whl",
        expected_sha256=expected_sha256,
        downloader=download,
        runner=run,
    )

    wheel_path = (
        tmp_path
        / "pinned-wheels"
        / "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
    )
    assert record == {
        "version": "0.0.35",
        "url": "https://example.invalid/xformers.whl",
        "path": str(wheel_path),
        "sha256": expected_sha256,
        "size_bytes": len(payload),
        "install_mode": "forced-local-wheel-no-deps",
        "pip_version": "pip 25.1 from /test/site-packages/pip (python 3.12)",
    }
    assert commands == [
        {
            "command": [
                "/usr/bin/python3",
                "-m",
                "pip",
                "--disable-pip-version-check",
                "--version",
            ],
            "cwd": None,
            "exit_code": 0,
            "stdout": "pip 25.1 from /test/site-packages/pip (python 3.12)\n",
            "stderr": "",
        },
        {
            "command": [
                "/usr/bin/python3",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--force-reinstall",
                "--no-deps",
                str(wheel_path),
            ],
            "cwd": None,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }
    ]
    assert report["setup_commands"] == commands


def test_real_pip_replaces_same_version_and_joins_pep610_provenance(
    tmp_path, monkeypatch
):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    wheel_name = "xformers-0.0.35-py3-none-any.whl"
    dependency_wheel = tmp_path / "wheels" / "dependency" / (
        "xformers_conformance_dependency-1.0-py3-none-any.whl"
    )
    wheel_a = tmp_path / "wheels" / "a" / wheel_name
    wheel_b = tmp_path / "wheels" / "b" / wheel_name
    _write_minimal_wheel(
        dependency_wheel,
        distribution="xformers-conformance-dependency",
        version="1.0",
        package_files={"xformers_conformance_dependency/__init__.py": ""},
    )
    _write_minimal_wheel(
        wheel_a,
        distribution="xformers",
        version="0.0.35",
        package_files={
            "xformers/__init__.py": "__version__ = '0.0.35'\nMARKER = 'wheel-a'\n",
            "xformers/ops/__init__.py": "MARKER = 'ops-a'\n",
        },
        requires=("xformers-conformance-dependency==1.0",),
    )
    _write_minimal_wheel(
        wheel_b,
        distribution="xformers",
        version="0.0.35",
        package_files={
            "xformers/__init__.py": "__version__ = '0.0.35'\nMARKER = 'wheel-b'\n",
            "xformers/ops/__init__.py": "MARKER = 'ops-b'\n",
        },
        requires=("xformers-conformance-dependency==2.0",),
    )

    baseline_python = _create_test_venv(tmp_path / "baseline-venv")
    _pip_install(baseline_python, dependency_wheel)
    _pip_install(baseline_python, wheel_a)
    _pip_install(baseline_python, wheel_b)
    assert _venv_import_record(baseline_python, Path.cwd()) == {
        "marker": "wheel-a",
        "ops_marker": "ops-a",
        "dependency_version": "1.0",
    }

    forced_python = _create_test_venv(tmp_path / "forced-venv")
    _pip_install(forced_python, dependency_wheel)
    _pip_install(forced_python, wheel_a)
    monkeypatch.setattr(witness, "XFORMERS_WHEEL_FILENAME", wheel_name)
    wheel_b_sha256 = _sha256(wheel_b)
    report = {"setup_commands": []}
    record = witness._install_pinned_xformers(
        python=str(forced_python),
        work_dir=tmp_path / "runtime",
        report=report,
        wheel_url=wheel_b.resolve().as_uri(),
        expected_sha256=wheel_b_sha256,
        downloader=lambda _url, destination: shutil.copyfile(wheel_b, destination),
    )

    assert record["pip_version"].startswith("pip ")
    assert report["setup_commands"][0]["command"][-1] == "--version"
    assert "--force-reinstall" in report["setup_commands"][1]["command"]
    assert "--no-deps" in report["setup_commands"][1]["command"]
    assert _venv_import_record(forced_python, Path.cwd()) == {
        "marker": "wheel-b",
        "ops_marker": "ops-b",
        "dependency_version": "1.0",
    }

    repo_root = Path(__file__).resolve().parents[1]
    provenance_code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "from scripts import source_cuda_native_image_to_glb_witness as w; "
        "import xformers, xformers.ops as ops; "
        f"r=w.read_xformers_install_provenance(xformers, ops, wheel_path={record['path']!r}, "
        f"expected_sha256={wheel_b_sha256!r}); "
        "print(json.dumps(r, sort_keys=True))"
    )
    completed = subprocess.run(
        [str(forced_python), "-I", "-c", provenance_code],
        check=True,
        capture_output=True,
        text=True,
    )
    provenance = json.loads(completed.stdout)
    assert provenance["wheel_path"] == record["path"]
    assert provenance["wheel_sha256"] == wheel_b_sha256
    assert provenance["distribution_name"] == "xformers"
    assert provenance["distribution_version"] == "0.0.35"
    assert provenance["distribution_files"] == {
        "xformers": "xformers/__init__.py",
        "xformers.ops": "xformers/ops/__init__.py",
    }
    assert provenance["direct_url"]["url"] == Path(record["path"]).resolve().as_uri()
    assert witness._direct_url_archive_sha256(provenance["direct_url"]) == wheel_b_sha256


def test_pinned_xformers_install_rejects_digest_before_pip(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    called = []

    def download(_url, destination):
        Path(destination).write_bytes(b"substituted-wheel")

    with pytest.raises(RuntimeError, match="xformers wheel digest mismatch"):
        witness._install_pinned_xformers(
            python="/usr/bin/python3",
            work_dir=tmp_path,
            report={"setup_commands": []},
            wheel_url="https://example.invalid/xformers.whl",
            expected_sha256="0" * 64,
            downloader=download,
            runner=lambda *_args, **_kwargs: called.append(True),
        )

    assert called == []


def test_native_image_xformers_provenance_owns_imports_from_exact_wheel(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness

    wheel_path = tmp_path / witness.XFORMERS_WHEEL_FILENAME
    wheel_path.write_bytes(b"exact-wheel")
    wheel_sha256 = _sha256(wheel_path)
    site_root = tmp_path / "site-packages"
    package = site_root / "xformers"
    ops_package = package / "ops"
    ops_package.mkdir(parents=True)
    xformers_path = package / "__init__.py"
    ops_path = ops_package / "__init__.py"
    xformers_path.write_text("")
    ops_path.write_text("")

    class Distribution:
        metadata = {"Name": "xformers"}
        version = "0.0.35"
        files = [Path("xformers/__init__.py"), Path("xformers/ops/__init__.py")]

        def locate_file(self, relative):
            return site_root / relative

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": wheel_path.resolve().as_uri(),
                    "archive_info": {"hashes": {"sha256": wheel_sha256}},
                }
            )

    record = witness.read_xformers_install_provenance(
        types.SimpleNamespace(
            __name__="xformers", __version__="0.0.35", __file__=str(xformers_path)
        ),
        types.SimpleNamespace(__name__="xformers.ops", __file__=str(ops_path)),
        wheel_path=wheel_path,
        expected_sha256=wheel_sha256,
        distribution_loader=lambda _name: Distribution(),
    )

    assert record["wheel_sha256"] == wheel_sha256
    assert record["module_paths"] == {
        "xformers": str(xformers_path.resolve()),
        "xformers.ops": str(ops_path.resolve()),
    }


@pytest.mark.parametrize("mutation", ["unknown_untracked", "tracked_modified"])
def test_post_build_cleanup_rejects_unattributable_source_changes_without_deleting_them(
    tmp_path, mutation
):
    from scripts.source_cuda_native_image_to_glb_witness import (
        _remove_expected_build_products,
    )

    root = tmp_path / "nvdiffrast"
    _write_clean_git_checkout(root)
    (root / "build").mkdir()
    (root / "build" / "extension.o").write_bytes(b"object")
    if mutation == "unknown_untracked":
        evidence = root / "unexpected.txt"
        evidence.write_text("not a build product\n")
    else:
        evidence = root / "setup.py"
        evidence.write_text("# source mutation\n")

    with pytest.raises(RuntimeError, match="unattributable post-build source changes"):
        _remove_expected_build_products(
            root,
            allowed_roots=("build", "nvdiffrast.egg-info"),
        )

    assert evidence.exists()
    assert (root / "build" / "extension.o").exists()


def test_parser_defaults_bind_the_authorized_native_route():
    from scripts.source_cuda_native_image_to_glb_witness import (
        CUMESH_COMMIT,
        CUMESH_REPOSITORY,
        DINOV3_REVISION,
        EXPECTED_TORCH_VERSION,
        MODEL_REVISION,
        NVDIFFRAST_COMMIT,
        REMBG_REVISION,
        SPARSE_DECODER_REVISION,
        TRELLIS_COMMIT,
        TRELLIS_REPOSITORY,
        build_parser,
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "--image",
            "image.png",
            "--expected-image-sha256",
            "a" * 64,
            "--output-dir",
            "outputs",
            "--work-dir",
            "runtime",
        ]
    )

    assert TRELLIS_REPOSITORY == "https://github.com/microsoft/TRELLIS.2.git"
    assert TRELLIS_COMMIT == "5565d240c4a494caaf9ece7a554542b76ffa36d3"
    assert CUMESH_REPOSITORY == "https://github.com/JeffreyXiang/CuMesh.git"
    assert CUMESH_COMMIT == "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87"
    assert EXPECTED_TORCH_VERSION == "2.10.0+cu128"
    assert MODEL_REVISION == "af44b45f2e35a493886929c6d786e563ec68364d"
    assert SPARSE_DECODER_REVISION == "25e0d31ffbebe4b5a97464dd851910efc3002d96"
    assert DINOV3_REVISION == "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    assert REMBG_REVISION == "5df4c9c76d8170882c34f6986e848ee07fd0ba43"
    assert NVDIFFRAST_COMMIT == "253ac4fcea7de5f396371124af597e6cc957bfae"
    assert args.pipeline_type == "512"
    assert args.seed == 42
    assert args.steps == 8
    assert args.target_faces == 350000
    assert args.texture_size == 1024
    assert args.attention_backend == "xformers"
    assert args.sparse_conv_backend == "flex_gemm"


def test_wrong_image_digest_fails_before_touching_existing_output(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import main

    image = tmp_path / "image.png"
    image.write_bytes(b"not-the-authorized-image")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    stale = output_dir / "consumer.glb"
    stale.write_bytes(b"preserve-until-request-validates")

    args = _base_args(tmp_path, image)
    args[args.index("--expected-image-sha256") + 1] = "0" * 64
    rc = main(args + ["--no-download"])

    report = json.loads((output_dir / "report.json").read_text())
    assert rc == 1
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "arguments_parsed"
    assert report["primary_output_status"] == "not_attempted"
    assert "image SHA256" in report["error"]
    assert stale.read_bytes() == b"preserve-until-request-validates"


def test_no_download_preflight_records_exact_route_without_primary_artifacts(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import main

    image = tmp_path / "image.png"
    image.write_bytes(b"authorized-image-fixture")
    args = _base_args(tmp_path, image)

    rc = main(args + ["--no-download"])

    output_dir = tmp_path / "outputs"
    report = json.loads((output_dir / "report.json").read_text())
    assert rc == 0
    assert report["status"] == "preflight_stopped"
    assert report["failure_phase"] is None
    assert report["last_trustworthy_phase"] == "request_validated"
    assert report["primary_output_status"] == "not_written_no_download"
    assert report["effective_route"]["device_type"] == "not_loaded_no_download"
    assert report["effective_route"]["trellis_source_clean"] == "not_checked_no_download"
    assert report["requested_route"]["native_conditioning"] is True
    assert report["requested_route"]["native_rng"] is True
    assert report["requested_route"]["pipeline_run_called_once"] is True
    assert report["requested_route"]["sampler_steps"] == {
        "shape": 8,
        "sparse": 8,
        "texture": 8,
    }
    assert report["requested_route"]["postprocess"] == {
        "decimation_target": 350000,
        "remesh": False,
        "texture_size": 1024,
    }
    assert report["expected_capture_order"] == list(EXPECTED_STAGES)
    assert list(output_dir.iterdir()) == [output_dir / "report.json"]


def test_recorder_publishes_each_boundary_to_progress_report(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import ArtifactRecorder

    observations = []

    def publish(recorder):
        observations.append(
            {
                "capture_order": list(recorder.capture_order),
                "artifacts": dict(recorder.artifacts),
            }
        )

    recorder = ArtifactRecorder(tmp_path / "outputs", on_capture=publish)
    recorder.save_image(
        "preprocessed_image",
        type(
            "FakeImage",
            (),
            {
                "mode": "RGBA",
                "size": (2, 2),
                "save": lambda self, path, format: Path(path).write_bytes(b"png"),
            },
        )(),
    )

    assert observations == [
        {
            "capture_order": ["preprocessed_image"],
            "artifacts": {
                "preprocessed_image": recorder.artifacts["preprocessed_image"]
            },
        }
    ]


def test_orientation_observer_does_not_mutate_native_extension_class():
    from scripts.source_cuda_native_image_to_glb_witness import (
        _restore_orientation_observer,
        install_orientation_observer,
    )

    class ImmutableNativeType(type):
        def __setattr__(cls, name, value):
            raise TypeError("immutable extension type")

    class NativeCuMesh(metaclass=ImmutableNativeType):
        def read(self):
            return [1, 2, 3], [[0, 1, 2]]

        def unify_face_orientations(self):
            return "native-return"

    class Recorder:
        def __init__(self):
            self.stages = []

        def save_npz(self, stage, arrays):
            self.stages.append((stage, arrays))

    module = types.SimpleNamespace(CuMesh=NativeCuMesh)
    recorder = Recorder()

    observer = install_orientation_observer(module, recorder)
    observed = module.CuMesh()
    result = observed.unify_face_orientations()
    _restore_orientation_observer(observer)

    assert result == "native-return"
    assert module.CuMesh is NativeCuMesh
    assert [stage for stage, _ in recorder.stages] == [
        "postprocess_stage11_pre_orientation",
        "postprocess_stage12_post_orientation",
    ]
    assert observer["state"] == {
        "call_count": 1,
        "native_method_return_preserved": True,
        "pre_readback_written": True,
        "post_readback_written": True,
    }


def test_actual_kaggle_runner_reaches_witness_request_validation(tmp_path, monkeypatch):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket
    from scripts import source_cuda_native_image_to_glb_witness as witness

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    image = capsule / "9_img.png"
    image.write_bytes(b"synthetic-transport-contract-image")
    run_id = "22222222-2222-4222-8222-222222222222"
    image_sha256 = _sha256(image)
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = witness.prepare_native_image_to_glb_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / "packet",
            dataset_id="operator/native-image-anchor-inputs",
            kernel_id="operator/native-image-anchor-cuda",
            title="Native Image Anchor CUDA",
            entrypoint=entrypoint.name,
            inputs=(entrypoint.name, image.name),
            run_id=run_id,
            expected_image_sha256=image_sha256,
            output_json="report.json",
            output_npz=None,
            expected_outputs=expected_outputs,
            entrypoint_args=(
                "--image",
                image.name,
                "--expected-image-sha256",
                image_sha256,
                "--run-id",
                run_id,
                "--output-dir",
                ".",
                "--work-dir",
                str(tmp_path / "runtime"),
                "--no-download",
            ),
        )
    )
    fake_torch = types.SimpleNamespace(
        __version__="2.10.0+cu128",
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Tesla T4",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    runner = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text().replace(
        'Path("/kaggle/input")',
        f"Path({str(packet.dataset_dir)!r})",
    )
    namespace = {"__name__": "runner_test"}
    exec(runner, namespace)

    rc = namespace["main"]()

    report = json.loads((work / "report.json").read_text())
    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc != 0
    assert report["status"] == "preflight_stopped"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert receipt["failure_phase"] == "output"
    assert receipt["run_id"] == run_id
    assert receipt["expected_image_sha256"] == image_sha256
    assert receipt["effective_command"][:3] == [
        sys.executable,
        entrypoint.name,
        "--output-json",
    ]
    assert set(packet.expected_outputs).issuperset(expected_outputs)
    assert receipt["effective_command"].count("--run-id") == 1
    assert receipt["effective_command"][receipt["effective_command"].index("--run-id") + 1] == run_id


def test_native_packet_rejects_missing_attempt_identities_before_output_mutation(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        prepare_packet,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    image = capsule / "9_img.png"
    image.write_bytes(b"native-packet-input")
    output_dir = tmp_path / "packet"
    output_dir.mkdir()
    marker = output_dir / "must-survive-identity-rejection.txt"
    marker.write_text("preserved")
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-image-anchor-inputs",
        kernel_id="operator/native-image-anchor-cuda",
        title="Native Image Anchor CUDA",
        entrypoint=entrypoint.name,
        inputs=(entrypoint.name, image.name),
        output_json="report.json",
        output_npz=None,
        expected_outputs=expected_outputs,
        entrypoint_args=(
            "--image",
            image.name,
            "--output-dir",
            ".",
            "--work-dir",
            str(tmp_path / "runtime"),
        ),
    )
    # On the reviewed parent there is no native constructor, so this exercises
    # the actual generic route and records its mutation as the fail-first witness.
    prepare_native = getattr(
        witness,
        "prepare_native_image_to_glb_packet",
        prepare_packet,
    )

    with pytest.raises(WitnessPacketError, match="run identity.*image identity"):
        prepare_native(packet)

    assert marker.read_text() == "preserved"


@pytest.mark.parametrize(
    ("declared_output_dir", "declared_work_dir", "assignment_form"),
    (
        (".", "/kaggle/working/native-pixal9-runtime", False),
        ("//kaggle/working", "/kaggle/working/native-pixal9-runtime", False),
        ("/kaggle/working", "//kaggle/working/native-pixal9-runtime", False),
        ("outputs", "outputs/runtime", True),
        ("same", "same", False),
        ("runtime/output", "runtime", False),
    ),
)
def test_native_packet_rejects_remote_output_work_overlap_before_output_mutation(
    tmp_path,
    declared_output_dir,
    declared_work_dir,
    assignment_form,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    image = capsule / "9_img.png"
    image.write_bytes(b"native-packet-input")
    image_sha256 = _sha256(image)
    output_dir = tmp_path / "packet"
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    path_arguments = (
        (
            f"--output-dir={declared_output_dir}",
            f"--work-dir={declared_work_dir}",
        )
        if assignment_form
        else (
            "--output-dir",
            declared_output_dir,
            "--work-dir",
            declared_work_dir,
        )
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-image-anchor-inputs",
        kernel_id="operator/native-image-anchor-cuda",
        title="Native Image Anchor CUDA",
        entrypoint=entrypoint.name,
        inputs=(entrypoint.name, image.name),
        output_json="report.json",
        output_npz=None,
        expected_outputs=expected_outputs,
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        expected_image_sha256=image_sha256,
        entrypoint_args=(
            "--image",
            image.name,
            "--expected-image-sha256",
            image_sha256,
            "--run-id",
            "31fce6b7-853b-4a0f-b99d-518be23ebabc",
            *path_arguments,
        ),
    )

    with pytest.raises(WitnessPacketError, match="Kaggle output and work directories overlap"):
        witness.prepare_native_image_to_glb_packet(packet)

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("declared_output_dir", "declared_work_dir"),
    (("outputs", "runtime"), ("outputs", "outputs-archive"), ("-1", "runtime")),
)
def test_native_packet_accepts_disjoint_assignment_form_paths(
    tmp_path,
    declared_output_dir,
    declared_work_dir,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    image = capsule / "9_img.png"
    image.write_bytes(b"native-packet-input")
    image_sha256 = _sha256(image)
    output_dir = tmp_path / "packet"
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-image-anchor-inputs",
        kernel_id="operator/native-image-anchor-cuda",
        title="Native Image Anchor CUDA",
        entrypoint=entrypoint.name,
        inputs=(entrypoint.name, image.name),
        output_json="report.json",
        output_npz=None,
        expected_outputs=expected_outputs,
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        expected_image_sha256=image_sha256,
        entrypoint_args=(
            "--image",
            image.name,
            "--expected-image-sha256",
            image_sha256,
            "--run-id",
            "31fce6b7-853b-4a0f-b99d-518be23ebabc",
            f"--output-dir={declared_output_dir}",
            f"--work-dir={declared_work_dir}",
        ),
    )

    assert witness.prepare_native_image_to_glb_packet(packet) is packet
    assert output_dir.is_dir()


@pytest.mark.parametrize(
    "path_arguments",
    (
        ("--output-dir", "--work-dir", "runtime"),
        ("--output-dir", "-x", "--work-dir", "runtime"),
        ("--", "--output-dir", "outputs", "--work-dir", "runtime"),
        (
            "--output-dir",
            "outputs",
            "--output-dir=other",
            "--work-dir",
            "runtime",
        ),
    ),
)
def test_native_packet_rejects_malformed_path_arguments_before_output_mutation(
    tmp_path,
    path_arguments,
):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
    )

    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    image = capsule / "9_img.png"
    image.write_bytes(b"native-packet-input")
    image_sha256 = _sha256(image)
    output_dir = tmp_path / "packet"
    output_dir.mkdir()
    marker = output_dir / "must-survive-argument-rejection.txt"
    marker.write_text("preserved")
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-image-anchor-inputs",
        kernel_id="operator/native-image-anchor-cuda",
        title="Native Image Anchor CUDA",
        entrypoint=entrypoint.name,
        inputs=(entrypoint.name, image.name),
        output_json="report.json",
        output_npz=None,
        expected_outputs=expected_outputs,
        run_id="31fce6b7-853b-4a0f-b99d-518be23ebabc",
        expected_image_sha256=image_sha256,
        entrypoint_args=(
            "--image",
            image.name,
            "--expected-image-sha256",
            image_sha256,
            "--run-id",
            "31fce6b7-853b-4a0f-b99d-518be23ebabc",
            *path_arguments,
        ),
    )

    with pytest.raises(WitnessPacketError):
        witness.prepare_native_image_to_glb_packet(packet)

    assert marker.read_text() == "preserved"


@pytest.mark.parametrize(
    "mutated_role",
    ("image", "model.safetensors", "config.json", "preprocessor_config.json"),
)
def test_admitted_inputs_reject_requested_path_substitution_before_use(
    tmp_path,
    mutated_role,
):
    from scripts.source_cuda_native_image_to_glb_witness import (
        admit_run_inputs,
        verify_admitted_inputs_before_use,
    )

    image = tmp_path / "image.png"
    image.write_bytes(b"image-authority")
    dino = tmp_path / "dino"
    dino.mkdir()
    dino_files = {
        "model.safetensors": b"model-authority",
        "config.json": b"config-authority",
        "preprocessor_config.json": b"preprocessor-authority",
    }
    for name, payload in dino_files.items():
        (dino / name).write_bytes(payload)
    args = types.SimpleNamespace(
        image=image,
        expected_image_sha256=_sha256(image),
        dinov3_model_path=dino,
        work_dir=tmp_path / "runtime",
        run_id="11111111-1111-4111-8111-111111111111",
    )
    report = {}
    admitted = admit_run_inputs(
        args,
        report,
        expected_dinov3_files={name: _sha256(dino / name) for name in dino_files},
    )
    requested = image if mutated_role == "image" else dino / mutated_role
    requested.write_bytes(b"substituted-after-admission")

    with pytest.raises(RuntimeError, match=mutated_role):
        verify_admitted_inputs_before_use(
            admitted,
            report,
            expected_dinov3_files={name: _sha256(admitted.dinov3_model_path / name) for name in dino_files},
        )

    assert Path(admitted.image).read_bytes() == b"image-authority"
    for name, payload in dino_files.items():
        assert (Path(admitted.dinov3_model_path) / name).read_bytes() == payload


def _write_completed_fixture(
    tmp_path: Path,
    *,
    run_id: str = "11111111-1111-4111-8111-111111111111",
    image_sha256: str = "a" * 64,
) -> Path:
    from scripts.source_cuda_native_image_to_glb_witness import (
        CUMESH_COMMIT,
        CUMESH_REPOSITORY,
        DINOV3_FILES,
        DINOV3_REPOSITORY,
        DINOV3_REVISION,
        FLEX_GEMM_COMMIT,
        FLEX_GEMM_REPOSITORY,
        MODEL_PIPELINE_SHA256,
        MODEL_REPOSITORY,
        MODEL_REVISION,
        NVDIFFRAST_COMMIT,
        NVDIFFRAST_REPOSITORY,
        REMBG_REPOSITORY,
        REMBG_REVISION,
        SPARSE_DECODER_REPOSITORY,
        SPARSE_DECODER_REVISION,
        TRELLIS_COMMIT,
        TRELLIS_REPOSITORY,
    )
    from PIL import Image
    import trimesh

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True)
    coords = np.asarray(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=np.int32,
    )
    voxel_coords = coords[:, 1:].copy()
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=np.int32,
    )
    arrays_by_stage = {
        "conditioning_512": {
            "cond": np.ones((1, 2, 3), dtype=np.float32),
            "neg_cond": np.zeros((1, 2, 3), dtype=np.float32),
        },
        "sparse_flow": {
            "noise": np.ones((1, 2, 2, 2, 2), dtype=np.float32),
            "sample_next": np.ones((8, 1, 2, 2, 2, 2), dtype=np.float32),
            "pred_x0": np.zeros((8, 1, 2, 2, 2, 2), dtype=np.float32),
        },
        "sparse_support": {"coords": coords},
        "shape_flow": {
            "noise_feats": features,
            "noise_coords": coords,
            "coords": coords,
            "sample_next": np.repeat(features[None], 8, axis=0),
            "pred_x0": np.repeat((features + 1)[None], 8, axis=0),
        },
        "shape_slat": {"shape_slat_feats": features, "shape_slat_coords": coords},
        "texture_flow": {
            "noise_feats": features,
            "noise_coords": coords,
            "coords": coords,
            "sample_next": np.repeat(features[None], 8, axis=0),
            "pred_x0": np.repeat((features + 1)[None], 8, axis=0),
        },
        "decoder_raw_mesh": {"vertices": vertices, "faces": faces},
        "texture_voxels": {
            "texture_voxels_feats": features,
            "texture_voxels_coords": coords,
        },
        "pipeline_filled_mesh": {
            "vertices": vertices,
            "faces": faces,
            "texture_coords": voxel_coords,
            "texture_attrs": np.ones((4, 6), dtype=np.float32),
        },
        "postprocess_stage11_pre_orientation": {"vertices": vertices, "faces": faces},
        "postprocess_stage12_post_orientation": {"vertices": vertices, "faces": faces},
    }
    artifacts = {}
    for index, stage in enumerate(EXPECTED_STAGES):
        suffix = ".png" if stage == "preprocessed_image" else ".glb" if stage == "consumer_glb" else ".npz"
        path = output_dir / f"{index:02d}-{stage}{suffix}"
        metadata = {}
        if stage == "preprocessed_image":
            image = Image.new("RGB", (2, 2), (80, 120, 160))
            image.save(path, format="PNG")
            metadata = {"mode": "RGB", "size": [2, 2]}
        elif stage == "consumer_glb":
            uv = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=Image.new("RGB", (2, 2), (200, 100, 50))
            )
            visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                visual=visual,
                process=False,
            )
            path.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))
        else:
            arrays = arrays_by_stage[stage]
            np.savez(path, **arrays)
            metadata = {
                "arrays": {
                    name: {
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                    }
                    for name, value in arrays.items()
                }
            }
        artifacts[stage] = {
            "path": path.name,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "run_id": run_id,
            **metadata,
        }

    report = {
        "schema": "trellis2mlx.source_cuda_native_image_to_glb.v1",
        "run_id": run_id,
        "status": "completed",
        "failure_phase": None,
        "last_trustworthy_phase": "consumer_glb_validated",
        "primary_output_status": "validated",
        "xformers_wheel": {
            "version": "0.0.35",
            "url": (
                "https://files.pythonhosted.org/packages/a4/85/"
                "6d71f9b16f2ac647877e66ed4af723b3fbd477806ab8b8a89d39a362b85f/"
                "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
            ),
            "path": (
                "/run/pinned-wheels/"
                "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
            ),
            "sha256": "ccc73c7db9890224ab05f5fb60e2034f9e6c8672a10be0cf00e95cbbae3eda7c",
            "size_bytes": 3264751,
            "install_mode": "forced-local-wheel-no-deps",
            "pip_version": "pip 25.1 from /run/site-packages/pip (python 3.12)",
        },
        "effective_route": {
            "device_type": "cuda",
            "cuda_device_name": "Tesla T4",
            "torch_version": "2.10.0+cu128",
            "xformers_build_identity": {
                "version": "0.0.35",
                "torch": "2.10.0+cu128",
                "cuda": 1208,
                "torch_cuda_arch_list": "7.5 8.0+PTX 8.0 9.0a",
                "package_from": "wheel-v0.0.35",
            },
            "xformers_import_provenance": {
                "mode": "pep610-local-wheel",
                "wheel_path": (
                    "/run/pinned-wheels/"
                    "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
                ),
                "wheel_sha256": (
                    "ccc73c7db9890224ab05f5fb60e2034f9e6c8672a10be0cf00e95cbbae3eda7c"
                ),
                "distribution_name": "xformers",
                "distribution_version": "0.0.35",
                "distribution_root": "/run/site-packages",
                "module_paths": {
                    "xformers": "/run/site-packages/xformers/__init__.py",
                    "xformers.ops": "/run/site-packages/xformers/ops/__init__.py",
                },
                "distribution_files": {
                    "xformers": "xformers/__init__.py",
                    "xformers.ops": "xformers/ops/__init__.py",
                },
                "direct_url": {
                    "url": (
                        "file:///run/pinned-wheels/"
                        "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
                    ),
                    "archive_info": {
                        "hashes": {
                            "sha256": (
                                "ccc73c7db9890224ab05f5fb60e2034f9e6c8672a10be0cf00e95cbbae3eda7c"
                            )
                        }
                    },
                },
            },
            "attention_backend": "xformers",
            "sparse_attention_backend": "xformers",
            "sparse_conv_backend": "flex_gemm",
            "trellis_commit": TRELLIS_COMMIT,
            "trellis_source_clean": True,
            "cumesh_commit": CUMESH_COMMIT,
            "cumesh_source_clean_before_build": True,
            "nvdiffrast_commit": "253ac4fcea7de5f396371124af597e6cc957bfae",
            "model_revision": "af44b45f2e35a493886929c6d786e563ec68364d",
            "sparse_decoder_revision": "25e0d31ffbebe4b5a97464dd851910efc3002d96",
            "dinov3_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
            "rembg_revision": "5df4c9c76d8170882c34f6986e848ee07fd0ba43",
            "pipeline_type": "512",
            "seed": 42,
            "sampler_steps": {"sparse": 8, "shape": 8, "texture": 8},
            "pipeline_run_call_count": 1,
            "native_conditioning": True,
            "native_rng": True,
            "observation_only_instrumentation": True,
            "run_id": run_id,
        },
        "requested_route": {"run_id": run_id, "image_sha256": image_sha256},
        "effective_inputs": {
            "run_id": run_id,
            "image": {"path": "/run/image.png", "sha256": image_sha256, "size_bytes": 100},
            "dinov3": {
                "path": "/run/dinov3",
                "files": {
                    name: {"path": f"/run/dinov3/{name}", "sha256": digest, "size_bytes": 100}
                    for name, digest in DINOV3_FILES.items()
                },
            },
        },
        "model_assets": {
            "trellis": {
                "repository": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "pipeline_json_sha256": MODEL_PIPELINE_SHA256,
            },
            "sparse_decoder": {
                "repository": SPARSE_DECODER_REPOSITORY,
                "revision": SPARSE_DECODER_REVISION,
            },
            "dinov3": {
                "repository": DINOV3_REPOSITORY,
                "revision": DINOV3_REVISION,
                "files": dict(DINOV3_FILES),
            },
            "rembg": {"repository": REMBG_REPOSITORY, "revision": REMBG_REVISION},
            "path_rewrite_only": True,
        },
        "source_identities_after_build": {
            "trellis": {"repository": TRELLIS_REPOSITORY, "commit": TRELLIS_COMMIT, "clean": True},
            "cumesh": {"repository": CUMESH_REPOSITORY, "commit": CUMESH_COMMIT, "clean": True},
            "flex_gemm": {"repository": FLEX_GEMM_REPOSITORY, "commit": FLEX_GEMM_COMMIT, "clean": True},
            "nvdiffrast": {"repository": NVDIFFRAST_REPOSITORY, "commit": NVDIFFRAST_COMMIT, "clean": True},
        },
        "capture_order": list(EXPECTED_STAGES),
        "orientation_observer": {
            "call_count": 1,
            "native_method_return_preserved": True,
            "pre_readback_written": True,
            "post_readback_written": True,
        },
        "artifacts": artifacts,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    return report_path


def test_completed_report_admission_reopens_every_boundary(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import (
        validate_completed_report,
    )

    report_path = _write_completed_fixture(tmp_path)
    admitted = validate_completed_report(report_path)

    assert admitted["status"] == "completed"
    assert admitted["primary_output_status"] == "validated"
    assert admitted["capture_order"] == list(EXPECTED_STAGES)


def test_completed_report_rejects_plaintext_npz_and_glb_bytes(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    record = report["artifacts"]["shape_flow"]
    path = report_path.parent / record["path"]
    path.write_bytes(b"plain text pretending to be npz")
    record["sha256"] = _sha256(path)
    record["size_bytes"] = path.stat().st_size
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="NPZ|GLB|PNG|artifact structure"):
        validate_completed_report(report_path)


def test_completed_report_rejects_malformed_glb_even_with_matching_receipt(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    record = report["artifacts"]["consumer_glb"]
    path = report_path.parent / record["path"]
    path.write_bytes(b"plain text pretending to be a glb")
    record["sha256"] = _sha256(path)
    record["size_bytes"] = path.stat().st_size
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="GLB"):
        validate_completed_report(report_path)


def test_completed_report_rejects_duplicate_stage_path_even_with_matching_receipt(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    duplicate = report["artifacts"]["shape_flow"]
    report["artifacts"]["texture_flow"] = dict(duplicate)
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="distinct|duplicate"):
        validate_completed_report(report_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_array", "NPZ keys"),
        ("retyped_array", "NPZ dtype"),
        ("empty_array", "NPZ array is empty"),
    ],
)
def test_completed_report_rejects_invalid_npz_array_contract(tmp_path, mutation, match):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    record = report["artifacts"]["conditioning_512"]
    path = report_path.parent / record["path"]
    with np.load(path, allow_pickle=False) as reopened:
        arrays = {name: np.ascontiguousarray(reopened[name]) for name in reopened.files}
    if mutation == "missing_array":
        arrays.pop("neg_cond")
    elif mutation == "retyped_array":
        arrays["cond"] = arrays["cond"].astype(np.float64)
    elif mutation == "empty_array":
        arrays["cond"] = np.empty((0, 2, 3), dtype=np.float32)
    np.savez(path, **arrays)
    record["sha256"] = _sha256(path)
    record["size_bytes"] = path.stat().st_size
    record["arrays"] = {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
        for name, value in arrays.items()
    }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_completed_report(report_path)


def test_completed_report_rejects_stale_bundle_from_another_run(tmp_path):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)

    with pytest.raises(ValueError, match="run identity mismatch"):
        validate_completed_report(
            report_path,
            expected_run_id="22222222-2222-4222-8222-222222222222",
        )


def test_current_packet_consumer_rejects_complete_prior_attempt_bundle(tmp_path):
    from scripts import source_cuda_native_image_to_glb_witness as witness
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        sha256_file,
    )

    old_run_id = "11111111-1111-4111-8111-111111111111"
    current_run_id = "22222222-2222-4222-8222-222222222222"
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    entrypoint = capsule / "source_cuda_native_image_to_glb_witness.py"
    entrypoint.write_text(Path(witness.__file__).read_text())
    image = capsule / "9_img.png"
    image.write_bytes(b"packet-owned-image-authority")
    image_sha256 = _sha256(image)
    expected_outputs = tuple(
        f"{index:02d}-{stage}{'.png' if stage == 'preprocessed_image' else '.glb' if stage == 'consumer_glb' else '.npz'}"
        for index, stage in enumerate(EXPECTED_STAGES)
    )

    def packet_for(run_id, name):
        return witness.prepare_native_image_to_glb_packet(
            KaggleCudaWitnessPacket(
                capsule_dir=capsule,
                output_dir=tmp_path / name,
                dataset_id="operator/native-image-anchor-inputs",
                kernel_id="operator/native-image-anchor-cuda",
                title="Native Image Anchor CUDA",
                entrypoint=entrypoint.name,
                inputs=(entrypoint.name, image.name),
                run_id=run_id,
                expected_image_sha256=image_sha256,
                output_json="report.json",
                output_npz=None,
                expected_outputs=expected_outputs,
                entrypoint_args=(
                    "--image",
                    image.name,
                    "--expected-image-sha256",
                    image_sha256,
                    "--run-id",
                    run_id,
                    "--output-dir",
                    ".",
                    "--work-dir",
                    str(tmp_path / f"runtime-{name}"),
                ),
            )
        )

    old_packet = packet_for(old_run_id, "old-packet")
    current_packet = packet_for(current_run_id, "current-packet")
    report_path = _write_completed_fixture(
        tmp_path / "old-bundle",
        run_id=old_run_id,
        image_sha256=image_sha256,
    )
    output_dir = report_path.parent
    def write_receipt(packet, bundle_dir):
        receipt_outputs = {
            name: {
                "exists": True,
                "sha256": sha256_file(bundle_dir / name),
                "size_bytes": (bundle_dir / name).stat().st_size,
            }
            for name in packet.outputs
        }
        manifest = packet.dataset_dir / "witness-manifest.json"
        receipt = {
            "schema": "trellis2mlx.kaggle_cuda_witness.receipt.v1",
            "status": "done",
            "failure_phase": None,
            "requested_dataset_id": packet.dataset_id,
            "requested_kernel_id": packet.kernel_id,
            "requested_accelerator": packet.accelerator,
            "source_identity": {
                "dataset_sources": [packet.dataset_id],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            },
            "run_id": packet.run_id,
            "expected_image_sha256": packet.expected_image_sha256,
            "cuda_available": True,
            "cuda_device": "Tesla T4",
            "input_manifest": {
                "sha256": sha256_file(manifest),
                "size_bytes": manifest.stat().st_size,
            },
            "outputs": receipt_outputs,
        }
        (bundle_dir / "kaggle_cuda_witness_receipt.json").write_text(
            json.dumps(receipt, sort_keys=True) + "\n"
        )

    write_receipt(old_packet, output_dir)

    with pytest.raises(WitnessPacketError, match="run identity|run_id"):
        witness.validate_downloaded_native_image_to_glb_outputs(
            current_packet,
            output_dir,
        )

    current_report_path = _write_completed_fixture(
        tmp_path / "current-bundle",
        run_id=current_run_id,
        image_sha256=image_sha256,
    )
    write_receipt(current_packet, current_report_path.parent)
    admitted = witness.validate_downloaded_native_image_to_glb_outputs(
        current_packet,
        current_report_path.parent,
    )
    assert admitted["report"]["status"] == "completed"
    assert admitted["report"]["run_id"] == current_run_id


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_model_assets", "model assets"),
        ("wrong_effective_image", "effective image"),
        ("missing_run_id", "run identity"),
        ("missing_xformers_wheel", "xformers wheel"),
        ("missing_xformers_pip_identity", "effective pip identity"),
        ("missing_xformers_provenance", "xformers import provenance"),
        ("foreign_xformers_module", "xformers import provenance|module"),
        ("foreign_xformers_wheel_path", "xformers import provenance|wheel path"),
        ("foreign_xformers_wheel_digest", "xformers import provenance|wheel digest"),
    ],
)
def test_completed_report_rejects_missing_or_wrong_authority(tmp_path, mutation, match):
    from scripts.source_cuda_native_image_to_glb_witness import validate_completed_report

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    if mutation == "missing_model_assets":
        report.pop("model_assets", None)
    elif mutation == "wrong_effective_image":
        report.setdefault("effective_inputs", {})["image"] = {
            "sha256": "0" * 64,
            "size_bytes": 1,
        }
    elif mutation == "missing_run_id":
        report.pop("run_id", None)
    elif mutation == "missing_xformers_wheel":
        report.pop("xformers_wheel", None)
    elif mutation == "missing_xformers_pip_identity":
        report["xformers_wheel"].pop("pip_version", None)
    elif mutation == "missing_xformers_provenance":
        report["effective_route"].pop("xformers_import_provenance", None)
    elif mutation == "foreign_xformers_module":
        report["effective_route"]["xformers_import_provenance"]["module_paths"][
            "xformers"
        ] = "/tmp/foreign/xformers/__init__.py"
    elif mutation == "foreign_xformers_wheel_path":
        report["effective_route"]["xformers_import_provenance"]["direct_url"]["url"] = (
            "file:///tmp/foreign/xformers-0.0.35.whl"
        )
    elif mutation == "foreign_xformers_wheel_digest":
        report["effective_route"]["xformers_import_provenance"]["direct_url"][
            "archive_info"
        ]["hashes"]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_completed_report(report_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_stage", "capture order"),
        ("wrong_digest", "digest mismatch"),
        ("wrong_device", "device_type"),
        ("dirty_source", "trellis_source_clean"),
        ("orientation_not_observed", "orientation observer"),
    ],
)
def test_completed_report_admission_rejects_false_closure(tmp_path, mutation, match):
    from scripts.source_cuda_native_image_to_glb_witness import (
        validate_completed_report,
    )

    report_path = _write_completed_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    if mutation == "missing_stage":
        report["capture_order"].remove("shape_flow")
    elif mutation == "wrong_digest":
        report["artifacts"]["shape_flow"]["sha256"] = "0" * 64
    elif mutation == "wrong_device":
        report["effective_route"]["device_type"] = "cpu"
    elif mutation == "dirty_source":
        report["effective_route"]["trellis_source_clean"] = False
    elif mutation == "orientation_not_observed":
        report["orientation_observer"]["call_count"] = 0
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        validate_completed_report(report_path)
