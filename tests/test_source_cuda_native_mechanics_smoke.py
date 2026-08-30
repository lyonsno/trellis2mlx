from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/source_cuda_native_mechanics_smoke.py"
RUN_ID = "11111111-1111-4111-8111-111111111111"
OTHER_RUN_ID = "22222222-2222-4222-8222-222222222222"
EXPECTED_CAPTURE_ORDER = (
    "postprocess_stage11_pre_orientation",
    "postprocess_stage12_post_orientation",
)
XFORMERS_SHA256 = "ccc73c7db9890224ab05f5fb60e2034f9e6c8672a10be0cf00e95cbbae3eda7c"


def xformers_wheel_path(runtime_root: Path) -> Path:
    return (
        runtime_root
        / "pinned-wheels"
        / "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
    ).resolve()


def xformers_provenance(runtime_root: Path, site_root: Path) -> dict:
    wheel_path = xformers_wheel_path(runtime_root)
    module_paths = {
        "xformers": (site_root / "xformers" / "__init__.py").resolve(),
        "xformers.ops": (site_root / "xformers" / "ops" / "__init__.py").resolve(),
    }
    return {
        "mode": "pep610-local-wheel",
        "wheel_path": str(wheel_path),
        "wheel_sha256": XFORMERS_SHA256,
        "distribution_name": "xformers",
        "distribution_version": "0.0.35",
        "distribution_root": str(site_root.resolve()),
        "module_paths": {name: str(path) for name, path in module_paths.items()},
        "distribution_files": {
            name: str(path.relative_to(site_root.resolve()))
            for name, path in module_paths.items()
        },
        "direct_url": {
            "url": wheel_path.as_uri(),
            "archive_info": {"hashes": {"sha256": XFORMERS_SHA256}},
        },
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("source_cuda_native_mechanics_smoke", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCuda:
    def __init__(self, *, available=True, name="Tesla T4", capability=(7, 5)):
        self.available = available
        self.name = name
        self.capability = capability

    def is_available(self):
        return self.available

    def get_device_name(self, _index):
        return self.name

    def get_device_capability(self, _index):
        return self.capability

    def synchronize(self):
        return None


def fake_torch(*, version="2.10.0+cu128", name="Tesla T4", capability=(7, 5)):
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda="12.8"),
        cuda=FakeCuda(name=name, capability=capability),
    )


def fake_imports(roots: dict[str, Path]):
    source_root = Path(roots["source_root"]).resolve()
    runtime_root = source_root.parent
    site_root = Path("/opt/native-mechanics/site-packages")
    module_paths = {
        "xformers": site_root / "xformers" / "__init__.py",
        "xformers_ops": site_root / "xformers" / "ops" / "__init__.py",
        "cumesh": site_root / "cumesh.so",
        "flex_gemm": site_root / "flex_gemm.so",
        "o_voxel": site_root / "o_voxel.py",
        "nvdiffrast": site_root / "nvdiffrast" / "torch" / "__init__.py",
        "attention_config": source_root / "trellis2" / "modules" / "attention" / "config.py",
        "sparse_config": source_root / "trellis2" / "modules" / "sparse" / "config.py",
    }

    def distribution_record(name: str, source: Path):
        module_path = module_paths[name]
        return {
            "mode": "pep610-direct-url",
            "source_root": str(Path(source).resolve()),
            "import_name": name,
            "module_path": str(module_path),
            "distribution_name": name.replace("_", "-"),
            "distribution_version": "0.0.test",
            "distribution_root": str(site_root),
            "distribution_file": str(module_path.relative_to(site_root)),
            "direct_url": {"url": Path(source).resolve().as_uri(), "dir_info": {}},
        }

    provenance = {
        "trellis": {
            "mode": "source-tree",
            "source_root": str(source_root),
            "module_paths": {
                "attention_config": str(module_paths["attention_config"]),
                "sparse_config": str(module_paths["sparse_config"]),
            },
        },
        "cumesh": distribution_record("cumesh", roots["cumesh_root"]),
        "flex_gemm": distribution_record("flex_gemm", roots["flex_root"]),
        "o_voxel": distribution_record("o_voxel", source_root / "o-voxel"),
        "nvdiffrast": distribution_record("nvdiffrast", roots["nvdiffrast_root"]),
    }
    return {
        "xformers": SimpleNamespace(
            __name__="xformers",
            __version__="0.0.35",
            __file__=str(module_paths["xformers"]),
        ),
        "xformers_ops": SimpleNamespace(
            __name__="xformers.ops",
            __file__=str(module_paths["xformers_ops"]),
        ),
        "xformers_import_provenance": xformers_provenance(runtime_root, site_root),
        "xformers_build_identity": {
            "version": "0.0.35",
            "torch": "2.10.0+cu128",
            "cuda": 1208,
            "torch_cuda_arch_list": "7.5 8.0+PTX 8.0 9.0a",
            "package_from": "wheel-v0.0.35",
        },
        "cumesh": SimpleNamespace(__name__="cumesh", __file__=str(module_paths["cumesh"])),
        "flex_gemm": SimpleNamespace(
            __name__="flex_gemm", __file__=str(module_paths["flex_gemm"])
        ),
        "o_voxel": SimpleNamespace(
            __name__="o_voxel", __file__=str(module_paths["o_voxel"])
        ),
        "nvdiffrast": SimpleNamespace(
            __name__="nvdiffrast.torch", __file__=str(module_paths["nvdiffrast"])
        ),
        "attention_config": SimpleNamespace(
            __name__="trellis2.modules.attention.config",
            __file__=str(module_paths["attention_config"]),
            BACKEND="xformers",
        ),
        "sparse_config": SimpleNamespace(
            __name__="trellis2.modules.sparse.config",
            __file__=str(module_paths["sparse_config"]),
            ATTN="xformers",
            CONV="flex_gemm",
        ),
        "build_import_provenance": provenance,
    }


def test_distribution_provenance_binds_imported_file_to_exact_pep610_source(
    tmp_path, monkeypatch
):
    module = load_module()
    source_root = (tmp_path / "CuMesh").resolve()
    distribution_root = (tmp_path / "site-packages").resolve()
    distribution_file = Path("cumesh") / "_C.cpython-test.so"
    module_path = distribution_root / distribution_file
    imported = SimpleNamespace(__name__="cumesh", __file__=str(module_path))

    class FakeDistribution:
        version = "0.0.test"
        files = [distribution_file]
        metadata = {"Name": "cumesh"}

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({"url": source_root.as_uri(), "dir_info": {}})

        def locate_file(self, relative):
            return distribution_root / relative

    distribution = FakeDistribution()
    monkeypatch.setattr(
        module.importlib_metadata,
        "packages_distributions",
        lambda: {"cumesh": ["cumesh"]},
    )
    monkeypatch.setattr(
        module.importlib_metadata,
        "distribution",
        lambda _name: distribution,
    )
    monkeypatch.setattr(
        module.importlib_metadata,
        "distributions",
        lambda: [distribution],
    )

    record = module._distribution_provenance(
        imported,
        label="cumesh",
        source_root=source_root,
    )

    assert record["module_path"] == str(module_path)
    assert record["distribution_file"] == str(distribution_file)
    assert record["direct_url"]["url"] == source_root.as_uri()
    with pytest.raises(RuntimeError, match="not owned"):
        module._distribution_provenance(
            imported,
            label="cumesh",
            source_root=tmp_path / "unrelated-source",
        )


def test_validate_runtime_identity_accepts_only_exact_effective_route():
    module = load_module()
    roots = {
        "source_root": Path("/tmp/TRELLIS.2"),
        "cumesh_root": Path("/tmp/CuMesh"),
        "flex_root": Path("/tmp/FlexGEMM"),
        "nvdiffrast_root": Path("/tmp/nvdiffrast"),
    }
    imported = fake_imports(roots)

    route = module.validate_runtime_identity(fake_torch(), imported, roots)

    assert route["torch_version"] == "2.10.0+cu128"
    assert route["cuda_capability"] == [7, 5]
    assert route["attention_backend"] == "xformers"
    assert route["sparse_conv_backend"] == "flex_gemm"
    assert route["xformers_build_identity"]["torch_cuda_arch_list"].split()[0] == "7.5"
    assert route["xformers_import_provenance"]["distribution_name"] == "xformers"

    with pytest.raises(RuntimeError, match="Torch runtime drift"):
        module.validate_runtime_identity(fake_torch(version="2.9.0+cu128"), imported, roots)
    with pytest.raises(RuntimeError, match="Tesla T4"):
        module.validate_runtime_identity(fake_torch(name="Tesla P100"), imported, roots)
    with pytest.raises(RuntimeError, match="SM75"):
        module.validate_runtime_identity(fake_torch(capability=(8, 0)), imported, roots)
    imported["sparse_config"] = SimpleNamespace(ATTN="xformers", CONV="spconv")
    with pytest.raises(RuntimeError, match="sparse convolution backend fallback"):
        module.validate_runtime_identity(fake_torch(), imported, roots)


def test_xformers_build_identity_binds_exact_torch_cuda_and_sm75(tmp_path):
    module = load_module()
    package = tmp_path / "xformers"
    package.mkdir()
    module_path = package / "__init__.py"
    module_path.write_text("")
    (package / "cpp_lib.json").write_text(
        json.dumps(
            {
                "version": {"torch": "2.10.0+cu128", "cuda": 1208},
                "env": {
                    "TORCH_CUDA_ARCH_LIST": "7.5 8.0+PTX 8.0 9.0a",
                    "XFORMERS_PACKAGE_FROM": "wheel-v0.0.35",
                },
            }
        )
    )
    xformers = SimpleNamespace(__version__="0.0.35", __file__=str(module_path))

    record = module.read_xformers_build_identity(xformers)

    assert record["torch"] == "2.10.0+cu128"
    assert record["cuda"] == 1208
    assert "7.5" in record["torch_cuda_arch_list"].split()

    payload = json.loads((package / "cpp_lib.json").read_text())
    payload["version"]["torch"] = "2.9.0+cu128"
    (package / "cpp_lib.json").write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="xformers build Torch drift"):
        module.read_xformers_build_identity(xformers)


def test_xformers_import_provenance_joins_both_modules_to_exact_local_wheel(tmp_path):
    module = load_module()
    runtime_root = (tmp_path / "runtime").resolve()
    site_root = (tmp_path / "site-packages").resolve()
    wheel_path = xformers_wheel_path(runtime_root)
    wheel_path.parent.mkdir(parents=True)
    wheel_path.write_bytes(b"exact-wheel")
    wheel_sha256 = sha256(wheel_path)
    package = site_root / "xformers"
    ops_package = package / "ops"
    ops_package.mkdir(parents=True)
    xformers_path = package / "__init__.py"
    ops_path = ops_package / "__init__.py"
    xformers_path.write_text("")
    ops_path.write_text("")
    xformers = SimpleNamespace(
        __name__="xformers", __version__="0.0.35", __file__=str(xformers_path)
    )
    xformers_ops = SimpleNamespace(__name__="xformers.ops", __file__=str(ops_path))

    class FakeDistribution:
        version = "0.0.35"
        files = [Path("xformers/__init__.py"), Path("xformers/ops/__init__.py")]
        metadata = {"Name": "xformers"}

        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps(
                {
                    "url": wheel_path.as_uri(),
                    "archive_info": {"hashes": {"sha256": wheel_sha256}},
                }
            )

        def locate_file(self, relative):
            return site_root / relative

    distribution = FakeDistribution()
    record = module.read_xformers_install_provenance(
        xformers,
        xformers_ops,
        wheel_path=wheel_path,
        expected_sha256=wheel_sha256,
        distribution_loader=lambda _name: distribution,
    )

    assert record["module_paths"] == {
        "xformers": str(xformers_path),
        "xformers.ops": str(ops_path),
    }
    assert record["distribution_files"] == {
        "xformers": "xformers/__init__.py",
        "xformers.ops": "xformers/ops/__init__.py",
    }
    assert record["direct_url"]["archive_info"]["hashes"]["sha256"] == wheel_sha256

    xformers.__file__ = str(tmp_path / "foreign" / "xformers" / "__init__.py")
    with pytest.raises(RuntimeError, match="not owned"):
        module.read_xformers_install_provenance(
            xformers,
            xformers_ops,
            wheel_path=wheel_path,
            expected_sha256=wheel_sha256,
            distribution_loader=lambda _name: distribution,
        )

    xformers.__file__ = str(xformers_path)
    distribution.read_text = lambda _name: json.dumps(
        {
            "url": (tmp_path / "foreign.whl").resolve().as_uri(),
            "archive_info": {"hashes": {"sha256": wheel_sha256}},
        }
    )
    with pytest.raises(RuntimeError, match="wheel path"):
        module.read_xformers_install_provenance(
            xformers,
            xformers_ops,
            wheel_path=wheel_path,
            expected_sha256=wheel_sha256,
            distribution_loader=lambda _name: distribution,
        )

    distribution.read_text = lambda _name: json.dumps(
        {
            "url": wheel_path.as_uri(),
            "archive_info": {"hashes": {"sha256": "0" * 64}},
        }
    )
    with pytest.raises(RuntimeError, match="wheel digest"):
        module.read_xformers_install_provenance(
            xformers,
            xformers_ops,
            wheel_path=wheel_path,
            expected_sha256=wheel_sha256,
            distribution_loader=lambda _name: distribution,
        )


def test_run_smoke_requires_real_xformers_attention_probe_before_orientation(tmp_path):
    module = load_module()
    report_path = tmp_path / "mechanics-report.json"
    roots = {
        "source_root": tmp_path / "work" / "TRELLIS.2",
        "cumesh_root": tmp_path / "work" / "CuMesh",
        "flex_root": tmp_path / "work" / "FlexGEMM",
        "nvdiffrast_root": tmp_path / "work" / "nvdiffrast",
    }
    native = SimpleNamespace(
        TRELLIS_COMMIT="5565d240c4a494caaf9ece7a554542b76ffa36d3",
        CUMESH_COMMIT="c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
        FLEX_GEMM_COMMIT="6dd94a859c26ee8246888502eada3dd8ad85532e",
        NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae",
    )
    events = []

    def attention_probe(_torch, _ops):
        events.append("xformers")
        return {
            "executed": True,
            "device_type": "cuda",
            "finite": True,
            "shape": [1, 4, 1, 8],
            "dtype": "float16",
        }

    def orientation_probe(*_args):
        events.append("orientation")
        raise RuntimeError("stop after ordering witness")

    with pytest.raises(RuntimeError, match="stop after ordering witness"):
        module.run_smoke(
            run_id=RUN_ID,
            output_json=report_path,
            work_dir=tmp_path / "work",
            native_module=native,
            prepare=lambda *_args: roots,
            importer=lambda _roots: (fake_torch(), fake_imports(_roots)),
            attention_probe=attention_probe,
            orientation_probe=orientation_probe,
        )

    report = json.loads(report_path.read_text())
    assert events == ["xformers", "orientation"]
    assert report["effective_route"]["xformers_attention_probe"]["executed"] is True
    assert report["last_trustworthy_phase"] == "xformers_attention_executed"


def test_run_smoke_writes_failure_report_before_primary_output(tmp_path):
    module = load_module()
    report_path = tmp_path / "mechanics-report.json"
    native = SimpleNamespace(
        TRELLIS_COMMIT="5565d240c4a494caaf9ece7a554542b76ffa36d3",
        CUMESH_COMMIT="c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
        FLEX_GEMM_COMMIT="6dd94a859c26ee8246888502eada3dd8ad85532e",
        NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae",
    )

    def explode(_native, _report):
        raise RuntimeError("compile route failed")

    with pytest.raises(RuntimeError, match="compile route failed"):
        module.run_smoke(
            run_id=RUN_ID,
            output_json=report_path,
            work_dir=tmp_path / "work",
            native_module=native,
            prepare=explode,
        )

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "prepare_runtime"
    assert report["primary_output_status"] == "not_written"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert "compile route failed" in report["error"]


def test_run_smoke_rejects_partial_orientation_observation(tmp_path):
    module = load_module()
    report_path = tmp_path / "mechanics-report.json"
    native = SimpleNamespace(
        EXPECTED_TORCH_VERSION="2.10.0+cu128",
        TRELLIS_COMMIT="5565d240c4a494caaf9ece7a554542b76ffa36d3",
        CUMESH_COMMIT="c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
        FLEX_GEMM_COMMIT="6dd94a859c26ee8246888502eada3dd8ad85532e",
        NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae",
    )

    def prepare(_native, report):
        report["source_identities_after_build"] = {}
        return {
            "source_root": tmp_path / "TRELLIS.2",
            "cumesh_root": tmp_path / "CuMesh",
            "flex_root": tmp_path / "FlexGEMM",
            "nvdiffrast_root": tmp_path / "nvdiffrast",
        }

    def imports(_roots):
        return fake_torch(), fake_imports(_roots)

    def partial_probe(*_args, **_kwargs):
        return {
            "state": {
                "call_count": 1,
                "native_method_return_preserved": True,
                "pre_readback_written": True,
                "post_readback_written": False,
            },
            "artifacts": {},
        }

    with pytest.raises(RuntimeError, match="orientation observer incomplete"):
        module.run_smoke(
            run_id=RUN_ID,
            output_json=report_path,
            work_dir=tmp_path / "work",
            native_module=native,
            prepare=prepare,
            importer=imports,
            attention_probe=lambda *_args: {
                "executed": True,
                "device_type": "cuda",
                "finite": True,
                "shape": [1, 4, 1, 8],
                "dtype": "float16",
            },
            orientation_probe=partial_probe,
        )

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "orientation_probe"
    assert report["primary_output_status"] == "not_written"


def test_no_download_preflight_binds_canonical_run_without_touching_runtime(
    tmp_path, monkeypatch
):
    module = load_module()
    native = SimpleNamespace(
        TRELLIS_COMMIT="5565d240c4a494caaf9ece7a554542b76ffa36d3",
        CUMESH_COMMIT="c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
        FLEX_GEMM_COMMIT="6dd94a859c26ee8246888502eada3dd8ad85532e",
        NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae",
    )
    monkeypatch.setitem(sys.modules, "source_cuda_native_image_to_glb_witness", native)
    report_path = tmp_path / "mechanics-report.json"

    rc = module.main(
        [
            "--run-id",
            RUN_ID,
            "--output-json",
            str(report_path),
            "--work-dir",
            str(tmp_path / "runtime"),
            "--no-download",
        ]
    )

    assert rc == 0
    report = json.loads(report_path.read_text())
    assert report["run_id"] == RUN_ID
    assert report["status"] == "preflight_stopped"
    assert report["last_trustworthy_phase"] == "request_validated"
    assert report["primary_output_status"] == "not_written_no_download"
    assert report["effective_route"] == {
        "device_type": "not_loaded_no_download",
        "native_dependencies": "not_built_no_download",
    }
    assert list(tmp_path.iterdir()) == [report_path]


@pytest.mark.parametrize(
    "run_id",
    ["not-a-uuid", "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA", ""],
)
def test_cli_rejects_noncanonical_run_identity_with_durable_report(
    tmp_path, monkeypatch, run_id
):
    module = load_module()
    native = SimpleNamespace(
        TRELLIS_COMMIT="5565d240c4a494caaf9ece7a554542b76ffa36d3",
        CUMESH_COMMIT="c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
        FLEX_GEMM_COMMIT="6dd94a859c26ee8246888502eada3dd8ad85532e",
        NVDIFFRAST_COMMIT="253ac4fcea7de5f396371124af597e6cc957bfae",
    )
    monkeypatch.setitem(sys.modules, "source_cuda_native_image_to_glb_witness", native)
    report_path = tmp_path / "mechanics-report.json"

    rc = module.main(
        [
            "--run-id",
            run_id,
            "--output-json",
            str(report_path),
            "--work-dir",
            str(tmp_path / "runtime"),
            "--no-download",
        ]
    )

    assert rc == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "request_validation"
    assert report["last_trustworthy_phase"] == "arguments_parsed"
    assert "canonical" in report["error"]


def write_valid_mechanics_bundle(root: Path, *, run_id: str) -> Path:
    module = load_module()
    root.mkdir(parents=True, exist_ok=True)
    runtime_root = (root / "runtime").resolve()
    roots = {
        "source_root": runtime_root / "TRELLIS.2",
        "cumesh_root": runtime_root / "CuMesh",
        "flex_root": runtime_root / "FlexGEMM",
        "nvdiffrast_root": runtime_root / "nvdiffrast",
    }
    imported = fake_imports(roots)
    module_paths = {
        name: str(Path(imported[name].__file__).resolve())
        for name in (
            "xformers",
            "xformers_ops",
            "cumesh",
            "flex_gemm",
            "o_voxel",
            "nvdiffrast",
            "attention_config",
            "sparse_config",
        )
    }
    vertices = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    faces = np.asarray(
        [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32
    )
    artifacts = {}
    for index, stage in enumerate(EXPECTED_CAPTURE_ORDER):
        path = root / f"{index:02d}-{stage}.npz"
        np.savez(path, vertices=vertices, faces=faces)
        artifacts[stage] = {
            "run_id": run_id,
            "path": path.name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "arrays": {
                "vertices": {"dtype": "float32", "shape": [4, 3]},
                "faces": {"dtype": "int32", "shape": [4, 3]},
            },
        }
    report = {
        "schema": module.SCHEMA,
        "run_id": run_id,
        "status": "completed",
        "failure_phase": None,
        "last_trustworthy_phase": "mechanics_artifacts_reopened_and_validated",
        "primary_output_status": "validated",
        "xformers_wheel": {
            "version": "0.0.35",
            "url": (
                "https://files.pythonhosted.org/packages/a4/85/"
                "6d71f9b16f2ac647877e66ed4af723b3fbd477806ab8b8a89d39a362b85f/"
                "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
            ),
            "path": str(
                runtime_root
                / "pinned-wheels"
                / "xformers-0.0.35-py39-none-manylinux_2_28_x86_64.whl"
            ),
            "sha256": "ccc73c7db9890224ab05f5fb60e2034f9e6c8672a10be0cf00e95cbbae3eda7c",
            "size_bytes": 3264751,
            "install_mode": "forced-local-wheel-no-deps",
            "pip_version": "pip 25.1 from /run/site-packages/pip (python 3.12)",
        },
        "requested_route": {
            "purpose": "native-runtime-mechanics-qualification-only",
            "work_dir": str(runtime_root),
            "torch_version": "2.10.0+cu128",
            "cuda_device_name": "Tesla T4",
            "cuda_capability": [7, 5],
            "attention_backend": "xformers",
            "sparse_attention_backend": "xformers",
            "sparse_conv_backend": "flex_gemm",
            "trellis_repository": "https://github.com/microsoft/TRELLIS.2.git",
            "trellis_commit": "5565d240c4a494caaf9ece7a554542b76ffa36d3",
            "cumesh_repository": "https://github.com/JeffreyXiang/CuMesh.git",
            "cumesh_commit": "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
            "flex_gemm_repository": "https://github.com/JeffreyXiang/FlexGEMM.git",
            "flex_gemm_commit": "6dd94a859c26ee8246888502eada3dd8ad85532e",
            "nvdiffrast_repository": "https://github.com/NVlabs/nvdiffrast.git",
            "nvdiffrast_commit": "253ac4fcea7de5f396371124af597e6cc957bfae",
        },
        "effective_route": {
            "device_type": "cuda",
            "cuda_device_name": "Tesla T4",
            "cuda_capability": [7, 5],
            "cuda_runtime_version": "12.8",
            "torch_version": "2.10.0+cu128",
            "xformers_version": "0.0.35",
            "xformers_build_identity": {
                "version": "0.0.35",
                "torch": "2.10.0+cu128",
                "cuda": 1208,
                "torch_cuda_arch_list": "7.5 8.0+PTX 8.0 9.0a",
                "package_from": "wheel-v0.0.35",
            },
            "xformers_import_provenance": imported["xformers_import_provenance"],
            "xformers_attention_probe": {
                "executed": True,
                "device_type": "cuda",
                "finite": True,
                "shape": [1, 4, 1, 8],
                "dtype": "float16",
            },
            "attention_backend": "xformers",
            "sparse_attention_backend": "xformers",
            "sparse_conv_backend": "flex_gemm",
            "source_roots": {
                name: str(path) for name, path in roots.items()
            },
            "module_paths": module_paths,
            "build_import_provenance": imported["build_import_provenance"],
        },
        "orientation_probe": {
            "state": dict(module.EXPECTED_ORIENTATION_STATE),
            "capture_order": list(EXPECTED_CAPTURE_ORDER),
            "artifacts": artifacts,
        },
        "source_identities_before_build": {
            "trellis": {
                "path": str(roots["source_root"]),
                "repository": "https://github.com/microsoft/TRELLIS.2.git",
                "commit": "5565d240c4a494caaf9ece7a554542b76ffa36d3",
                "clean": True,
            },
            "cumesh": {
                "path": str(roots["cumesh_root"]),
                "repository": "https://github.com/JeffreyXiang/CuMesh.git",
                "commit": "c4ad6125924fcedfd13f0bd61520ca2d24eb7a87",
                "clean": True,
            },
            "flex_gemm": {
                "path": str(roots["flex_root"]),
                "repository": "https://github.com/JeffreyXiang/FlexGEMM.git",
                "commit": "6dd94a859c26ee8246888502eada3dd8ad85532e",
                "clean": True,
            },
            "nvdiffrast": {
                "path": str(roots["nvdiffrast_root"]),
                "repository": "https://github.com/NVlabs/nvdiffrast.git",
                "commit": "253ac4fcea7de5f396371124af597e6cc957bfae",
                "clean": True,
            },
        },
        "claim_ceiling": (
            "qualifies the exact T4 native build/import/backend/orientation-observer mechanics; "
            "does not qualify model assets, inference, postprocess quality, or final GLB"
        ),
    }
    report["source_identities_after_build"] = json.loads(
        json.dumps(report["source_identities_before_build"])
    )
    report["source_build_products_removed"] = {
        "nvdiffrast": ["build", "nvdiffrast.egg-info"]
    }
    report_path = root / "mechanics-report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    return report_path


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing_after", "source identity"),
        ("dirty_after", "source identity"),
        ("wrong_repository_before", "source identity"),
        ("wrong_path_after", "source identity"),
        ("missing_build_product_record", "build-product cleanup"),
        ("unknown_build_product_record", "build-product cleanup"),
    ],
)
def test_completed_report_rejects_unbound_source_checkout_identity(
    tmp_path, mutation, match
):
    module = load_module()
    report_path = write_valid_mechanics_bundle(tmp_path, run_id=RUN_ID)
    report = json.loads(report_path.read_text())
    if mutation == "missing_after":
        report.pop("source_identities_after_build")
    elif mutation == "dirty_after":
        report["source_identities_after_build"]["cumesh"]["clean"] = False
    elif mutation == "wrong_repository_before":
        report["source_identities_before_build"]["trellis"]["repository"] = (
            "https://example.invalid/TRELLIS.2.git"
        )
    elif mutation == "wrong_path_after":
        report["source_identities_after_build"]["flex_gemm"]["path"] = (
            "/tmp/not-the-imported-flex-root"
        )
    elif mutation == "missing_build_product_record":
        report.pop("source_build_products_removed")
    elif mutation == "unknown_build_product_record":
        report["source_build_products_removed"]["nvdiffrast"].append("unknown")
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        module.validate_completed_mechanics_report(
            report_path,
            expected_run_id=RUN_ID,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("disconnected_root", "requested work directory|source root"),
        ("unrelated_cumesh_import", "import provenance|module path"),
        ("unrelated_cumesh_direct_url", "direct URL"),
        ("missing_nvdiffrast_import", "module paths|nvdiffrast|import provenance"),
        ("missing_xformers_build_identity", "xformers build identity"),
        ("missing_xformers_probe", "xformers attention probe"),
        ("missing_xformers_wheel", "xformers wheel"),
        ("missing_xformers_pip_identity", "effective pip identity"),
        ("missing_xformers_module_path", "module paths|xformers"),
        ("missing_xformers_provenance", "xformers import provenance"),
        ("foreign_xformers_module", "xformers import provenance|module"),
        ("foreign_xformers_wheel_path", "xformers import provenance|wheel path"),
        ("foreign_xformers_wheel_digest", "xformers import provenance|wheel digest"),
    ],
)
def test_completed_report_rejects_disconnected_build_and_import_provenance(
    tmp_path, mutation, match
):
    module = load_module()
    report_path = write_valid_mechanics_bundle(tmp_path, run_id=RUN_ID)
    report = json.loads(report_path.read_text())
    if mutation == "disconnected_root":
        report["effective_route"]["source_roots"]["cumesh_root"] = (
            "/tmp/clean-but-disconnected/CuMesh"
        )
        for phase in ("before_build", "after_build"):
            report[f"source_identities_{phase}"]["cumesh"]["path"] = (
                "/tmp/clean-but-disconnected/CuMesh"
            )
    elif mutation == "unrelated_cumesh_import":
        report["effective_route"]["module_paths"]["cumesh"] = (
            "/tmp/unrelated-site-packages/cumesh.so"
        )
    elif mutation == "unrelated_cumesh_direct_url":
        report["effective_route"]["build_import_provenance"]["cumesh"][
            "direct_url"
        ] = {"url": "file:///tmp/unrelated/CuMesh", "dir_info": {}}
    elif mutation == "missing_nvdiffrast_import":
        report["effective_route"]["module_paths"].pop("nvdiffrast", None)
    elif mutation == "missing_xformers_build_identity":
        report["effective_route"].pop("xformers_build_identity")
    elif mutation == "missing_xformers_probe":
        report["effective_route"].pop("xformers_attention_probe")
    elif mutation == "missing_xformers_wheel":
        report.pop("xformers_wheel")
    elif mutation == "missing_xformers_pip_identity":
        report["xformers_wheel"].pop("pip_version")
    elif mutation == "missing_xformers_module_path":
        report["effective_route"]["module_paths"].pop("xformers")
    elif mutation == "missing_xformers_provenance":
        report["effective_route"].pop("xformers_import_provenance")
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
        module.validate_completed_mechanics_report(
            report_path,
            expected_run_id=RUN_ID,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("wrong_route", "effective route"),
        ("wrong_observer", "orientation observer"),
        ("wrong_dtype", "vertices dtype"),
        ("out_of_bounds", "face index"),
        ("nan_vertex", "finite"),
    ],
)
def test_completed_report_rejects_semantically_invalid_mechanics(
    tmp_path, mutation, match
):
    module = load_module()
    report_path = write_valid_mechanics_bundle(tmp_path, run_id=RUN_ID)
    report = json.loads(report_path.read_text())
    first = tmp_path / report["orientation_probe"]["artifacts"][EXPECTED_CAPTURE_ORDER[0]]["path"]
    with np.load(first, allow_pickle=False) as archive:
        vertices = archive["vertices"].copy()
        faces = archive["faces"].copy()
    if mutation == "wrong_route":
        report["effective_route"]["sparse_conv_backend"] = "spconv"
    elif mutation == "wrong_observer":
        report["orientation_probe"]["state"]["native_method_return_preserved"] = False
    elif mutation == "wrong_dtype":
        vertices = vertices.astype(np.float64)
    elif mutation == "out_of_bounds":
        faces[0, 0] = len(vertices)
    elif mutation == "nan_vertex":
        vertices[0, 0] = np.nan
    if mutation in {"wrong_dtype", "out_of_bounds", "nan_vertex"}:
        np.savez(first, vertices=vertices, faces=faces)
        record = report["orientation_probe"]["artifacts"][EXPECTED_CAPTURE_ORDER[0]]
        record["sha256"] = sha256(first)
        record["size_bytes"] = first.stat().st_size
        record["arrays"] = {
            "vertices": {"dtype": str(vertices.dtype), "shape": list(vertices.shape)},
            "faces": {"dtype": str(faces.dtype), "shape": list(faces.shape)},
        }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match=match):
        module.validate_completed_mechanics_report(
            report_path,
            expected_run_id=RUN_ID,
        )


def prepare_mechanics_packet(
    tmp_path: Path,
    *,
    run_id: str,
    name: str,
    no_download: bool = False,
    enable_internet: bool = True,
    accelerator: str = "NvidiaTeslaT4",
):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket

    capsule = tmp_path / "capsule"
    capsule.mkdir(exist_ok=True)
    smoke = capsule / "source_cuda_native_mechanics_smoke.py"
    witness = capsule / "source_cuda_native_image_to_glb_witness.py"
    authority = capsule / "witness_authority.py"
    smoke.write_text(SCRIPT.read_text())
    witness.write_text(
        Path(__file__).parents[1]
        .joinpath("scripts/source_cuda_native_image_to_glb_witness.py")
        .read_text()
    )
    authority.write_text(
        Path(__file__).parents[1]
        .joinpath("trellmlx/witness_authority.py")
        .read_text()
    )
    module = load_module()
    return module.prepare_native_mechanics_packet(
        KaggleCudaWitnessPacket(
            capsule_dir=capsule,
            output_dir=tmp_path / name,
            dataset_id="operator/native-mechanics-inputs",
            kernel_id="operator/native-mechanics-cuda",
            title="Native Mechanics CUDA",
            entrypoint=smoke.name,
            inputs=(smoke.name, witness.name, authority.name),
            run_id=run_id,
            enable_internet=enable_internet,
            accelerator=accelerator,
            output_json="mechanics-report.json",
            output_npz=None,
            expected_outputs=tuple(
                f"{index:02d}-{stage}.npz"
                for index, stage in enumerate(EXPECTED_CAPTURE_ORDER)
            ),
            entrypoint_args=(
                "--run-id",
                run_id,
                "--work-dir",
                "/kaggle/working/native-mechanics-runtime",
                *(("--no-download",) if no_download else ()),
            ),
        )
    )


@pytest.mark.parametrize(
    ("enable_internet", "accelerator", "match"),
    [
        (False, "NvidiaTeslaT4", "internet"),
        (True, "NvidiaTeslaP100", "Tesla T4"),
    ],
)
def test_mechanics_packet_rejects_non_executable_route_before_output_mutation(
    tmp_path, enable_internet, accelerator, match
):
    from trellmlx.kaggle_cuda_witness import KaggleCudaWitnessPacket, WitnessPacketError

    module = load_module()
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    smoke = capsule / "source_cuda_native_mechanics_smoke.py"
    witness = capsule / "source_cuda_native_image_to_glb_witness.py"
    smoke.write_text(SCRIPT.read_text())
    witness.write_text(
        Path(__file__).parents[1]
        .joinpath("scripts/source_cuda_native_image_to_glb_witness.py")
        .read_text()
    )
    output_dir = tmp_path / "packet"
    output_dir.mkdir()
    marker = output_dir / "must-survive-route-rejection.txt"
    marker.write_text("preserved")
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-mechanics-inputs",
        kernel_id="operator/native-mechanics-cuda",
        title="Native Mechanics CUDA",
        entrypoint=smoke.name,
        inputs=(smoke.name, witness.name),
        run_id=RUN_ID,
        enable_internet=enable_internet,
        accelerator=accelerator,
        output_json="mechanics-report.json",
        output_npz=None,
        expected_outputs=tuple(
            f"{index:02d}-{stage}.npz"
            for index, stage in enumerate(EXPECTED_CAPTURE_ORDER)
        ),
        entrypoint_args=(
            "--run-id",
            RUN_ID,
            "--work-dir",
            "/kaggle/working/native-mechanics-runtime",
        ),
    )

    with pytest.raises(WitnessPacketError, match=match):
        module.prepare_native_mechanics_packet(packet)

    assert marker.read_text() == "preserved"


def test_mechanics_packet_rejects_missing_run_identity_before_output_mutation(tmp_path):
    from trellmlx.kaggle_cuda_witness import (
        KaggleCudaWitnessPacket,
        WitnessPacketError,
        prepare_packet,
    )

    module = load_module()
    capsule = tmp_path / "capsule"
    capsule.mkdir()
    smoke = capsule / "source_cuda_native_mechanics_smoke.py"
    witness = capsule / "source_cuda_native_image_to_glb_witness.py"
    smoke.write_text(SCRIPT.read_text())
    witness.write_text(
        Path(__file__).parents[1]
        .joinpath("scripts/source_cuda_native_image_to_glb_witness.py")
        .read_text()
    )
    output_dir = tmp_path / "packet"
    output_dir.mkdir()
    marker = output_dir / "must-survive-run-identity-rejection.txt"
    marker.write_text("preserved")
    packet = KaggleCudaWitnessPacket(
        capsule_dir=capsule,
        output_dir=output_dir,
        dataset_id="operator/native-mechanics-inputs",
        kernel_id="operator/native-mechanics-cuda",
        title="Native Mechanics CUDA",
        entrypoint=smoke.name,
        inputs=(smoke.name, witness.name),
        output_json="mechanics-report.json",
        output_npz=None,
        expected_outputs=tuple(
            f"{index:02d}-{stage}.npz"
            for index, stage in enumerate(EXPECTED_CAPTURE_ORDER)
        ),
        entrypoint_args=(
            "--work-dir",
            "/kaggle/working/native-mechanics-runtime",
        ),
    )
    prepare_mechanics = getattr(
        module,
        "prepare_native_mechanics_packet",
        prepare_packet,
    )

    with pytest.raises(WitnessPacketError, match="run identity"):
        prepare_mechanics(packet)

    assert marker.read_text() == "preserved"


def test_corrupt_orientation_capture_cannot_claim_semantic_validation(tmp_path):
    module = load_module()
    report_path = tmp_path / "mechanics-report.json"
    native = SimpleNamespace(
        TRELLIS_COMMIT=module.TRELLIS_COMMIT,
        CUMESH_COMMIT=module.CUMESH_COMMIT,
        FLEX_GEMM_COMMIT=module.FLEX_GEMM_COMMIT,
        NVDIFFRAST_COMMIT=module.NVDIFFRAST_COMMIT,
    )

    def prepare(_native, _report):
        return {
            "source_root": Path("/tmp/TRELLIS.2"),
            "cumesh_root": Path("/tmp/CuMesh"),
            "flex_root": Path("/tmp/FlexGEMM"),
            "nvdiffrast_root": Path("/tmp/nvdiffrast"),
        }

    def imports(_roots):
        return fake_torch(), fake_imports(_roots)

    def corrupt_probe(_native, _cumesh, _torch, output_dir, run_id):
        artifacts = {}
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64
        )
        faces = np.asarray([[0, 1, 2]], dtype=np.int32)
        for index, stage in enumerate(EXPECTED_CAPTURE_ORDER):
            path = output_dir / f"{index:02d}-{stage}.npz"
            np.savez(path, vertices=vertices, faces=faces)
            artifacts[stage] = {
                "run_id": run_id,
                "path": path.name,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "arrays": {
                    "vertices": {"dtype": "float64", "shape": [3, 3]},
                    "faces": {"dtype": "int32", "shape": [1, 3]},
                },
            }
        return {
            "state": dict(module.EXPECTED_ORIENTATION_STATE),
            "capture_order": list(EXPECTED_CAPTURE_ORDER),
            "artifacts": artifacts,
        }

    with pytest.raises(ValueError, match="vertices dtype"):
        module.run_smoke(
            run_id=RUN_ID,
            output_json=report_path,
            work_dir=tmp_path / "runtime",
            native_module=native,
            prepare=prepare,
            importer=imports,
            attention_probe=lambda *_args: {
                "executed": True,
                "device_type": "cuda",
                "finite": True,
                "shape": [1, 4, 1, 8],
                "dtype": "float16",
            },
            orientation_probe=corrupt_probe,
        )

    report = json.loads(report_path.read_text())
    assert report["status"] == "failed"
    assert report["failure_phase"] == "orientation_probe"
    assert report["last_trustworthy_phase"] == "xformers_attention_executed"
    assert report["primary_output_status"] == "not_written"


def write_receipt(packet, output_dir: Path) -> None:
    from trellmlx.kaggle_cuda_witness import sha256_file

    outputs = {
        name: {
            "exists": True,
            "sha256": sha256_file(output_dir / name),
            "size_bytes": (output_dir / name).stat().st_size,
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
        "expected_image_sha256": None,
        "cuda_available": True,
        "cuda_device": "Tesla T4",
        "input_manifest": {
            "sha256": sha256_file(manifest),
            "size_bytes": manifest.stat().st_size,
        },
        "outputs": outputs,
    }
    (output_dir / "kaggle_cuda_witness_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n"
    )


def test_current_packet_consumer_rejects_complete_prior_mechanics_bundle(tmp_path):
    module = load_module()
    old_packet = prepare_mechanics_packet(tmp_path, run_id=RUN_ID, name="old-mechanics")
    current_packet = prepare_mechanics_packet(
        tmp_path, run_id=OTHER_RUN_ID, name="current-mechanics"
    )
    old_report = write_valid_mechanics_bundle(tmp_path / "old-output", run_id=RUN_ID)
    write_receipt(old_packet, old_report.parent)

    with pytest.raises(ValueError, match="run identity|run_id"):
        module.validate_downloaded_native_mechanics_outputs(
            current_packet,
            old_report.parent,
        )

    current_report = write_valid_mechanics_bundle(
        tmp_path / "current-output", run_id=OTHER_RUN_ID
    )
    write_receipt(current_packet, current_report.parent)
    admitted = module.validate_downloaded_native_mechanics_outputs(
        current_packet,
        current_report.parent,
    )
    assert admitted["report"]["run_id"] == OTHER_RUN_ID


def test_actual_generated_runner_binds_mechanics_run_identity(tmp_path, monkeypatch):
    packet = prepare_mechanics_packet(
        tmp_path,
        run_id=RUN_ID,
        name="runner-mechanics",
        no_download=True,
    )
    runner_text = (packet.kernel_dir / "run_kaggle_cuda_witness.py").read_text()
    assert runner_text.count(RUN_ID) >= 1
    manifest = json.loads((packet.dataset_dir / "witness-manifest.json").read_text())
    assert manifest["run_id"] == RUN_ID
    assert manifest["enable_internet"] is True
    assert manifest["accelerator"] == "NvidiaTeslaT4"
    reloaded = __import__(
        "trellmlx.kaggle_cuda_witness", fromlist=["load_prepared_packet"]
    ).load_prepared_packet(packet.output_dir)
    assert reloaded.enable_internet is True
    assert reloaded.accelerator == "NvidiaTeslaT4"
    assert manifest["entrypoint_args"].count("--run-id") == 1
    assert manifest["entrypoint_args"][
        manifest["entrypoint_args"].index("--run-id") + 1
    ] == RUN_ID
    fake_torch = SimpleNamespace(
        __version__="2.10.0+cu128",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "Tesla T4",
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    work = tmp_path / "runner-work"
    work.mkdir()
    monkeypatch.chdir(work)
    namespace = {"__name__": "runner_test"}
    exec(
        runner_text.replace(
            'Path("/kaggle/input")',
            f"Path({str(packet.dataset_dir)!r})",
        ),
        namespace,
    )

    rc = namespace["main"]()

    report = json.loads((work / "mechanics-report.json").read_text())
    receipt = json.loads((work / "kaggle_cuda_witness_receipt.json").read_text())
    assert rc != 0
    assert report["status"] == "preflight_stopped"
    assert report["run_id"] == RUN_ID
    assert receipt["run_id"] == RUN_ID
    assert receipt["effective_command"].count("--run-id") == 1
    assert receipt["effective_command"][
        receipt["effective_command"].index("--run-id") + 1
    ] == RUN_ID
