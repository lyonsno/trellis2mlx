"""One private T4 image-to-GLB attempt using pinned official TRELLIS.2.

The parent builds the retained CUDA runtime; a fresh child runs inference.
Every completed stage and sampler input survives a later failure. No local
MLX prediction, mesh converter, or remesher participates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time
import traceback
from types import SimpleNamespace


MODEL_REVISION = "af44b45f2e35a493886929c6d786e563ec68364d"
SPARSE_REVISION = "25e0d31ffbebe4b5a97464dd851910efc3002d96"
SOURCE_COMMIT = "5565d240c4a494caaf9ece7a554542b76ffa36d3"
PIPELINE_SHA = "222c359ab1ed9bc6735a640a34f95d47f8681b9bc4aaa101bfb80274676253c6"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def validate_request(request, root):
    expected = {"pipeline_type": "1024_cascade", "seed": 42, "steps": 12,
                "target_faces": 1000000, "texture_size": 4096,
                "remesh": True, "remesh_band": 1, "remesh_project": 0,
                "max_num_tokens": 49152}
    if request.get("settings") != expected:
        raise ValueError("request must name the complete intended 1024 source settings")
    if request.get("source_commit") != SOURCE_COMMIT or request.get("model_revision") != MODEL_REVISION:
        raise ValueError("source/model revision mismatch")
    for name, expected_sha in request["files"].items():
        path = Path(root) / name
        if Path(name).name != name or digest(path) != expected_sha:
            raise ValueError(f"input identity mismatch: {name}")
    if request["files"].get("source_cuda_1024_capacity.py") != digest(__file__):
        raise ValueError("executed producer differs from the admitted producer")


def find_request(explicit=None):
    candidates = [Path(explicit)] if explicit else list(Path("/kaggle/input").rglob("full1024-request.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one mounted request, found {len(candidates)}")
    return candidates[0]


def validate_completion(report):
    route = report.get("effective_route", {})
    if route.get("cuda_device") != "Tesla T4" or route.get("capability") != [7, 5]:
        raise ValueError("effective device is not the requested T4")
    if route.get("source_commit") != SOURCE_COMMIT or route.get("resolution") != 1024:
        raise ValueError("effective source/resolution mismatch")
    if route.get("attention") != "sdpa" or route.get("sparse_attention") != "sdpa" or route.get("sparse_conv") != "flex_gemm":
        raise ValueError("effective source backend mismatch")
    if report.get("pipeline_run_calls") != 1 or report.get("sampler_calls") != {"sparse": 1, "shape": 2, "texture": 1}:
        raise ValueError("incomplete official 1024 cascade")
    if not all(report.get("artifacts", {}).get(name) for name in ("raw_mesh.npz", "shape_terminal.npz", "texture_terminal.npz", "output.glb")):
        raise ValueError("missing complete inference or GLB artifact")
    if report.get("glb", {}).get("faces", 0) <= 0:
        raise ValueError("empty final geometry")


def run_child(request_path, output, runtime):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from PIL import Image

    root = request_path.parent
    request = json.loads(request_path.read_text())
    validate_request(request, root)
    report = {"schema": "trellis2mlx.full1024-source-capacity.v1", "status": "running",
              "phase": "runtime_import", "request": request, "artifacts": {},
              "memory": [], "sampler_calls": {}, "pipeline_run_calls": 0}
    report_path = output / "report.json"

    def publish():
        write_json(report_path, report)

    def memory(boundary):
        row = {"boundary": boundary, "time": time.time(), "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        try:
            free, total = torch.cuda.mem_get_info()
            row.update(free_bytes=free, total_bytes=total, allocated_bytes=torch.cuda.memory_allocated(),
                       reserved_bytes=torch.cuda.memory_reserved(), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        except Exception as exc:
            row["error"] = str(exc)
        report["memory"].append(row)
        publish()

    def artifact(path):
        report["artifacts"][path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
        publish()

    def save_arrays(name, **arrays):
        path = output / name
        converted = {key: np.asarray(value) for key, value in arrays.items()}
        if any(v.dtype.kind == "f" and not np.isfinite(v).all() for v in converted.values()):
            # Preserve the disputed arrays before refusing their use as success.
            np.savez(path, **converted)
            artifact(path)
            raise ValueError(f"nonfinite output in {name}")
        np.savez(path, **converted)
        artifact(path)

    def tensor_arrays(value, prefix):
        if hasattr(value, "feats") and hasattr(value, "coords"):
            return {prefix + "_feats": value.feats.detach().float().cpu().numpy(),
                    prefix + "_coords": value.coords.detach().cpu().numpy()}
        if torch.is_tensor(value):
            return {prefix: value.detach().float().cpu().numpy()}
        return {}

    publish()
    try:
        if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "Tesla T4":
            raise ValueError("requested Tesla T4 is unavailable")
        os.environ.update(ATTN_BACKEND="sdpa", SPARSE_ATTN_BACKEND="sdpa", SPARSE_CONV_BACKEND="flex_gemm")
        sys.path.insert(0, str(runtime / "TRELLIS.2"))
        import o_voxel
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.modules.attention import config as attention_config
        from trellis2.modules.sparse import config as sparse_config
        report["effective_route"] = {"cuda_device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__,
            "cuda": torch.version.cuda, "source_commit": SOURCE_COMMIT,
            "attention": attention_config.BACKEND, "sparse_attention": sparse_config.ATTN,
            "sparse_conv": sparse_config.CONV, "model_revision": MODEL_REVISION,
            "preprocessing": "official-BiRefNet-and-DINOv3", "low_vram": True}
        memory("runtime-import")
        report["phase"] = "download_public_model_assets"
        publish()
        model_root = Path(snapshot_download("microsoft/TRELLIS.2-4B", revision=MODEL_REVISION,
            allow_patterns=["pipeline.json", "ckpts/*512*", "ckpts/*1024*", "ckpts/ss_flow*", "ckpts/*dec_next*"]))
        sparse_root = Path(snapshot_download("microsoft/TRELLIS-image-large", revision=SPARSE_REVISION,
            allow_patterns=["ckpts/ss_dec_conv3d_16l8_fp16.*"]))
        if digest(model_root / "pipeline.json") != PIPELINE_SHA:
            raise ValueError("official pipeline configuration identity mismatch")
        view = output.parent / "model-view"
        view.mkdir()
        config = json.loads((model_root / "pipeline.json").read_text())
        for key, value in config["args"]["models"].items():
            config["args"]["models"][key] = str(sparse_root / "ckpts/ss_dec_conv3d_16l8_fp16") if key == "sparse_structure_decoder" else str(model_root / value)
        for kind, prefix in (("image_cond_model", "dino"), ("rembg_model", "rembg")):
            directory = view / prefix
            directory.mkdir()
            for name in request["files"]:
                if name.startswith(prefix + "__"):
                    (directory / name.split("__", 1)[1]).symlink_to((root / name).resolve())
            config["args"][kind]["args"]["model_name"] = str(directory)
        write_json(view / "pipeline.json", config)
        report["model_path_configuration"] = config
        report["model_files"] = {str(path.relative_to(base)): {"sha256": digest(path), "bytes": path.stat().st_size}
            for base in (model_root, sparse_root) for path in base.rglob("*") if path.is_file()}
        report["phase"] = "load_pipeline"
        publish()
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(view))
        if pipeline.low_vram is not True:
            raise ValueError("official low-VRAM mode was not selected")
        pipeline.cuda()
        memory("pipeline-loaded")

        # Observers only retain inputs/results and query memory; native calls own arithmetic.
        for name, sampler in (("sparse", pipeline.sparse_structure_sampler), ("shape", pipeline.shape_slat_sampler), ("texture", pipeline.tex_slat_sampler)):
            original = sampler.sample
            def observe(model, noise, *args, _original=original, _name=name, **kwargs):
                index = report["sampler_calls"].get(_name, 0) + 1
                report["sampler_calls"][_name] = index
                phase = f"{_name}-sampler-{index}"
                report["phase"] = phase
                arrays = tensor_arrays(noise, "noise")
                for key in ("cond", "neg_cond"):
                    arrays.update(tensor_arrays(kwargs.get(key), key))
                save_arrays(phase + "-input.npz", **arrays)
                memory(phase + "-begin")
                result = _original(model, noise, *args, **kwargs)
                save_arrays(phase + "-terminal.npz", **tensor_arrays(result.samples, "sample"))
                report["last_completed_stage"] = phase
                memory(phase + "-end")
                return result
            sampler.sample = observe

        original_preprocess = pipeline.preprocess_image
        def preprocess(image):
            report["phase"] = "official_preprocess"
            memory("preprocess-begin")
            result = original_preprocess(image)
            path = output / "preprocessed.png"
            result.save(path)
            artifact(path)
            report["last_completed_stage"] = "official_preprocess"
            return result
        pipeline.preprocess_image = preprocess
        original_decode = pipeline.decode_latent
        def decode(shape, texture, resolution):
            report["phase"] = "decode_latent"
            report["effective_route"]["resolution"] = resolution
            save_arrays("shape_terminal.npz", **tensor_arrays(shape, "shape"))
            save_arrays("texture_terminal.npz", **tensor_arrays(texture, "texture"))
            memory("decode-begin")
            result = original_decode(shape, texture, resolution)
            if len(result) != 1:
                raise ValueError("expected one official decoded mesh")
            mesh = result[0]
            save_arrays("raw_mesh.npz", vertices=mesh.vertices.detach().cpu().numpy(), faces=mesh.faces.detach().cpu().numpy(),
                attrs=mesh.attrs.detach().cpu().numpy(), coords=mesh.coords.detach().cpu().numpy(), voxel_size=mesh.voxel_size)
            report["raw_counts"] = {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}
            report["last_completed_stage"] = "raw_mesh"
            memory("decode-end")
            return result
        pipeline.decode_latent = decode

        report["phase"] = "pipeline_run"
        report["pipeline_run_calls"] += 1
        publish()
        with torch.no_grad():
            mesh = pipeline.run(Image.open(root / "kiln.png"), seed=42, pipeline_type="1024_cascade", max_num_tokens=49152,
                sparse_structure_sampler_params={"steps": 12}, shape_slat_sampler_params={"steps": 12}, tex_slat_sampler_params={"steps": 12})[0]
            report["phase"] = "official_remesh_and_glb"
            memory("finishing-begin")
            glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                aabb=[[-.5, -.5, -.5], [.5, .5, .5]], decimation_target=1000000, texture_size=4096,
                remesh=True, remesh_band=1, remesh_project=0, verbose=True)
            glb.export(output / "output.glb", extension_webp=True)
            artifact(output / "output.glb")
            helper = load_helper(root)
            report["glb"] = helper.validate_glb(output / "output.glb", 4096)
            memory("finishing-end")
        validate_completion(report)
        report.update(status="completed", phase="completed")
        return 0
    except Exception as exc:
        report.update(status="failed", failure_phase=report["phase"], error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        memory("failure")
        traceback.print_exc()
        return 1
    finally:
        publish()


def load_helper(root):
    path = root / "source_cuda_finishing.py"
    spec = importlib.util.spec_from_file_location("retained_cuda_finishing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("/kaggle/working/full1024-report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/full1024"))
    parser.add_argument("--runtime", type=Path, default=Path("/kaggle/working/full1024-runtime"))
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args(argv)
    output, runtime = args.output_dir, args.runtime
    if args.child:
        return run_child(find_request(args.request), output, runtime)
    report = {"schema": "trellis2mlx.full1024-source-parent.v1", "status": "running", "phase": "input_admission", "producer_sha256": digest(__file__)}
    started = time.monotonic()
    code = 1
    owned_output = False
    if args.output_json.exists():
        args.output_json = args.output_json.with_name(args.output_json.name + f".refusal-{time.time_ns()}.json")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output_json, report)
    try:
        output.mkdir(exist_ok=False)
        owned_output = True
        request_path = find_request(args.request)
        validate_request(json.loads(request_path.read_text()), request_path.parent)
        helper = load_helper(request_path.parent)
        original_command = helper.command
        def command(argv, record, cwd=None):
            record["current_command"] = list(map(str, argv))
            write_json(args.output_json, record)
            try:
                return original_command(argv, record, cwd)
            finally:
                write_json(args.output_json, record)
        helper.command = command
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        helper.bootstrap(SimpleNamespace(runtime=runtime), report)
        command([sys.executable, "-m", "pip", "install", "transformers", "huggingface-hub", "kornia", "timm", "scipy", "zstandard", "imageio", "imageio-ffmpeg", "lpips", "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"], report)
        command(["git", "submodule", "update", "--init", "--recursive"], report, runtime / "TRELLIS.2")
        command([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", runtime / "TRELLIS.2/o-voxel"], report)
        report["phase"] = "source_child"
        write_json(args.output_json, report)
        with (output / "child.log").open("w") as log:
            child = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--child", "--request", str(request_path), "--output-dir", str(output), "--runtime", str(runtime)], stdout=log, stderr=subprocess.STDOUT)
        report["child_exit_code"] = child.returncode
        child_path = output / "report.json"
        if child_path.is_file():
            report["child"] = json.loads(child_path.read_text())
        if child.returncode != 0 or report.get("child", {}).get("status") != "completed":
            raise RuntimeError("source child failed; complete partial evidence retained")
        validate_completion(report["child"])
        report.update(status="completed", phase="completed")
        code = 0
    except Exception as exc:
        report.update(status="failed", failure_phase=report["phase"], error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        # Includes partial arrays and failure report even after a child signal.
        if owned_output:
            bundle = args.output_json.with_suffix(".evidence.tar")
            with tarfile.open(bundle, "x") as archive:
                archive.add(output, arcname="full1024")
            report["bundle"] = {"path": str(bundle), "sha256": digest(bundle), "bytes": bundle.stat().st_size}
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output_json, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
