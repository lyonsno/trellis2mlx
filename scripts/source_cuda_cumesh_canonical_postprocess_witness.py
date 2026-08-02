"""Run full source-CUDA geometry with canonical consumed adjacency order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.source_cuda_cumesh_postprocess_witness import (
        CUMESH_COMMIT,
        WitnessError,
        _run_setup_command,
        execute_geometry_sequence,
        prepare_release_runtime,
        run_witness as run_base_witness,
        sha256_file,
    )
    from trellmlx.canonical_cumesh import (
        mesh_state_digest_observer,
        simplify_with_canonical_adjacency_step_loop,
    )
except ModuleNotFoundError:
    from source_cuda_cumesh_postprocess_witness import (
        CUMESH_COMMIT,
        WitnessError,
        _run_setup_command,
        execute_geometry_sequence,
        prepare_release_runtime,
        run_witness as run_base_witness,
        sha256_file,
    )
    from canonical_cumesh import (
        mesh_state_digest_observer,
        simplify_with_canonical_adjacency_step_loop,
    )


INSTRUMENTATION_SCHEMA = (
    "trellis2mlx.cumesh_canonical_adjacency_instrumentation.v1"
)
INSTRUMENTED_FILES = (
    "src/connectivity.cu",
    "src/cumesh.h",
    "src/ext.cpp",
    "src/simplify.cu",
)
ADJACENCY_ORDER = "ascending-face-id-per-vertex"


def _porcelain_changed_files(status: str) -> list[str]:
    return sorted(line[3:] for line in status.splitlines() if len(line) >= 4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--report-json",
        "--output-json",
        dest="report_json",
        required=True,
        type=Path,
    )
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--target-faces", required=True, type=int)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--instrumentation-patch", required=True, type=Path)
    parser.add_argument("--expected-patch-sha256", required=True)
    parser.add_argument(
        "--record-simplify-step-digests",
        action="store_true",
    )
    return parser


def _instrumentation_callback(
    patch: Path,
    expected_patch_sha256: str,
) -> Callable[[Path, dict[str, Any]], dict[str, Any]]:
    patch = Path(patch).resolve(strict=False)

    def apply(cumesh_root: Path, report: dict[str, Any]) -> dict[str, Any]:
        if not patch.is_file():
            raise WitnessError(f"instrumentation patch does not exist: {patch}")
        actual_patch_sha256 = sha256_file(patch)
        if actual_patch_sha256 != expected_patch_sha256:
            raise WitnessError(
                "instrumentation patch SHA256 mismatch: "
                f"expected {expected_patch_sha256}, got {actual_patch_sha256}"
            )
        _run_setup_command(
            ["git", "-C", str(cumesh_root), "apply", "--check", str(patch)],
            report,
        )
        _run_setup_command(
            ["git", "-C", str(cumesh_root), "apply", str(patch)],
            report,
        )
        _run_setup_command(
            ["git", "-C", str(cumesh_root), "diff", "--check"],
            report,
        )
        status = _git_output(
            cumesh_root,
            "status",
            "--porcelain",
            "--untracked-files=no",
        )
        changed_files = _porcelain_changed_files(status)
        if changed_files != sorted(INSTRUMENTED_FILES):
            raise WitnessError(
                "canonical instrumentation changed unexpected CuMesh files: "
                f"expected {sorted(INSTRUMENTED_FILES)}, got {changed_files}"
            )
        return {
            "schema": INSTRUMENTATION_SCHEMA,
            "patch_path": str(patch),
            "patch_sha256": actual_patch_sha256,
            "changed_files": changed_files,
            "base_commit": CUMESH_COMMIT,
            "diagnostic_only": True,
            "read_only_trace": False,
            "semantic_change": (
                "sort_each_vertex_face_segment_before_consumed_simplify_step"
            ),
            "adjacency_order": ADJACENCY_ORDER,
        }

    return apply


def _git_output(root: Path, *args: str) -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise WitnessError(
            f"git {' '.join(args)} failed: "
            f"{completed.stderr or completed.stdout}"
        )
    return completed.stdout.rstrip()


def _build_canonical_simplification_runner(
    *,
    record_step_digests: bool,
) -> Callable[[Any, int], dict[str, Any]]:
    def runner(mesh: Any, target_faces: int) -> dict[str, Any]:
        trace = simplify_with_canonical_adjacency_step_loop(
            mesh,
            int(target_faces),
            step_observer=(
                mesh_state_digest_observer
                if record_step_digests
                else None
            ),
        )
        return {
            "simplifier_route": (
                "canonical-adjacency-source-cuda-step-loop"
            ),
            "adjacency_order": ADJACENCY_ORDER,
            "reuse_vertex_face_adjacency": True,
            "record_step_digests": record_step_digests,
            "simplifier_step_trace": trace,
        }

    return runner


def _build_canonical_sequence_runner(
    *,
    record_step_digests: bool,
) -> Callable[..., None]:
    simplification_runner = _build_canonical_simplification_runner(
        record_step_digests=record_step_digests,
    )

    def runner(
        mesh: Any,
        target_faces: int,
        snapshot: Callable[..., None],
    ) -> None:
        execute_geometry_sequence(
            mesh,
            int(target_faces),
            snapshot,
            simplification_runner=simplification_runner,
        )

    return runner


def run_witness(
    *,
    input_ply: Path,
    output_dir: Path,
    report_json: Path,
    expected_input_sha256: str,
    target_faces: int,
    work_dir: Path,
    instrumentation_patch: Path,
    expected_patch_sha256: str,
    record_simplify_step_digests: bool = False,
    runtime_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    instrumentation = _instrumentation_callback(
        instrumentation_patch,
        expected_patch_sha256,
    )

    def build_runtime(**kwargs):
        if runtime_factory is None:
            runtime = prepare_release_runtime(
                **kwargs,
                cumesh_instrumentation=instrumentation,
            )
        else:
            runtime = runtime_factory(**kwargs)
        identity = runtime.effective_route.get("cumesh_instrumentation")
        if not isinstance(identity, dict):
            raise WitnessError(
                "effective CUDA route omitted canonical instrumentation identity"
            )
        if identity.get("schema") != INSTRUMENTATION_SCHEMA:
            raise WitnessError(
                "effective CUDA route used the wrong instrumentation schema"
            )
        if identity.get("patch_sha256") != expected_patch_sha256:
            raise WitnessError(
                "effective CUDA route used the wrong instrumentation patch"
            )
        runtime.effective_route.update({
            "geometry_route": (
                "release-trellis2-cumesh-canonical-adjacency-non-remesh"
            ),
            "adjacency_order": ADJACENCY_ORDER,
            "reuse_vertex_face_adjacency": True,
            "record_simplify_step_digests": (
                record_simplify_step_digests
            ),
        })
        return runtime

    sequence_runner = _build_canonical_sequence_runner(
        record_step_digests=record_simplify_step_digests,
    )
    return run_base_witness(
        input_ply=input_ply,
        output_dir=output_dir,
        report_json=report_json,
        expected_input_sha256=expected_input_sha256,
        target_faces=int(target_faces),
        work_dir=work_dir,
        runtime_factory=build_runtime,
        sequence_runner=sequence_runner,
        requested_route_overrides={
            "geometry_route": (
                "release-trellis2-cumesh-canonical-adjacency-non-remesh"
            ),
            "adjacency_order": ADJACENCY_ORDER,
            "reuse_vertex_face_adjacency": True,
            "record_simplify_step_digests": (
                record_simplify_step_digests
            ),
            "instrumentation_patch": str(instrumentation_patch),
            "expected_patch_sha256": expected_patch_sha256,
        },
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_witness(
            input_ply=args.input_ply,
            output_dir=args.output_dir,
            report_json=args.report_json,
            expected_input_sha256=args.expected_input_sha256,
            target_faces=args.target_faces,
            work_dir=args.work_dir,
            instrumentation_patch=args.instrumentation_patch,
            expected_patch_sha256=args.expected_patch_sha256,
            record_simplify_step_digests=(
                args.record_simplify_step_digests
            ),
        )
    except Exception:
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
