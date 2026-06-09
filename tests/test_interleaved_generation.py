"""Contracts for in-process stage-major generation interleaving."""

from pathlib import Path

import pytest


def test_generation_job_requires_explicit_conditioning_route(tmp_path):
    from trellmlx.interleaved_generation import GenerationJob

    with pytest.raises(ValueError, match="image or explicit random conditioning"):
        GenerationJob(job_id="missing", images=(), seed=1, output_path=tmp_path / "out.glb")

    random_job = GenerationJob(
        job_id="random",
        images=(),
        seed=1,
        output_path=tmp_path / "random.glb",
        random_conditioning=True,
    )

    assert random_job.conditioning_route == "random"


def test_stage_major_plan_groups_jobs_by_stage_before_advancing(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        InterleavedBatchPlan,
    )

    jobs = (
        GenerationJob(
            job_id="seed-101",
            images=("subject.png",),
            seed=101,
            output_path=tmp_path / "seed-101.glb",
        ),
        GenerationJob(
            job_id="seed-202",
            images=("subject.png",),
            seed=202,
            output_path=tmp_path / "seed-202.glb",
        ),
    )
    plan = InterleavedBatchPlan(
        jobs=jobs,
        stages=("image_conditioning", "sparse_structure", "mesh_export"),
    )

    assert [
        (invocation.stage, invocation.job_id, invocation.seed)
        for invocation in plan.iter_invocations()
    ] == [
        ("image_conditioning", "seed-101", 101),
        ("image_conditioning", "seed-202", 202),
        ("sparse_structure", "seed-101", 101),
        ("sparse_structure", "seed-202", 202),
        ("mesh_export", "seed-101", 101),
        ("mesh_export", "seed-202", 202),
    ]


def test_plan_rejects_duplicate_job_ids(tmp_path):
    from trellmlx.interleaved_generation import GenerationJob, InterleavedBatchPlan

    jobs = (
        GenerationJob(
            job_id="dup",
            images=("a.png",),
            seed=1,
            output_path=tmp_path / "one.glb",
        ),
        GenerationJob(
            job_id="dup",
            images=("b.png",),
            seed=2,
            output_path=tmp_path / "two.glb",
        ),
    )

    with pytest.raises(ValueError, match="duplicate job_id"):
        InterleavedBatchPlan(jobs=jobs)


def test_job_trace_records_stage_timings_and_output_counts(tmp_path):
    from trellmlx.interleaved_generation import (
        GenerationJob,
        GenerationStageResult,
        new_job_traces,
    )

    job = GenerationJob(
        job_id="seed-101",
        images=("subject.png",),
        seed=101,
        output_path=tmp_path / "seed-101.glb",
        resolution=512,
        texture_size=1024,
    )
    traces = new_job_traces((job,))
    trace = traces["seed-101"].record_stage(
        GenerationStageResult(
            stage="sparse_structure",
            elapsed_seconds=1.25,
            output_counts={"lr_voxels": 3072},
        )
    )

    assert trace.job_id == "seed-101"
    assert trace.seed == 101
    assert trace.output_path == str(tmp_path / "seed-101.glb")
    assert trace.stage_results[0].stage == "sparse_structure"
    assert trace.stage_results[0].output_counts == {"lr_voxels": 3072}
    assert trace.config["resolution"] == 512
    assert trace.config["texture_size"] == 1024


def test_generation_job_can_be_built_from_process_batch_job(tmp_path):
    from trellmlx.batch_inference import BatchJob
    from trellmlx.interleaved_generation import GenerationJob

    batch_job = BatchJob(
        images=("subject.png",),
        seed=101,
        output_path=tmp_path / "seed-101.glb",
        resolution=512,
        target_faces=50_000,
        no_cleanup=True,
    )

    generation_job = GenerationJob.from_batch_job(batch_job)

    assert generation_job.job_id == "seed-101"
    assert generation_job.images == ("subject.png",)
    assert generation_job.seed == 101
    assert generation_job.output_path == Path(tmp_path / "seed-101.glb")
    assert generation_job.resolution == 512
    assert generation_job.target_faces == 50_000
    assert generation_job.no_cleanup is True


def test_stage_seed_is_deterministic_per_job_and_stage():
    from trellmlx.interleaved_generation import derive_stage_seed

    assert derive_stage_seed(job_seed=101, stage_index=3) == derive_stage_seed(
        job_seed=101,
        stage_index=3,
    )
    assert derive_stage_seed(job_seed=101, stage_index=3) != derive_stage_seed(
        job_seed=202,
        stage_index=3,
    )
    assert derive_stage_seed(job_seed=101, stage_index=3) != derive_stage_seed(
        job_seed=101,
        stage_index=4,
    )
    assert 0 <= derive_stage_seed(job_seed=101, stage_index=3) < 2**32
