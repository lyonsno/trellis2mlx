"""Stage-major generation contracts for in-process batch execution.

This module does not tensor-batch TRELLIS internals. It defines the portable
job/state spine needed to run stage N for multiple jobs before advancing to
stage N+1, which is the safe model-handle reuse boundary before segmented
sparse tensor batching exists.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence


StageArtifactValue = bool | int | float | str
_STAGE_ARTIFACT_TYPES = (bool, int, float, str)


DEFAULT_STAGE_SEQUENCE: tuple[str, ...] = (
    "image_conditioning",
    "sparse_structure",
    "lr_shape_latent",
    "hr_coordinates",
    "hr_shape_latent",
    "shape_decode",
    "mesh_extract",
    "texture_latent",
    "texture_decode",
    "texture_bake",
    "export",
)


def derive_stage_seed(*, job_seed: int, stage_index: int) -> int:
    """Derive a deterministic per-job/per-stage uint32 seed."""

    digest = hashlib.blake2s(f"{int(job_seed)}:{int(stage_index)}".encode("ascii"), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def _validate_artifacts(artifacts: Mapping[str, StageArtifactValue]) -> dict[str, StageArtifactValue]:
    validated: dict[str, StageArtifactValue] = {}
    for key, value in artifacts.items():
        if not isinstance(key, str):
            raise ValueError("artifact keys must be strings")
        if not isinstance(value, _STAGE_ARTIFACT_TYPES):
            raise ValueError("artifact values must be bool, int, float, or str")
        validated[key] = value
    return validated


@dataclass(frozen=True)
class GenerationJob:
    """Single in-process generation request.

    `random_conditioning` is explicit so a missing image cannot silently become
    random conditioning in a batch run.
    """

    job_id: str
    images: tuple[str, ...]
    seed: int
    output_path: Path
    resolution: int = 1024
    max_tokens: int = 49152
    target_faces: int = 200_000
    compile: bool = False
    quantize: int = 0
    no_rembg: bool = False
    no_cleanup: bool = False
    keep_largest: bool = False
    texture_size: int = 1024
    texture_backend: str = "gpu"
    random_conditioning: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "output_path", Path(self.output_path))
        if not self.images and not self.random_conditioning:
            raise ValueError("GenerationJob requires an image or explicit random conditioning")
        if self.images and self.random_conditioning:
            raise ValueError("GenerationJob cannot use images and random conditioning together")

    @property
    def conditioning_route(self) -> str:
        return "random" if self.random_conditioning else "image"

    @classmethod
    def from_batch_job(cls, batch_job, *, job_id: str | None = None) -> "GenerationJob":
        """Build a stage-pipeline job from the process-level batch job shape."""

        seed = int(batch_job.seed)
        return cls(
            job_id=job_id or f"seed-{seed}",
            images=tuple(batch_job.images),
            seed=seed,
            output_path=Path(batch_job.output_path),
            resolution=batch_job.resolution,
            max_tokens=batch_job.max_tokens,
            target_faces=batch_job.target_faces,
            compile=batch_job.compile,
            quantize=batch_job.quantize,
            no_rembg=batch_job.no_rembg,
            no_cleanup=batch_job.no_cleanup,
            keep_largest=getattr(batch_job, "keep_largest", False),
            texture_size=getattr(batch_job, "texture_size", 1024),
            texture_backend=getattr(batch_job, "texture_backend", "gpu"),
        )

    def config_dict(self) -> dict[str, int | bool | str]:
        return {
            "resolution": self.resolution,
            "max_tokens": self.max_tokens,
            "target_faces": self.target_faces,
            "compile": self.compile,
            "quantize": self.quantize,
            "no_rembg": self.no_rembg,
            "no_cleanup": self.no_cleanup,
            "keep_largest": self.keep_largest,
            "texture_size": self.texture_size,
            "texture_backend": self.texture_backend,
            "conditioning_route": self.conditioning_route,
        }


@dataclass(frozen=True)
class GenerationStageInvocation:
    """A scheduled `(stage, job)` unit for a stage-major runner."""

    stage: str
    stage_index: int
    job_id: str
    seed: int
    stage_seed: int
    images: tuple[str, ...]
    output_path: str
    conditioning_route: str


@dataclass(frozen=True)
class InterleavedBatchPlan:
    """Stage-major ordering for future in-process batch generation."""

    jobs: tuple[GenerationJob, ...]
    stages: tuple[str, ...] = DEFAULT_STAGE_SEQUENCE

    def __init__(
        self,
        jobs: Sequence[GenerationJob],
        stages: Sequence[str] = DEFAULT_STAGE_SEQUENCE,
    ) -> None:
        jobs_tuple = tuple(jobs)
        stages_tuple = tuple(stages)
        if not jobs_tuple:
            raise ValueError("InterleavedBatchPlan requires at least one job")
        if not stages_tuple:
            raise ValueError("InterleavedBatchPlan requires at least one stage")

        seen: set[str] = set()
        for job in jobs_tuple:
            if job.job_id in seen:
                raise ValueError(f"duplicate job_id: {job.job_id}")
            seen.add(job.job_id)

        object.__setattr__(self, "jobs", jobs_tuple)
        object.__setattr__(self, "stages", stages_tuple)

    def iter_invocations(self) -> Iterator[GenerationStageInvocation]:
        for stage_index, stage in enumerate(self.stages):
            for job in self.jobs:
                yield GenerationStageInvocation(
                    stage=stage,
                    stage_index=stage_index,
                    job_id=job.job_id,
                    seed=job.seed,
                    stage_seed=derive_stage_seed(job_seed=job.seed, stage_index=stage_index),
                    images=job.images,
                    output_path=str(job.output_path),
                    conditioning_route=job.conditioning_route,
                )


@dataclass(frozen=True)
class GenerationStageResult:
    """Observed result for one job at one generation stage."""

    stage: str
    elapsed_seconds: float
    output_counts: Mapping[str, int] = field(default_factory=dict)
    failure_phase: str | None = None

    def __post_init__(self) -> None:
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        object.__setattr__(self, "output_counts", dict(self.output_counts))


@dataclass(frozen=True)
class GenerationJobTrace:
    """Accumulated per-stage trace for one generation job."""

    job_id: str
    seed: int
    images: tuple[str, ...]
    output_path: str
    config: Mapping[str, int | bool | str]
    stage_results: tuple[GenerationStageResult, ...] = ()

    @classmethod
    def from_job(cls, job: GenerationJob) -> "GenerationJobTrace":
        return cls(
            job_id=job.job_id,
            seed=job.seed,
            images=job.images,
            output_path=str(job.output_path),
            config=job.config_dict(),
        )

    def record_stage(self, result: GenerationStageResult) -> "GenerationJobTrace":
        return replace(self, stage_results=(*self.stage_results, result))


def new_job_traces(jobs: Iterable[GenerationJob]) -> dict[str, GenerationJobTrace]:
    """Create empty traces keyed by job id."""

    traces: dict[str, GenerationJobTrace] = {}
    for job in jobs:
        if job.job_id in traces:
            raise ValueError(f"duplicate job_id: {job.job_id}")
        traces[job.job_id] = GenerationJobTrace.from_job(job)
    return traces


@dataclass(frozen=True)
class StageExecutionContext:
    """Runtime-only reusable handles available to every stage handler.

    Handles may be Python objects in this implementation, but metadata stays
    scalar and portable so the same lifecycle contract can be reimplemented in
    MLX Swift.
    """

    run_id: str
    handles: Mapping[str, object] = field(default_factory=dict)
    handle_metadata: Mapping[str, Mapping[str, StageArtifactValue]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("StageExecutionContext requires a run_id")
        handles = dict(self.handles)
        metadata = {handle_id: _validate_artifacts(values) for handle_id, values in self.handle_metadata.items()}
        unknown_metadata = sorted(set(metadata) - set(handles))
        if unknown_metadata:
            raise ValueError(f"metadata for unknown handle_id: {', '.join(unknown_metadata)}")
        object.__setattr__(self, "handles", handles)
        object.__setattr__(self, "handle_metadata", metadata)

    @property
    def handle_ids(self) -> tuple[str, ...]:
        return tuple(self.handles)

    def require_handle(self, handle_id: str) -> object:
        try:
            return self.handles[handle_id]
        except KeyError as exc:
            raise KeyError(f"missing stage execution handle: {handle_id}") from exc


@dataclass(frozen=True)
class StageRunnerOutput:
    """State update returned by a stage handler.

    `artifacts` is intentionally a small scalar map so the contract stays easy
    to port to MLX Swift and easy to serialize in future reports.
    """

    result: GenerationStageResult
    artifacts: Mapping[str, StageArtifactValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", _validate_artifacts(self.artifacts))


@dataclass(frozen=True)
class JobState:
    """Portable per-job state snapshot consumed by stage handlers."""

    job_id: str
    seed: int
    images: tuple[str, ...]
    output_path: str
    config: Mapping[str, int | bool | str]
    next_stage_index: int = 0
    stage_results: tuple[GenerationStageResult, ...] = ()
    artifacts: Mapping[str, StageArtifactValue] = field(default_factory=dict)
    failure_phase: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "config", dict(self.config))
        object.__setattr__(self, "artifacts", _validate_artifacts(self.artifacts))

    @classmethod
    def from_job(cls, job: GenerationJob) -> "JobState":
        return cls(
            job_id=job.job_id,
            seed=job.seed,
            images=job.images,
            output_path=str(job.output_path),
            config=job.config_dict(),
        )

    @property
    def ok(self) -> bool:
        return self.failure_phase is None

    def record_stage(
        self,
        invocation: GenerationStageInvocation,
        output: StageRunnerOutput,
    ) -> "JobState":
        if not self.ok:
            raise ValueError(f"cannot record stage for failed job: {self.job_id}")
        if invocation.job_id != self.job_id:
            raise ValueError(f"invocation job_id {invocation.job_id} does not match state {self.job_id}")
        if invocation.stage_index != self.next_stage_index:
            raise ValueError(
                f"stage index mismatch for {self.job_id}: expected {self.next_stage_index}, "
                f"got {invocation.stage_index}"
            )
        if output.result.stage != invocation.stage:
            raise ValueError(
                f"result stage {output.result.stage} does not match invocation stage {invocation.stage}"
            )

        return replace(
            self,
            next_stage_index=self.next_stage_index + 1,
            stage_results=(*self.stage_results, output.result),
            artifacts=_validate_artifacts({**self.artifacts, **output.artifacts}),
            failure_phase=output.result.failure_phase,
        )

    def record_artifacts(self, artifacts: Mapping[str, StageArtifactValue]) -> "JobState":
        return replace(self, artifacts=_validate_artifacts({**self.artifacts, **artifacts}))


StageHandler = Callable[
    [GenerationStageInvocation, JobState, StageExecutionContext],
    StageRunnerOutput | GenerationStageResult,
]
StageContextFactory = Callable[[InterleavedBatchPlan], StageExecutionContext]
StageContextCloser = Callable[[StageExecutionContext], None]


def _default_context_factory(plan: InterleavedBatchPlan) -> StageExecutionContext:
    return StageExecutionContext(run_id="stage-runner", handles={})


def _noop_context_closer(context: StageExecutionContext) -> None:
    return None


@dataclass(frozen=True)
class StageRunResult:
    """Final state for a stage-major runner invocation."""

    job_states: Mapping[str, JobState]
    context: StageExecutionContext
    context_closed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_states", dict(self.job_states))

    @property
    def ok(self) -> bool:
        return all(state.ok for state in self.job_states.values())


class StageRunner:
    """Execute a stage-major plan against explicit per-stage handlers."""

    def __init__(
        self,
        plan: InterleavedBatchPlan,
        *,
        handlers: Mapping[str, StageHandler],
        context_factory: StageContextFactory | None = None,
        context_closer: StageContextCloser | None = None,
    ) -> None:
        missing = [stage for stage in plan.stages if stage not in handlers]
        if missing:
            raise ValueError(f"missing stage handlers: {', '.join(missing)}")
        self.plan = plan
        self.handlers = dict(handlers)
        self.context_factory = context_factory or _default_context_factory
        self.context_closer = context_closer or _noop_context_closer

    def run(self, initial_states: Mapping[str, JobState] | None = None) -> StageRunResult:
        planned_ids = {job.job_id for job in self.plan.jobs}
        if initial_states is None:
            states = {job.job_id: JobState.from_job(job) for job in self.plan.jobs}
        else:
            provided_ids = set(initial_states)
            missing_ids = [job.job_id for job in self.plan.jobs if job.job_id not in provided_ids]
            if missing_ids:
                raise ValueError(f"missing initial state for job_id: {', '.join(missing_ids)}")
            extra_ids = sorted(provided_ids - planned_ids)
            if extra_ids:
                raise ValueError(f"unexpected initial state job_id: {', '.join(extra_ids)}")
            states = {}
            for job in self.plan.jobs:
                state = initial_states[job.job_id]
                if state.job_id != job.job_id:
                    raise ValueError(
                        f"initial state key {job.job_id} does not match state job_id {state.job_id}"
                    )
                states[job.job_id] = state

        context = self.context_factory(self.plan)
        context_closed = False
        try:
            for invocation in self.plan.iter_invocations():
                state = states[invocation.job_id]
                if not state.ok:
                    continue
                output = self.handlers[invocation.stage](invocation, state, context)
                if isinstance(output, GenerationStageResult):
                    output = StageRunnerOutput(result=output)
                states[invocation.job_id] = state.record_stage(invocation, output)
        finally:
            self.context_closer(context)
            context_closed = True

        return StageRunResult(job_states=states, context=context, context_closed=context_closed)
