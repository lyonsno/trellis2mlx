# Pause and continue a Trellis run

On a machine with Greenroom installed, an ordinary `python generate.py ...`
automatically submits the exact CLI command to the shared queue and waits for
its terminal continuation. No queue, pause, identity or checkpoint flags are
required. The foreground client relays model logs and writes a unique
`_control/*.admission.json` beside the requested output. Ctrl-C requests
cancellation of its current job, following a completed handoff when necessary.
If a worker claims or completes the job during cancellation, the client refreshes
that state and follows any committed successor. Its receipt distinguishes accepted
running cancellation, cancellation before start, already terminal work and failure
to cancel; client exit alone does not establish process quiescence.
On a machine without Greenroom, standalone generation remains available.
An old managed route that did not admit checkpoint support fails before loading
models; it cannot silently occupy the worker without pause support.
Each automatic submission registers a unique `_control/*.job` output directory;
the caller's containing directory never becomes job-owned storage.

An admitted Greenroom checkpoint job running `generate.py` uses managed
checkpoint/exit/continuation automatically. No Trellis queue, pause-report or
checkpoint-save flags are needed. The context comes from the actual scheduler
request; a stale or wrong request fails before inference. Private virtual
environments discover the SDK through the installed `gpu-greenroom` command.

Completed conditioning, sampling, decode, extraction, cleanup, unwrap and bake
calls are saved under a unique `_control/replay-*` folder beside the output.
Euler sampling also retains the last completed step, including diagnostic
captures and the actual MLX random key. A pause during an unfinished model
block resumes from that step. The producer unwinds model frames and drains MLX
before staging the successor. Greenroom waits for the whole process group to
exit before admitting other work. Resume reloads model weights but reuses the
saved completed results. Queue-wide pause, per-job holds and Quick passage
use the scheduler's existing controls; cancellation does not create a successor.

The saved state binds the source files, arguments, input files, weight files
and MLX version. It includes optional NPZ sidecar presence/content, resumed raw
mesh and texture inputs, and numerical `TRELLIS2MLX_*` environment settings.
Changed identity or tensor bytes fail visibly. These internal
snapshots use exact safetensors/NumPy arrays without pickle. They are distinct
from `--save-checkpoints` research outputs. Pending, failed, witness and explicit
research snapshots remain retained without automatic expiry. After an ordinary
final export succeeds, only this controller's recorded, unchanged tensor scratch
files are removed. Unknown files, changed files and final assets are preserved;
no scheduled cleaner or retention timer is installed.
The same rule covers ordinary submissions with no output class; an explicit
witness or intermediate class keeps its scratch.

Opaque extraction, cleanup, unwrap and bake calls currently yield after the
call returns. Model block boundaries are interruptible, with unfinished work
replayed from the saved prefix. Compiled model calls retain their surrounding
step boundary. This does not promise a maximum pause latency for opaque calls.
The separate original-order Metal finisher has its own checkpoint integration.
An outer controller launching child CLIs must own the continuation; the CLI
rejects a mismatched admitted entrypoint instead of silently duplicating it.
Existing outer harnesses that spawn the CLI therefore need a continuation
adapter before using this managed route. The ordinary typed generation routes
invoke `generate.py` directly and inherit its source declaration.

The source-owned `.gpu-greenroom-producer.json` declares `generate.py` to
Greenroom versions supporting producer checkpoint declarations. Normal typed
`submit` and structured `submit-command` then admit the capability without a
caller flag. Callers without `--agent-id` retain an unrecorded caller identity;
Greenroom assigns checkpoint custody to the job, preserving caller attribution.
Prepared jobs preserve input validation and rerun their original CPU
preflight for a successor. These paths require the corresponding worker/SDK
capabilities. Older Greenroom installations must be updated before shared CLI
generation; unsupported managed execution fails visibly. A plain resident
queue flag below does not establish managed admission. The declaration applies to this exact
entrypoint; it does not claim arbitrary custom controllers or standalone
finisher scripts are checkpoint-capable.

## Legacy resident Pause/Play

Opt in when submitting a new `generate.py` command to Greenroom:

```sh
python generate.py --image warrior.webp --output /results/warrior.glb \
  --greenroom-queue-dir /path/to/the/actual/greenroom-queue
```

Use the queue's existing Pause and Play controls. Trellis checks the explicitly
selected queue's `paused` marker before sampling and after every completed Euler
step, and between the normal pipeline's conditioning, sampling, decode, mesh
cleanup, UV unwrap, texture bake and export stages. The active invocation also
checks inside ordinary uncompiled shape/texture flow blocks and sparse decoder
levels/blocks. Direct Euler consumers inside `pause_session` inherit the control;
callers do not have to forward a callback through each sampler invocation.

Pause finishes already-submitted MLX work on the default stream before
acknowledging `paused`. Play removes the marker through Greenroom's normal,
epoch-checked control and continues the same process; it does not reload the
model or rerun completed steps. A new pause epoch keeps the run paused.

This is a **resident computational pause**, not checkpoint-and-exit, memory
release or a GPU ownership handoff. Model tensors and the worker's execution
lock remain held. Other queued jobs cannot start while this job is parked. Do
not start another heavy workload merely because the pause report says `paused`.
A submitted kernel must finish before the next boundary can honor Pause.
Compiled whole-flow blocks, dense sparse-structure decoding, simplification,
unwrap and texture bake check at their surrounding step/stage boundary. VS3D's separate editing sampler is not hooked
per step; its surrounding pipeline boundaries remain cooperative. No new wait
or GPU barrier is added when pause support is off or the queue is playing.

Set `TRELLIS2MLX_GREENROOM_QUEUE_DIR` in a structured-command environment to
select the queue without extra CLI arguments. `TRELLIS2MLX_GREENROOM_PAUSE_REPORT`
selects a report path; CLI arguments override these environment defaults. Queue
paths are explicit and are never inferred from the working directory.

Status defaults to a unique `_control/<output-stem>.<run-id>.pause.json` beside
the output, allowing repeated invocations to use the same output name. Supply
`--greenroom-pause-report /results/run-id/pause.json` to choose another path. The
path must be fresh, so an older process's paused acknowledgment cannot masquerade
as this run. The report includes the run UUID, PID, source and queue paths, device,
last observed stage/step, pause epoch, drain state and explicit memory/lock
retention. Normal returns, including requested early-stage stops, report
`finished`; that means the invocation ended, not that a GLB necessarily exists.
Exceptions report `failed` or `stopped`. A killed process may leave stale state;
the report is not a heartbeat, process-liveness proof or Greenroom job receipt.

The existing Greenroom monitor still conservatively calls the subprocess running
while it is resident. Its queue acknowledgment describes admission, not the
Trellis drain acknowledgment. Monitor rendering of that separate acknowledgment
is not part of this hook. Runs launched before these hooks were installed cannot
gain them retroactively. The older `--checkpoint-stop-file` still means save and
exit; it is a different control and is unchanged.

For a direct sampler or decoder assay, scope the existing objects explicitly:

```python
from trellmlx.greenroom_pause import pause_session

with pause_session(queue_dir, fresh_report_path):
    result = flow_euler_sample(model, noise, cond, neg_cond, **settings)
    decoded = decoder(result, coords)
```

This resident protocol reads the actual Greenroom queue Pause marker. It does
not claim command-v2 `cooperative_checkpoint` support, which means a different
checkpoint/exit/queued-continuation protocol. The pause status report records
that memory and the worker lock remain held.
