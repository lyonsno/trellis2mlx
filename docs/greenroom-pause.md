# Pause and continue a resident Trellis run

Opt in when submitting a new `generate.py` command to Greenroom:

```sh
python generate.py --image warrior.webp --output /results/warrior.glb \
  --greenroom-queue-dir /path/to/the/actual/greenroom-queue
```

Use the queue's existing Pause and Play controls. Trellis checks the explicitly
selected queue's `paused` marker before sampling and after every completed Euler
step, and between the normal pipeline's conditioning, sampling, decode, mesh
cleanup, UV unwrap, texture bake and export stages. HR replay and recovery and
finalizer resume also consume the hook. Existing step-checkpoint callbacks run
before the pause check, so those saved steps remain available.

Pause finishes already-submitted MLX work on the default stream before
acknowledging `paused`. Play removes the marker through Greenroom's normal,
epoch-checked control and continues the same process; it does not reload the
model or rerun completed steps. A new pause epoch keeps the run paused.

This is a **resident computational pause**, not checkpoint-and-exit, memory
release or a GPU ownership handoff. Model tensors and the worker's execution
lock remain held. Other queued jobs cannot start while this job is parked. Do
not start another heavy workload merely because the pause report says `paused`.
A decode, simplifier, unwrap or texture bake already underway must finish before
the next boundary can honor Pause. VS3D's separate editing sampler is not hooked
per step; its surrounding pipeline boundaries remain cooperative. No new wait
or GPU barrier is added when pause support is off or the queue is playing.

Status defaults to `_control/<output-stem>.pause.json` beside the output. Supply
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
