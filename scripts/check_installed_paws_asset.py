"""CPU observer for an ordinary installed CLI's pause/passage/final-GLB check.

The model child admits itself through Greenroom; this observer owns no GPU
lease. Its report is lifecycle evidence, never numerical or visual parity.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid


ROUTE = "trellis2mlx/ordinary-cli/automatic-checkpoint"
BUCKETS = ("pending", "running", "done", "failed", "cancelled", "deferred")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def source(root, expected):
    root = Path(root).resolve()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if sha != expected or dirty:
        raise ValueError(f"source mismatch or dirty checkout: {root}")
    return {"root": str(root), "commit": sha, "dirty": False}


def lineage_helpers(root, sha):
    """Load the immutable queue-only functions without model-library imports.

    Importing greenroom_entry normally imports its checkpoint module and MLX.
    These two functions need only Path/json/hash and the existing queue API.
    Source identity is retained so this is production code, not a second edge
    implementation or a synthetic replacement for the continuation contract.
    """
    path = "trellmlx/greenroom_entry.py"
    raw = subprocess.check_output(["git", "show", sha + ":" + path], cwd=root)
    names = {"_successor", "_successor_record"}
    nodes = [node for node in ast.parse(raw).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError("production lineage helpers missing from pinned source")
    namespace = {"Path": Path, "json": json, "_hash": digest}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(Path(root) / path), "exec"), namespace)
    return SimpleNamespace(_successor=namespace["_successor"]), {
        "path": path, "revision": sha, "sha256": hashlib.sha256(raw).hexdigest(),
        "functions": sorted(names), "mode": "immutable production function projection; no module imports"}


def validate_request(request, plan):
    if (request.get("repo_root") != plan["producer_root"]
            or request.get("command_argv") != plan["argv"]
            or request.get("command_cwd") != plan["cwd"]
            or request.get("route_identity") != ROUTE
            or request.get("agent_id") is not None
            or request.get("cooperative_checkpoint") is not True
            or not request.get("params", {}).get("_producer_checkpoint")):
        raise ValueError("wrong ordinary automatic-checkpoint request")


def validate_receipt(snapshot, plan):
    state, receipt = snapshot["state"], snapshot["receipt"]
    worker = receipt.get("worker", {})
    if (state["status"] != "done" or state.get("exit_code") != 0
            or receipt.get("status") != "done" or receipt.get("exit_code") != 0
            or receipt.get("effective_argv") != plan["argv"]
            or receipt.get("effective_cwd") != plan["cwd"]
            or receipt.get("requested_route") != ROUTE
            or worker.get("source_root") != plan["sdk_root"]
            or worker.get("commit") != plan["sdk_sha"]
            or worker.get("git_dirty") is not False):
        raise ValueError("terminal worker route/source or successful execution mismatch")


def validate_pause(producer, plan, job):
    if (producer.get("mode") != "checkpoint-exit-continuation"
            or producer.get("queue_dir") != plan["queue"] or producer.get("job_id") != job
            or producer.get("source_root") != plan["producer_root"]
            or producer.get("sdk") != str(Path(plan["sdk_root"]) / "gpu_queue/cooperative.py")):
        raise ValueError("producer pause state has wrong route/source/job")


def validate_glb(path):
    payload = Path(path).read_bytes()
    if len(payload) < 20 or struct.unpack_from("<4sII", payload) != (b"glTF", 2, len(payload)):
        raise ValueError("missing or malformed GLB container")
    length, kind = struct.unpack_from("<I4s", payload, 12)
    if kind != b"JSON" or 20 + length + 8 > len(payload):
        raise ValueError("missing GLB JSON/binary chunks")
    scene = json.loads(payload[20:20 + length])
    binary_length, binary_kind = struct.unpack_from("<I4s", payload, 20 + length)
    if binary_kind != b"BIN\0" or 28 + length + binary_length != len(payload):
        raise ValueError("incomplete GLB binary chunk")
    binary = payload[28 + length:]
    if scene.get("asset", {}).get("version") != "2.0" or not scene.get("scenes"):
        raise ValueError("missing glTF scene")
    views, accessors = scene["bufferViews"], scene["accessors"]
    for view in views:
        start, size = view.get("byteOffset", 0), view["byteLength"]
        if view.get("buffer", 0) != 0 or start < 0 or size <= 0 or start + size > len(binary):
            raise ValueError("GLB buffer view outside binary payload")
    primitives = [p for mesh in scene.get("meshes", []) for p in mesh.get("primitives", [])]
    if not primitives:
        raise ValueError("GLB has no mesh primitives")

    def values(accessor, kind, width):
        if accessor.get("type") != kind or accessor.get("sparse"):
            raise ValueError("unsupported or sparse geometry accessor")
        fmt = {5126: "f", 5125: "I", 5123: "H", 5121: "B"}[accessor["componentType"]]
        row = struct.Struct("<" + fmt * width)
        view = views[accessor["bufferView"]]
        offset, stride, count = accessor.get("byteOffset", 0), view.get("byteStride", row.size), accessor["count"]
        if (count < 1 or offset < 0 or stride < row.size
                or offset + (count - 1) * stride + row.size > view["byteLength"]):
            raise ValueError("geometry accessor exceeds its buffer view")
        start = view.get("byteOffset", 0) + offset
        return [row.unpack_from(binary, start + index * stride) for index in range(count)]

    for primitive in primitives:
        positions = accessors[primitive["attributes"]["POSITION"]]
        indices = accessors[primitive["indices"]]
        if (positions["count"] < 3 or positions["componentType"] != 5126
                or indices["count"] < 3 or indices["count"] % 3 or primitive.get("mode", 4) != 4):
            raise ValueError("GLB has empty geometry")
        xyz, triangles = values(positions, "VEC3", 3), values(indices, "SCALAR", 1)
        if not all(math.isfinite(value) for point in xyz for value in point):
            raise ValueError("GLB geometry is nonfinite")
        if any(index[0] >= len(xyz) for index in triangles):
            raise ValueError("GLB indices exceed its positions")
        if len(set(xyz)) < 3:
            raise ValueError("GLB geometry is collapsed")
        material = scene["materials"][primitive["material"]]
        texture = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        image = scene["images"][scene["textures"][texture]["source"]]
        view = views[image["bufferView"]]
        start, size = view.get("byteOffset", 0), view["byteLength"]
        image_bytes = binary[start:start + size]
        if not (image_bytes.startswith(b"\x89PNG\r\n\x1a\n") or image_bytes.startswith(b"\xff\xd8")):
            raise ValueError("missing embedded base-color image")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            "primitives": len(primitives), "images": len(scene["images"]),
            "claim": "textured GLB structure; visual quality requires separate inspection"}


def wait_terminal(read, job, record=lambda value: value, poll=time.sleep):
    while True:
        snapshot = read(job)
        status = snapshot["state"]["status"]
        if status in ("done", "failed", "cancelled") and snapshot["bucket"] == status:
            if "receipt" not in snapshot:
                raise ValueError("published terminal record has no execution receipt")
            return record(snapshot)
        poll(.5)


def run(plan, report_path):
    report = {"schema": "trellis2mlx.installed-paws-asset.v1", "status": "running",
              "phase": "preflight", "plan": plan, "events": [], "started_at": time.time()}
    caller = None
    queue = None
    held = None
    expected = None
    entry = None

    def write(**changes):
        report.update(changes, updated_at=time.time())
        dump(report_path, report)

    def record(snapshot):
        report["events"].append(snapshot)
        write()
        return snapshot

    def read(job):
        with queue._coordination_lock():
            matches = [queue.queue_dir / b / job for b in BUCKETS
                       if (queue.queue_dir / b / job / "request.json").is_file()]
            if len(matches) != 1:
                raise ValueError("missing or ambiguous exact job record: " + job)
            path = matches[0]
            result = {"at": time.time(), "path": str(path), "bucket": path.parent.name,
                      "request": json.loads((path / "request.json").read_text()),
                      "request_sha256": digest(path / "request.json"),
                      "state": json.loads((path / "status.json").read_text())}
            for filename, key in (("receipt.json", "receipt"), ("checkpoint-handoff.json", "handoff")):
                if (path / filename).is_file():
                    result[key] = json.loads((path / filename).read_text())
                    result[key + "_sha256"] = digest(path / filename)
            return result

    def action(job, verb):
        snapshot = read(job)
        validate_request(snapshot["request"], plan)
        state = snapshot["state"]
        # Bypass applies only to this observer's voluntary per-job hold. It
        # lets ordinary FIFO jobs pass; it never removes a queue-wide pause.
        return controls.perform(queue, job_id=job, request_digest=snapshot["request_sha256"],
            expected_status=state["status"], expected_started_at=state.get("started_at"),
            request_id=uuid.uuid4().hex, requested_by=plan["control_owner"] + " exact own-job consumer",
            action=verb, allow_bypass=True)

    write()
    try:
        root = Path(plan["run_root"])
        output = Path(plan["output"])
        if output.exists() or (output.parent / "_control").exists():
            raise ValueError("stale output or admission directory; choose a fresh run")
        output.parent.mkdir(parents=True, exist_ok=True)
        report["producer"] = source(plan["producer_root"], plan["producer_sha"])
        report["sdk"] = source(plan["sdk_root"], plan["sdk_sha"])
        if digest(plan["input"]) != plan["input_sha256"]:
            raise ValueError("source image changed")
        entry, report["lineage_helper_source"] = lineage_helpers(plan["producer_root"], plan["producer_sha"])
        sys.path.insert(0, plan["sdk_root"])
        from gpu_queue.queue import GPUQueue
        from gpu_queue.models import JobRequest
        from gpu_queue import job_controls as controls
        import gpu_queue
        if Path(gpu_queue.__file__).resolve().parent.parent != Path(plan["sdk_root"]).resolve():
            raise ValueError("wrong installed SDK module")
        queue = GPUQueue(Path(plan["queue"]))
        env = {k: v for k, v in os.environ.items() if k not in
               {"PYTHONPATH", "GPU_GREENROOM_CONTEXT", "GPU_GREENROOM_JOB_ID", "GREENROOM_JOB_ID",
                "GPU_GREENROOM_QUEUE_DIR", "GPU_GREENROOM_REPORT_PATH", "GPU_GREENROOM_RESUME_CHECKPOINT"}
               and not k.startswith("TRELLIS2MLX_")}
        with (root / "caller.stdout.log").open("w") as stdout, (root / "caller.stderr.log").open("w") as stderr:
            caller = subprocess.Popen(plan["argv"], cwd=plan["cwd"], env=env, stdout=stdout, stderr=stderr)
            write(phase="automatic-admission", caller_pid=caller.pid)
            while True:
                admissions = list((output.parent / "_control").glob(output.stem + ".*.admission.json"))
                if len(admissions) > 1:
                    raise ValueError("ambiguous automatic admission")
                if admissions:
                    admission = json.loads(admissions[0].read_text())
                    if admission.get("job_id"):
                        break
                if caller.poll() is not None:
                    raise RuntimeError("ordinary caller ended before automatic admission")
                time.sleep(.5)
            parent = admission["job_id"]
            initial = record(read(parent))
            validate_request(initial["request"], plan)
            expected = JobRequest.from_json(json.dumps(initial["request"]))
            write(phase="wait-for-mid-shape", admission_path=str(admissions[0]), parent_job=parent)
            while True:
                snapshot = read(parent)
                if snapshot["state"]["status"] in ("done", "failed", "cancelled"):
                    raise RuntimeError("model ended before intended mid-shape pause")
                states = list((output.parent / "_control").glob("*.pause.json"))
                if len(states) > 1:
                    raise ValueError("ambiguous producer pause state")
                if states:
                    producer = json.loads(states[0].read_text())
                    if producer.get("stage") == "sampler_2" and producer.get("completed_step", -1) >= 0:
                        validate_pause(producer, plan, parent)
                        report["pause_at"] = producer
                        report["pause_request"] = action(parent, "pause")
                        held = parent
                        write(phase="wait-for-process-exit")
                        break
                time.sleep(.25)
            terminal = wait_terminal(read, parent, record)
            validate_receipt(terminal, plan)
            edge = terminal.get("handoff", {})
            if edge.get("status") != "committed" or edge.get("job_id") != parent:
                raise ValueError("missing committed outgoing pause edge")
            observed = queue.get_job(parent)
            successor = entry._successor(queue, expected, parent, observed, [])
            if successor != edge.get("next_job_id"):
                raise ValueError("successor custody mismatch")
            if queue._process_group_alive(terminal["state"].get("child_process_group")) is not False:
                raise ValueError("parent process group is not confirmed dead")
            child = record(read(successor))
            if child["bucket"] != "pending":
                raise ValueError("successor did not retain the own-job hold")
            held = successor
            report["parent_group_exited"] = True
            probe = JobRequest(job_type="command", input_path="", agent_id=plan["control_owner"],
                repo_root=plan["producer_root"], command_cwd=str(root),
                command_argv=[sys.executable, "-c", "import os,time,json;from pathlib import Path;"
                              "Path('passage.json').write_text(json.dumps({'pid':os.getpid(),'at':time.time()}))"],
                output_dir=str(root / "probe"), output_class="witness", service_class="normal",
                route_identity="trellis2mlx/paws-final-asset/ordinary-passage-probe")
            queue.submit(probe)
            write(phase="ordinary-queued-passage", probe_job=probe.job_id, successor_job=successor)
            passage = wait_terminal(read, probe.job_id, record)
            if (passage["state"]["status"] != "done" or passage["receipt"].get("exit_code") != 0
                    or passage["state"]["started_at"] < terminal["state"]["finished_at"]
                    or not (root / "passage.json").is_file() or read(successor)["bucket"] != "pending"):
                raise ValueError("normal queued passage/held successor not established")
            report["resume_request"] = action(successor, "resume")
            held = None
            write(phase="resumed-final-asset")
            current, seen = successor, [parent]
            while True:
                terminal = wait_terminal(read, current, record)
                validate_receipt(terminal, plan)
                next_job = entry._successor(queue, expected, current, queue.get_job(current), seen)
                if next_job is None:
                    break
                seen.append(current)
                current = next_job
            while caller.poll() is None:
                time.sleep(.5)
            if caller.returncode != 0:
                raise RuntimeError("ordinary caller failed after terminal continuation")
            report["glb"] = validate_glb(output)
            report["terminal_admission"] = json.loads(admissions[0].read_text())
            if report["terminal_admission"].get("terminal_job") != current:
                raise ValueError("foreground caller did not follow actual terminal job")
            source(plan["producer_root"], plan["producer_sha"])
            source(plan["sdk_root"], plan["sdk_sha"])
            write(status="done", phase="complete", terminal_job=current, finished_at=time.time())
    except BaseException as error:
        write(status="failed", failure_phase=report["phase"], error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if caller is not None and caller.poll() is None:
            # Foreground SIGTERM invokes its source-owned exact-job cancellation.
            caller.terminate()
            report["caller_cleanup_exit"] = caller.wait()
        if held is not None:
            try:
                snapshot = read(held)
                if snapshot["bucket"] in ("pending", "running"):
                    report["hold_release"] = action(held, "resume")
                else:
                    report["hold_terminal"] = snapshot["bucket"]
            except Exception as error:
                report["hold_release_error"] = str(error)
        write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # Even malformed/missing plans have a durable phase/error return.
    try:
        plan = json.loads(args.plan.read_text())
    except BaseException as error:
        dump(args.report, {"status": "failed", "failure_phase": "read-plan", "error": str(error)})
        raise
    run(plan, args.report)


if __name__ == "__main__":
    main()
