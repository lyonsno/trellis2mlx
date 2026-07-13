#!/usr/bin/env python3
"""Build a semantic, route-identified atlas from causal TRELLIS.2 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class AtlasContractError(ValueError):
    """Raised when an input cannot support the claimed atlas coordinate."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the TRELLIS.2 causal basin atlas")
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--alpha", required=True, action="append", type=Path)
    parser.add_argument("--composition", required=True, type=Path)
    parser.add_argument("--operation", type=Path)
    parser.add_argument("--operation-replays", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    atlas = build_atlas(
        prefix_path=args.prefix,
        alpha_paths=args.alpha,
        composition_path=args.composition,
        operation_path=args.operation,
        operation_replays_path=args.operation_replays,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(atlas, indent=2) + "\n", encoding="utf-8")
    args.output_html.write_text(render_html(atlas), encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": atlas["schema"],
                "chart_count": len(atlas["charts"]),
                "node_count": sum(len(chart["nodes"]) for chart in atlas["charts"].values()),
                "output_json": str(args.output_json),
                "output_html": str(args.output_html),
            },
            sort_keys=True,
        )
    )
    return 0


def build_atlas(
    *,
    prefix_path: Path,
    alpha_paths: list[Path],
    composition_path: Path,
    operation_path: Path | None,
    operation_replays_path: Path | None = None,
) -> dict[str, Any]:
    if not alpha_paths:
        raise AtlasContractError("at least one attention-alpha summary is required")

    prefix = _load_json(prefix_path)
    alphas = [_load_json(path) for path in alpha_paths]
    composition = _load_json(composition_path)
    operation = _load_json(operation_path) if operation_path is not None else None
    operation_replays = (
        _load_json(operation_replays_path) if operation_replays_path is not None else None
    )

    sources = [
        _source_identity("source_prefix", prefix_path, prefix),
        *[
            _source_identity(f"attention_alpha_{index}", path, payload)
            for index, (path, payload) in enumerate(zip(alpha_paths, alphas, strict=True))
        ],
        _source_identity("composition", composition_path, composition),
    ]
    if operation_path is not None and operation is not None:
        sources.append(_source_identity("block_operation", operation_path, operation))
    if operation_replays_path is not None and operation_replays is not None:
        sources.append(
            _source_identity("operation_replays", operation_replays_path, operation_replays)
        )

    charts = {
        "source_prefix": _build_prefix_chart(prefix),
        "attention_alpha": _build_alpha_chart(alphas),
        "composition": _build_composition_chart(composition),
    }
    if operation is not None:
        charts["block_operation"] = _build_operation_chart(operation)
    if operation_replays is not None:
        charts["operation_replay"] = _build_operation_replay_chart(operation_replays)

    return {
        "schema": "trellis2mlx.causal_basin_atlas.v1",
        "coordinate_authority": "named routed interventions and explicit continuations",
        "embedding_authority": "none",
        "evidence_limit": (
            "Coordinates encode interventions, not global manifold distance; summary-only nodes "
            "remain non-authoritative until their effective route identity is visible."
        ),
        "sources": sources,
        "charts": charts,
    }


def _build_prefix_chart(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _require_rows(payload, "source-prefix")
    baseline_rows = [row for row in rows if row.get("boundary") == "mlx_baseline"]
    if len(baseline_rows) != 1:
        raise AtlasContractError("source-prefix chart requires exactly one mlx_baseline row")

    nodes: list[dict[str, Any]] = []
    for row in rows:
        if row.get("boundary") == "mlx_baseline":
            block = -1
            artifact = payload.get("baseline_path")
            label = "MLX baseline"
        else:
            if "block" not in row:
                raise AtlasContractError("source-prefix row is missing block coordinate")
            block = int(row["block"])
            artifact = row.get("path")
            label = f"source through block {block}"
        _require_metrics(row, ("pred_final_source_mean_abs", "pred_final_remaining_norm_ratio"))
        nodes.append(
            {
                "id": f"prefix:{block}",
                "chart": "source_prefix",
                "label": label,
                "coordinate": {"block": block},
                "placement": "on_axis",
                "artifact": artifact,
                "route_identity": _route_identity(artifact),
                "intervention": {
                    "kind": "natural_source_cuda_prefix_then_mlx_suffix",
                    "source_prefix_through_block": block if block >= 0 else None,
                    "mlx_suffix_from_block": block + 1,
                },
                "metrics": _copy_metrics(row),
            }
        )
    nodes.sort(key=lambda node: node["coordinate"]["block"])
    if len({node["coordinate"]["block"] for node in nodes}) != len(nodes):
        raise AtlasContractError("source-prefix chart has duplicate block coordinates")

    edges = []
    for left, right in zip(nodes, nodes[1:]):
        added = right["coordinate"]["block"] - left["coordinate"]["block"]
        edges.append(
            {
                "id": f"{left['id']}->{right['id']}",
                "from": left["id"],
                "to": right["id"],
                "kind": "source_prefix_extension",
                "intervention": {"added_blocks": added},
            }
        )
    return {
        "title": "Longitudinal source-prefix chart",
        "x_axis": {"semantic": "source-prefix depth (last natural source-CUDA block retained)", "field": "block"},
        "y_axis": {"semantic": "remaining source endpoint error norm", "field": "pred_final_remaining_norm_ratio"},
        "nodes": nodes,
        "edges": edges,
    }


def _build_alpha_chart(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_alpha: dict[float, dict[str, Any]] = {}
    for payload in payloads:
        for row in _alpha_rows(payload):
            if "alpha" not in row:
                raise AtlasContractError("attention-alpha row is missing alpha coordinate")
            alpha = float(row["alpha"])
            _require_metrics(row, ("pred_final_source_mean_abs", "move_to_source_norm_ratio"))
            by_alpha[alpha] = row

    nodes = []
    for alpha, row in sorted(by_alpha.items()):
        artifact = row.get("artifact") or row.get("path")
        nodes.append(
            {
                "id": f"attention-alpha:{alpha:g}",
                "chart": "attention_alpha",
                "label": f"alpha {alpha:g}",
                "coordinate": {"alpha": alpha},
                "placement": "on_axis",
                "artifact": artifact,
                "route_identity": _route_identity(artifact),
                "intervention": {
                    "kind": "block1_source_cuda_attention_delta_scale",
                    "block": 1,
                    "stage": "attention_raw",
                    "scale": alpha,
                },
                "metrics": _copy_metrics(row),
            }
        )
    if not nodes:
        raise AtlasContractError("attention-alpha chart has no rows")
    edges = [
        {
            "id": f"{left['id']}->{right['id']}",
            "from": left["id"],
            "to": right["id"],
            "kind": "attention_delta_scale",
            "intervention": {
                "delta_alpha": right["coordinate"]["alpha"] - left["coordinate"]["alpha"]
            },
        }
        for left, right in zip(nodes, nodes[1:])
    ]
    return {
        "title": "Transverse block1 attention chart",
        "x_axis": {"semantic": "attention delta scale", "field": "alpha"},
        "y_axis": {"semantic": "source endpoint mean absolute error", "field": "pred_final_source_mean_abs"},
        "nodes": nodes,
        "edges": edges,
    }


def _build_composition_chart(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _require_rows(payload, "composition")
    nodes = []
    for index, row in enumerate(rows):
        name = str(row.get("name", f"composition_{index}"))
        artifact = row.get("path")
        nodes.append(
            {
                "id": f"composition:{name}",
                "chart": "composition",
                "label": name.replace("_", " "),
                "coordinate": {"ordinal": index},
                "placement": "off_chart",
                "artifact": artifact,
                "route_identity": _route_identity(artifact),
                "intervention": {"kind": "multi_site_source_tensor_composition", "name": name},
                "metrics": _copy_metrics(row),
            }
        )
    return {
        "title": "Composed and misregistered interventions",
        "x_axis": {"semantic": "cosine to source displacement", "field": "cosine_to_source_direction"},
        "y_axis": {"semantic": "source endpoint mean absolute error", "field": "pred_final_source_mean_abs"},
        "nodes": nodes,
        "edges": [],
    }


def _build_operation_chart(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _require_rows(payload, "block-operation")
    branch_indices: dict[str, int] = {}
    nodes = []
    for row in rows:
        branch = str(row.get("branch", "both"))
        stage = row.get("stage")
        if not stage:
            raise AtlasContractError("block-operation row is missing stage coordinate")
        stage_index = branch_indices.get(branch, 0)
        branch_indices[branch] = stage_index + 1
        nodes.append(
            {
                "id": f"operation:{branch}:{stage_index}:{stage}",
                "chart": "block_operation",
                "label": f"{branch} {stage}",
                "coordinate": {"branch": branch, "stage_index": stage_index, "stage": stage},
                "placement": "on_axis",
                "artifact": payload.get("candidate_path"),
                "route_identity": _route_identity(payload.get("candidate_path")),
                "intervention": {
                    "kind": "exact_common_input_operation_boundary",
                    "comparison_class": payload.get("comparison_class"),
                    "branch": branch,
                    "stage": stage,
                },
                "metrics": _copy_metrics(row),
            }
        )
    edges = []
    for branch in branch_indices:
        branch_nodes = [node for node in nodes if node["coordinate"]["branch"] == branch]
        for left, right in zip(branch_nodes, branch_nodes[1:]):
            edges.append(
                {
                    "id": f"{left['id']}->{right['id']}",
                    "from": left["id"],
                    "to": right["id"],
                    "kind": "operation_continuation",
                    "intervention": {
                        "from_stage": left["coordinate"]["stage"],
                        "to_stage": right["coordinate"]["stage"],
                    },
                }
            )
    return {
        "title": "Operation chart",
        "x_axis": {"semantic": "ordered operation boundary", "field": "stage_index"},
        "y_axis": {"semantic": "source mean absolute delta", "field": "source_mean_abs", "scale": "log1p"},
        "nodes": nodes,
        "edges": edges,
    }


def _build_operation_replay_chart(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "done":
        raise AtlasContractError("operation replay summary is not status done")
    rows = payload.get("replay_rows")
    if not isinstance(rows, list) or len(rows) < 2:
        raise AtlasContractError("operation replay summary needs at least two replay_rows")
    nodes = []
    for row in rows:
        _require_metrics(row, ("intervention_depth", "pred_final_source_mean_abs"))
        name = str(row.get("name", "unnamed"))
        artifact = row.get("artifact")
        nodes.append(
            {
                "id": f"operation-replay:{name}",
                "chart": "operation_replay",
                "label": name.replace("_", " "),
                "coordinate": {"intervention_depth": int(row["intervention_depth"])},
                "placement": "on_axis",
                "artifact": artifact,
                "route_identity": _route_identity(artifact),
                "intervention": {
                    "kind": "block29_causal_boundary_replay",
                    "manifest_identity": row.get("manifest_identity"),
                },
                "metrics": _copy_metrics(row),
            }
        )
    nodes.sort(key=lambda node: node["coordinate"]["intervention_depth"])
    edges = [
        {
            "id": f"{left['id']}->{right['id']}",
            "from": left["id"],
            "to": right["id"],
            "kind": "causal_boundary_replay",
            "intervention": {"from": left["label"], "to": right["label"]},
        }
        for left, right in zip(nodes, nodes[1:])
    ]
    return {
        "title": "Block29 causal replay endpoints",
        "x_axis": {
            "semantic": "deepest exact source operation boundary",
            "field": "intervention_depth",
        },
        "y_axis": {
            "semantic": "guided endpoint source mean absolute error",
            "field": "pred_final_source_mean_abs",
        },
        "nodes": nodes,
        "edges": edges,
    }


def _alpha_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if isinstance(rows, list):
        return [dict(row) for row in rows]
    points = payload.get("points")
    if not isinstance(points, dict):
        raise AtlasContractError("attention-alpha summary has neither rows nor points")
    result = []
    for coordinate, point in points.items():
        row = dict(point)
        row.setdefault("alpha", float(coordinate))
        pred_final = row.get("arrays", {}).get("pred_final", {})
        aliases = {
            "pred_final_source_mean_abs": "source_mean_abs_after",
            "move_to_source_norm_ratio": "move_norm_over_source_displacement_norm",
            "cosine_to_source_direction": "move_vs_source_displacement_cosine",
            "projection_fraction": "projection_fraction_of_source_displacement",
            "pred_final_changed_values": "move_nonzero",
        }
        for target, source in aliases.items():
            if target not in row and source in pred_final:
                row[target] = pred_final[source]
        result.append(row)
    return result


def _copy_metrics(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {"artifact", "path", "name", "boundary", "block", "alpha", "branch", "stage"}
    return {
        key: value
        for key, value in row.items()
        if key not in excluded and isinstance(value, (int, float, str, bool, type(None)))
    }


def _route_identity(artifact: Any) -> dict[str, Any]:
    if not artifact:
        return {"status": "missing", "authority": "summary_only", "reason": "artifact path absent"}
    path = Path(str(artifact))
    if not path.is_file():
        return {
            "status": "missing",
            "authority": "summary_only",
            "artifact": str(path),
            "reason": "artifact is not locally present",
        }
    roots = [path.parent, path.parent.parent]
    for root in roots:
        route_path = root / "route_identity.json"
        if route_path.is_file():
            route = _load_json(route_path)
            effective = route.get("route") if isinstance(route.get("route"), dict) else route
            return {
                "status": "visible",
                "authority": "route_identity_file",
                "path": str(route_path),
                "sha256": _sha256(route_path),
                "requested_route": route.get("requested_route") or route.get("requested_stop"),
                "effective_route": effective.get("effective_route") or effective.get("family"),
                "effective_device": (
                    effective.get("effective_device")
                    or effective.get("effective_device_type")
                    or effective.get("device")
                    or effective.get("backend")
                ),
                "effective_attention_backend": (
                    effective.get("effective_attention_backend")
                    or effective.get("attention_backend")
                ),
                "intervention_manifest": effective.get(
                    "shape_flow_block_injection_manifest_path"
                ),
            }
    return {
        "status": "unverified",
        "authority": "artifact_only",
        "artifact": str(path),
        "reason": "artifact exists but route_identity.json is absent",
    }


def _source_identity(role: str, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "schema": payload.get("schema"),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        raise AtlasContractError(f"missing JSON input: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AtlasContractError(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AtlasContractError(f"JSON input must be an object: {path}")
    return payload


def _require_rows(payload: dict[str, Any], label: str) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise AtlasContractError(f"{label} summary has no nonempty rows array")
    if not all(isinstance(row, dict) for row in rows):
        raise AtlasContractError(f"{label} rows must be objects")
    return rows


def _require_metrics(row: dict[str, Any], names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in row]
    if missing:
        raise AtlasContractError(f"row is missing required metrics: {', '.join(missing)}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_html(atlas: dict[str, Any]) -> str:
    data = json.dumps(atlas, separators=(",", ":")).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRELLIS.2 Causal Basin Atlas</title>
<style>
:root{color-scheme:dark;--bg:#101214;--panel:#171a1d;--line:#343a40;--text:#f2f3f5;--muted:#aab0b7;--red:#ff5b57;--cyan:#3ddbd9;--yellow:#ffd166;--green:#66d17a;--violet:#b39cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:0}header{padding:20px 24px 16px;border-bottom:1px solid var(--line);display:flex;gap:24px;align-items:end;justify-content:space-between}h1{margin:0;font-size:22px;font-weight:650;letter-spacing:0}header p{margin:0;color:var(--muted);max-width:720px}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 340px;min-height:calc(100vh - 86px)}main{min-width:0}.chart-section{padding:18px 24px 22px;border-bottom:1px solid var(--line)}.chart-section h2{font-size:14px;margin:0 0 4px}.axis-note{color:var(--muted);font-size:12px;margin-bottom:10px}.chart{display:block;width:100%;height:260px;background:var(--panel);border:1px solid var(--line)}aside{border-left:1px solid var(--line);padding:18px;min-width:0;background:#131619}aside h2{font-size:14px;margin:0 0 12px}.detail{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);font-size:12px}svg text{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;fill:var(--muted);letter-spacing:0}.axis{stroke:#59616a;stroke-width:1}.edge{stroke:#6c747d;stroke-width:1.5;fill:none}.node{cursor:pointer;stroke:#101214;stroke-width:2}.node:focus{outline:none;stroke:#fff;stroke-width:3}.label{pointer-events:none}.limit{color:var(--yellow)}@media(max-width:860px){header{align-items:start;flex-direction:column}.workspace{grid-template-columns:1fr}aside{border-left:0;border-top:1px solid var(--line)}.chart-section{padding:16px;overflow-x:auto}.chart{height:240px;min-width:760px}}
</style>
</head>
<body>
<header><div><h1>TRELLIS.2 Causal Basin Atlas</h1><p>Named interventions, exact continuation edges, and route-visible evidence.</p></div><p class="limit">No PCA or learned embedding defines these coordinates.</p></header>
<div class="workspace"><main id="charts"></main><aside><h2>Selected evidence</h2><div class="detail" id="detail">Select a routed state.</div></aside></div>
<script type="application/json" id="atlas-data">__ATLAS_JSON__</script>
<script>
const atlas=JSON.parse(document.getElementById('atlas-data').textContent);const root=document.getElementById('charts');const detail=document.getElementById('detail');const NS='http:'+'//www.w3.org/2000/svg';
const colors={source_prefix:'#3ddbd9',attention_alpha:'#ff5b57',composition:'#ffd166',block_operation:'#b39cff',operation_replay:'#66d17a'};
const el=(name,attrs={})=>{const n=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))n.setAttribute(k,String(v));return n};
const metric=(node,key,fallback=0)=>{const v=node.metrics[key];return typeof v==='number'&&Number.isFinite(v)?v:fallback};
const extent=(values)=>{let lo=Math.min(...values),hi=Math.max(...values);if(lo===hi){lo-=.5;hi+=.5}return[lo,hi]};
const scale=(v,a,b,c,d)=>c+(v-a)*(d-c)/(b-a);const show=node=>{detail.textContent=JSON.stringify(node,null,2)};
const overlaps=(a,b)=>!(a.r<b.l||a.l>b.r||a.b<b.t||a.t>b.b);
function placeLabel(svg,q,X,Y,boxes,W,p){const px=X(q.x),py=Y(q.y),text=q.n.label,width=Math.min(230,text.length*6.8+4);const anchor=px>W-p.r-180?'end':'start';const x=anchor==='end'?px-9:px+9;const offsets=q.n.coordinate.branch==='neg'?[16,30,-10,-24,44]:[-10,-24,16,30,-38,44];for(const dy of offsets){const box={l:anchor==='end'?x-width:x,r:anchor==='end'?x:x+width,t:py+dy-10,b:py+dy+3};if(box.l<p.l||box.r>W-p.r||boxes.some(other=>overlaps(box,other)))continue;const t=el('text',{x,y:py+dy,class:'label','text-anchor':anchor});t.textContent=text;svg.append(t);boxes.push(box);return}}
function draw(chartKey,chart){const section=document.createElement('section');section.className='chart-section';const title=document.createElement('h2');title.textContent=chart.title;const note=document.createElement('div');note.className='axis-note';note.textContent=chart.x_axis.semantic+' / '+chart.y_axis.semantic;const svg=el('svg',{class:'chart',viewBox:'0 0 1000 260',role:'img','aria-label':chart.title});section.append(title,note,svg);root.append(section);const W=1000,H=260,p={l:78,r:34,t:28,b:46};let points=[];
if(chartKey==='source_prefix')points=chart.nodes.map(n=>({n,x:n.coordinate.block,y:metric(n,'pred_final_remaining_norm_ratio')}));
else if(chartKey==='attention_alpha')points=chart.nodes.map(n=>({n,x:n.coordinate.alpha,y:metric(n,'pred_final_source_mean_abs')}));
else if(chartKey==='composition')points=chart.nodes.map(n=>({n,x:metric(n,'cosine_to_source_direction'),y:metric(n,'pred_final_source_mean_abs')}));
else if(chartKey==='operation_replay')points=chart.nodes.map(n=>({n,x:n.coordinate.intervention_depth,y:metric(n,'pred_final_source_mean_abs')}));
else points=chart.nodes.map(n=>({n,x:n.coordinate.stage_index,y:Math.log10(metric(n,'source_mean_abs',0)+1e-9)+(n.coordinate.branch==='neg'?.08:0)}));
if(!points.length)return;const [xmin,xmax]=extent(points.map(q=>q.x)),[ymin,ymax]=extent(points.map(q=>q.y));const X=v=>scale(v,xmin,xmax,p.l,W-p.r),Y=v=>scale(v,ymin,ymax,H-p.b,p.t);svg.append(el('line',{x1:p.l,y1:H-p.b,x2:W-p.r,y2:H-p.b,class:'axis'}),el('line',{x1:p.l,y1:p.t,x2:p.l,y2:H-p.b,class:'axis'}));
for(let i=0;i<=4;i++){const value=ymin+(ymax-ymin)*i/4,y=Y(value);svg.append(el('line',{x1:p.l-4,y1:y,x2:p.l,y2:y,class:'axis'}));const tick=el('text',{x:p.l-8,y:y+4,'text-anchor':'end'});tick.textContent=chartKey==='block_operation'?value.toFixed(1):value.toPrecision(3);svg.append(tick)}
for(let i=0;i<=4;i++){const value=xmin+(xmax-xmin)*i/4,x=X(value);svg.append(el('line',{x1:x,y1:H-p.b,x2:x,y2:H-p.b+4,class:'axis'}));const tick=el('text',{x,y:H-p.b+16,'text-anchor':'middle'});tick.textContent=chartKey==='source_prefix'?Math.round(value):value.toPrecision(3);svg.append(tick)}
for(const edge of chart.edges){const a=points.find(q=>q.n.id===edge.from),b=points.find(q=>q.n.id===edge.to);if(a&&b)svg.append(el('line',{x1:X(a.x),y1:Y(a.y),x2:X(b.x),y2:Y(b.y),class:'edge'}))}
const boxes=[];for(const q of points){let fill=colors[chartKey];if(chartKey==='block_operation'&&q.n.coordinate.branch==='neg')fill='#66d17a';const circle=el('circle',{cx:X(q.x),cy:Y(q.y),r:7,fill,class:'node',tabindex:0});const tip=el('title');tip.textContent=q.n.label;circle.append(tip);circle.addEventListener('click',()=>show(q.n));circle.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' ')show(q.n)});svg.append(circle)}for(const q of points)placeLabel(svg,q,X,Y,boxes,W,p);
const xl=el('text',{x:(p.l+W-p.r)/2,y:H-10,'text-anchor':'middle'});xl.textContent=chart.x_axis.semantic;svg.append(xl)}
for(const [key,chart] of Object.entries(atlas.charts))draw(key,chart);
</script>
</body>
</html>
'''
    return template.replace("__ATLAS_JSON__", data)


if __name__ == "__main__":
    raise SystemExit(main())
