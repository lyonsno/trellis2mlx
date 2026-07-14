"""Render the admitted block29 alpha-beta intervention surface without projection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.summarize_shape_block_intervention_grid import (
    COMPARED_ARRAYS,
    COMPARISON_CLASS,
    SUMMARY_SCHEMA,
    _route_vector,
)


REPORT_SCHEMA = "trellis2mlx.shape_block_intervention_grid_render.v1"


class GridRenderContractError(ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--output-report", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    summary_sha = _sha256(args.summary_json) if args.summary_json.is_file() else None
    phase = "read_summary"
    try:
        if not args.summary_json.is_file() or args.summary_json.stat().st_size == 0:
            raise GridRenderContractError(f"summary is missing or blank: {args.summary_json}")
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
        phase = "validate_summary"
        effective_route = validate_summary(summary)
        summary = dict(summary)
        summary["route_vector"] = effective_route
        phase = "render_html"
        html = render_html(summary, summary_sha256=str(summary_sha))
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        args.output_html.write_text(html, encoding="utf-8")
        if args.output_html.stat().st_size == 0:
            raise GridRenderContractError("rendered HTML is blank")
        report = {
            "schema": REPORT_SCHEMA,
            "status": "done",
            "summary_json": str(args.summary_json),
            "summary_sha256": summary_sha,
            "summary_schema": summary["schema"],
            "summary_status": summary["status"],
            "grid_index": summary["grid_index"],
            "grid_index_sha256": summary["grid_index_sha256"],
            "effective_route": summary["route_vector"],
            "output_html": str(args.output_html),
            "output_html_sha256": _sha256(args.output_html),
        }
    except Exception as exc:
        if args.output_html.exists():
            args.output_html.unlink()
        report = {
            "schema": REPORT_SCHEMA,
            "status": "failed",
            "failure_phase": phase,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "summary_json": str(args.summary_json),
            "summary_sha256": summary_sha,
            "output_html": str(args.output_html),
            "last_trustworthy_evidence": {
                "summary_exists": args.summary_json.is_file(),
                "summary_size_bytes": (
                    args.summary_json.stat().st_size if args.summary_json.is_file() else None
                ),
            },
        }
        _write_json(args.output_report, report)
        return 1
    _write_json(args.output_report, report)
    return 0


def validate_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise GridRenderContractError("summary must be an object")
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise GridRenderContractError(f"summary must use schema {SUMMARY_SCHEMA}")
    if summary.get("status") != "done":
        raise GridRenderContractError("summary status is not done")
    if summary.get("comparison_class") != COMPARISON_CLASS:
        raise GridRenderContractError("summary comparison class is wrong")
    if summary.get("compared_arrays") != list(COMPARED_ARRAYS):
        raise GridRenderContractError("summary does not carry the canonical twelve arrays")
    _require_sha(summary.get("grid_index_sha256"), "grid index")
    _require_sha(summary.get("source_trace_sha256"), "source trace")

    route = summary.get("route_vector")
    try:
        effective_route = _route_vector(route, name="summary")
    except (TypeError, ValueError) as exc:
        raise GridRenderContractError(
            f"summary route is not the complete admitted MLX Metal/fast route: {exc}"
        ) from exc

    axes = summary.get("axes")
    if not isinstance(axes, dict):
        raise GridRenderContractError("summary axes are missing")
    alphas = _axis(axes.get("alpha"), "alpha")
    betas = _axis(axes.get("beta"), "beta")
    expected_coordinates = {(alpha, beta) for alpha in alphas for beta in betas}

    points = summary.get("points")
    if not isinstance(points, list) or summary.get("point_count") != len(points):
        raise GridRenderContractError("summary point count is stale")
    observed_coordinates = set()
    point_digests: dict[tuple[float, float], dict[str, str]] = {}
    for point in points:
        if not isinstance(point, dict) or not isinstance(point.get("coordinate"), dict):
            raise GridRenderContractError("summary point is malformed")
        coordinate = (
            _finite(point["coordinate"].get("alpha"), "point alpha"),
            _finite(point["coordinate"].get("beta"), "point beta"),
        )
        if coordinate in observed_coordinates:
            raise GridRenderContractError("summary has duplicate point coordinates")
        observed_coordinates.add(coordinate)
        _require_sha(point.get("artifact_sha256"), "point artifact")
        _require_sha(point.get("manifest_sha256"), "point manifest")
        state_digests = point.get("state_digests")
        source_metrics = point.get("source_metrics")
        if not isinstance(state_digests, dict) or set(state_digests) != set(COMPARED_ARRAYS):
            raise GridRenderContractError("point state digests do not cover canonical arrays")
        if not isinstance(source_metrics, dict) or set(source_metrics) != set(COMPARED_ARRAYS):
            raise GridRenderContractError("point source metrics do not cover canonical arrays")
        for array_name in COMPARED_ARRAYS:
            _require_sha(state_digests[array_name], f"{array_name} state")
            _validate_metrics(
                source_metrics[array_name],
                f"point {point.get('name')} {array_name} source metrics",
                ("mean_abs", "max_abs", "relative_norm", "nonzero"),
            )
        point_digests[coordinate] = state_digests
    if observed_coordinates != expected_coordinates:
        raise GridRenderContractError("summary points are not the full Cartesian product")

    geometry = summary.get("coordinate_geometry")
    if not isinstance(geometry, dict):
        raise GridRenderContractError("coordinate geometry is missing")
    expected_system = {
        "alpha": "source_delta_scale at block29 after_self",
        "beta": "source_delta_scale at block29 cross_attention_raw",
        "projection": "none",
    }
    if geometry.get("coordinate_system") != expected_system:
        raise GridRenderContractError("coordinate system is not the accepted causal chart")
    if geometry.get("sorted_axes") != {"alpha": sorted(alphas), "beta": sorted(betas)}:
        raise GridRenderContractError("geometry axes differ from summary axes")
    _validate_quotients(geometry.get("quotient_classes"), expected_coordinates, point_digests)
    _validate_cells(geometry.get("cells"), tuple(sorted(alphas)), tuple(sorted(betas)))
    return effective_route


def _validate_quotients(
    quotients: Any,
    expected_coordinates: set[tuple[float, float]],
    point_digests: dict[tuple[float, float], dict[str, str]],
) -> None:
    if not isinstance(quotients, dict) or set(quotients) != set(COMPARED_ARRAYS):
        raise GridRenderContractError("quotient classes do not cover canonical arrays")
    for array_name in COMPARED_ARRAYS:
        groups = quotients[array_name]
        if not isinstance(groups, list) or not groups:
            raise GridRenderContractError(f"{array_name} quotient classes are missing")
        observed = set()
        for group in groups:
            if not isinstance(group, dict) or not isinstance(group.get("coordinates"), list):
                raise GridRenderContractError(f"{array_name} quotient class is malformed")
            digest = group.get("state_digest")
            _require_sha(digest, f"{array_name} quotient")
            coordinates = []
            for raw in group["coordinates"]:
                coordinate = (
                    _finite(raw.get("alpha"), "quotient alpha"),
                    _finite(raw.get("beta"), "quotient beta"),
                )
                if point_digests.get(coordinate, {}).get(array_name) != digest:
                    raise GridRenderContractError(f"{array_name} quotient digest contradicts point")
                coordinates.append(coordinate)
            if group.get("point_count") != len(coordinates) or len(coordinates) != len(set(coordinates)):
                raise GridRenderContractError(f"{array_name} quotient point count is stale")
            if observed.intersection(coordinates):
                raise GridRenderContractError(f"{array_name} quotient classes overlap")
            observed.update(coordinates)
        if observed != expected_coordinates:
            raise GridRenderContractError(f"{array_name} quotient classes are incomplete")


def _validate_cells(cells: Any, alphas: tuple[float, ...], betas: tuple[float, ...]) -> None:
    expected_bounds = {
        (alpha0, alpha1, beta0, beta1)
        for alpha0, alpha1 in zip(alphas, alphas[1:])
        for beta0, beta1 in zip(betas, betas[1:])
    }
    if not isinstance(cells, list) or len(cells) != len(expected_bounds):
        raise GridRenderContractError("coordinate cells are incomplete")
    observed_bounds = set()
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("bounds"), dict):
            raise GridRenderContractError("coordinate cell is malformed")
        alpha = cell["bounds"].get("alpha")
        beta = cell["bounds"].get("beta")
        if not isinstance(alpha, list) or len(alpha) != 2 or not isinstance(beta, list) or len(beta) != 2:
            raise GridRenderContractError("coordinate cell bounds are malformed")
        bounds = tuple(_finite(value, "cell bound") for value in (*alpha, *beta))
        if bounds in observed_bounds:
            raise GridRenderContractError("coordinate cells have duplicate bounds")
        observed_bounds.add(bounds)
        arrays = cell.get("arrays")
        if not isinstance(arrays, dict) or set(arrays) != set(COMPARED_ARRAYS):
            raise GridRenderContractError("cell arrays do not cover canonical arrays")
        for array_name in COMPARED_ARRAYS:
            witness = arrays[array_name]
            lower = witness.get("lower_corner_tangents", {})
            _validate_metrics(lower.get("alpha"), f"{array_name} alpha tangent")
            _validate_metrics(lower.get("beta"), f"{array_name} beta tangent")
            if lower.get("cosine") is not None:
                _finite(lower.get("cosine"), f"{array_name} tangent cosine")
            transport = witness.get("opposite_edge_transport", {})
            for axis in ("alpha", "beta"):
                item = transport.get(axis, {})
                _validate_metrics(item.get("difference"), f"{array_name} {axis} transport")
                if item.get("cosine") is not None:
                    _finite(item.get("cosine"), f"{array_name} {axis} transport cosine")
            _validate_metrics(witness.get("mixed_second_difference"), f"{array_name} mixed term")
    if observed_bounds != expected_bounds:
        raise GridRenderContractError("coordinate cell bounds do not match adjacent axes")


def _validate_metrics(value: Any, label: str, keys: tuple[str, ...] = ("mean_abs", "max_abs", "l2_norm", "nonzero")) -> None:
    if not isinstance(value, dict):
        raise GridRenderContractError(f"{label} are missing")
    for key in keys:
        _finite(value.get(key), f"{label} {key}")


def _axis(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise GridRenderContractError(f"{name} axis is missing")
    axis = tuple(_finite(item, f"{name} axis") for item in value)
    if len(axis) != len(set(axis)):
        raise GridRenderContractError(f"{name} axis has duplicates")
    return axis


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GridRenderContractError(f"{label} must be finite numeric")
    return float(value)


def _require_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise GridRenderContractError(f"{label} SHA256 is invalid")


def render_html(summary: dict[str, Any], *, summary_sha256: str) -> str:
    _require_sha(summary_sha256, "summary")
    effective_route = validate_summary(summary)
    summary = dict(summary)
    summary["route_vector"] = effective_route
    options = "".join(
        f'<option value="{name}"{" selected" if name == "pos_final_output" else ""}>{name}</option>'
        for name in COMPARED_ARRAYS
    )
    payload = json.dumps(
        {"summary_sha256": summary_sha256, "summary": summary},
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en" data-summary-sha256="__SUMMARY_SHA__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRELLIS.2 Block29 Intervention Surface</title>
<style>
:root{color-scheme:dark;--bg:#101214;--panel:#171a1d;--line:#3a4148;--text:#f2f4f5;--muted:#a9b0b7;--cyan:#42d6d0;--red:#ff625d;--yellow:#f5c451;--green:#68d391;--violet:#b7a0ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:0}header{padding:18px 22px;border-bottom:1px solid var(--line);display:flex;align-items:end;justify-content:space-between;gap:24px}h1{font-size:20px;margin:0 0 5px;letter-spacing:0}p{margin:0;color:var(--muted)}.truth{color:var(--yellow);text-align:right}.controls{padding:12px 22px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:end;flex-wrap:wrap;background:#131619}label{display:grid;gap:5px;color:var(--muted);font-size:11px}select{min-height:34px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--text);padding:5px 28px 5px 8px;font:inherit}.layout{display:grid;grid-template-columns:minmax(0,1fr) 340px;min-height:calc(100vh - 126px)}main{padding:18px 22px;min-width:0}.chart-wrap{position:relative;min-height:620px;border:1px solid var(--line);background:var(--panel)}svg{display:block;width:100%;height:620px}.axis{stroke:#707983;stroke-width:1}.gridline{stroke:#30363d;stroke-width:1}.cell{stroke:#171a1d;stroke-width:3;cursor:pointer}.cell:focus,.point:focus{outline:none;stroke:#fff;stroke-width:4}.point{cursor:pointer;stroke:#0d0f10;stroke-width:3}.quotient{fill:none;stroke:var(--violet);stroke-width:2;stroke-dasharray:7 4;pointer-events:none}.label{fill:var(--muted);font:11px ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:0}.cell-value{fill:#fff;font:10px ui-monospace,SFMono-Regular,Menlo,monospace;pointer-events:none;letter-spacing:0}.legend{position:absolute;left:14px;top:14px;padding:8px;border:1px solid var(--line);background:#101214dd;color:var(--muted);font-size:10px}.legend-section+.legend-section{margin-top:9px;padding-top:8px;border-top:1px solid var(--line)}.legend-bar{width:150px;height:8px;margin:5px 0;background:linear-gradient(90deg,#26343b,#42d6d0,#f5c451,#ff625d)}aside{border-left:1px solid var(--line);padding:18px;background:#131619;min-width:0;max-height:calc(100vh - 126px);overflow:auto;position:sticky;top:0}aside h2{font-size:13px;margin:0 0 10px}.detail{white-space:pre-wrap;overflow-wrap:anywhere;color:var(--muted);font-size:11px}.digest{margin-top:18px;padding-top:12px;border-top:1px solid var(--line);overflow-wrap:anywhere;color:#7f8992;font-size:10px}@media(max-width:900px){header{align-items:start;flex-direction:column}.truth{text-align:left}.layout{grid-template-columns:1fr}.chart-wrap{min-height:0}.chart-wrap svg{height:auto;aspect-ratio:640/620}.legend{position:static;border-width:1px 0 0}.label{font-size:16px}.cell-value{font-size:14px}aside{position:static;max-height:520px;border-left:0;border-top:1px solid var(--line)}}
</style>
</head>
<body>
<header><div><h1>TRELLIS.2 Block29 Intervention Surface</h1><p>source_delta_scale at block29 after_self x source_delta_scale at block29 cross_attention_raw</p></div><p class="truth">No PCA or learned embedding defines these coordinates.</p></header>
<div class="controls"><label>Tensor<select id="tensor">__OPTIONS__</select></label><label>Cell field<select id="cell-field"><option value="mixed">mixed second difference</option><option value="tangent">lower-corner tangent cosine</option><option value="alpha_transport">alpha opposite-edge transport</option><option value="beta_transport">beta opposite-edge transport</option></select></label><label>Scalar<select id="scalar"><option value="l2_norm">L2 norm</option><option value="mean_abs">mean absolute</option></select></label></div>
<div class="layout"><main><div class="chart-wrap"><svg id="chart" viewBox="0 0 900 620" role="img" aria-label="Direct alpha-beta intervention surface"></svg><div class="legend"><div class="legend-section"><div id="legend-title">cell magnitude</div><div class="legend-bar"></div><div><span id="legend-min">0</span><span style="float:right" id="legend-max">1</span></div></div><div class="legend-section"><div>point source mean absolute</div><div class="legend-bar"></div><div><span id="point-legend-min">0</span><span style="float:right" id="point-legend-max">1</span></div></div></div></div></main><aside><h2>Evidence and quotient classes</h2><div class="detail" id="detail"></div><div class="digest" id="digest"></div></aside></div>
<script type="application/json" id="grid-data">__GRID_JSON__</script>
<script>
const payload=JSON.parse(document.getElementById('grid-data').textContent),summary=payload.summary,geometry=summary.coordinate_geometry,NS='http:'+'//www.w3.org/2000/svg';
const svg=document.getElementById('chart'),tensor=document.getElementById('tensor'),field=document.getElementById('cell-field'),scalar=document.getElementById('scalar'),detail=document.getElementById('detail'),digest=document.getElementById('digest');
digest.textContent=`summary ${payload.summary_sha256}\ngrid ${summary.grid_index_sha256}\nroute ${summary.route_vector.backend}/${summary.route_vector.attention_backend}`;
const el=(name,attrs={})=>{const n=document.createElementNS(NS,name);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,String(v));return n},finite=(v,label)=>{if(typeof v!=='number'||!Number.isFinite(v))throw new Error(`non-finite ${label}`);return v};
const A=geometry.sorted_axes.alpha,B=geometry.sorted_axes.beta,mobile=window.matchMedia('(max-width:900px)').matches,W=mobile?640:900,H=620,p=mobile?{l:100,r:34,t:34,b:96}:{l:90,r:50,t:46,b:80},loA=Math.min(...A),hiA=Math.max(...A),loB=Math.min(...B),hiB=Math.max(...B),X=a=>p.l+(a-loA)*(W-p.l-p.r)/(hiA-loA||1),Y=b=>H-p.b-(b-loB)*(H-p.t-p.b)/(hiB-loB||1);svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
const cellValue=(cell,name)=>{const w=cell.arrays[name],key=scalar.value;if(field.value==='mixed')return finite(w.mixed_second_difference[key],'mixed term');if(field.value==='tangent')return Math.abs(finite(w.lower_corner_tangents.cosine??0,'tangent cosine'));const axis=field.value==='alpha_transport'?'alpha':'beta';return finite(w.opposite_edge_transport[axis].difference[key],axis+' transport')};
const pointValue=(point,name)=>finite(point.source_metrics[name].mean_abs,'point source mean');const color=(v,lo,hi)=>{const t=hi===lo?.5:Math.max(0,Math.min(1,(v-lo)/(hi-lo)));if(t<.5){const q=t*2;return `rgb(${Math.round(38+28*q)},${Math.round(52+162*q)},${Math.round(59+149*q)})`}const q=(t-.5)*2;return `rgb(${Math.round(66+189*q)},${Math.round(214-116*q)},${Math.round(208-115*q)})`};
const show=value=>{detail.textContent=JSON.stringify(value,null,2)};
function draw(){svg.replaceChildren();const name=tensor.value,cells=geometry.cells,cellValues=cells.map(c=>cellValue(c,name)),pointValues=summary.points.map(q=>pointValue(q,name)),cellLo=Math.min(...cellValues),cellHi=Math.max(...cellValues),pointLo=Math.min(...pointValues),pointHi=Math.max(...pointValues);document.getElementById('legend-min').textContent=cellLo.toExponential(2);document.getElementById('legend-max').textContent=cellHi.toExponential(2);document.getElementById('point-legend-min').textContent=pointLo.toExponential(2);document.getElementById('point-legend-max').textContent=pointHi.toExponential(2);document.getElementById('legend-title').textContent=field.options[field.selectedIndex].text+' / '+scalar.options[scalar.selectedIndex].text;
for(const a of A){svg.append(el('line',{x1:X(a),y1:p.t,x2:X(a),y2:H-p.b,class:'gridline'}));const t=el('text',{x:X(a),y:H-p.b+22,'text-anchor':'middle',class:'label'});t.textContent=a;svg.append(t)}for(const b of B){svg.append(el('line',{x1:p.l,y1:Y(b),x2:W-p.r,y2:Y(b),class:'gridline'}));const t=el('text',{x:p.l-12,y:Y(b)+4,'text-anchor':'end',class:'label'});t.textContent=b;svg.append(t)}
for(const [i,c]of cells.entries()){const [a0,a1]=c.bounds.alpha,[b0,b1]=c.bounds.beta,v=cellValues[i],r=el('rect',{x:X(a0)+2,y:Y(b1)+2,width:Math.max(1,X(a1)-X(a0)-4),height:Math.max(1,Y(b0)-Y(b1)-4),fill:color(v,cellLo,cellHi),class:'cell',tabindex:0,'aria-label':`cell alpha ${a0} to ${a1}, beta ${b0} to ${b1}, value ${v}`});r.addEventListener('click',()=>show({tensor:name,cell:c.bounds,witness:c.arrays[name]}));r.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' ')show({tensor:name,cell:c.bounds,witness:c.arrays[name]})});svg.append(r);const t=el('text',{x:(X(a0)+X(a1))/2,y:(Y(b0)+Y(b1))/2+4,'text-anchor':'middle',class:'cell-value'});t.textContent=v.toExponential(2);svg.append(t)}
const groups=geometry.quotient_classes[name];for(const group of groups.filter(g=>g.point_count>1)){const points=group.coordinates.map(c=>[X(c.alpha),Y(c.beta)]),path=points.map((q,i)=>(i?'L':'M')+q[0]+','+q[1]).join(' ');svg.append(el('path',{d:path,class:'quotient'}))}
for(const [i,q]of summary.points.entries()){const v=pointValues[i],c=el('circle',{cx:X(q.coordinate.alpha),cy:Y(q.coordinate.beta),r:10,fill:color(v,pointLo,pointHi),class:'point',tabindex:0,'aria-label':`${q.name}, source mean ${v}`});c.addEventListener('click',()=>show({tensor:name,point:{name:q.name,coordinate:q.coordinate,artifact_sha256:q.artifact_sha256,manifest_sha256:q.manifest_sha256,control_role:q.control_role,control_exact:q.control_exact,source_metrics:q.source_metrics[name],state_digest:q.state_digests[name]},quotient_class:groups.find(g=>g.coordinates.some(k=>k.alpha===q.coordinate.alpha&&k.beta===q.coordinate.beta))}));c.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' ')c.click()});svg.append(c)}
svg.append(el('line',{x1:p.l,y1:H-p.b,x2:W-p.r,y2:H-p.b,class:'axis'}),el('line',{x1:p.l,y1:p.t,x2:p.l,y2:H-p.b,class:'axis'}));const xl=el('text',{x:(p.l+W-p.r)/2,y:H-24,'text-anchor':'middle',class:'label'});xl.textContent=geometry.coordinate_system.alpha;svg.append(xl);const yl=el('text',{x:22,y:(p.t+H-p.b)/2,transform:`rotate(-90 22 ${(p.t+H-p.b)/2})`,'text-anchor':'middle',class:'label'});yl.textContent=geometry.coordinate_system.beta;svg.append(yl);show({tensor:name,quotient_classes:groups.map(g=>({state_digest:g.state_digest,point_count:g.point_count,coordinates:g.coordinates}))})}
for(const control of[tensor,field,scalar])control.addEventListener('change',draw);draw();
</script>
</body>
</html>
'''
    return (
        template.replace("__SUMMARY_SHA__", summary_sha256)
        .replace("__OPTIONS__", options)
        .replace("__GRID_JSON__", payload)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
