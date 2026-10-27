"""Experimental correspondence and warp for a known three-section turned part.

The caller supplies the authored axis and ordered feature family. This is not
unknown-part recognition. Inputs and results are ordinary JSON-compatible dicts.
"""
import argparse
import hashlib
import json
import struct
import traceback
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit


def radius(profile, x):
    r = np.asarray(profile['radii'])
    knots = np.asarray(profile['knots'])
    widths = np.asarray(profile['widths'])
    u = expit((np.asarray(x)[..., None] - knots[1:3]) / widths)
    return r[0] + u @ np.diff(r)


def center(profile, x):
    return (np.asarray(profile['center']) +
            (np.asarray(x)[..., None] - profile.get('centerOrigin', 0.)) * profile['slope'])


def warp_with_jacobian(points, generated, authored):
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    gk, ak = np.asarray(generated['knots']), np.asarray(authored['knots'])
    if not (np.all(np.diff(gk) > 0) and np.all(np.diff(ak) > 0)):
        raise ValueError('Ordered, distinct axial stations are required')
    # Linear continuation outside the observed extent; never clamp end faces.
    segment = np.clip(np.searchsorted(gk, x, side='right') - 1, 0, 2)
    dx = np.diff(ak)[segment] / np.diff(gk)[segment]
    target_x = ak[segment] + (x - gk[segment]) * dx
    rg, ra = radius(generated, x), radius(authored, target_x)
    if not (np.all(rg > 0) and np.all(ra > 0)):
        raise ValueError('Positive recovered and authored radii are required')
    scale = ra / rg
    residual = points[:, 1:3] - center(generated, x)
    out = np.column_stack([target_x, center(authored, target_x) + residual * scale[:, None]])

    def derivative(profile, values):
        r, k, w = np.asarray(profile['radii']), np.asarray(profile['knots']), np.asarray(profile['widths'])
        u = expit((values[:, None] - k[1:3]) / w)
        return (u * (1 - u) / w) @ np.diff(r)

    ds = (derivative(authored, target_x) * dx * rg - ra * derivative(generated, x)) / rg ** 2
    jac = np.zeros((len(x), 3, 3))
    jac[:, 0, 0] = dx
    jac[:, 1, 1] = jac[:, 2, 2] = scale
    jac[:, 1:3, 0] = (np.asarray(authored['slope']) * dx[:, None] -
                         np.asarray(generated['slope']) * scale[:, None] + residual * ds[:, None])
    return out, jac


def warp_points(points, generated, authored):
    """Map the recovered profile to its authored counterpart."""
    return warp_with_jacobian(points, generated, authored)[0]


def fit_profile(mesh, authored):
    """Recover the ordered radii/transitions using every triangle's area.

    Axial cap normals have zero cylindrical-wall weight. Angular detail and
    bevels retain intermediate weight. Correspondence starts from the authored
    order and coarse registration, rather than recognizing an arbitrary part.
    """
    pts = np.asarray(mesh.triangles_center)
    area = np.asarray(mesh.area_faces)
    wall_weight = (1 - np.asarray(mesh.face_normals)[:, 0] ** 2) ** 2
    weights = area * wall_weight
    if not np.any(weights > 0):
        raise ValueError('No cylindrical-wall evidence')
    sw = np.sqrt(weights / weights.mean())
    lo, hi = mesh.bounds[:, 0]
    span, mid = hi - lo, (hi + lo) / 2
    if not span > 0:
        raise ValueError('No axial extent')
    ak = np.asarray(authored['knots'])
    fraction = (ak - ak[0]) / (ak[-1] - ak[0])
    f1, f2 = fraction[1], (fraction[2] - fraction[1]) / (1 - fraction[1])
    initial_center = center(authored, np.array([mid]))[0]
    initial = np.r_[initial_center, authored['slope'], np.log(authored['radii']),
                    logit([f1, f2]), np.log(np.asarray(authored['widths']) * span / (ak[-1] - ak[0]))]

    def decode(p):
        b1 = lo + span * expit(p[7])
        b2 = b1 + (hi - b1) * expit(p[8])
        return dict(knots=[float(lo), float(b1), float(b2), float(hi)],
                    radii=np.exp(p[4:7]).tolist(), widths=np.exp(p[9:11]).tolist(),
                    center=p[:2].tolist(), slope=p[2:4].tolist(), centerOrigin=float(mid))

    def residual(p):
        profile = decode(p)
        radial = np.linalg.norm(pts[:, 1:3] - center(profile, pts[:, 0]), axis=1)
        # Weak log-parameter prior stabilizes a feature with little wall area;
        # the returned surface error and per-feature evidence remain visible.
        return np.r_[(radial - radius(profile, pts[:, 0])) * sw,
                     .001 * (p - initial)]

    fitted = least_squares(residual, initial, loss='soft_l1', f_scale=.005, x_scale='jac')
    profile = decode(fitted.x)
    r = np.linalg.norm(pts[:, 1:3] - center(profile, pts[:, 0]), axis=1) - radius(profile, pts[:, 0])
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError('Profile solve failed: ' + fitted.message)
    return profile, dict(success=bool(fitted.success), message=fitted.message,
                         nfev=int(fitted.nfev), triangleCount=len(pts),
                         weightedWallRMS=float(np.sqrt(np.average(r*r, weights=weights))),
                         wallWeightSum=float(weights.sum()), allTriangleArea=float(area.sum()),
                         method='ordered-three-section-axisymmetric-wall-fit; source-axis/frame prior')


def export_warp(input_path, output_path, position, generated, authored):
    """Bake a nonlinear warp into GLB positions/normals; retain UV/images/indices."""
    raw = Path(input_path).read_bytes()
    jn, jt = struct.unpack_from('<II', raw, 12)
    doc = json.loads(raw[20:20+jn])
    bn, bt = struct.unpack_from('<II', raw, 20+jn)
    if jt != 0x4e4f534a or bt != 0x004e4942 or len(doc['nodes']) != 1 or len(doc['meshes']) != 1:
        raise ValueError('Expected the retained single-node binary GLB route')
    binary = bytearray(raw[28+jn:28+jn+bn])
    original = bytes(binary)
    node = doc['nodes'][0]
    if any(k in node for k in ['rotation', 'scale', 'matrix']):
        raise ValueError('Input must have its linear transform baked')
    primitive = doc['meshes'][0]['primitives'][0]

    def array(index):
        ac = doc['accessors'][index]; bv = doc['bufferViews'][ac['bufferView']]
        if ac['componentType'] != 5126 or ac['type'] != 'VEC3' or 'sparse' in ac:
            raise ValueError('Expected float32 VEC3 position/normal accessor')
        return np.ndarray((ac['count'], 3), dtype='<f4', buffer=binary,
                          offset=bv.get('byteOffset', 0)+ac.get('byteOffset', 0),
                          strides=(bv.get('byteStride', 12), 4))

    pos = array(primitive['attributes']['POSITION'])
    world = np.array(pos, dtype=float) + position + np.asarray(node.get('translation', [0, 0, 0]))
    corrected, jac = warp_with_jacobian(world, generated, authored)
    determinants = np.linalg.det(jac)
    if not np.isfinite(corrected).all() or not np.all(determinants > 0):
        raise ValueError('Nonfinite or orientation-reversing warp')
    placement = (corrected.min(0) + corrected.max(0)) / 2
    pos[:] = corrected - placement
    accessor = doc['accessors'][primitive['attributes']['POSITION']]
    accessor['min'], accessor['max'] = pos.min(0).tolist(), pos.max(0).tolist()
    normal = array(primitive['attributes']['NORMAL'])
    n = np.linalg.solve(np.swapaxes(jac, 1, 2), np.asarray(normal, dtype=float)[..., None])[..., 0]
    length = np.linalg.norm(n, axis=1)
    n[length > 0] /= length[length > 0, None]
    normal[:] = n
    preserved = []
    indices = [primitive['indices'], primitive['attributes']['TEXCOORD_0']]
    views = [doc['accessors'][i]['bufferView'] for i in indices] + [i['bufferView'] for i in doc.get('images', [])]
    for i in views:
        bv = doc['bufferViews'][i]; start = bv.get('byteOffset', 0); end = start+bv['byteLength']
        if bytes(binary[start:end]) != original[start:end]:
            raise ValueError('Warp modified an index/UV/image buffer')
        preserved.append(hashlib.sha256(original[start:end]).hexdigest())
    node.pop('translation', None)
    doc.setdefault('extras', {})['knownAxialRectification'] = dict(generated=generated, authored=authored,
        sourceSha256=hashlib.sha256(raw).hexdigest(), preservedBufferSha256=preserved)
    jb = json.dumps(doc, separators=(',', ':')).encode(); jb += b' '*((-len(jb)) % 4)
    binary.extend(b'\0'*((-len(binary)) % 4))
    output = (struct.pack('<III', 0x46546c67, 2, 28+len(jb)+len(binary)) +
              struct.pack('<II', len(jb), 0x4e4f534a)+jb+struct.pack('<II', len(binary), 0x004e4942)+binary)
    dest = Path(output_path); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.read_bytes() != output:
        raise FileExistsError('Choose a new output path for a changed result: ' + str(dest))
    dest.write_bytes(output)
    return dict(position=placement.tolist(), vertexCount=len(pos),
                bounds=[corrected.min(0).tolist(), corrected.max(0).tolist()],
                jacobianDeterminantMin=float(determinants.min()),
                jacobianDeterminantMax=float(determinants.max()),
                sha256=hashlib.sha256(output).hexdigest(), preservedBufferSha256=preserved)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import trimesh
    result = dict(ok=False, phase='plan', plan=str(args.plan), parts=[])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        plan = json.loads(args.plan.read_text())
        result['sourceSha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for part in plan['parts']:
            result['phase'] = 'fit:' + part['name']
            input_hash = hashlib.sha256(Path(part['input']).read_bytes()).hexdigest()
            if input_hash != part['inputSha256']:
                raise ValueError('Input identity changed: ' + part['input'])
            mesh = trimesh.load(part['input'], force='scene', process=False).to_geometry()
            mesh.apply_translation(part['position'])
            generated, fit = fit_profile(mesh, part['authored'])
            result['phase'] = 'export:' + part['name']
            output = export_warp(part['input'], part['output'], part['position'], generated, part['authored'])
            result['parts'].append(dict(name=part['name'], input=part['input'], output=part['output'],
                                        inputSha256=input_hash, generated=generated, authored=part['authored'],
                                        fit=fit, export=output))
        result.update(ok=True, phase='complete')
    except BaseException as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        args.report.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
