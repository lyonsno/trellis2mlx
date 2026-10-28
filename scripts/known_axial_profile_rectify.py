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


def validate_profile(profile):
    """Each named section must dominate its own interval's midpoint."""
    k, w = np.asarray(profile['knots']), np.asarray(profile['widths'])
    if not (np.all(np.diff(k) > 0) and np.all(w > 0)):
        raise ValueError('Ordered, distinct axial stations are required')
    mid = (k[:-1] + k[1:]) / 2
    u = expit((mid[:, None]-k[1:3])/w)
    basis = np.column_stack([1-u[:, 0], u[:, 0]-u[:, 1], u[:, 1]])
    if not np.all(np.diag(basis) > .5):
        raise ValueError('Three sections are not separately observed: transition dominance failed')
    return basis.tolist()


def warp_with_jacobian(points, generated, authored):
    validate_profile(generated)
    validate_profile(authored)
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


def outer_sections(mesh, spacing):
    """Intersect the complete mesh, retaining connected contours and rejections.

    A circular contour must surround its own center. The largest such contour
    supplies the exterior radius; nested bore contours remain in the evidence.
    The axial quadrature spacing is caller-visible, not a triangle/sample cap.
    """
    import trimesh
    lo, hi = mesh.bounds[:, 0]
    xs = np.linspace(lo, hi, int(np.ceil((hi-lo)/spacing))+2)[1:-1]
    rows = []
    for x in xs:
        lines = trimesh.intersections.mesh_plane(mesh, [1, 0, 0], [x, 0, 0])
        points = lines[:, :, 1:3].reshape(-1, 2)
        row = dict(x=float(x), segmentCount=len(lines), candidates=[], selected=None)
        rows.append(row)
        if not len(lines):
            continue
        _, ids = np.unique(np.rint(points/5e-6).astype(np.int64), axis=0, return_inverse=True)
        parent = list(range(int(ids.max())+1))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i

        for i, j in ids.reshape(-1, 2):
            parent[find(int(i))] = find(int(j))
        labels = np.array([find(int(i)) for i in ids[::2]])
        for label in np.unique(labels):
            segments = lines[labels == label, :, 1:3]
            p = segments.reshape(-1, 2)
            w = np.repeat(np.linalg.norm(segments[:, 1]-segments[:, 0], axis=1), 2)
            candidate = dict(segmentCount=len(segments), valid=False)
            row['candidates'].append(candidate)
            if len(p) < 3 or np.linalg.matrix_rank(p-p.mean(0)) < 2 or not w.sum() > 0:
                candidate['reason'] = 'Underconstrained profile'; continue
            mean = np.average(p, axis=0, weights=w)
            q = p-mean
            design = np.column_stack([2*q, np.ones(len(q))])
            solution = np.linalg.lstsq(design*np.sqrt(w[:, None]),
                                       np.sum(q*q, axis=1)*np.sqrt(w), rcond=None)[0]
            c = mean+solution[:2]
            rsq = solution[2]+np.sum(solution[:2]**2)
            if not rsq > 0:
                candidate['reason'] = 'Nonpositive radius'; continue
            r = np.sqrt(rsq)
            angles = np.sort(np.mod(np.arctan2(p[:, 1]-c[1], p[:, 0]-c[0]), 2*np.pi))
            gap = np.diff(np.r_[angles, angles[0]+2*np.pi]).max()
            candidate.update(center=c.tolist(), radius=float(r),
                             radialRMS=float(np.sqrt(np.average((np.linalg.norm(p-c, axis=1)-r)**2, weights=w))),
                             angularCoverage=float(2*np.pi-gap), valid=bool(gap < np.pi))
            if not candidate['valid']:
                candidate['reason'] = 'Contour does not surround fitted center'
        valid = [i for i, c in enumerate(row['candidates']) if c['valid']]
        if valid:
            row['selected'] = max(valid, key=lambda i: row['candidates'][i]['radius'])
    return rows


def fit_profile(mesh, authored):
    """Fit the known ordered exterior features, distinct from nested bores."""
    lo, hi = mesh.bounds[:, 0]
    span, mid = hi-lo, (hi+lo)/2
    ak = np.asarray(authored['knots'])
    spacing = float(np.min(np.diff(ak))/8)
    sections = outer_sections(mesh, spacing)
    observed = [s for s in sections if s['selected'] is not None]
    if len(observed) < 3:
        raise ValueError('Insufficient separately observed exterior sections')
    x = np.array([s['x'] for s in observed])
    candidates = [s['candidates'][s['selected']] for s in observed]
    radii = np.array([c['radius'] for c in candidates])
    centers = np.array([c['center'] for c in candidates])
    line = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x-mid]), centers, rcond=None)[0]
    # Globally initialize the two change points on the complete sampled
    # exterior profile, rather than depending on tiny transition gradients.
    prefix = np.r_[0., np.cumsum(radii)]
    square = np.r_[0., np.cumsum(radii*radii)]

    def cost(i, j):
        return square[j]-square[i]-(prefix[j]-prefix[i])**2/(j-i)

    best = min((cost(0, i)+cost(i, j)+cost(j, len(x)), i, j)
               for i in range(1, len(x)-1) for j in range(i+1, len(x)))
    _, i, j = best
    b1, b2 = (x[i-1]+x[i])/2, (x[j-1]+x[j])/2
    levels = [radii[:i].mean(), radii[i:j].mean(), radii[j:].mean()]
    f1, f2 = (b1-lo)/span, (b2-b1)/(hi-b1)
    initial = np.r_[np.log(levels), logit([f1, f2]), np.log([spacing, spacing])]

    def decode(p):
        b1 = lo+span*expit(p[3]); b2 = b1+(hi-b1)*expit(p[4])
        return dict(knots=[float(lo), float(b1), float(b2), float(hi)],
                    radii=np.exp(p[:3]).tolist(), widths=np.exp(p[5:7]).tolist(),
                    center=line[0].tolist(), slope=line[1].tolist(), centerOrigin=float(mid))

    fitted = least_squares(lambda p: np.r_[radii-radius(decode(p), x), .0001*(p-initial)],
                           initial, loss='soft_l1', f_scale=.005, x_scale='jac')
    profile = decode(fitted.x)
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError('Profile solve failed: '+fitted.message)
    dominance = validate_profile(profile)
    return profile, dict(success=True, message=fitted.message, nfev=int(fitted.nfev),
                         triangleCount=len(mesh.faces), sectionSpacing=spacing,
                         profileRMS=float(np.sqrt(np.mean((radii-radius(profile, x))**2))),
                         midpointFeatureBasis=dominance, sections=sections,
                         method='ordered-three-section-exterior-contour-fit; source-axis/frame prior')


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
