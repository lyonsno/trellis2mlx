"""Local warp from measured, known coaxial bore sections to authored sections."""
import numpy as np


def warp_with_jacobian(points, anchors, fade_radius):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not anchors:
        raise ValueError('Finite XYZ points and measured bore anchors are required')
    data = np.array([[a['x'], *a['center'], a['radius'], *a['targetCenter'], a['targetRadius']]
                     for a in anchors], dtype=float)
    if not np.isfinite(data).all() or np.any(np.diff(data[:,0]) <= 0) or np.any(data[:,[3,6]] <= 0):
        raise ValueError('Distinct ordered stations and positive bore radii are required')
    if not np.isfinite(fade_radius) or fade_radius <= 1.25*data[:,3].max():
        raise ValueError('Fade radius must enclose the measured bore band')
    x = points[:,0]
    values = np.column_stack([np.interp(x,data[:,0],data[:,i]) for i in range(1,7)])
    slopes = np.zeros_like(values)
    if len(data) > 1:
        segment = np.clip(np.searchsorted(data[:,0],x,side='right')-1,0,len(data)-2)
        slopes = np.diff(data[:,1:],axis=0)[segment]/np.diff(data[:,0])[segment,None]
        slopes[(x < data[0,0]) | (x >= data[-1,0])] = 0
    c, rg, target, ra = values[:,:2], values[:,2], values[:,3:5], values[:,5]
    dc, drg, dt, dra = slopes[:,:2], slopes[:,2], slopes[:,3:5], slopes[:,5]
    d = points[:,1:]-c
    rho = np.linalg.norm(d,axis=1)
    unit = np.divide(d,rho[:,None],out=np.zeros_like(d),where=rho[:,None]>0)
    band = 1.25*rg
    u = np.clip((rho-band)/(fade_radius-band),0,1)
    w = 1-3*u*u+2*u*u*u
    dwdu = -6*u+6*u*u
    dwdr = dwdu/(fade_radius-band)
    drdx = -np.einsum('ni,ni->n',unit,dc)
    dudx = (drdx-1.25*drg*(1-u))/(fade_radius-band)
    dwdx = np.where((u>0)&(u<1),dwdu*dudx,0)
    scale = ra/rg
    ds = dra/rg-ra*drg/(rg*rg)
    delta = target-c+(scale-1)[:,None]*d
    out = points.copy(); out[:,1:] += w[:,None]*delta
    jac = np.broadcast_to(np.eye(3),(len(points),3,3)).copy()
    jac[:,1:,1:] += (w*(scale-1))[:,None,None]*np.eye(2)
    jac[:,1:,1:] += delta[:,:,None]*(dwdr[:,None]*unit)[:,None,:]
    jac[:,1:,0] = (dwdx[:,None]*delta + w[:,None]*(dt-scale[:,None]*dc+ds[:,None]*d))
    if not np.all(np.linalg.det(jac)>0):
        raise ValueError('Requested bore correction reverses or collapses the surface neighborhood')
    return out,jac


def main():
    import argparse, hashlib, importlib.util, json, traceback
    from pathlib import Path
    import trimesh
    from scripts.known_axial_profile_rectify import export_deformation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=dict(ok=False,phase='plan',parts=[])
    args.report.parent.mkdir(parents=True,exist_ok=True)
    sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    try:
        plan=json.loads(args.plan.read_text())
        result['sourceSha256']=sha(__file__)
        result['planSha256']=sha(args.plan)
        diagnostic=plan['sectionDiagnostic']
        if sha(diagnostic['path'])!=diagnostic['sha256']:
            raise ValueError('Section diagnostic identity changed')
        spec=importlib.util.spec_from_file_location('observed_bore_section',diagnostic['path'])
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        section=module.section
        for part in plan['parts']:
            result['phase']='observe:'+part['name']
            for key in ['input','source']:
                if sha(part[key])!=part[key+'Sha256']:
                    raise ValueError('Input identity changed: '+part[key])
            mesh=trimesh.load(part['input'],force='scene',process=False).to_geometry()
            mesh.apply_translation(part['position'])
            source=trimesh.load(part['source'],force='scene',process=False).to_geometry()
            source.apply_translation(part['sourcePosition'])
            anchors=[]; evidence=[]
            for feature in part['features']:
                before=section(mesh,feature); authored=section(source,feature)
                evidence.append(dict(before=before,authored=authored))
                if not before['observed'] or not authored['observed']:
                    raise ValueError('Known bore correspondence unresolved: '+feature['feature'])
                anchors.append(dict(x=feature['sectionX'],center=before['selected']['centerYZ'],
                    radius=before['selected']['radius'],targetCenter=authored['selected']['centerYZ'],
                    targetRadius=authored['selected']['radius']))
            result['phase']='export:'+part['name']
            exported=export_deformation(part['input'],part['output'],part['position'],
                lambda p:warp_with_jacobian(p,anchors,part['fadeRadius']),
                'knownBoreRectification',dict(anchors=anchors,fadeRadius=part['fadeRadius'],
                    sectionDiagnostic=diagnostic,authoredSha256=part['sourceSha256']))
            output=trimesh.load(part['output'],force='scene',process=False).to_geometry()
            output.apply_translation(exported['position'])
            for feature,row in zip(part['features'],evidence):
                row['after']=section(output,feature)
                if not row['after']['observed']:
                    raise ValueError('Corrected bore became unresolved: '+feature['feature'])
            result['parts'].append(dict(name=part['name'],input=part['input'],output=part['output'],
                inputSha256=part['inputSha256'],sourceSha256=part['sourceSha256'],
                sourcePosition=part['sourcePosition'],position=part['position'],
                anchors=anchors,fadeRadius=part['fadeRadius'],sections=evidence,export=exported))
        result.update(ok=True,phase='complete')
    except BaseException as exc:
        result.update(error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        args.report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(ok=result['ok'],phase=result['phase'],error=result.get('error'),
            parts=[dict(name=p['name'],sections=[dict(feature=e['after']['feature'],
                before=e['before']['selected']['radius'],after=e['after']['selected']['radius'],
                target=e['authored']['selected']['radius'],centerError=e['after']['selected']['centerError'])
                for e in p['sections']]) for p in result['parts']]),indent=2),flush=True)


if __name__=='__main__':
    main()
