"""Características geométricas: metros, longitudes y comparación en ambos sentidos."""
import json
import time
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from pyproj import CRS, Transformer
from scipy.spatial.distance import cdist
from numba import njit
from .data import ROOT, load_data

BASE_FEATURES = ['length_est','length_real','relative_length_delta','start_gap_m','end_gap_m',
                 'mean_est_to_real_m','mean_real_to_est_m','hausdorff_discrete_m']

def project_pair(a,b):
    center=np.concatenate([a,b]).mean(axis=0)
    local=CRS.from_proj4(f'+proj=aeqd +lat_0={center[0]} +lon_0={center[1]} +datum=WGS84 +units=m')
    t=Transformer.from_crs('EPSG:4326',local,always_xy=True)
    return [np.column_stack(t.transform(p[:,1],p[:,0])) for p in [a,b]]

def clean_line(points):
    keep=np.r_[True,np.linalg.norm(np.diff(points,axis=0),axis=1)>1e-6]
    p=points[keep]
    if len(p)<2: raise ValueError('Ruta degenerada sin longitud espacial')
    return LineString(p)

def sample_line(line, n=64, midpoints=False):
    fractions=(np.arange(n)+.5)/n if midpoints else np.linspace(0,1,n)
    return shapely.get_coordinates(shapely.line_interpolate_point(line,fractions,normalized=True))

@njit(cache=True)
def discrete_frechet_from_distances(dist):
    n,m=dist.shape
    out=np.empty((n,m))
    for i in range(n):
        for j in range(m):
            if i==0 and j==0: out[i,j]=dist[i,j]
            elif i==0: out[i,j]=max(out[i,j-1],dist[i,j])
            elif j==0: out[i,j]=max(out[i-1,j],dist[i,j])
            else: out[i,j]=max(min(out[i-1,j],out[i-1,j-1],out[i,j-1]),dist[i,j])
    return out[-1,-1]

def longest_run(mask):
    best=current=0
    for flag in mask:
        current=current+1 if flag else 0
        best=max(best,current)
    return best/len(mask)

def features_xy(a,b):
    est,real=clean_line(a),clean_line(b)
    le,lr=est.length,real.length
    scale=max((le+lr)/2,1.)
    f=dict(length_est=le,length_real=lr,relative_length_delta=(lr-le)/scale,
           absolute_length_delta_m=abs(lr-le),length_ratio_real_est=lr/max(le,1),
           start_gap_m=float(np.linalg.norm(a[0]-b[0])),end_gap_m=float(np.linalg.norm(a[-1]-b[-1])))
    for name,line,other in [('est_to_real',est,real),('real_to_est',real,est)]:
        n=int(np.clip(np.ceil(line.length/20),16,1600))
        pts=sample_line(line,n,midpoints=True)
        ds=shapely.distance(shapely.points(pts),other)
        f[f'mean_{name}_m']=float(ds.mean())
        f[f'p95_{name}_m']=float(np.quantile(ds,.95))
        f[f'mean_{name}_relative']=float(ds.mean()/scale)
        f[f'p95_{name}_relative']=float(np.quantile(ds,.95)/scale)
        for threshold in [20,50,100]:
            mask=ds>threshold
            f[f'fraction_{name}_over_{threshold}m']=float(mask.mean())
            f[f'longest_{name}_over_{threshold}m']=longest_run(mask)
    f['hausdorff_discrete_m']=float(shapely.hausdorff_distance(est,real,densify=.25))
    f['hausdorff_relative']=f['hausdorff_discrete_m']/scale
    f['frechet_64_m']=float(discrete_frechet_from_distances(cdist(sample_line(est),sample_line(real))))
    f['frechet_relative']=f['frechet_64_m']/scale
    for radius in [20,50,100]:
        ba,bb=est.buffer(radius,quad_segs=4),real.buffer(radius,quad_segs=4)
        inter=ba.intersection(bb).area
        f[f'buffer_iou_{radius}m']=inter/(ba.area+bb.area-inter)
    for name,line,points in [('est',est,a),('real',real,b)]:
        f[f'straightness_{name}']=float(np.linalg.norm(points[-1]-points[0])/line.length)
        d=np.diff(sample_line(line),axis=0)
        headings=np.arctan2(d[:,1],d[:,0])
        turns=np.abs((np.diff(headings)+np.pi)%(2*np.pi)-np.pi)
        f[f'total_turn_{name}_rad']=float(turns.sum())
        f[f'mean_turn_{name}_rad']=float(turns.mean())
    f['start_gap_relative']=f['start_gap_m']/scale
    f['end_gap_relative']=f['end_gap_m']/scale
    return f

def build_features():
    df,routes=load_data()
    path=ROOT/'data/cache/geometry_features.json'
    if path.exists():
        out=pd.DataFrame.from_dict(json.loads(path.read_text()),orient='index')
        assert set(out.index)==set(df.index)
        return out.loc[df.index]
    rows={}; start=time.perf_counter()
    for k,(jid,(a,b)) in enumerate(routes.items()):
        rows[jid]=features_xy(*project_pair(a,b))
        if (k+1)%300==0:print(f'Geometría: {k+1}/{len(routes)} viajes',flush=True)
    out=pd.DataFrame.from_dict(rows,orient='index')
    assert np.isfinite(out.to_numpy()).all()
    path.write_text(json.dumps(rows))
    (ROOT/'results/routes/geometry_manifest.json').write_text(json.dumps(dict(
        n_features=len(out.columns),features=list(out.columns),baseline=BASE_FEATURES,
        seconds=time.perf_counter()-start,units='metros en proyección azimutal local por pareja',
        mean_sampling='puntos medios equiespaciados en longitud: 20 m, mínimo 16 y máximo 1600',
        frechet='Fréchet discreto sobre 64 puntos por ruta; aproximación a la curva continua',
        hausdorff='Hausdorff discreto con densificación 0.25',
        iou='corredores geométricos; no red viaria ni orden'),indent=2))
    return out

if __name__=='__main__': build_features()
