"""Versioned repair: undefined zero-residual correlations are invalid draws."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import rankdata

R=Path(__file__).resolve().parents[1]
C=R/'config/STUDY1_ACCURACY_V2.json'
O=R/'runs/study1_accuracy_v2'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def statistic(x,y,c):
    if np.unique(x).size<2 or np.unique(y).size<2:
        return np.nan,'constant_outcome'
    c=np.asarray(c); c=c[:,None] if c.ndim==1 else c
    d=np.column_stack([np.ones(len(x))]+[rankdata(col) for col in c.T])
    if len(x)<3 or np.linalg.matrix_rank(d)<d.shape[1]:
        return np.nan,'rank_deficient_design'
    residuals=[]
    for label,v in [('x',x),('y',y)]:
        ranked=rankdata(v)
        residual=ranked-d@np.linalg.lstsq(d,ranked,rcond=None)[0]
        threshold=10*np.finfo(float).eps*max(d.shape)*np.linalg.norm(ranked-ranked.mean())
        if np.linalg.norm(residual)<=threshold:
            return np.nan,'zero_residual_'+label
        residuals.append(residual-residual.mean())
    a,b=residuals
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),'valid'

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--accepted-contract-sha',required=True); args=parser.parse_args()
    assert sha(C)==args.accepted_contract_sha
    contract=json.loads(C.read_text())
    assert sha(Path(__file__))==contract['producer_code_sha256']
    for name,h in contract['inputs'].items(): assert sha(R/name)==h
    O.mkdir(exist_ok=False)
    old=R/'runs/study1_reuse_v1/accuracy_conditioned'
    z=np.load(old/'inputs.npz',allow_pickle=False);draws=np.load(old/'family_draws.npy',allow_pickle=False)
    previous=json.loads((old/'results.json').read_text())
    blocks=[np.flatnonzero(z['families']==f) for f in sorted(set(z['families']))]
    report={'producer':'/root','status':'PENDING_INDEPENDENT_REVIEW','contract_sha256':sha(C),
            'execution_class':'CORRECTED_ESTIMAND_NOT_HISTORICAL_REPLAY','domains':{},
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__}}
    arrays={}
    for dom in ['code','math']:
        x,y,a,c=[z[dom+'_'+k] for k in ['composite','theta','accuracy','controls']]
        report['domains'][dom]={}
        for key,yy,cc in [('q4_rho_g_t__a',y,a),('q5_rho_g_t__a_C',y,np.column_stack([a,c])),('q6_rho_g_a__C',a,c)]:
            point,reason=statistic(x,yy,cc)
            assert reason=='valid'
            vals=np.full(len(draws),np.nan);reasons=[]
            for i,dd in enumerate(draws):
                ix=np.concatenate([blocks[k] for k in dd])
                vals[i],why=statistic(x[ix],yy[ix],cc[ix]);reasons.append(why)
            valid=np.isfinite(vals); ci=np.percentile(vals[valid],[2.5,97.5]).tolist()
            from collections import Counter
            record={'rho':point,'ci':ci,'n_valid':int(valid.sum()),'n_invalid':int((~valid).sum()),
                    'invalid_draw_indices':np.flatnonzero(~valid).tolist(),'reason_counts':dict(Counter(reasons)),
                    'previous_ci':previous['domains'][dom][key]['ci'],
                    'ci_excludes_zero':bool(ci[0]>0 or ci[1]<0)}
            report['domains'][dom][key]=record
            arrays[dom+'_'+key]=vals
            print(dom,key,record['n_invalid'],ci,flush=True)
    np.savez_compressed(O/'bootstrap_values.npz',**arrays)
    (O/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,h in contract['inputs'].items(): assert sha(R/name)==h
    receipt={'producer':'/root','contract_sha256':sha(C),'code_sha256':sha(Path(__file__)),
             'inputs':contract['inputs'],'outputs':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
    (O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
