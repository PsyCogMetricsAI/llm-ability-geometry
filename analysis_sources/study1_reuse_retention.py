#!/usr/bin/env python3
"""Retention reconstruction: reuse matching cached halves, calculate only gaps.

Stages pilot -> fill -> statistics are resumable. No LLM inference or extraction.
All NPZ arrays are read-only mmap views of existing ZIP_STORED NPY members.
"""
import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import struct
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from study1_reuse_bind import R,G,PrimitiveUnpickler,digest,pc1
from study1_reuse_analyze import rho,bootstrap,permute,historical_verdict
from study1_reuse_bos import spectral

CP=R/'reports/study1_reuse_retention_contract.json'
OUT=R/'runs/study1_reuse_v1/retention'


def read_contract():
    return json.loads(CP.read_text())


def save_json(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def split_ids(bank,seed):
    order=np.random.default_rng(seed).permutation(len(bank));h=len(bank)//2
    return ({bank[i] for i in order[:h]},{bank[i] for i in order[h:]})


def mapped_hidden(asset):
    path=Path(asset['path']);st=path.stat()
    assert st.st_size==asset['bytes'] and st.st_mtime_ns==asset['mtime_ns'],str(path)
    with zipfile.ZipFile(path) as z:
        member=z.getinfo('hidden_by_layer.npy')
        assert member.compress_type==zipfile.ZIP_STORED
    with path.open('rb') as f:
        f.seek(member.header_offset);header=f.read(30)
        assert header[:4]==b'PK\x03\x04'
        namelen,extralen=struct.unpack('<HH',header[26:30])
        f.seek(member.header_offset+30+namelen+extralen)
        version=np.lib.format.read_magic(f)
        if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
        elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
        else:raise ValueError(f'Unsupported NPY version {version}')
        offset=f.tell()
    assert not fortran and list(shape)==asset['shape'] and dtype.str==asset['dtype']
    mapped=np.memmap(path,dtype=dtype,mode='r',offset=offset,shape=shape,order='C')
    with np.load(path,allow_pickle=False) as z:ids=[str(v) for v in z['item_ids']]
    assert ids==[str(v) for v in asset['item_ids']]
    return mapped,ids


def chi_from_layers(ds,idx):
    neighbors=[]
    for d in ds:
        sub=d[np.ix_(idx,idx)].copy();np.fill_diagonal(sub,np.inf)
        neighbors.append(np.argsort(sub,axis=1)[:,:10])
    vals=[]
    for a,b in zip(neighbors[:-1],neighbors[1:]):
        overlap=np.sum(a[:,:,None]==b[:,None,:],axis=(1,2))
        vals.append(float(np.mean(overlap/10)))
    return float(np.mean(vals))


def geometric_features(hbl,ds,idx):
    if len(idx)<12:return {'n':len(idx),'eff_rank':None,'participation_ratio':None,'chi':None}
    er,pr=spectral(hbl[idx,0,:])
    return {'n':len(idx),'eff_rank':er,'participation_ratio':pr,'chi':chi_from_layers(ds,idx)}


def cache_features(cache,seed,half):
    h=0 if half=='A' else 1
    return {'n':cache['n_items']//2,'eff_rank':cache['halves'][seed]['eff_rank'][h],
            'participation_ratio':cache['halves'][seed]['participation_ratio'][h],
            'chi':cache['halves'][seed]['chi'][h]}


def geometry_job(job):
    dom,tag=job;contract=read_contract();contract_sha=digest(CP);code_sha=digest(__file__)
    folder=OUT/'geometry';folder.mkdir(parents=True,exist_ok=True);target=folder/f'{dom}__{tag}.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old.get('contract_sha256')==contract_sha and old.get('code_sha256')==code_sha and old.get('status')=='COMPLETE':
            return {'domain':dom,'model':tag,'resumed':True,'elapsed_seconds':old['elapsed_seconds']}
    start=time.monotonic();dc=contract['domains'][dom];asset=contract['observed_NPZ_bindings'][f'{tag}|{dom}']
    hbl,ids=mapped_hidden(asset);positions={v:i for i,v in enumerate(ids)};n,L,_=hbl.shape
    n_late=max(1,int(np.ceil((L-1)*.25)))
    with Path(dc['cache_path']).open('rb') as f:met=PrimitiveUnpickler(f).load()['metrics'][tag]
    assert met['n_items']==n and met['n_layers']==L
    selected={};matching=[]
    for seed in contract['seeds']:
        halves=split_ids(dc['bank_item_order'],seed)
        perm=np.random.default_rng(seed).permutation(n);half=n//2
        cached=[perm[:half],perm[half:2*half]]
        for j,label in enumerate(['A','B']):
            idx=np.array(sorted(positions[i] for i in halves[j] if i in positions),dtype=int)
            key=f'{seed}|{label}';selected[key]=idx
            if set(idx.tolist())==set(cached[j].tolist()):matching.append(key)
    # Original gpu_backend explicitly supports this float64 scipy CPU fallback.
    ds=[]
    for layer in range(n_late+1):
        x=np.asarray(hbl[:,layer,:],dtype=np.float64)
        ds.append(cdist(x,x))
    validation=None;cache_allowed=bool(matching);validation_value=None
    if matching:
        key=matching[0];seed,label=key.split('|')
        validation_value=geometric_features(hbl,ds,selected[key]);saved=cache_features(met,int(seed),label)
        diffs={k:abs(validation_value[k]-saved[k]) for k in ['eff_rank','participation_ratio','chi']}
        rels={k:diffs[k]/max(abs(saved[k]),1e-12) for k in ['eff_rank','participation_ratio']}
        cache_allowed=bool(diffs['chi']<=1e-12 and all(v<=1e-5 for v in rels.values()))
        validation={'half':key,'recomputed':validation_value,'saved_cache':saved,'abs_differences':diffs,
                    'spectral_relative_differences':rels,'pass':cache_allowed,
                    'reason':'same input set; ordering invariance diagnosed on first matching half, remaining same-set halves retain saved readouts'}
    records={};computed=0;reused=0
    for key,idx in selected.items():
        seed,label=key.split('|')
        if cache_allowed and key in matching:
            values=cache_features(met,int(seed),label);source='REUSED_SAME_SET_CACHE_AFTER_MODEL_DIAGNOSTIC';reused+=1
        else:
            values=validation_value if validation and validation['half']==key else geometric_features(hbl,ds,idx)
            source='RECOMPUTED_FROM_SAVED_ACTIVATIONS_CPU';computed+=1
        records[key]=dict(values,source=source,row_indices=idx.tolist(),item_ids=[ids[i] for i in idx])
    result={'status':'COMPLETE','domain':dom,'model':tag,'contract_sha256':contract_sha,'code_sha256':code_sha,
            'source_npz':asset,'cache_sha256':digest(dc['cache_path']),
            'shape':list(hbl.shape),'late_layer_count':n_late+1,'distance_backend':'scipy.cdist float64 Euclidean',
            'cached_candidate_count':len(matching),'reused_half_count':reused,'new_half_count':computed,
            'cache_order_validation':validation,'halves':records,'elapsed_seconds':time.monotonic()-start}
    save_json(target,result)
    print(f'[retention geometry] {dom}/{tag}: reused {reused}, new {computed}, {result["elapsed_seconds"]:.1f}s',flush=True)
    return {'domain':dom,'model':tag,'resumed':False,'elapsed_seconds':result['elapsed_seconds'],'reused':reused,'new':computed}


def theta_job(job):
    dom,seed,label=job;contract=read_contract();contract_sha=digest(CP);code_sha=digest(__file__)
    folder=OUT/'theta';folder.mkdir(parents=True,exist_ok=True);target=folder/f'{dom}__{seed}__{label}.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old.get('contract_sha256')==contract_sha and old.get('code_sha256')==code_sha and old.get('status')=='COMPLETE':
            return {'domain':dom,'seed':seed,'half':label,'resumed':True,'elapsed_seconds':old['elapsed_seconds']}
    import girth
    from girth.utilities import default_options
    start=time.monotonic();dc=contract['domains'][dom];persons=dc['theta_fit_model_order']
    matrix=json.loads((G/'response_matrix_geom.json').read_text())
    chosen=split_ids(dc['bank_item_order'],seed)[0 if label=='A' else 1]
    items=sorted(chosen,key=int);x=np.array([[matrix[t][dom][i] for i in items] for t in persons],dtype=int)
    keep=x.var(0)>0;ip=x[:,keep].T
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');est=girth.twopl_mml(ip)
        theta=np.asarray(girth.ability_eap(ip,est['Difficulty'],est['Discrimination']),dtype=float)
    options=default_options();options['distribution']='scipy.stats.norm(0,1).pdf (original default)'
    finite=bool(np.isfinite(theta).all());variance=float(np.var(theta))
    proxy=float(min(max(variance/(variance+max(1e-9,1-variance)),0),1)) if finite else None
    # Preserve non-finite fit results explicitly, preventing downstream use.
    clean=lambda arr:[float(v) if np.isfinite(v) else None for v in np.asarray(arr).ravel()]
    result={'status':'COMPLETE' if finite else 'FAILED_NONFINITE_THETA','domain':dom,'seed':seed,'half':label,
            'contract_sha256':contract_sha,'code_sha256':code_sha,'model_order':persons,
            'all_item_order':items,'kept_item_order':[i for i,k in zip(items,keep) if k],
            'n_items_kept':int(keep.sum()),'theta':clean(theta),
            'Difficulty':clean(est['Difficulty']),'Discrimination':clean(est['Discrimination']),
            'all_theta_finite':finite,'original_s1_clamped_reliability_proxy':proxy,
            'convergence_status':'not supplied by girth API; no convergence assertion',
            'warning_messages':sorted(set(str(w.message) for w in ws)),'effective_defaults':options,
            'library_sources':{str(Path(inspect.getsourcefile(f))):digest(inspect.getsourcefile(f)) for f in [girth.twopl_mml,girth.grm_mml,girth.ability_eap,default_options]},
            'elapsed_seconds':time.monotonic()-start}
    save_json(target,result)
    print(f'[retention theta] {dom}/{seed}/{label}: k={int(keep.sum())}, finite={finite}, {result["elapsed_seconds"]:.1f}s',flush=True)
    if not finite:raise ValueError(f'Nonfinite theta {dom}/{seed}/{label}; persisted failure record')
    return {'domain':dom,'seed':seed,'half':label,'resumed':False,'elapsed_seconds':result['elapsed_seconds']}


def preflight():
    contract=read_contract()
    for a in contract['assets']:assert digest(a['path'])==a['sha256'],a['path']
    for a in contract['observed_NPZ_bindings'].values():
        st=Path(a['path']).stat();assert st.st_size==a['bytes'] and st.st_mtime_ns==a['mtime_ns'],a['path']
    return contract


def run_pilot():
    # One full-bank model in both domains exercises cached and newly computed halves.
    geometry=[geometry_job((d,'afm_4_5b')) for d in ['code','math']]
    with ProcessPoolExecutor(max_workers=2) as ex:
        theta=list(ex.map(theta_job,[(d,20260608,'A') for d in ['code','math']]))
    report={'status':'PILOT_PRODUCED_PENDING_INDEPENDENT_DIAGNOSTIC_REVIEW','geometry':geometry,'theta':theta,
            'projection_note':'These are observed single model/fit elapsed times, not a guaranteed full runtime; compute sizes vary. Completed pilot assets count toward final62 model files and80 fits.'}
    save_json(OUT/'pilot_receipt.json',report);print(json.dumps(report),flush=True)


def run_fill(workers):
    contract=read_contract()
    jobs=[(d,t) for d in ['code','math'] for t in contract['domains'][d]['geometry_model_order']]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        geometry=[future.result() for future in as_completed([ex.submit(geometry_job,j) for j in jobs])]
    jobs=[(d,s,h) for d in ['code','math'] for s in contract['seeds'] for h in ['A','B']]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        theta=[future.result() for future in as_completed([ex.submit(theta_job,j) for j in jobs])]
    save_json(OUT/'fill_receipt.json',{'status':'INPUTS_COMPLETE_PENDING_STATISTICS_AND_INDEPENDENT_REVIEW','geometry':geometry,'theta':theta})


def run_statistics():
    contract=read_contract();contract_sha=digest(CP);records={};arrays={};boot_values={};old=json.loads((G/'ext_retention_rerun_20260913/decouple_results_rerun_20260913.json').read_text())
    rng=np.random.default_rng(20260605);family_draws=np.array([rng.choice(12,size=12,replace=True) for _ in range(10000)])
    result={'producer':'scientific-analysis','status':'PRODUCED_PENDING_INDEPENDENT_NUMERIC_AND_SCIENTIFIC_REVIEW',
            'contract_sha256':contract_sha,'code_sha256':digest(__file__),'seeds':contract['seeds'],'domains':{},
            'interpretation':'Descriptive signal retention under original31model protocol; no construct realism or causal independence claim.'}
    for dom in ['code','math']:
        dc=contract['domains'][dom];tags=dc['geometry_model_order'];co=json.loads(Path(dc['covariates_path']).read_text())
        families=np.array([co[t]['family'] for t in tags]);fams=sorted(set(families));codes={f:i for i,f in enumerate(fams)}
        c=np.array([[co[t]['scale'],co[t]['d_model'],codes[co[t]['family']],co[t]['fluency']] for t in tags])
        geo={t:json.loads((OUT/'geometry'/f'{dom}__{t}.json').read_text()) for t in tags}
        assert all(x['status']=='COMPLETE' and x['contract_sha256']==contract_sha for x in geo.values())
        arrays[f'{dom}_model_order']=np.array(tags);arrays[f'{dom}_families']=families;arrays[f'{dom}_C']=c
        cells={};full={};nails={};domain_checks=[]
        for seed in contract['seeds']:
            for direction,gh,thh in [('geoA_thB','A','B'),('geoB_thA','B','A')]:
                key=f'{seed}|{direction}';tf=json.loads((OUT/'theta'/f'{dom}__{seed}__{thh}.json').read_text())
                assert tf['status']=='COMPLETE' and tf['contract_sha256']==contract_sha
                lookup=dict(zip(tf['model_order'],tf['theta']));y=np.array([lookup[t] for t in tags])
                members={k:np.array([geo[t]['halves'][f'{seed}|{gh}'][source] for t in tags],dtype=float) for k,source in [('eff_rank','eff_rank'),('participation_ratio','participation_ratio'),('g5no_chi','chi')]}
                mask=np.all(np.column_stack([np.isfinite(v) for v in members.values()]),axis=1)
                assert np.isfinite(y).all()
                x=pc1({k:v[mask] for k,v in members.items()});yy=y[mask];cc=c[mask]
                raw=rho(x,yy);partial=rho(x,yy,cc)
                cells[key]={'raw':raw,'partial':partial,'n_models':int(mask.sum()),'dropped':[t for t,ok in zip(tags,mask) if not ok]}
                prefix=f'{dom}_{seed}_{direction}';arrays[prefix+'_x']=x;arrays[prefix+'_theta']=yy;arrays[prefix+'_mask']=mask
                for k,v in members.items():arrays[prefix+'_'+k]=v
                historical=old['domains'][dom]['cells'][key]
                domain_checks.append({'cell':key,'raw_difference':raw-historical['raw'],'partial_difference':partial-historical['partial'],
                                      'n_models_equal':int(mask.sum())==historical['n_models'],
                                      'matches_1e10':abs(raw-historical['raw'])<=1e-10 and abs(partial-historical['partial'])<=1e-10 and int(mask.sum())==historical['n_models']})
                if seed==contract['seeds'][0]:
                    bare,bs=bootstrap(x,yy,cc[:,2],families[mask],family_draws);boot_values[prefix+'_bare']=bs
                    ctrl,bs=bootstrap(x,yy,cc,families[mask],family_draws);boot_values[prefix+'_controlled']=bs
                    prng=np.random.default_rng(20260605);perms=np.array([prng.permutation(len(yy)) for _ in range(10000)])
                    p,pv=permute(x,yy,cc,perms);boot_values[prefix+'_permutation']=pv;arrays[prefix+'_perm_indices']=perms
                    full[direction]={'raw_rho':raw,'partial_rho_controlling_C':partial,'bare_cluster_ci':bare['ci'],
                        'controlled_cluster_ci':ctrl['ci'],'bare_ci_excludes_zero':bare['ci_excludes_zero'],
                        'controlled_ci_excludes_zero':ctrl['ci_excludes_zero'],'permutation_p':p['p'],
                        'verdict':historical_verdict(bare['ci_excludes_zero'],ctrl['ci_excludes_zero'],p['p'],len(yy),tf['n_items_kept']),
                        'n_models':len(yy),'bootstrap_valid_counts':[bare['n_valid'],ctrl['n_valid']],
                        'permutation_valid_count':p['n_valid'],'permutation_exceed_count':p['exceed_count']}
                    csv_path=G/f'theta_mirt_out_decouple/theta_{dom}_{thh}.csv'
                    rows=list(csv.DictReader(csv_path.open()));col='theta' if 'theta' in rows[0] else [k for k in rows[0] if k!='model'][0]
                    mr={r['model']:float(r[col]) for r in rows};common=sorted(set(mr)&set(lookup));assert len(common)==32
                    nails[thh]={'n_models':len(common),'model_order':common,'girth_mirt_rank_rho':rho(np.array([lookup[t] for t in common]),np.array([mr[t] for t in common])),
                                'scope':'Saved mirt seed0 scores compared with newly reconstructed girth; mirt fits not repeated'}
        med=float(np.median([x['partial'] for x in cells.values()]));ref=contract['statistics']['retention_reference'][dom];ret=abs(med)/abs(ref)
        result['domains'][dom]={'cells':cells,'median_partial':med,'full_run_partial_ref':ref,'retention':ret,
             'retention_percent':100*ret,'same_sign':bool(np.sign(med)==np.sign(ref) and med!=0),
             'seed0_full':full,'seed0_verdicts_concordant':full['geoA_thB']['verdict']==full['geoB_thA']['verdict'],
             'seed0_mirt_nails':nails,'historical_cell_comparisons':domain_checks,
             'historical_all_cells_match':all(x['matches_1e10'] for x in domain_checks),
             'historical_retention_difference':ret-old['domains'][dom]['retention'],
             'geometry_cache_reused_half_count':sum(g['reused_half_count'] for g in geo.values()),
             'geometry_new_half_count':sum(g['new_half_count'] for g in geo.values())}
        print(f'[retention statistics] {dom}: retained={ret*100:.9f}%, all40 old cells match={result["domains"][dom]["historical_all_cells_match"]}',flush=True)
    arrays['family_bootstrap_draws']=family_draws
    np.savez_compressed(OUT/'per_model_statistical_inputs.npz',**arrays)
    np.savez_compressed(OUT/'seed0_sampling_statistics.npz',**boot_values)
    result['environment']={'python':platform.python_version(),'numpy':np.__version__,
        'threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}}
    preflight();result['source_metadata_and_small_hashes_unchanged']=True
    save_json(OUT/'results.json',result)
    save_json(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','fill','statistics'],required=True);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    assert 1<=args.workers<=4
    preflight();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'input_contract.json').write_bytes(CP.read_bytes())
    if args.stage=='pilot':run_pilot()
    elif args.stage=='fill':run_fill(args.workers)
    else:run_statistics()


if __name__=='__main__':main()
