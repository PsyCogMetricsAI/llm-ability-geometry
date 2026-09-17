#!/usr/bin/env python3
"""Recalculate Study1 statistics from bound saved per-model data and metric caches.

Historical final statistics are opened only by compare_history(), after production.
No legacy producer is imported; original data are never written.
"""
import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata
from study1_reuse_bind import R, G, PrimitiveUnpickler, digest, pc1

SEED = 20260605


def pearson(x, y):
    a, b = np.asarray(x, float), np.asarray(y, float)
    a, b = a-a.mean(), b-b.mean()
    denominator = math.sqrt(np.dot(a, a)*np.dot(b, b))
    return float(np.dot(a, b)/denominator) if denominator else float('nan')


def design(c):
    c = np.asarray(c, float)
    if c.ndim == 1:
        c = c[:, None]
    return np.column_stack([np.ones(len(c))]+[rankdata(c[:, j]) for j in range(c.shape[1])])


def rho(x, y, c=None):
    a, b = rankdata(x), rankdata(y)
    if c is not None:
        d = design(c)
        a = a-d@np.linalg.lstsq(d, a, rcond=None)[0]
        b = b-d@np.linalg.lstsq(d, b, rcond=None)[0]
    return pearson(a, b)


def bootstrap(x, y, c, families, draws):
    family_order = sorted(set(families.tolist()))
    blocks = [np.where(families == f)[0] for f in family_order]
    values = np.full(len(draws), np.nan)
    for row, choices in enumerate(draws):
        idx = np.concatenate([blocks[k] for k in choices])
        xb, yb = x[idx], y[idx]
        if len(idx) < 3 or len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue
        cb = None if c is None else c[idx]
        if cb is not None:
            cc = cb[:, None] if cb.ndim == 1 else cb
            if any(len(np.unique(cc[:, j])) < 2 for j in range(cc.shape[1])):
                continue
            d = design(cc)
            if np.linalg.matrix_rank(d) < d.shape[1]:
                continue
        values[row] = rho(xb, yb, cb)
    finite = values[np.isfinite(values)]
    ci = np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else [None, None]
    return {'rho':rho(x, y, c), 'ci':ci, 'n_valid':int(len(finite)),
            'n_invalid':int(len(values)-len(finite)),
            'ci_excludes_zero':bool(len(finite) and (ci[0] > 0 or ci[1] < 0))}, values


def permute(x, y, c, draws):
    observed = rho(x, y, c)
    values = np.array([rho(x, y[idx], c) for idx in draws])
    finite = values[np.isfinite(values)]
    exceed = int(np.sum(np.abs(finite) >= abs(observed)-1e-12))
    return {'p':(exceed+1)/(len(finite)+1), 'exceed_count':exceed,
            'n_valid':int(len(finite)), 'n_invalid':int(len(values)-len(finite))}, values


def sb(r):
    return float(2*r/(1+r)) if np.isfinite(r) and r > -1 else float('nan')


def chi(curve, convention='late_quarter'):
    n = max(1, int(np.ceil(len(curve)*.25)))
    if convention == 'late_quarter': return float(np.mean(curve[:n]))
    if convention == 'all_adjacent': return float(np.mean(curve))
    if convention == 'deepest_pair': return float(curve[0])
    mid = len(curve)//2
    return float(np.mean(curve[max(0, mid-n//2):mid+max(1,n//2)]))


def historical_verdict(bare, controlled, p, n, k):
    if controlled and bare and p < .05: return 'CONFIRM'
    if bare and not controlled: return 'SHADOW-ONLY'
    return 'REFUTE' if n >= 30 and k >= 40 else 'AMBIGUOUS'


def compare_history(results):
    comparisons = []
    for dom in ['code', 'math']:
        historical = json.loads((G/f'convergence_verdict_{dom}.json').read_text())
        for key in ['n_models','n_families','n_items','raw_rho','partial_rho_controlling_C',
                    'bare_cluster_ci','controlled_cluster_ci','bare_ci_n_valid',
                    'controlled_ci_n_valid','permutation_p','rel_geo','rel_th',
                    'disattenuation_ceiling','verdict']:
            new, old = results['domains'][dom][key], historical[key]
            difference = None if isinstance(new, str) else float(np.max(np.abs(np.asarray(new)-np.asarray(old))))
            comparisons.append({'domain':dom,'field':key,'new':new,'historical':old,
                                'max_abs_difference':difference,'matches_tolerance':new == old if isinstance(new,str) else difference <= 1e-10})
    historical = json.loads((G/'code_dropfluency_50panel.json').read_text())
    for key, target in [('drop_fluency_C3_partial','rho'),('drop_fluency_C3_controlled_ci','ci'),('drop_fluency_C3_perm_p','permutation_p')]:
        new, old = results['domains']['code']['drop_fluency'][target], historical[key]
        difference = float(np.max(np.abs(np.asarray(new)-np.asarray(old))))
        comparisons.append({'domain':'code','field':key,'new':new,'historical':old,'max_abs_difference':difference,'matches_tolerance':difference <= 1e-10})
    return comparisons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=R/'runs/study1_reuse_v1/main')
    args = ap.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(R/'runs/study1_reuse_v1'), 'Output must be isolated within assigned run directory'
    out.mkdir(parents=True, exist_ok=True)
    binding_path = R/'config/REUSE_BINDINGS_STUDY1.json'
    binding = json.loads(binding_path.read_text())
    for item in binding['asset_bindings'].values():
        assert digest(item['path']) == item['sha256'], f"Input changed: {item['path']}"
    tags = binding['model_order']
    cov = json.loads((G/'covariates_50.json').read_text())['models']
    theta = json.loads((G/'theta_hat_panel51.json').read_text())
    matrix = json.loads((G/'response_matrix_full57.json').read_text())
    families = np.array([cov[t]['code']['family'] for t in tags])
    # Same seed is reset independently for every historic bootstrap/permutation.
    boot_rng = np.random.default_rng(SEED)
    family_draws = np.array([boot_rng.choice(13, size=13, replace=True) for _ in range(40000)])
    perm_rng = np.random.default_rng(SEED)
    permutation_draws = np.array([perm_rng.permutation(50) for _ in range(10000)])
    np.savez_compressed(out/'resampling_indices.npz', family_draws=family_draws,
                        permutation_indices=permutation_draws, families=families,
                        family_order=np.array(sorted(set(families))), model_order=np.array(tags))
    results = {'producer':'scientific-analysis', 'status':'PRODUCED_PENDING_INDEPENDENT_NUMERIC_AND_SCIENTIFIC_REVIEW',
               'binding_sha256':digest(binding_path), 'code_sha256':digest(__file__),
               'seed':SEED, 'bootstrap_replicates':10000, 'cross_domain_bootstrap_replicates':40000,
               'permutation_replicates':10000, 'model_order':tags, 'domains':{},
               'scope':'Statistics conditional on saved observations and cached metrics; not full scoring/activation replay.',
               'execution_classes':{'primary_associations_reliability_dropfluency':'HISTORICAL_ALGORITHM_REIMPLEMENTATION_FROM_BOUND_INPUTS',
                                    'P2-B-005':'RECONSTRUCTED_FROM_DECLARED_ESTIMAND',
                                    'P2-B-009-THETA':'RECONSTRUCTED_FROM_DECLARED_ESTIMAND',
                                    'P2-B-009-GEO':'RECONSTRUCTED_FROM_DECLARED_ESTIMAND',
                                    'P2-B-010':'PARTIAL_COMPUTABLE_DIAGNOSTICS_ONLY'},
               'limitations':binding['limitations']}
    input_arrays, sample_arrays, inputs = {}, {}, {}
    for dom in ['code', 'math']:
        print(f'[{dom}] loading bound saved inputs', flush=True)
        with Path(binding['caches'][dom]['path']).open('rb') as stream:
            cache = PrimitiveUnpickler(stream).load()
        m, seeds = cache['metrics'], cache['seeds']
        members = {'eff_rank':np.array([m[t]['g1']['eff_rank'] for t in tags]),
                   'participation_ratio':np.array([m[t]['g1']['participation_ratio'] for t in tags]),
                   'g5no_chi':np.array([chi(m[t]['curve']) for t in tags])}
        x = pc1(members)
        y = np.array([theta[dom][t] for t in tags])
        c = np.array([[cov[t][dom][k] for k in ['scale_log10_params','d_model','family_code','fluency_mean_logprob']] for t in tags])
        accuracy = np.array([np.mean(list(matrix[t][dom].values())) for t in tags])
        assert all(np.isfinite(v).all() for v in [x,y,c,accuracy,*members.values()])
        inputs[dom] = (x, y, c, accuracy)
        for key, values in dict(members, composite=x, theta=y, controls=c, accuracy=accuracy).items():
            input_arrays[f'{dom}_{key}'] = values
        reliabilities, half_composites = [], []
        chi_reliabilities = []
        for seed in seeds:
            half = []
            for h in [0,1]:
                mh = {key:np.array([m[t]['halves'][seed]['chi' if key == 'g5no_chi' else key][h] for t in tags]) for key in members}
                half.append(pc1(mh))
                for key, values in mh.items(): input_arrays[f'{dom}_seed{seed}_half{h}_{key}'] = values
            half_composites.append(half)
            reliabilities.append(sb(rho(*half)))
            chi_reliabilities.append(sb(rho(np.array([m[t]['halves'][seed]['chi'][0] for t in tags]),np.array([m[t]['halves'][seed]['chi'][1] for t in tags]))))
        input_arrays[f'{dom}_half_composites'] = np.array(half_composites)
        bare, draws = bootstrap(x, y, c[:,2], families, family_draws[:10000])
        sample_arrays[f'{dom}_family_only_bootstrap'] = draws
        controlled, draws = bootstrap(x, y, c, families, family_draws[:10000])
        sample_arrays[f'{dom}_controlled_bootstrap'] = draws
        permutation, draws = permute(x, y, c, permutation_draws)
        sample_arrays[f'{dom}_controlled_permutation'] = draws
        rgeo, rtheta = float(np.mean(reliabilities)), float(theta['rel'][dom])
        d = {'n_models':50,'n_families':13,'n_items':theta['n_items'][dom],
             'raw_rho':rho(x,y),'family_only_partial_rho':bare['rho'],
             'partial_rho_controlling_C':controlled['rho'],
             'bare_cluster_ci':bare['ci'],'bare_ci_n_valid':bare['n_valid'],
             'bare_ci_excludes_zero':bare['ci_excludes_zero'],
             'controlled_cluster_ci':controlled['ci'],'controlled_ci_n_valid':controlled['n_valid'],
             'controlled_ci_excludes_zero':controlled['ci_excludes_zero'],
             'permutation_p':permutation['p'],'permutation_details':permutation,
             'rel_geo':rgeo,'rel_th':rtheta,'disattenuation_ceiling':math.sqrt(rgeo*rtheta),
             'geometry_reliability_per_seed':reliabilities,'chi_reliability_per_seed':chi_reliabilities,
             'theta_accuracy_rho':rho(y,accuracy),
             'accuracy_partial_rho':rho(x,accuracy,c),
             'theta_geometry_given_accuracy':rho(x,y,accuracy),
             'theta_geometry_given_accuracy_controls':rho(x,y,np.column_stack([accuracy,c])),
             'drop_scale_partial_rho':rho(x,y,c[:,1:]),
             'scale_width_Pearson':pearson(c[:,0],c[:,1]),
             'magnitude_reduction_percent':100*(abs(rho(x,y))-abs(controlled['rho']))/abs(rho(x,y)),
             'g5_orthogonality_abs_rho':abs(rho(pc1({k:v for k,v in members.items() if k != 'g5no_chi'}),members['g5no_chi'])),
             'g5_gate_status':'pass-conditional; token-n uncomputed',
             'member_partial_rho':{k:rho(v,y,c) for k,v in members.items()},
             'layer_conventions':{conv:rho(pc1(dict(members,g5no_chi=np.array([chi(m[t]['curve'],conv) for t in tags]))),y,c) for conv in ['late_quarter','all_adjacent','deepest_pair','mid_quarter']}}
        d['verdict'] = historical_verdict(bare['ci_excludes_zero'], controlled['ci_excludes_zero'],permutation['p'],50,d['n_items'])
        if dom == 'code':
            nf, draws = bootstrap(x,y,c[:,:3],families,family_draws[:10000])
            sample_arrays['code_dropfluency_bootstrap'] = draws
            nfperm, draws = permute(x,y,c[:,:3],permutation_draws)
            sample_arrays['code_dropfluency_permutation'] = draws
            d['drop_fluency'] = dict(nf,permutation_p=nfperm['p'],permutation_details=nfperm)
        results['domains'][dom] = d
        print(f"[{dom}] raw={d['raw_rho']:.8f}, partial={d['partial_rho_controlling_C']:.8f}, historical algorithm={d['verdict']}",flush=True)
    results['cross_domain'] = {}
    for k,col in [('theta',1),('geometry',0)]:
        stat, draws = bootstrap(inputs['code'][col],inputs['math'][col],None,families,family_draws)
        results['cross_domain'][k] = stat
        sample_arrays[f'cross_domain_{k}_bootstrap'] = draws
    np.savez_compressed(out/'per_model_inputs.npz',model_order=np.array(tags),families=families,**input_arrays)
    np.savez_compressed(out/'resampling_statistics.npz',**sample_arrays)
    results['historical_comparisons'] = compare_history(results)
    results['historical_comparison_all_matched'] = all(x['matches_tolerance'] for x in results['historical_comparisons'])
    results['environment'] = {'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
                              'threads':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
                              'platform':platform.platform(),'command':sys.argv}
    for item in binding['asset_bindings'].values():
        assert digest(item['path']) == item['sha256'], f"Original input mutated: {item['path']}"
    results['bound_input_no_mutation_check'] = True
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (out/'artifact_hashes.json').write_text(json.dumps({p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'artifact_hashes.json'},indent=2)+'\n')
    print(f'Output: {out}; historical numerical match={results["historical_comparison_all_matched"]}',flush=True)


if __name__ == '__main__': main()
