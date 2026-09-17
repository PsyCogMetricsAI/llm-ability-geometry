#!/usr/bin/env python3
"""Bind the saved exact50 inputs; never import or write historical analysis code."""
import hashlib
import json
import pickle
import platform
from pathlib import Path

import numpy as np
import scipy

R = Path(__file__).resolve().parents[1]
P = R.parent
G = P / 'Imports/geometry'


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'Global forbidden: {module}.{name}')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pc1(members):
    names = sorted(members)
    cols = [np.asarray(members[n], float) for n in names]
    z = np.column_stack([(x-x.mean())/(x.std() or 1.) for x in cols])
    u, s, vt = np.linalg.svd(z-z.mean(0), full_matrices=False)
    return u[:, 0]*s[0]*(-1 if vt[0, names.index('eff_rank')] < 0 else 1)


def main():
    files = {
        'rho': G/'rho_inputs_50.json', 'covariates': G/'covariates_50.json',
        'matrix': G/'response_matrix_full57.json', 'theta': G/'theta_hat_panel51.json',
        'resource_audit': R/'reports/RESOURCE_REUSE_AUDIT.json',
        'schema_audit': R/'inputs_manifest/INPUT_SCHEMA_AUDIT.json',
        'theta_independent_review': R/'verifier/V_NUM_THETA_REPLAY.md',
        'cache_generator': G/'run_offline_b.py', 'geometry_definitions': G/'offline_b_analyze.py',
        'composite_exporter': G/'export_rho_inputs_50.py',
        'statistics_source': G/'offline_analyze_docs_20260607/s5_convergence_analyze.py',
        'theta_split_source': G/'recompute_decision_3_7.py',
        'binding_code': Path(__file__),
    }
    rho = json.loads(files['rho'].read_text())['models']
    cov = json.loads(files['covariates'].read_text())['models']
    theta = json.loads(files['theta'].read_text())
    matrix = json.loads(files['matrix'].read_text())
    tags = sorted(rho)
    assert len(tags) == 50 and tags == sorted(cov)
    families = [cov[t]['code']['family'] for t in tags]
    assert len(set(families)) == 13
    family_codes = {f:i for i,f in enumerate(sorted(set(families)))}
    checks, caches = {}, {}
    for dom in ['code', 'math']:
        path = R/f'recovered/Imports/geometry/caches/_metrics_cache_panel_geom_panel53_{dom}.pkl'
        files[f'cache_{dom}'] = path
        files[f'gate_{dom}'] = G/f'g5no_gates_{dom}.json'
        files[f'historical_{dom}'] = G/f'convergence_verdict_{dom}.json'
        with path.open('rb') as stream:
            cache = PrimitiveUnpickler(stream).load()
        assert sorted(cache['metrics']) == tags
        assert cache['seeds'] == list(range(20260607, 20260647))
        assert sorted(set(theta[dom])-set(tags)) == ['llama_3_1_70b_instruct']
        m = cache['metrics']
        members = {'eff_rank': [m[t]['g1']['eff_rank'] for t in tags],
                   'participation_ratio': [m[t]['g1']['participation_ratio'] for t in tags],
                   'g5no_chi': [float(np.mean(m[t]['curve'][:max(1,int(np.ceil(len(m[t]['curve'])*.25)))])) for t in tags]}
        delta = float(np.max(np.abs(pc1(members)-[rho[t][dom]['geometry_composite_zpc1'] for t in tags])))
        assert delta < 1e-12
        assert all(theta[dom][t] == rho[t][dom]['theta_hat_2pl_eap'] for t in tags)
        assert all(cov[t][dom]['family'] == families[i] and cov[t][dom]['family_code'] == family_codes[families[i]] for i,t in enumerate(tags))
        assert all(np.isfinite(v).all() for v in members.values())
        assert all(set(m[t]['halves']) == set(cache['seeds']) for t in tags)
        assert all(set(matrix[t][dom].values()) <= {0,1} for t in tags)
        checks[dom] = {'models':50, 'family_count':13, 'seeds':40,
                       'composite_export_max_abs_difference':delta,
                       'theta_export_max_abs_difference':0,
                       'geometry_layer_order':'deepest first; item answer-tail cloud',
                       'gate':json.loads(files[f'gate_{dom}'].read_text())['verdict'],
                       'behavior_item_counts':sorted(set(len(matrix[t][dom]) for t in tags)),
                       'fit_item_count':theta['n_items'][dom],
                       'historical_theta_converged':theta['converged'][dom]}
        caches[dom] = {'path':str(path), 'seed_order':cache['seeds'], 'model_order':tags}
    ids = ['P2-A-003','P2-A-004','P2-A-006-CODE','P2-A-006-MATH','P2-A-008-CODE','P2-A-008-MATH',
           'P2-B-001-CODE','P2-B-001-MATH','P2-B-002','P2-B-005','P2-B-008','P2-B-009-THETA','P2-B-009-GEO','P2-B-010']
    contract = {
        'schema':'P2-STUDY1-REUSE-BINDINGS-v1', 'producer':'scientific-analysis',
        'status':'PENDING_INDEPENDENT_LEAF_CONTRACT_REVIEW', 'dag_nodes':['U1A','U2A','U2B'],
        'subgoals':['S1','S2','S3'], 'dod':['D1','D2','D3'],
        'execution_boundary':'Saved model scores, exact50 metric and split-half caches, aligned covariates, and already-replayed panel51 theta. No LLM inference, activation extraction or repeated theta fitting.',
        'result_ids_for_leaf_review':ids, 'model_order':tags, 'family_order':sorted(set(families)),
        'asset_bindings':{k:{'path':str(v),'sha256':digest(v),'bytes':v.stat().st_size} for k,v in files.items()},
        'checks':checks, 'caches':caches,
        'statistical_contract':{
            'composite':'sorted columns z-score ddof0; centered SVD PC1; eff_rank loading positive; conditional G5 admitted exactly as historical run',
            'geometry_reliability':'40 seeds; independently standardize and fit each half composite, correlate panel ranks, apply 2r/(1+r), mean over seeds. Cached halves are item halves, not token halves.',
            'raw_rho':'Pearson of average-tie ranks',
            'partial':'rank each variable and every C column; residualize both ranks against intercept plus ranked C by least-squares; Pearson residuals',
            'controls':['log10_params','d_model','ordinal_sorted_family_code','mean_logprob'],
            'bare_interval':'family-ordinal-controlled partial statistic; NOT a CI for raw_rho',
            'bootstrap':{'seed':20260605,'replicates':10000,'cross_domain_replicates':40000,'resample_unit':'13 complete families with replacement; sorted family labels; model order retained within blocks','rerank':True,'PCA_refit':False,'CI_percentiles':[2.5,97.5],'invalid':'skip constant x/y/C or rank-deficient ranked design; retain counts and signed draws'},
            'permutation':{'seed':20260605,'replicates':10000,'algorithm':'unrestricted permutation of theta over models; C and geometry fixed; two-sided >= abs(obs)-1e-12; plus-one correction','interpretation':'historical computational replay only; exchangeability and inferential validity unresolved'},
            'theta_accuracy':'mean of stored zero-coded full response bank (code170/math1333), not only variable columns; exact50 lookup',
            'cross_domain':'raw Spearman of exact50 paired theta or geometry; signed family bootstrap; no covariate residualization',
            'sensitivity':'drop fluency, drop scale, four chi layer conventions; preserve each as separate diagnostic',
        },
        'tolerance':{'point_and_CI_abs':1e-10,'permutation_count':'exact match','displayed_only':'half of final displayed decimal unit; full values retained'},
        'outputs':'runs/study1_reuse_v1; all per-model inputs, half arrays, bootstrap family-index draws, permutation indices, statistics and comparison json; no original writes',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'threads':1,'rng':'numpy Generator PCG64'},
        'scientific_status':'UNRESOLVED; replay acceptance is not N4 evidence release',
        'limitations':[
            'Four historical 2PL fits are nonconverged and F1 unavailable; do not silently repair or label convergence.',
            'Exact50 membership frozen; omission reason for llama_3_1_70b_instruct is undocumented.',
            'Cache-to-export equality is established; a full activation-to-cache rebuild has not been performed. Input geometry producer revision/BOS metadata remain incompletely recorded.',
            'Ordinal family code is not categorical family adjustment. Unrestricted permutations are not family-aware conditional null tests.',
            'Historical bare CI belongs to family-only controlled rho, not the displayed raw rho.',
            'G5 token-n gate remains uncomputed; convention checks do not establish 4/4 passage.',
            'Source scoring replay, girth full/split-half fits, disjoint-retention inputs and BOS/cloud-size branches are separate leaves; none is marked complete by this contract.',
            'A005 original girth-mirt cross-check source not yet matched. A007 saved theta split vectors need location or minimal CPU fitting. A001/A002 historical custom/synthetic claims not covered.',
        ],
        'leaf_scope':{
            'exact50_association':'eligible for technical replay after independent acceptance',
            'cache_split_half':'eligible for technical replay after independent acceptance; independent cache numerical checks still required for scientific release',
            'B010':'orthogonality and layer-convention diagnostic components only; token-n claim remains unresolved',
            'girth_and_retention':'not yet authorized by this leaf contract',
        },
    }
    target = R/'config/REUSE_BINDINGS_STUDY1.json'
    target.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
    print(target)
    print(json.dumps(checks,ensure_ascii=False))


if __name__ == '__main__':
    main()
