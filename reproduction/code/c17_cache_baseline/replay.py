#!/usr/bin/env python3
"""C17 cached-baseline replay (worker C10b).

Standalone producer that recomputes the frozen C17 baseline statistics from
saved per-fit / per-draw artifacts so a final clean run does not need to
re-execute the accepted large fits.  It never refits IRT models, never imports
the historical analysis modules, never writes into the analysis tree, and it
fails closed on a non-empty output directory, a missing input, or a hash
mismatch (including a source mutation detected during the run).

Usage:
    python code/c17_cache_baseline/replay.py --analysis-root <R> --out <NEW_DIR> \
        [--branch all|study1|lofo|empirical|medical|theory]

Reference level of every produced number is a *cache replay* of the accepted
producer artifacts, not a fresh scientific fit.  Old terminal reports are read
only as comparators, after the fresh numbers exist.
"""
from __future__ import annotations

import os

for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import argparse
import hashlib
import json
import math
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import rankdata, spearmanr

SCHEMA = 'C17-CACHE-BASELINE-REPLAY-v1'
BRANCHES = ('study1', 'lofo', 'empirical', 'medical', 'theory', 'bos', 'retention')
TOL_RULE = 'abs<=1e-9+1e-6*abs(reference)'


def dig(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    """JSON-safe copy; non-finite floats become null (comparisons run first)."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def jget(obj, pointer):
    cur = obj
    for part in pointer.strip('/').split('/'):
        if part == '':
            continue
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def close(a, b):
    return abs(a - b) <= 1e-9 + 1e-6 * abs(b)


# --------------------------------------------------------------------------
# statistics re-implemented independently of the historical producers
# --------------------------------------------------------------------------
def pearson(x, y):
    a, b = np.asarray(x, float), np.asarray(y, float)
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den else float('nan')


def design(c):
    c = np.asarray(c, float)
    if c.ndim == 1:
        c = c[:, None]
    return np.column_stack([np.ones(len(c))] + [rankdata(c[:, j]) for j in range(c.shape[1])])


def rho(x, y, c=None):
    a, b = rankdata(x), rankdata(y)
    if c is not None:
        d = design(c)
        a = a - d @ np.linalg.lstsq(d, a, rcond=None)[0]
        b = b - d @ np.linalg.lstsq(d, b, rcond=None)[0]
    return pearson(a, b)


def sb(r):
    return float(2 * r / (1 + r)) if np.isfinite(r) and r > -1 else float('nan')


def family_blocks(families):
    return [np.where(np.asarray(families) == f)[0] for f in sorted(set(np.asarray(families).tolist()))]


def boot_stat(x, y, c, families, choices):
    blocks = family_blocks(families)
    idx = np.concatenate([blocks[k] for k in choices])
    return rho(x[idx], y[idx], None if c is None else np.asarray(c)[idx])


def ci_from_draws(values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    ci = np.percentile(finite, [2.5, 97.5]).tolist() if finite.size else [None, None]
    return {'ci': ci, 'n_valid': int(finite.size), 'n_invalid': int(values.size - finite.size),
            'ci_excludes_zero': bool(finite.size and (ci[0] > 0 or ci[1] < 0))}


def perm_from_draws(observed, values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    exceed = int(np.sum(np.abs(finite) >= abs(observed) - 1e-12))
    return {'p': (exceed + 1) / (finite.size + 1), 'exceed_count': exceed,
            'n_valid': int(finite.size), 'n_invalid': int(values.size - finite.size)}


def pc1(members):
    names = sorted(members)
    cols = [np.asarray(members[n], float) for n in names]
    z = np.column_stack([(x - x.mean()) / (x.std() or 1.) for x in cols])
    u, s, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    return u[:, 0] * s[0] * (-1 if vt[0, names.index('eff_rank')] < 0 else 1)


def pc1_sorted_first(members):
    """PC1 oriented by the loading of the alphabetically first member."""
    names = sorted(members)
    cols = [np.asarray(members[n], float) for n in names]
    z = np.column_stack([(x - x.mean()) / (x.std() or 1.) for x in cols])
    u, s, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
    return u[:, 0] * s[0] * (-1 if vt[0, 0] < 0 else 1)


def chi(curve, convention='late_quarter'):
    n = max(1, int(np.ceil(len(curve) * .25)))
    if convention == 'late_quarter':
        return float(np.mean(curve[:n]))
    if convention == 'all_adjacent':
        return float(np.mean(curve))
    if convention == 'deepest_pair':
        return float(curve[0])
    mid = len(curve) // 2
    return float(np.mean(curve[max(0, mid - n // 2):mid + max(1, n // 2)]))


def historical_verdict(bare, controlled, p, n, k):
    if controlled and bare and p < .05:
        return 'CONFIRM'
    if bare and not controlled:
        return 'SHADOW-ONLY'
    return 'REFUTE' if n >= 30 and k >= 40 else 'AMBIGUOUS'


def bh(pvalues):
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    out = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        out[i] = min(prev, 1.0)
    return out.tolist()


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f'global forbidden: {module}.{name}')


# --------------------------------------------------------------------------
# replay context
# --------------------------------------------------------------------------
class Replay:
    def __init__(self, root, out, pins=None):
        self.R = Path(root).resolve()
        self.P = self.R.parent
        self.out = Path(out).resolve()
        self.pins = pins
        self.manifest = {}
        self._abs = {}
        self.checks = []
        self.comparators = []
        self.gaps = []
        self.branch_results = {}
        self.t0 = time.time()

    # -- manifest ---------------------------------------------------------
    def rel(self, path):
        path = Path(path).resolve()
        for root, tag in ((self.R, 'R'), (self.P, 'P')):
            try:
                return f'{tag}/{path.relative_to(root)}'
            except ValueError:
                continue
        raise ValueError(f'consumed path outside analysis roots: {path}')

    def register(self, path, role, level, declared_sha256=None):
        path = Path(path)
        rel = self.rel(path)
        if not path.exists():
            raise FileNotFoundError(f'missing required input: {rel}')
        sha = dig(path)
        if self.pins is not None and rel not in self.pins:
            raise ValueError(f'--expected-manifest pin violation: consumed source not pinned: {rel}')
        if self.pins is not None and self.pins[rel] != sha:
            raise ValueError(f'--expected-manifest pin mismatch: {rel} pinned {self.pins[rel]} got {sha}')
        if declared_sha256 is not None and sha != declared_sha256:
            raise ValueError(f'declared hash mismatch: {rel} expected {declared_sha256} got {sha}')
        prev = self.manifest.get(rel)
        if prev is not None and prev['sha256'] != sha:
            raise ValueError(f'input mutated mid-run: {rel}')
        self.manifest[rel] = {'path': rel, 'role': role, 'level': level,
                              'sha256': sha, 'bytes': path.stat().st_size}
        self._abs[rel] = path
        return path

    def load_bytes(self, path, role, level, declared_sha256=None):
        path = self.register(path, role, level, declared_sha256)
        return path.read_bytes()

    def load_json(self, path, role, level, declared_sha256=None):
        return json.loads(self.load_bytes(path, role, level, declared_sha256).decode('utf-8'))

    def load_npz(self, path, role, level, declared_sha256=None):
        path = self.register(path, role, level, declared_sha256)
        return np.load(path, allow_pickle=False)

    def load_npy(self, path, role, level, declared_sha256=None):
        path = self.register(path, role, level, declared_sha256)
        return np.load(path, allow_pickle=False)

    def load_pickle(self, path, role, level, declared_sha256=None):
        path = self.register(path, role, level, declared_sha256)
        with path.open('rb') as stream:
            return PrimitiveUnpickler(stream).load()

    def verify_manifest(self):
        mutated = [rel for rel, entry in self.manifest.items() if dig(self._abs[rel]) != entry['sha256']]
        if mutated:
            raise RuntimeError(f'source mutation detected (writeback guard): {mutated}')
        return True

    # -- evidence rows ----------------------------------------------------
    def check(self, branch, name, passed, detail=None, pointer=None,
              max_abs_difference=None, tolerance=None, exact=False):
        row = {'branch': branch, 'check': name, 'passed': bool(passed), 'exact_required': bool(exact)}
        if pointer is not None:
            row['pointer'] = pointer
        if detail is not None:
            row['detail'] = detail
        if max_abs_difference is not None:
            row['max_abs_difference'] = float(max_abs_difference)
        if tolerance is not None:
            row['tolerance'] = float(tolerance)
        self.checks.append(row)
        return row

    def compare(self, branch, pointer, new, reference, artifact, exact=False):
        diff = None
        if exact or isinstance(reference, (str, bool)) or reference is None \
                or isinstance(new, (str, bool)) or new is None:
            matched = (new == reference)
        elif isinstance(reference, dict):
            matched = set(map(str, reference)) == set(map(str, new))
            if matched:
                for key in reference:
                    if not self.compare(branch, f'{pointer}/{key}', new[key], reference[key], artifact):
                        matched = False
        elif isinstance(reference, list):
            if not isinstance(new, (list, tuple)) or len(new) != len(reference):
                matched = False
            else:
                matched = True
                for i, ref in enumerate(reference):
                    if not self.compare(branch, f'{pointer}/{i}', new[i], ref, artifact):
                        matched = False
        elif isinstance(reference, (int, float, np.number)):
            a, b = float(new), float(reference)
            diff = abs(a - b)
            matched = close(a, b)
        else:
            matched = (new == reference)
        row = {'branch': branch, 'pointer': pointer, 'reference_artifact': artifact,
               'rule': 'exact' if exact else TOL_RULE, 'new': safe(new), 'reference': safe(reference),
               'max_abs_difference': diff, 'matched': bool(matched)}
        self.comparators.append(row)
        return bool(matched)

    def gap(self, branch, item, note):
        self.gaps.append({'branch': branch, 'item': item, 'note': note})

    def dump(self, path, obj):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(safe(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return path


def read_json(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------
# branch: study1 (convergence / reliability / retained panel statistics)
# --------------------------------------------------------------------------
STUDY1_DOMAIN_FIELDS = [
    'n_models', 'n_families', 'n_items', 'raw_rho', 'family_only_partial_rho',
    'partial_rho_controlling_C', 'bare_cluster_ci', 'bare_ci_n_valid',
    'bare_ci_excludes_zero', 'controlled_cluster_ci', 'controlled_ci_n_valid',
    'controlled_ci_excludes_zero', 'permutation_p', 'permutation_details', 'rel_geo',
    'rel_th', 'disattenuation_ceiling', 'geometry_reliability_per_seed',
    'chi_reliability_per_seed', 'theta_accuracy_rho', 'accuracy_partial_rho',
    'theta_geometry_given_accuracy', 'theta_geometry_given_accuracy_controls',
    'drop_scale_partial_rho', 'scale_width_Pearson', 'magnitude_reduction_percent',
    'g5_orthogonality_abs_rho', 'g5_gate_status', 'member_partial_rho',
    'layer_conventions', 'verdict',
]


def branch_study1(rp):
    branch = 'study1'
    R = rp.R
    main = R / 'runs/study1_reuse_v1/main'
    mz = rp.load_npz(main / 'per_model_inputs.npz', 'per_model_inputs', 'primary_cache_input')
    rz = rp.load_npz(main / 'resampling_statistics.npz', 'resampling_statistics', 'primary_cache_input')
    iz = rp.load_npz(main / 'resampling_indices.npz', 'resampling_indices', 'primary_cache_input')
    binding_path = R / 'config/REUSE_BINDINGS_STUDY1.json'
    binding = rp.load_json(binding_path, 'frozen_rule_binding', 'frozen_rule_config')
    rp.register(R / 'code/study1_reuse_analyze.py', 'source_code_reference', 'reference_code_only')
    rp.register(R / 'code/study1_reuse_bind.py', 'source_code_reference', 'reference_code_only')
    old_path = main / 'results.json'
    old = rp.load_json(old_path, 'accepted_producer_summary', 'comparator_only')
    theta = rp.load_json(rp.P / 'Imports/geometry/theta_hat_panel51.json',
                         'upstream_panel_theta', 'primary_cache_input')

    declared = {}
    for item in binding['asset_bindings'].values():
        declared[Path(item['path']).resolve()] = item['sha256']

    tags = [str(t) for t in mz['model_order'].tolist()]
    exact_ids = (tags == [str(t) for t in iz['model_order'].tolist()]
                 and tags == [str(t) for t in binding['model_order']]
                 and len(tags) == 50 and len(set(tags)) == 50)
    rp.check(branch, 'model_order_identity', exact_ids,
             detail=f'{len(tags)} unique model tags identical across per_model_inputs, indices and frozen binding',
             pointer='/model_order', exact=True)
    fam_mz = [str(f) for f in mz['families'].tolist()]
    fam_iz = [str(f) for f in iz['families'].tolist()]
    rp.check(branch, 'family_identity', fam_mz == fam_iz and sorted(set(fam_mz)) == sorted(str(f) for f in iz['family_order']),
             detail='50 model families and 13 sorted family labels identical across inputs and indices', exact=True)
    fd = iz['family_draws']
    pix = iz['permutation_indices']
    idx_range_ok = bool(fd.min() >= 0 and fd.max() < 13 and pix.shape == (10000, 50))
    rows_perm = bool(np.all(np.sort(pix, axis=1) == np.arange(50)[None, :]))
    rp.check(branch, 'resampling_index_domain', idx_range_ok and rows_perm,
             detail='family draws in [0,13); every permutation row is a permutation of 0..49', exact=True)

    domains = {}
    cache_checks = []
    for dom in ('code', 'math'):
        x = mz[f'{dom}_composite'].astype(float)
        y = mz[f'{dom}_theta'].astype(float)
        c = mz[f'{dom}_controls'].astype(float)
        acc = mz[f'{dom}_accuracy'].astype(float)
        members = {k: mz[f'{dom}_{k}'].astype(float) for k in ('eff_rank', 'participation_ratio', 'g5no_chi')}

        cache_path = R / f'recovered/Imports/geometry/caches/_metrics_cache_panel_geom_panel53_{dom}.pkl'
        cache = rp.load_pickle(cache_path, 'saved_metric_cache', 'secondary_linkage_input',
                               declared.get(cache_path.resolve()))
        rp.register(R / f'config/REUSE_BINDINGS_STUDY1.json', 'frozen_rule_binding', 'frozen_rule_config')
        cache_tags = [str(t) for t in sorted(cache['metrics'])]
        rp.check(branch, f'{dom}_cache_model_identity', cache_tags == sorted(tags),
                 detail='saved metric cache covers the exact same 50 model tags', exact=True)
        cache_members = {
            'eff_rank': np.array([cache['metrics'][t]['g1']['eff_rank'] for t in tags]),
            'participation_ratio': np.array([cache['metrics'][t]['g1']['participation_ratio'] for t in tags]),
            'g5no_chi': np.array([chi(cache['metrics'][t]['curve']) for t in tags]),
        }
        delta = float(np.max(np.abs(pc1(cache_members) - x)))
        cache_checks.append({'domain': dom, 'max_abs_difference_vs_saved_composite': delta})
        rp.check(branch, f'{dom}_composite_cache_linkage', delta <= 1e-9 + 1e-6 * float(np.max(np.abs(x))),
                 detail='fresh pc1 of saved metric cache equals per_model_inputs composite',
                 pointer=f'/domains/{dom}/composite', max_abs_difference=delta,
                 tolerance=1e-9 + 1e-6 * float(np.max(np.abs(x))))
        for key in ('eff_rank', 'participation_ratio', 'g5no_chi'):
            dmax = float(np.max(np.abs(cache_members[key] - members[key])))
            rp.check(branch, f'{dom}_{key}_cache_linkage', dmax <= 1e-12,
                     pointer=f'/domains/{dom}/{key}', max_abs_difference=dmax, tolerance=1e-12)
        dtheta = float(np.max(np.abs(np.array([theta[dom][t] for t in tags]) - y)))
        rp.check(branch, f'{dom}_theta_export_linkage', dtheta <= 0.0,
                 pointer=f'/domains/{dom}/theta_hat_2pl_eap', max_abs_difference=dtheta, tolerance=0.0)

        seeds = [int(s) for s in binding['caches'][dom]['seed_order']]
        half = mz[f'{dom}_half_composites'].astype(float)
        rel_seed = [sb(rho(half[s, 0], half[s, 1])) for s in range(half.shape[0])]
        chi_rel = [sb(rho(mz[f'{dom}_seed{sd}_half0_g5no_chi'], mz[f'{dom}_seed{sd}_half1_g5no_chi']))
                   for sd in seeds]
        # half-array regeneration: rebuild each half composite from its member halves
        half_delta = 0.0
        for s, sd in enumerate(seeds):
            for h in (0, 1):
                rebuilt = pc1({k: mz[f'{dom}_seed{sd}_half{h}_{k}'] for k in members})
                half_delta = max(half_delta, float(np.max(np.abs(rebuilt - half[s, h]))))
        rp.check(branch, f'{dom}_half_composite_regeneration', half_delta <= 1e-12,
                 detail='seed half composites rebuilt from saved member halves',
                 pointer=f'/domains/{dom}/geometry_reliability_per_seed',
                 max_abs_difference=half_delta, tolerance=1e-12)

        bare_point = rho(x, y, c[:, 2])
        bare = dict(ci_from_draws(rz[f'{dom}_family_only_bootstrap']))
        bare['rho'] = bare_point
        controlled_point = rho(x, y, c)
        controlled = dict(ci_from_draws(rz[f'{dom}_controlled_bootstrap']))
        controlled['rho'] = controlled_point
        perm = perm_from_draws(controlled_point, rz[f'{dom}_controlled_permutation'])
        rel_geo = float(np.mean(rel_seed))
        rel_th = float(theta['rel'][dom])
        n_items = int(theta['n_items'][dom])
        rp.check(branch, f'{dom}_n_items_rule_source', n_items == int(binding['checks'][dom]['fit_item_count']),
                 detail=f'n_items={n_items} from saved panel theta, identical to frozen binding fit_item_count',
                 pointer=f'/domains/{dom}/n_items', exact=True)
        d = {
            'n_models': 50, 'n_families': 13, 'n_items': n_items,
            'raw_rho': rho(x, y), 'family_only_partial_rho': bare_point,
            'partial_rho_controlling_C': controlled_point,
            'bare_cluster_ci': bare['ci'], 'bare_ci_n_valid': bare['n_valid'],
            'bare_ci_excludes_zero': bare['ci_excludes_zero'],
            'controlled_cluster_ci': controlled['ci'], 'controlled_ci_n_valid': controlled['n_valid'],
            'controlled_ci_excludes_zero': controlled['ci_excludes_zero'],
            'permutation_p': perm['p'], 'permutation_details': perm,
            'rel_geo': rel_geo, 'rel_th': rel_th,
            'disattenuation_ceiling': math.sqrt(rel_geo * rel_th),
            'geometry_reliability_per_seed': rel_seed, 'chi_reliability_per_seed': chi_rel,
            'theta_accuracy_rho': rho(y, acc), 'accuracy_partial_rho': rho(x, acc, c),
            'theta_geometry_given_accuracy': rho(x, y, acc),
            'theta_geometry_given_accuracy_controls': rho(x, y, np.column_stack([acc, c])),
            'drop_scale_partial_rho': rho(x, y, c[:, 1:]),
            'scale_width_Pearson': pearson(c[:, 0], c[:, 1]),
            'magnitude_reduction_percent': 100 * (abs(rho(x, y)) - abs(controlled_point)) / abs(rho(x, y)),
            'g5_orthogonality_abs_rho': abs(rho(pc1({k: v for k, v in members.items() if k != 'g5no_chi'}), members['g5no_chi'])),
            'g5_gate_status': 'pass-conditional; token-n uncomputed',
            'member_partial_rho': {k: rho(v, y, c) for k, v in members.items()},
            'layer_conventions': {conv: rho(pc1(dict(members, g5no_chi=np.array(
                [chi(cache['metrics'][t]['curve'], conv) for t in tags]))), y, c)
                for conv in ('late_quarter', 'all_adjacent', 'deepest_pair', 'mid_quarter')},
        }
        d['verdict'] = historical_verdict(bare['ci_excludes_zero'], controlled['ci_excludes_zero'],
                                          perm['p'], 50, n_items)
        if dom == 'code':
            nf_point = rho(x, y, c[:, :3])
            nf = dict(ci_from_draws(rz['code_dropfluency_bootstrap']))
            nf['rho'] = nf_point
            nfp = perm_from_draws(nf_point, rz['code_dropfluency_permutation'])
            d['drop_fluency'] = dict(nf, permutation_p=nfp['p'], permutation_details=nfp)
        domains[dom] = d

    cross = {}
    for key, xa, ya in (('theta', mz['code_theta'].astype(float), mz['math_theta'].astype(float)),
                        ('geometry', mz['code_composite'].astype(float), mz['math_composite'].astype(float))):
        st = dict(ci_from_draws(rz[f'cross_domain_{key}_bootstrap']))
        st['rho'] = rho(xa, ya)
        cross[key] = st

    # representative original-estimator regeneration (input linkage)
    regen = []
    fam_arr = np.asarray(fam_mz)
    samples = {'controlled_bootstrap': ('controlled', [0, 1, 17, 4321, 9999]),
               'family_only_bootstrap': ('family_only', [0, 2, 777, 5000, 9999]),
               'controlled_permutation': ('permutation', [0, 1, 7, 9999])}
    for dom in ('code', 'math'):
        x = mz[f'{dom}_composite'].astype(float)
        y = mz[f'{dom}_theta'].astype(float)
        c = mz[f'{dom}_controls'].astype(float)
        for stream, (kind, rows) in samples.items():
            saved = rz[f'{dom}_{stream}']
            for row in rows:
                if kind == 'controlled':
                    fresh = boot_stat(x, y, c, fam_arr, fd[row])
                elif kind == 'family_only':
                    fresh = boot_stat(x, y, c[:, 2], fam_arr, fd[row])
                else:
                    fresh = rho(x, y[pix[row]], c)
                diff = abs(fresh - float(saved[row]))
                regen.append({'domain': dom, 'stream': stream, 'row': row, 'fresh': fresh,
                              'saved': float(saved[row]), 'abs_difference': diff})
    for key, xa, ya, rows in (('theta', mz['code_theta'].astype(float), mz['math_theta'].astype(float), [0, 20000, 39999]),
                              ('geometry', mz['code_composite'].astype(float), mz['math_composite'].astype(float), [0, 12345, 39999])):
        saved = rz[f'cross_domain_{key}_bootstrap']
        for row in rows:
            fresh = boot_stat(xa, ya, None, fam_arr, fd[row])
            diff = abs(fresh - float(saved[row]))
            regen.append({'domain': 'cross_domain_' + key, 'stream': 'cross_domain_bootstrap',
                          'row': row, 'fresh': fresh, 'saved': float(saved[row]), 'abs_difference': diff})
    worst = max(r['abs_difference'] for r in regen)
    rp.check(branch, 'representative_resampling_regeneration', worst <= 1e-9,
             detail=f'{len(regen)} saved draws regenerated from per-model arrays + saved indices',
             max_abs_difference=worst, tolerance=1e-9)

    matched = 0
    total = 0
    for dom in ('code', 'math'):
        fields = STUDY1_DOMAIN_FIELDS + (['drop_fluency'] if dom == 'code' else [])
        for field in fields:
            total += 1
            matched += rp.compare(branch, f'/domains/{dom}/{field}', domains[dom][field],
                                  old['domains'][dom][field], 'R/runs/study1_reuse_v1/main/results.json')
    for key in ('theta', 'geometry'):
        for field in ('rho', 'ci', 'n_valid', 'n_invalid', 'ci_excludes_zero'):
            total += 1
            matched += rp.compare(branch, f'/cross_domain/{key}/{field}', cross[key][field],
                                  old['cross_domain'][key][field], 'R/runs/study1_reuse_v1/main/results.json')

    summary = {
        'schema': SCHEMA + '/study1', 'branch': 'study1', 'status': 'CACHED_BASELINE_REPLAY_PRODUCED',
        'replay_level': 'CACHED_BASELINE_REPLAY (no scoring, activation, cache or theta-fit rerun)',
        'scope': 'fresh recomputation from runs/study1_reuse_v1 saved per-model inputs, per-draw statistics and saved indices; '
                 'metric caches only for layer-convention diagnostics and linkage checks',
        'seed': int(old['seed']), 'bootstrap_replicates': 10000, 'cross_domain_bootstrap_replicates': 40000,
        'permutation_replicates': 10000, 'model_order': tags, 'families': fam_mz,
        'domains': domains, 'cross_domain': cross,
        'execution_classes': {'primary_associations_reliability_dropfluency':
                              'CACHE_REPLAY_REIMPLEMENTATION_FROM_SAVED_INPUTS_AND_DRAWS',
                              'P2-B-005': 'RECONSTRUCTED_FROM_DECLARED_ESTIMAND',
                              'P2-B-009-THETA': 'RECONSTRUCTED_FROM_DECLARED_ESTIMAND',
                              'P2-B-009-GEO': 'RECONSTRUCTED_FROM_DECLARED_ESTIMAND',
                              'P2-B-010': 'PARTIAL_COMPUTABLE_DIAGNOSTICS_ONLY'},
        'limitations': binding['limitations'],
        'cache_linkage': cache_checks,
        'representative_regeneration': regen,
        'historical_numeric_claims_in_producer_report': 'comparator_only',
    }
    return {'branch': branch, 'status': summary['status'], 'artifacts': {'study1_baseline.json': summary},
            'summary': {'comparator_matched': matched, 'comparator_total': total,
                        'checks_passed': sum(1 for c in rp.checks if c['branch'] == branch and c['passed']),
                        'checks_total': sum(1 for c in rp.checks if c['branch'] == branch),
                        'worst_regeneration_difference': worst}}


# --------------------------------------------------------------------------
# branch: lofo + retained study2 legacy aggregation and C9 ledger audit
# --------------------------------------------------------------------------
def branch_lofo(rp):
    branch = 'lofo'
    R = rp.R
    lofo_dir = R / 'runs/study2_reuse_v1/lofo'
    receipt = rp.load_json(lofo_dir / 'receipt.json', 'lofo_receipt', 'hash_declaration', None)
    declared = {lofo_dir / name: sha for name, sha in receipt['outputs'].items()}
    inputs = rp.load_json(lofo_dir / 'inputs.json', 'lofo_inputs', 'primary_cache_input',
                          declared.get(lofo_dir / 'inputs.json'))
    null = rp.load_npy(lofo_dir / 'permutation_raw_rho.npy', 'lofo_permutation_statistics',
                       'primary_cache_input', declared.get(lofo_dir / 'permutation_raw_rho.npy'))
    indices = rp.load_npy(lofo_dir / 'permutation_indices.npy', 'lofo_permutation_indices',
                          'primary_cache_input', declared.get(lofo_dir / 'permutation_indices.npy'))
    old = rp.load_json(lofo_dir / 'results.json', 'accepted_producer_summary', 'comparator_only',
                       declared.get(lofo_dir / 'results.json'))
    rp.register(R / 'code/study2_reuse_lofo.py', 'source_code_reference', 'reference_code_only')

    tags = [str(t) for t in inputs['model_order']]
    x = np.asarray(inputs['PR'], float)
    y = np.asarray(inputs['theta'], float)
    z = np.asarray(inputs['log_params'], float)
    f = np.asarray([str(v) for v in inputs['family']])
    exact_ids = (len(tags) == 43 and len(set(tags)) == 43 and len(x) == len(y) == len(z) == len(f) == 43
                 and len(set(f.tolist())) == 13)
    rp.check(branch, 'input_identity', exact_ids,
             detail='43 unique science model rows, 13 families, aligned PR/theta/log_params', exact=True)
    rp.check(branch, 'permutation_index_shape', indices.shape == (2000, 43)
             and bool(np.all(np.sort(indices, axis=1) == np.arange(43)[None, :])),
             detail='2000 unrestricted permutations of the 43 model rows', exact=True)
    recalc = np.array([spearmanr(x, y[ix]).statistic for ix in indices])
    dmax = float(np.max(np.abs(recalc - null)))
    rp.check(branch, 'saved_null_array_linkage', dmax <= 1e-9 + 1e-6 * float(np.max(np.abs(null))),
             detail='all 2000 saved permutation statistics regenerated from saved indices',
             max_abs_difference=dmax, tolerance=1e-9 + 1e-6 * float(np.max(np.abs(null))))

    families = sorted(set(f.tolist()))
    observed_raw = float(spearmanr(x, y).statistic)
    observed_par = rho(x, y, z)

    def partial_rank(xa, ya, za):
        d = np.column_stack([np.ones(len(xa)), rankdata(za)])
        xr, yr = rankdata(xa), rankdata(ya)
        ex = xr - d @ np.linalg.lstsq(d, xr, rcond=None)[0]
        ey = yr - d @ np.linalg.lstsq(d, yr, rcond=None)[0]
        ex, ey = ex - ex.mean(), ey - ey.mean()
        return float(ex @ ey / math.sqrt(float(ex @ ex) * float(ey @ ey)))

    rows = []
    for family in families:
        keep = f != family
        rows.append({'family_dropped': family, 'n_dropped': int((~keep).sum()),
                     'n_remaining': int(keep.sum()),
                     'rho_raw': float(spearmanr(x[keep], y[keep]).statistic),
                     'rho_partial_logparams': partial_rank(x[keep], y[keep], z[keep])})
    raw = np.array([r['rho_raw'] for r in rows])
    par = np.array([r['rho_partial_logparams'] for r in rows])

    def summary(values):
        return {'min': float(values.min()), 'max': float(values.max()),
                'min_abs': float(np.abs(values).min()), 'max_abs': float(np.abs(values).max()),
                'all_negative': bool((values < 0).all()),
                'all_abs_ge_0.4': bool((np.abs(values) >= .4).all())}

    sr, sp = summary(raw), summary(par)
    finite = null[np.isfinite(null)]
    exceed_two = int(np.sum(np.abs(finite) >= abs(observed_raw)))
    p_two = (1 + exceed_two) / (len(finite) + 1)
    p_neg = float((1 + np.count_nonzero(null <= observed_raw)) / (len(null) + 1))
    permutation = {'n_perm': int(indices.shape[0]), 'p_two_sided': p_two, 'p_one_sided_neg': p_neg,
                   'null_mean': float(null.mean()), 'null_sd': float(null.std(ddof=1)),
                   'null_abs_max': float(np.abs(null).max()), 'exceed_count_two_sided': exceed_two,
                   'n_finite_permutations': int(finite.size), 'n_nonfinite_permutations': int(null.size - finite.size)}
    verdict = {'raw_survives': sr['all_negative'] and sr['all_abs_ge_0.4'] and p_two < .05,
               'partial_logparams_survives': sp['all_negative'] and sp['all_abs_ge_0.4'] and p_two < .05,
               'sign_ok_raw': sr['all_negative'], 'mag_ok_raw': sr['all_abs_ge_0.4'],
               'sign_ok_partial': sp['all_negative'], 'mag_ok_partial': sp['all_abs_ge_0.4'],
               'perm_ok': p_two < .05}
    summary_out = {
        'schema': SCHEMA + '/lofo', 'branch': 'lofo', 'status': 'CACHED_BASELINE_REPLAY_PRODUCED',
        'replay_level': 'CACHED_BASELINE_REPLAY (no theta refit; deletion sensitivity recomputed from saved measurements)',
        'n_models': len(tags), 'n_families': len(families), 'families': families,
        'seed': int(old['seed']), 'n_perm': int(indices.shape[0]),
        'observed': {'raw_rho': observed_raw, 'partial_rho_logparams': observed_par,
                     'matches_convergence_verdict_science_-0.606614':
                         bool(abs(observed_raw + .6066143159166415) < 1e-6)},
        'lofo': rows, 'lofo_raw_summary': sr, 'lofo_partial_logparams_summary': sp,
        'permutation': permutation, 'verdict': verdict,
        'scientific_limits': receipt['scientific_limits'],
        'null_array_regenerated_from_saved_indices': recalc.tolist(),
    }

    # comparator: every pointer/field of the accepted producer summary
    matched = total = 0
    for field in ('n_models', 'n_families', 'families', 'seed', 'n_perm'):
        total += 1
        matched += rp.compare(branch, f'/{field}', summary_out[field], old[field],
                              'R/runs/study2_reuse_v1/lofo/results.json')
    for field in ('raw_rho', 'partial_rho_logparams'):
        total += 1
        matched += rp.compare(branch, f'/observed/{field}', summary_out['observed'][field],
                              old['observed'][field], 'R/runs/study2_reuse_v1/lofo/results.json')
    for i, row in enumerate(rows):
        for field in ('family_dropped', 'n_dropped', 'n_remaining', 'rho_raw', 'rho_partial_logparams'):
            total += 1
            matched += rp.compare(branch, f'/lofo/{i}/{field}', row[field], old['lofo'][i][field],
                                  'R/runs/study2_reuse_v1/lofo/results.json',
                                  exact=field in ('family_dropped', 'n_dropped', 'n_remaining'))
    for name, s in (('lofo_raw_summary', sr), ('lofo_partial_logparams_summary', sp)):
        for field in ('min', 'max', 'min_abs', 'max_abs', 'all_negative', 'all_abs_ge_0.4'):
            total += 1
            matched += rp.compare(branch, f'/{name}/{field}', s[field], old[name][field],
                                  'R/runs/study2_reuse_v1/lofo/results.json',
                                  exact=field in ('all_negative', 'all_abs_ge_0.4'))
    for field in ('n_perm', 'p_two_sided', 'p_one_sided_neg', 'null_mean', 'null_sd', 'null_abs_max'):
        total += 1
        matched += rp.compare(branch, f'/permutation/{field}', permutation[field], old['permutation'][field],
                              'R/runs/study2_reuse_v1/lofo/results.json', exact=(field == 'n_perm'))
    for field in ('raw_survives', 'partial_logparams_survives', 'sign_ok_raw', 'mag_ok_raw',
                  'sign_ok_partial', 'mag_ok_partial', 'perm_ok'):
        total += 1
        matched += rp.compare(branch, f'/verdict/{field}', verdict[field], old['verdict'][field],
                              'R/runs/study2_reuse_v1/lofo/results.json', exact=True)

    legacy = retained_study2_legacy(rp)
    audit = ledger_audit(rp, legacy)
    return {'branch': branch, 'status': summary_out['status'],
            'artifacts': {'lofo_baseline.json': summary_out,
                          'retained_study2_legacy.json': legacy,
                          'c9_ledger_coverage.json': audit},
            'summary': {'comparator_matched': matched, 'comparator_total': total,
                        'checks_passed': sum(1 for c in rp.checks if c['branch'] == branch and c['passed']),
                        'checks_total': sum(1 for c in rp.checks if c['branch'] == branch),
                        'retained_legacy_cells': len(legacy['cells']),
                        'ledger_records_audited': audit['record_count']}}


def retained_study2_legacy(rp):
    """Bounded aggregation of retained (non-new28) study2 inference arrays."""
    R = rp.R
    branch = 'lofo'
    inf = R / 'runs/study2_reuse_v1/inference'
    selection = rp.load_json(inf / 'U2A_SELECTION.json', 'retained_legacy_selection', 'primary_cache_input')
    cells = []
    targets = [tuple(t) for t in selection['targets']] + [('eff_rank_pr', 'medical')]
    for feature, domain in targets:
        key = f'u2a__{domain}__{feature}'
        cell_path = inf / 'cells' / f'{key}.json'
        draw_path = inf / 'draws' / f'{key}.npz'
        cell = rp.load_json(cell_path, 'retained_legacy_cell', 'primary_cache_input')
        if cell['status'].startswith('BLOCKED') or not draw_path.exists():
            cells.append({'key': key, 'domain': domain, 'feature': feature, 'kind': 'U2A',
                          'status': cell['status'], 'replayed': False,
                          'reason': 'saved cell is blocked (no persisted draws); retained as blocked, not replayed'})
            rp.gap(branch, key, f"retained cell not replayable: status={cell['status']} "
                                f"(expected for the medical NaN-ranking legacy defect)")
            continue
        z = rp.load_npz(draw_path, 'retained_legacy_draws', 'primary_cache_input')
        inp = rp.load_json(inf / 'inputs' / f'{domain}.json', 'retained_legacy_inputs', 'secondary_linkage_input')
        tags = [str(t) for t in inp['tags']]
        acc = np.asarray(inp['accuracy'], float)
        x = z['x'].astype(float)
        y = z['theta'].astype(float)
        c = z['controls'].astype(float)
        fam = z['families']
        rp.check(branch, f'{key}_array_identity',
                 len(tags) == len(x) == len(y) == len(acc) and len(set(fam.tolist())) == cell['n_families']
                 and len(y) == cell['n_models'],
                 detail='saved draw arrays align with saved inputs and cell counts', exact=True)
        row = {'key': key, 'domain': domain, 'feature': feature, 'kind': 'U2A',
               'n_models': int(len(y)), 'n_families': int(len(set(fam.tolist()))),
               'raw_rho': rho(x, y), 'spearman_theta_accuracy': rho(y, acc),
               'passes_gate': None}
        rp.compare(branch, f'/retained_legacy/{key}/raw_rho', row['raw_rho'], cell['raw_rho'],
                   'R/runs/study2_reuse_v1/inference/cells/' + key + '.json')
        rp.compare(branch, f'/retained_legacy/{key}/spearman_theta_accuracy',
                   row['spearman_theta_accuracy'], cell['spearman_theta_accuracy'],
                   'R/runs/study2_reuse_v1/inference/cells/' + key + '.json')
        for name, controls in (('acc', acc), ('acc_plus_C', np.column_stack([acc, c]))):
            point = rho(x, y, controls)
            boot = ci_from_draws(z[name + '_bootstrap'])
            boot['rho'] = point
            perm = perm_from_draws(point, z[name + '_permutation'])
            entry = {'partial_rho': point, 'bootstrap': boot, 'permutation': perm}
            row[name] = entry
            ref = cell[name]
            rp.compare(branch, f'/retained_legacy/{key}/{name}/partial_rho', point, ref['partial_rho'],
                       'R/runs/study2_reuse_v1/inference/cells/' + key + '.json')
            for field in ('rho', 'ci', 'n_valid', 'n_invalid', 'ci_excludes_zero'):
                rp.compare(branch, f'/retained_legacy/{key}/{name}/bootstrap/{field}',
                           boot[field], ref['bootstrap'][field],
                           'R/runs/study2_reuse_v1/inference/cells/' + key + '.json',
                           exact=field in ('n_valid', 'n_invalid', 'ci_excludes_zero'))
            for field in ('p', 'exceed_count', 'n_valid', 'n_invalid'):
                rp.compare(branch, f'/retained_legacy/{key}/{name}/permutation/{field}',
                           perm[field], ref['permutation'][field],
                           'R/runs/study2_reuse_v1/inference/cells/' + key + '.json',
                           exact=field in ('exceed_count', 'n_valid', 'n_invalid'))
        row['passes_gate'] = row['acc']['bootstrap']['ci_excludes_zero']
        rp.compare(branch, f'/retained_legacy/{key}/passes_gate', row['passes_gate'], cell['passes_gate'],
                   'R/runs/study2_reuse_v1/inference/cells/' + key + '.json', exact=True)
        cells.append(row)

    # H3 / DROP3 geometry cells (retained: P2-C-004; H3 geometry also underlies P2-C-005 sign diagnostics)
    geom = {}
    cell_specs = (('h3__code__eff_rank_pr', 'code', 'code'), ('h3__math__eff_rank_pr', 'math', 'math'),
                  ('h3__science__eff_rank_pr', 'science', 'science'),
                  ('h3__medical__eff_rank_pr', 'medical', 'medical'),
                  ('drop3__science__eff_rank_pr', 'science_drop3', 'science_drop3'))
    item_counts = {'science': 448, 'medical': 1273, 'code': 591, 'math': 1333, 'science_drop3': 448}
    for key, domain, geom_name in cell_specs:
        cell = rp.load_json(inf / 'cells' / f'{key}.json', 'retained_legacy_cell', 'primary_cache_input')
        if cell['status'].startswith('BLOCKED') or not (inf / 'draws' / f'{key}.npz').exists():
            cells.append({'key': key, 'domain': domain, 'kind': cell['kind'], 'status': cell['status'],
                          'replayed': False,
                          'reason': 'saved cell is blocked (no persisted draws); retained as blocked, not replayed'})
            rp.gap(branch, key, f"retained cell not replayable: status={cell['status']} "
                                f"(expected for the medical NaN-ranking legacy defect)")
            continue
        z = rp.load_npz(inf / 'draws' / f'{key}.npz', 'retained_legacy_draws', 'primary_cache_input')
        inp_path = inf / 'inputs' / f'{domain}.json'
        inp = rp.load_json(inp_path, 'retained_legacy_inputs', 'secondary_linkage_input')
        x = z['x'].astype(float)
        y = z['theta'].astype(float)
        c = z['controls'].astype(float)
        fam = z['families']
        exact = (len(y) == cell['n_models'] and len(set(fam.tolist())) == cell['n_families']
                 and len(inp['theta']) == len(y) and bool(np.allclose(np.asarray(inp['theta'], float), y, rtol=0, atol=1e-12)))
        rp.check(branch, f'{key}_array_identity', exact,
                 detail='saved draws align with retained inputs theta and cell counts', exact=True)
        labels = {fam_i: i for i, fam_i in enumerate(sorted(set(fam.tolist())))}
        bare_controls = np.array([labels[v] for v in fam.tolist()], float)
        point_bare = rho(x, y, bare_controls)
        point_ctl = rho(x, y, c)
        bare_boot = ci_from_draws(z['bare_bootstrap'])
        bare_boot['rho'] = point_bare
        ctl_boot = ci_from_draws(z['controlled_bootstrap'])
        ctl_boot['rho'] = point_ctl
        entry = {'key': key, 'domain': domain, 'kind': cell['kind'],
                 'raw_rho': rho(x, y), 'bare_rho': point_bare, 'partial_rho': point_ctl,
                 'bare_bootstrap': bare_boot,
                 'controlled_bootstrap': ctl_boot,
                 'controlled_ci': ctl_boot['ci'],
                 'permutation': perm_from_draws(point_ctl, z['permutation']),
                 'rel_geo': float(cell['rel_geo']), 'rel_th': float(cell['rel_th'])}
        entry['disattenuation_ceiling'] = math.sqrt(entry['rel_geo'] * entry['rel_th'])
        entry['verdict'] = historical_verdict(entry['bare_bootstrap']['ci_excludes_zero'],
                                              entry['controlled_bootstrap']['ci_excludes_zero'],
                                              entry['permutation']['p'], len(y), item_counts[domain])
        for field, value in (('raw_rho', entry['raw_rho']), ('partial_rho', entry['partial_rho']),
                             ('bare_bootstrap', entry['bare_bootstrap']),
                             ('controlled_bootstrap', entry['controlled_bootstrap']),
                             ('controlled_ci', entry['controlled_ci']),
                             ('permutation_p', entry['permutation']['p']),
                             ('rel_geo', entry['rel_geo']), ('rel_th', entry['rel_th']),
                             ('disattenuation_ceiling', entry['disattenuation_ceiling']),
                             ('verdict', entry['verdict'])):
            rp.compare(branch, f'/retained_legacy/{key}/{field}', value, cell[field],
                       'R/runs/study2_reuse_v1/inference/cells/' + key + '.json',
                       exact=(field == 'verdict'))
        geom[key] = entry
        cells.append(entry)
        if key.endswith('__eff_rank_pr'):
            h = rp.load_json(inf / 'h3_geometry' / f'{geom_name}.json', 'h3_geometry_rows', 'primary_cache_input')
            a = np.array([row['a'] for row in h['rows']])
            b = np.array([row['b'] for row in h['rows']])
            fresh = float(np.clip(2 * rho(a, b) / (1 + rho(a, b)), 0, 1))
            rp.compare(branch, f'/retained_legacy/{key}/rel_geo_from_split_halves', fresh, cell['rel_geo'],
                       'R/runs/study2_reuse_v1/inference/h3_geometry/' + geom_name + '.json')

    # P2-C-005 sign reconstruction from saved arrays
    h = rp.load_json(inf / 'h3_geometry/science.json', 'h3_geometry_rows', 'primary_cache_input')
    inp = rp.load_json(inf / 'inputs/science.json', 'retained_legacy_inputs', 'secondary_linkage_input')
    pr = np.array([row['pr'] for row in h['rows']])
    th = np.asarray(inp['theta'], float)
    usable = np.asarray(inp['usable_n'], float)
    c = np.asarray(inp['C'], float)
    sign = {'spearman_PR_theta': rho(pr, th), 'spearman_PR_usable_n': rho(pr, usable),
            'spearman_theta_usable_n': rho(th, usable)}
    parts = {'partial_PR_theta_4cov': rho(pr, th, c),
             'partial_PR_theta_4cov_plus_usable_n': rho(pr, th, np.column_stack([c, usable])),
             'partial_PR_theta_usable_n_only': rho(pr, th, usable)}
    ref = rp.load_json(inf / 'SCI_SIGN_RECONSTRUCTED.json', 'accepted_producer_summary', 'comparator_only')
    for field in sign:
        rp.compare(branch, f'/retained_legacy/sign/{field}', sign[field], ref['sign'][field],
                   'R/runs/study2_reuse_v1/inference/SCI_SIGN_RECONSTRUCTED.json')
    for field in parts:
        rp.compare(branch, f'/retained_legacy/sign/{field}', parts[field], ref['usable_n_partial'][field],
                   'R/runs/study2_reuse_v1/inference/SCI_SIGN_RECONSTRUCTED.json')

    return {'schema': SCHEMA + '/retained_study2_legacy', 'status': 'RETAINED_LEGACY_STATS_AGGREGATED',
            'replay_level': 'CACHED_BASELINE_REPLAY (bounded aggregation of saved per-cell draws; no refit)',
            'replaced_grid_policy': '28-grid cells and medical 28-grid associations are replaced by the new28 node '
                                    '(other worker); only retained (non-replaced) estimands are aggregated here.',
            'cells': cells, 'sign_diagnostics': {'sign': sign, 'usable_n_partial': parts},
            'sign_scope': ref['scope'],
            'notes': ['P2-C-004 DROP3 and P2-C-005 usable-n sign diagnostics recomputed from saved h3 geometry rows and saved inputs.',
                      'D-family accuracy-conditioned cells recomputed from saved per-cell draws including BH-free p-values; '
                      'the historical U2A selection remains diagnostic-only (current BH grid unresolved).']}


def ledger_audit(rp, legacy):
    R = rp.R
    ledger = rp.load_json(R / 'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json', 'ledger_context', 'comparator_only')
    rows = []
    for rec in ledger['records']:
        fam = rec['family']
        if fam not in ('B', 'C', 'D'):
            continue
        rid = rec['result_id']
        if fam == 'D':
            if rid.startswith('P2-D-ACC'):
                owner, coverage = 'C10b lofo retained', 'AGGREGATED_FROM_SAVED_DRAWS'
            elif rid == 'P2-D-001':
                owner, coverage = 'C10b lofo retained', 'AGGREGATED_FROM_SAVED_DRAWS (gate aggregate over u2a cells)'
            else:
                owner, coverage = 'C10b lofo retained', \
                    'AUDITED_PARTIAL (original 50-model diagnostics limited to saved arrays; no silent discard)'
        elif fam == 'B':
            if rid == 'P2-B-010':
                owner, coverage = 'C7 (other worker)', 'REPLACED_BY_C17'
            elif rid == 'P2-B-006':
                owner, coverage = 'C10b bos', 'REPLAYED_FRESH (14 saved-step statistics + sequential differences)'
            elif rid in ('P2-B-007-MATH', 'P2-B-007-CODE'):
                owner, coverage = 'C10b retention', 'REPLAYED_FRESH (40 disjoint-split cells per domain + seed0 draws)'
            elif rid == 'P2-B-003':
                owner, coverage = 'C10b study1', 'REPLAYED (composite members + PR-only member partials)'
            else:
                owner, coverage = 'C10b study1', 'REPLAYED_FRESH'
        elif rid.startswith('P2-C-LOFO'):
            owner, coverage = 'C10b lofo', 'REPLAYED_FRESH'
        elif rid == 'P2-C-004':
            owner, coverage = 'C10b lofo retained', 'AGGREGATED_FROM_SAVED_DRAWS'
        elif rid == 'P2-C-005':
            owner, coverage = 'C10b lofo retained', 'RECOMPUTED_FROM_SAVED_ARRAYS'
        elif rid.startswith('P2-C-GRID'):
            owner, coverage = 'C2 new28 (other worker)', 'REPLACED_BY_C17'
        elif rid.startswith('P2-C-NBIAS'):
            owner, coverage = 'C6 simulations (other worker)', 'RETAINED_OTHER_WORKER'
        elif rid in ('P2-C-002',):
            owner, coverage = 'C2 new28 (other worker)', 'REPLACED_BY_C17'
        elif rid.startswith('P2-C-TAB'):
            owner, coverage = 'C10 render', 'PENDING_RENDER'
        elif rid == 'P2-C-001':
            owner, coverage = 'C10b lofo retained', 'PARTIAL (observed science PR rho recomputed; full-panel PR chain not in C10b scope)'
        elif rid == 'P2-C-003':
            owner, coverage = 'not C10b', 'AUDITED_NO_C10B_BRANCH (code shadow-only recurrence chain requires study2 code feature panel)'
        else:
            owner, coverage = 'UNMAPPED', 'NEEDS_REVIEW'
        rows.append({'result_id': rid, 'family': fam, 'title': rec['title'],
                     'c17_disposition': rec['c17_disposition'], 'owner': owner, 'coverage': coverage})
    unmapped = [r['result_id'] for r in rows if r['owner'] == 'UNMAPPED']
    rp.check('lofo', 'c9_ledger_coverage_complete', not unmapped,
             detail=f'{len(rows)} B/C/D ledger records mapped; unmapped={unmapped}', exact=True)
    owners = {}
    for r in rows:
        owners[r['owner']] = owners.get(r['owner'], 0) + 1
    return {'schema': SCHEMA + '/c9_ledger_coverage', 'ledger_artifact': 'R/reports/C17_FINAL_EVIDENCE_LEDGER_v1.json',
            'record_count': len(rows), 'owner_counts': owners, 'unmapped': unmapped, 'records': rows,
            'retained_legacy_cells_replayed': len(legacy['cells'])}


# --------------------------------------------------------------------------
# branch: empirical AN-A residual repair
# --------------------------------------------------------------------------
def spectra_from(y, prob):
    rr = (y - prob) / np.sqrt(np.maximum(prob * (1 - prob), 1e-12))
    out = {}
    for mode, a in (('capped', np.clip(rr, -4, 4)), ('uncapped', rr)):
        a = a - a.mean(1, keepdims=True)
        ev = np.linalg.eigvalsh(a @ a.T / (a.shape[1] - 1))[::-1]
        out[mode + '_eigenvalues'] = ev
        out[mode + '_share'] = float(ev[0] / ev.sum())
    return out


def partial_early(x, y, c):
    if any(np.unique(v).size < 2 for v in (x, y, c)):
        return np.nan, 'constant_input'
    rx, ry, rc = [rankdata(v, method='average') for v in (x, y, c)]
    d = np.column_stack([np.ones(len(x)), rc])
    rank = np.linalg.matrix_rank(d)
    if rank < 2:
        return np.nan, 'rank_deficient_design'
    if any(np.linalg.matrix_rank(np.column_stack([d, v])) == rank for v in (rx, ry)):
        return np.nan, 'zero_residual_rank'
    ex = rx - d @ np.linalg.lstsq(d, rx, rcond=None)[0]
    ey = ry - d @ np.linalg.lstsq(d, ry, rcond=None)[0]
    ex, ey = ex - ex.mean(), ey - ey.mean()
    sx, sy = np.linalg.norm(ex), np.linalg.norm(ey)
    if min(sx, sy) <= 100 * np.finfo(float).eps * max(d.shape) * max(
            np.linalg.norm(rx), np.linalg.norm(ry), 1):
        return np.nan, 'zero_residual_norm'
    return float(ex @ ey / (sx * sy)), 'valid'


def branch_empirical(rp):
    branch = 'empirical'
    R = rp.R
    base = R / 'runs/new_empirical_repair'
    old = rp.load_json(base / 'report.json', 'accepted_producer_summary', 'comparator_only')
    rp.register(R / 'code/new_empirical_repair.py', 'source_code_reference', 'reference_code_only')
    early_inputs = rp.load_npz(R / 'runs/empirical_repair_feasibility/early16_inputs.npz',
                               'early16_inputs', 'primary_cache_input')
    early_saved = rp.load_npz(base / 'early16_training_sensitivity.npz',
                              'early16_sensitivity_draws', 'primary_cache_input')
    old_early = old['early16']['training_sensitivity']

    early_models = [str(m) for m in early_inputs['models']]
    fam = np.asarray([str(v) for v in early_saved['families']])
    order = sorted(set(fam.tolist()))
    blocks = [np.flatnonzero(fam == v) for v in order]
    rp.check(branch, 'early16_family_identity',
             order == old_early['family_order'] and [len(b) for b in blocks] == old_early['block_sizes'],
             detail='7 training families, block sizes identical to accepted producer', exact=True)
    pc1v = early_inputs['pc1'].astype(float)
    accv = early_inputs['accuracy'].astype(float)
    lp = early_inputs['logprob'].astype(float)
    point, status = partial_early(pc1v, accv, lp)
    stats = early_saved['statistics'].astype(float)
    good = np.isfinite(stats)
    ci = np.percentile(stats[good], [2.5, 97.5]).tolist() if good.any() else [None, None]
    rp.compare(branch, '/early16/point', point, old_early['point'],
               'R/runs/new_empirical_repair/report.json')
    rp.compare(branch, '/early16/ci95', ci, old_early['ci95'],
               'R/runs/new_empirical_repair/report.json')
    rp.compare(branch, '/early16/valid', int(good.sum()), old_early['valid'],
               'R/runs/new_empirical_repair/report.json', exact=True)
    from collections import Counter
    statuses = Counter(str(s) for s in early_saved['status'])
    rp.compare(branch, '/early16/status_counts', dict(sorted(statuses.items())),
               dict(sorted(old_early['status_counts'].items())), 'R/runs/new_empirical_repair/report.json', exact=True)
    # representative regeneration of saved sensitivity rows
    regen = []
    for b in (0, 1, 4321, 9999):
        idx = np.concatenate([blocks[k] for k in early_saved['family_choices'][b]])
        padded = np.full(early_saved['model_indices_padded'].shape[1], -1, dtype=np.int16)
        padded[:len(idx)] = idx
        padded_ok = bool(np.array_equal(padded, early_saved['model_indices_padded'][b])
                         and int(early_saved['lengths'][b]) == len(idx))
        value, error = partial_early(pc1v[idx], accv[idx], lp[idx])
        regen.append({'row': b, 'model_indices_padded_exact': padded_ok, 'status': error,
                      'fresh': value, 'saved': float(stats[b]),
                      'abs_difference': abs(value - float(stats[b])) if np.isfinite(value) else None})
    rp.check(branch, 'early16_representative_regeneration',
             all(r['model_indices_padded_exact'] for r in regen)
             and all(r['abs_difference'] is not None and r['abs_difference'] <= 1e-12 for r in regen),
             detail='4 saved rows regenerated from family choices + saved inputs; padded index arrays rebuilt exactly',
             max_abs_difference=max(r['abs_difference'] for r in regen if r['abs_difference'] is not None))

    domains = {}
    matched = total = 0
    for dom in ('code', 'math'):
        d = base / dom
        inputs = rp.load_npz(d / 'inputs.npz', 'empirical_inputs', 'primary_cache_input')
        observed = rp.load_npz(d / 'observed.npz', 'empirical_observed_fit', 'primary_cache_input')
        diag = rp.load_json(d / 'observed_diagnostics.json', 'empirical_diagnostics', 'primary_cache_input')
        Y = inputs['responses'].astype(float)
        raw = inputs['raw_responses'].astype(float)
        mask = inputs['retained_mask'].astype(bool)
        n_models = int(inputs['models'].shape[0])
        rp.check(branch, f'{dom}_retained_mask_identity',
                 bool(np.array_equal(mask, (raw.sum(0) > 0) & (raw.sum(0) < n_models)))
                 and int(mask.sum()) == Y.shape[1] and list(Y.shape) == old['residual'][dom]['retained_shape'],
                 detail='saved retained mask equals recomputed item filter; retained shape matches producer',
                 exact=True)
        theta = observed['theta'].astype(float)
        beta = observed['beta'].astype(float)
        prob = observed['probability'].astype(float)
        prob_identity = float(np.max(np.abs(prob - 1 / (1 + np.exp(-(theta[:, None] + beta[None, :]))))))
        rp.check(branch, f'{dom}_probability_identity', prob_identity <= 1e-12,
                 detail='saved probability array equals expit(theta+beta) at 1e-12',
                 pointer=f'/{dom}/probability', max_abs_difference=prob_identity, tolerance=1e-12)
        row_sums = np.max(np.abs((Y - prob).sum(1)))
        rp.check(branch, f'{dom}_marginal_score_check',
                 row_sums <= float(diag['score_max']) + 1e-9,
                 detail='max |person marginal score| recomputed from response/probability arrays is within the accepted score screen',
                 max_abs_difference=row_sums, tolerance=float(diag['score_max']))
        sp = spectra_from(Y, prob)
        for mode in ('capped', 'uncapped'):
            rp.compare(branch, f'/{dom}/observed/{mode}_share', sp[mode + '_share'],
                       observed[mode + '_share'], 'R/runs/new_empirical_repair/' + dom + '/observed.npz')
            dmax = float(np.max(np.abs(sp[mode + '_eigenvalues'] - observed[mode + '_eigenvalues'])))
            rp.check(branch, f'{dom}_{mode}_spectrum_regeneration',
                     dmax <= 1e-9 + 1e-6 * float(np.max(np.abs(observed[mode + '_eigenvalues']))),
                     detail='spectra recomputed from saved responses and probabilities',
                     max_abs_difference=dmax)
        draw_file = rp.load_npz(d / 'null_draws.npz', 'empirical_null_draws', 'primary_cache_input')
        draws = draw_file['responses']
        diagnostics = rp.load_json(d / 'null_diagnostics.json', 'empirical_null_diagnostics', 'primary_cache_input')
        shares = {'capped': [], 'uncapped': []}
        invalid = []
        representative = {0, 1, 2, 50, 100, 150, 199}
        spec_delta = 0.0
        mask_delta = 0
        for b in range(200):
            z = rp.load_npz(d / f'null_fit_{b:03}.npz', 'empirical_null_fit', 'primary_cache_input')
            fresh_mask = (draws[b].sum(0) > 0) & (draws[b].sum(0) < n_models)
            if not np.array_equal(z['retained_mask'].astype(bool), fresh_mask):
                mask_delta += 1
            if 'probability' in z.files:
                shares['capped'].append(float(z['capped_share']))
                shares['uncapped'].append(float(z['uncapped_share']))
                if b in representative:
                    p2 = 1 / (1 + np.exp(-(z['theta'][:, None] + z['beta'][None, :])))
                    spec_delta = max(spec_delta, float(np.max(np.abs(p2 - z['probability']))))
                    fresh = spectra_from(draws[b][:, fresh_mask].astype(float), z['probability'])
                    for mode in ('capped', 'uncapped'):
                        spec_delta = max(spec_delta, abs(fresh[mode + '_share'] - float(z[mode + '_share'])))
            else:
                invalid.append(b)
        rp.check(branch, f'{dom}_null_mask_linkage', mask_delta == 0,
                 detail='per-draw retained masks regenerated from null_draws exactly for all 200 draws', exact=True)
        rp.check(branch, f'{dom}_null_spectrum_representative',
                 spec_delta <= 1e-9 + 1e-6,
                 detail='7 representative null fits: probability identity and spectra regenerated from saved arrays',
                 max_abs_difference=spec_delta, tolerance=1e-9 + 1e-6)
        rp.check(branch, f'{dom}_null_invalid_set',
                 invalid == [r['draw'] for r in diagnostics if not r['valid']],
                 detail=f'{len(invalid)} fits without persisted probability arrays exactly match invalid diagnostics',
                 exact=True)
        stats_out = {}
        for mode in ('capped', 'uncapped'):
            a = np.array(shares[mode])
            obs_share = float(sp[mode + '_share'])
            exc = int(np.sum(a >= obs_share))
            fail = 200 - len(a)
            stats_out[mode] = {
                'observed_leading_share': obs_share,
                'valid_null_percentiles_2_5_50_95_97_5': np.percentile(a, [2.5, 50, 95, 97.5]).tolist() if len(a) else None,
                'exceed_count': exc, 'upper_tail_p': (1 + exc) / 201 if fail == 0 else None,
                'p_bounds': [(1 + exc) / 201, (1 + exc + fail) / 201],
                'n_valid_null_fits': int(len(a)), 'n_failed_null_fits': int(fail),
                'interpretation': 'conditional on successful fits' if fail else 'all prespecified draws retained'}
            ref = old['residual'][dom]['statistics'][mode]
            for field in ('observed_leading_share', 'valid_null_percentiles_2_5_50_95_97_5',
                          'exceed_count', 'upper_tail_p', 'p_bounds'):
                total += 1
                matched += rp.compare(branch, f'/{dom}/statistics/{mode}/{field}', stats_out[mode][field],
                                      ref[field], 'R/runs/new_empirical_repair/report.json',
                                      exact=field == 'exceed_count')
            total += 1
            matched += rp.compare(branch, f'/{dom}/statistics/{mode}/n_valid_null_fits',
                                  stats_out[mode]['n_valid_null_fits'], old['residual'][dom]['null_valid'],
                                  'R/runs/new_empirical_repair/report.json', exact=True)
        domains[dom] = {'status': 'CACHED_BASELINE_REPLAY', 'input_shape': raw.shape,
                        'retained_shape': Y.shape, 'null_valid': int(old['residual'][dom]['null_valid']),
                        'observed_fit_valid': bool(diag['valid']),
                        'observed_diagnostics_recomputed': {'marginal_score_max': float(row_sums),
                                                            'probability_identity_max_diff': prob_identity},
                        'statistics': stats_out}
        total += 1
        matched += rp.compare(branch, f'/{dom}/input_shape', list(raw.shape),
                              old['residual'][dom]['input_shape'], 'R/runs/new_empirical_repair/report.json', exact=True)
        total += 1
        matched += rp.compare(branch, f'/{dom}/retained_shape', list(Y.shape),
                              old['residual'][dom]['retained_shape'], 'R/runs/new_empirical_repair/report.json', exact=True)

    summary = {
        'schema': SCHEMA + '/empirical', 'branch': 'empirical', 'status': 'CACHED_BASELINE_REPLAY_PRODUCED',
        'replay_level': 'CACHED_BASELINE_REPLAY (no IRT refit; 200+200 saved null fits aggregated, 400 IRT fits not repeated)',
        'analysis_type': 'NEW_ANALYSIS_CACHE_REPLAY', 'domains': domains,
        'early16': {'point': point, 'point_status': status, 'ci95': ci, 'valid': int(good.sum()), 'B': int(len(stats)),
                    'family_order': order, 'block_sizes': [len(b) for b in blocks],
                    'status_counts': dict(sorted(statuses.items())),
                    'representative_regeneration': regen},
        'early16_scope': 'primary reused from runs/empirical_repair_feasibility_v2 probe as discovery input; '
                         'training-family sensitivity recomputed from saved inputs and saved per-draw statistics',
        'limitations': ['Case resampling over 7 training families only; CI is a training-sensitivity diagnostic.',
                        'Residual spectra are conditional on successful fits; failure counts are retained.',
                        'No construct-validity or geometry-inference release follows from this cache replay.'],
    }
    for mode in ('capped', 'uncapped'):
        total += 1
        matched += rp.compare(branch, f'/early16/{mode}_observed_leading_share',
                              domains['code']['statistics'][mode]['observed_leading_share'],
                              old['residual']['code']['statistics'][mode]['observed_leading_share'],
                              'R/runs/new_empirical_repair/report.json')
    return {'branch': branch, 'status': summary['status'],
            'artifacts': {'empirical_baseline.json': summary},
            'summary': {'comparator_matched': matched, 'comparator_total': total,
                        'checks_passed': sum(1 for c in rp.checks if c['branch'] == branch and c['passed']),
                        'checks_total': sum(1 for c in rp.checks if c['branch'] == branch)}}


# --------------------------------------------------------------------------
# branch: medical regularized MAP/Rasch cache replay
# --------------------------------------------------------------------------
def medical_grid(q):
    from scipy.special import roots_legendre
    t, w = roots_legendre(q)
    t, w = t * 6, w * 6
    lw = np.log(w) - t * t / 2 - .5 * np.log(2 * np.pi)
    return t, lw


def medical_recompute(Y, a, b, q):
    from scipy.special import expit
    t, lw = medical_grid(q)
    lin = a[:, None] * (t[None, :] - b[:, None])
    lp = -np.logaddexp(0, -lin)
    ln = -np.logaddexp(0, lin)
    ll = Y.T @ lp + (1 - Y).T @ ln
    from scipy.special import logsumexp
    norm = logsumexp(ll + lw, axis=1)
    post = np.exp(ll + lw - norm[:, None])
    theta = post @ t
    var = post @ (t * t) - theta ** 2
    err = Y @ post - expit(lin) * post.sum(0)[None, :]
    gb = a * err.sum(1)
    ga = -a * np.sum(err * (t[None, :] - b[:, None]), axis=1)
    expected = None
    return {'logML': norm, 'theta': theta, 'var': var, 'posterior': post, 'gb': gb, 'ga': ga,
            'a': a, 'b': b, 'q': q}


def projected_gradient(g, z, bounds):
    g = np.array(g, float)
    for j, (lo, hi) in enumerate(bounds):
        if z[j] <= lo + 1e-7 and g[j] > 0:
            g[j] = 0.0
        if z[j] >= hi - 1e-7 and g[j] < 0:
            g[j] = 0.0
    return g


def branch_medical(rp):
    branch = 'medical'
    R = rp.R
    provenance = {}
    free_failures = []
    matched = total = 0
    runs = [('runs/new_medical_repair', 'config/CT_MEDICAL_REPAIR.json', 'first_repair'),
            ('runs/new_medical_repair_refined', 'config/CT_MEDICAL_REFINEMENT.json', 'refined_repair')]
    for rel_dir, cfg_rel, tag in runs:
        rundir = R / rel_dir
        cfg_path = R / cfg_rel
        cfg = rp.load_json(cfg_path, 'medical_fit_contract', 'frozen_rule_config')
        declared = {}
        for name, info in cfg['inputs'].items():
            for key, sha in info['hashes'].items():
                declared[(R / info[key]).resolve()] = sha
        rp.register(R / ('code/new_medical_repair_refined.py' if 'refined' in rel_dir else 'code/new_medical_repair.py'),
                    'source_code_reference', 'reference_code_only')
        variants = {}
        for npz_path in sorted(rundir.glob('medical_*.npz')):
            label = npz_path.stem
            report_path = rundir / f'{label}.json'
            report = rp.load_json(report_path, 'medical_fit_report', 'comparator_only')
            z = rp.load_npz(npz_path, 'medical_fit_arrays', 'primary_cache_input', report['arrays_sha256'])
            Y = z['Y'].astype(float)
            a = z['discrimination'].astype(float)
            b = z['difficulty'].astype(float)
            saved_theta = z['theta'].astype(float)
            saved_var = z['variance'].astype(float)
            saved_logml = z['logML'].astype(float)
            post = z['posterior'].astype(float)
            q = int(post.shape[1])
            rec = medical_recompute(Y, a, b, q)
            from scipy.special import expit
            t_grid, _ = medical_grid(q)
            expected = expit(a[:, None] * (t_grid[None, :] - b[:, None])) @ post.T
            score_gap = float(np.max(np.abs((Y - expected).sum(0))))
            rows = [np.max(np.abs(post.sum(1) - 1)), float(np.max(np.abs(rec['theta'] - saved_theta))),
                    float(np.max(np.abs(rec['var'] - saved_var))),
                    float(np.max(np.abs(rec['logML'] - saved_logml)))]
            grad = rec['gb'] if report['kind'] == 'rasch' else np.r_[rec['ga'], rec['gb']]
            if report['kind'] == 'map':
                params = z['parameters'].astype(float)
                grad = grad.copy()
                grad[:len(a)] += params[:len(a)] / .25
                grad[len(a):] += params[len(a):] / 9
                objective = -float(rec['logML'].sum()) + .5 * float(np.sum((params[:len(a)] / .5) ** 2)) \
                    + .5 * float(np.sum((params[len(a):] / 3) ** 2))
            else:
                objective = -float(rec['logML'].sum())
            grad_delta = float(np.max(np.abs(grad - z['gradient'].astype(float))))
            rp.check(branch, f'{tag}_{label}_posterior_normalization', rows[0] <= 1e-12,
                     max_abs_difference=rows[0], tolerance=1e-12)
            rp.check(branch, f'{tag}_{label}_theta_variance_logML_recompute',
                     max(rows[1:]) <= 1e-9 + 1e-6 * float(np.max(np.abs(saved_logml))),
                     detail=f'q={q}; theta/var/logML recomputed from persisted posterior and quadrature grid',
                     max_abs_difference=max(rows[1:]))
            rp.check(branch, f'{tag}_{label}_gradient_recompute', grad_delta <= 1e-8,
                     detail='saved gradient reproduced from responses, posterior and parameters',
                     max_abs_difference=grad_delta, tolerance=1e-8)
            params = z['parameters'].astype(float)
            saved_grad = z['gradient'].astype(float)
            prior_b = params[len(a):] / 9 if report['kind'] == 'map' else np.zeros(len(a))
            b_part_fresh = a * (Y - expected).sum(1) + prior_b
            saved_b = saved_grad if report['kind'] == 'rasch' else saved_grad[len(a):]
            prob_grad_delta = float(np.max(np.abs(b_part_fresh - saved_b)))
            bounds = ([(np.log(.2), np.log(5))] * len(a) + [(-10., 10.)] * len(a)
                      if report['kind'] != 'rasch' else [(-10., 10.)] * len(a))
            pg_fresh = float(np.max(np.abs(projected_gradient(grad, params, bounds))))
            rp.check(branch, f'{tag}_{label}_fitted_probability_gradient_consistency',
                     prob_grad_delta <= 1e-8,
                     detail='model-implied fitted item probabilities recomputed from saved parameters/posterior '
                            'reproduce the saved item-parameter gradient against the response matrix',
                     max_abs_difference=prob_grad_delta, tolerance=1e-8)
            total += 1
            matched += rp.compare(branch, f'/{tag}/{label}/projected_gradient_max', pg_fresh,
                                  report['projected_gradient_max'], f'R/{rel_dir}/{label}.json')
            matched += rp.compare(branch, f'/{tag}/{label}/logML', float(rec['logML'].sum()),
                                  report['logML'], f'R/{rel_dir}/{label}.json')
            total += 1
            matched += rp.compare(branch, f'/{tag}/{label}/objective', objective, report['objective'],
                                  f'R/{rel_dir}/{label}.json')
            total += 1
            matched += rp.compare(branch, f'/{tag}/{label}/boundary_counts',
                                  [int(z['a_boundary'].sum()), int(z['b_boundary'].sum())],
                                  [report['slope_boundary_count'], report['difficulty_boundary_count']],
                                  f'R/{rel_dir}/{label}.json', exact=True)
            total += 1
            rho_ta = float(spearmanr(saved_theta, Y.mean(0)).statistic)
            total += 1
            matched += rp.compare(branch, f'/{tag}/{label}/theta_accuracy_rho', rho_ta,
                                  report['theta_accuracy_rho'], f'R/{rel_dir}/{label}.json')
            rel_post = float(np.var(saved_theta, ddof=1) / (np.var(saved_theta, ddof=1) + saved_var.mean()))
            total += 1
            matched += rp.compare(branch, f'/{tag}/{label}/posterior_reliability', rel_post,
                                  report['posterior_reliability'], f'R/{rel_dir}/{label}.json')
            if report['kind'] == '2pl' and not report['numeric_screen_pass']:
                free_failures.append({'run': tag, 'label': label, 'score_max': report.get('projected_gradient_max'),
                                      'numeric_screen_pass': False,
                                      'note': 'unpenalized free-2PL failure preserved; not re-fit here'})
            variants[label] = {'kind': report['kind'], 'shape': list(Y.shape), 'q': q,
                               'numeric_screen_pass': report['numeric_screen_pass'],
                               'score_acceptance_candidate': report.get('score_acceptance_candidate', False),
                               'logML': float(rec['logML'].sum()), 'objective': objective,
                               'theta_accuracy_rho': rho_ta, 'posterior_reliability': rel_post,
                               'fitted_probability_unconstrained_residual_gap_max': score_gap,
                               'max_abs_differences': {'posterior_row_sums': rows[0], 'theta': rows[1],
                                                       'variance': rows[2], 'logML_array': rows[3],
                                                       'gradient': grad_delta}}
        source_ok = True
        source_shapes = {}
        for name, info in cfg['inputs'].items():
            m = rp.load_npz(R / info['matrix'], 'medical_source_matrix', 'primary_cache_input',
                            declared.get((R / info['matrix']).resolve()))
            rp.load_json(R / info['metadata'], 'medical_source_metadata', 'primary_cache_input',
                         declared.get((R / info['metadata']).resolve()))
            Ycfg = m['matrix'][m['fit_mask']].astype(float)
            first_label = sorted(k for k in variants if k.startswith(name + '_'))[0]
            saved_Y = rp.load_npz(rundir / f'{first_label}.npz', 'medical_fit_arrays', 'primary_cache_input')
            same = Ycfg.shape == saved_Y['Y'].shape and bool(np.array_equal(Ycfg, saved_Y['Y'].astype(float)))
            source_ok = source_ok and same
            source_shapes[name] = {'config_shape': list(Ycfg.shape), 'saved_shape': list(saved_Y['Y'].shape),
                                   'identical': bool(same)}
        rp.check(branch, f'{tag}_source_response_identity', source_ok,
                 detail='saved fit Y equals declared source matrix restricted by the saved fit mask',
                 pointer=f'/{tag}/source_matrix', exact=True)
        provenance[tag] = {'contract': cfg_rel, 'source_shapes': source_shapes, 'variants': variants}

    # start-stability check from saved arrays (both runs)
    stability = {}
    for rel_dir, _, tag in runs:
        rundir = R / rel_dir
        if (rundir / 'medical_A_2pl_1.0.npz').exists() and (rundir / 'medical_A_2pl_0.7.npz').exists():
            one = rp.load_npz(rundir / 'medical_A_2pl_1.0.npz', 'medical_fit_arrays', 'primary_cache_input')
            two = rp.load_npz(rundir / 'medical_A_2pl_0.7.npz', 'medical_fit_arrays', 'primary_cache_input')
            stability[tag] = {'theta_maxdiff': float(np.max(np.abs(one['theta'] - two['theta']))),
                              'logML_perperson_difference':
                                  float(abs(one['logML'].sum() - two['logML'].sum()) / one['Y'].shape[1])}
            ref = rp.load_json(rundir / 'summary.json', 'accepted_producer_summary', 'comparator_only')
            ref_stab = ref['variants']['medical_A']['unpenalized_start_stability']
            for field in ('theta_maxdiff', 'logML_perperson_difference'):
                total += 1
                matched += rp.compare(branch, f'/{tag}/unpenalized_start_stability/{field}',
                                      stability[tag][field], ref_stab[field],
                                      f'R/{rel_dir}/summary.json')
    summary = {
        'schema': SCHEMA + '/medical', 'branch': 'medical', 'status': 'CACHED_BASELINE_REPLAY_PRODUCED',
        'replay_level': 'CACHED_BASELINE_REPLAY (no fit call; bounded numerical check of saved theta/'
                        'fitted probabilities/objective against saved scores and response data)',
        'fit_provenance': provenance,
        'unpenalized_free_2pl_failures_preserved': free_failures,
        'unpenalized_start_stability_recomputed': stability,
        'fresh_medical_associations_not_duplicated':
            {'owner': 'C10o / new28', 'pointer': 'R/reports/NEW_MEDICAL_ASSOCIATIONS.json',
             'policy': 'fresh association cells are produced by the medical-association worker, not here'},
        'missing_source_fields': ['Fitted item probabilities were not persisted as a separate array; they are '
                                  'recomputed exactly from the persisted posterior over the same quadrature grid.',
                                  'No fresh or re-estimated medical IRT parameters are produced by this cache replay.'],
        'fitted_probability_note': 'the model-implied fitted probability matrix is recomputed from the saved '
                                   'parameters and posterior; its item-gradient identity against the response '
                                   'matrix matches the saved gradient at <=1e-8. The unconstrained residual gap is '
                                   'reported per fit: it is large exactly for the prior-regularized MAP fits and for '
                                   'the boundary-hitting free-2PL failures, whose accepted screen is the projected '
                                   'gradient (recomputed here and compared to the saved report).',
        'scientific_limits': ['Numerical provenance screen only; no construct-validity or geometry-inference release.',
                              'The free-2PL failure is preserved as historical outcome, not repaired.'],
    }
    return {'branch': branch, 'status': summary['status'],
            'artifacts': {'medical_baseline.json': summary},
            'summary': {'comparator_matched': matched, 'comparator_total': total,
                        'checks_passed': sum(1 for c in rp.checks if c['branch'] == branch and c['passed']),
                        'checks_total': sum(1 for c in rp.checks if c['branch'] == branch)}}


# --------------------------------------------------------------------------
# branch: theory (independent formulas / closed-form checks)
# --------------------------------------------------------------------------
def branch_theory(rp):
    branch = 'theory'
    R = rp.R
    saved = rp.load_json(R / 'runs/theory_execution_v1/result.json', 'accepted_producer_summary', 'comparator_only')
    rp.register(R / 'code/theory_execution_v1.py', 'source_code_reference', 'reference_code_only')
    for rel, sha in saved['bindings'].items():
        path = R / rel
        if path.exists():
            rp.register(path, 'theory_binding', 'frozen_rule_config', sha)
        else:
            rp.gap(branch, rel, 'binding declared by accepted theory producer is missing; not used as input')

    import sympy as sp
    x, b, t = sp.symbols('x b t', real=True)
    p = 1 / (1 + sp.exp(b - t))
    u, v, ex, ey = sp.symbols('u v ex ey', positive=True)
    c, a, gamma = sp.symbols('c a gamma', real=True)
    checks = {
        'rasch_score_derivative': sp.simplify(sp.diff(x * sp.log(p) + (1 - x) * sp.log(1 - p), t) - (x - p)),
        'rasch_information': sp.simplify(sp.diff(p, t) - p * (1 - p)),
        'mirt_mirror_expected_total': sp.simplify(1 / (1 + sp.exp(t)) + 1 / (1 + sp.exp(-t)) - 1),
        'pearson_attenuation_squared': sp.simplify(c * c / ((u + ex) * (v + ey))
                                                   - (c * c / (u * v)) * (u / (u + ex)) * (v / (v + ey))),
        'noisy_confounder_covariance': sp.simplify(a * b * u - (a * u) * (b * u) / (u + v) - a * b * u * v / (u + v)),
        'mechanical_edge_conditional_covariance': sp.simplify(
            (sp.Matrix([[a, 1, 0]]) * sp.diag(u, ex, ey) * sp.Matrix([b, gamma, 1]))[0]
            - (a * u) * (b * u) / u - gamma * ex),
    }
    symbols_ok = {key: (str(val) == '0' and saved['checks'][key] == '0') for key, val in checks.items()}
    rp.check(branch, 'symbolic_identity_regeneration', all(symbols_ok.values()),
             detail='independent sympy simplification reproduces the accepted zero identities', exact=True)
    for check_key, check_value in checks.items():
        rp.compare(branch, f'/checks/{check_key}', str(check_value), saved['checks'][check_key],
                   'R/runs/theory_execution_v1/result.json', exact=True)

    rng = np.random.default_rng(20260916)
    X = rng.normal(size=(20, 5))
    Y = rng.normal(size=(20, 4))
    Q = np.linalg.qr(rng.normal(size=(5, 5)))[0]
    perm = rng.permutation(20)

    def cka(A, B):
        A = A - A.mean(0)
        B = B - B.mean(0)
        K, L = A @ A.T, B @ B.T
        return float(np.sum(K * L) / np.sqrt(np.sum(K * K) * np.sum(L * L)))

    base = cka(X, Y)
    numerics = {'cka_base': base,
                'orthogonal_scaled_error': abs(cka(-3 * X @ Q, 7 * Y) - base),
                'common_permutation_error': abs(cka(X[perm], Y[perm]) - base),
                'independent_row_permutation_difference': abs(cka(X[perm], Y) - base),
                'anisotropic_scaling_difference': abs(cka(X * np.array([30, 1, 1, 1, 1]), Y) - base),
                'confounder_residual_example': float((a * b * u * v / (u + v)).subs({a: 1, b: 1, u: 1, v: 1})),
                'correlated_errors_counterexample': saved['numerics']['correlated_errors_counterexample']}
    for num_key, num_value in numerics.items():
        if isinstance(num_value, dict):
            rp.compare(branch, f'/numerics/{num_key}', num_value, saved['numerics'][num_key],
                       'R/runs/theory_execution_v1/result.json', exact=True)
        else:
            rp.compare(branch, f'/numerics/{num_key}', num_value, saved['numerics'][num_key],
                       'R/runs/theory_execution_v1/result.json')
    invariants = (numerics['orthogonal_scaled_error'] < 1e-12
                  and numerics['common_permutation_error'] < 1e-12
                  and numerics['independent_row_permutation_difference'] > 1e-6
                  and numerics['anisotropic_scaling_difference'] > 1e-6)
    rp.check(branch, 'cka_invariance_thresholds', invariants and bool(saved['passed']),
             detail='invariance identities and counterexample inequalities hold on the frozen seed', exact=True)

    summary = {
        'schema': SCHEMA + '/theory', 'branch': 'theory', 'status': 'CACHED_BASELINE_REPLAY_PRODUCED',
        'replay_level': 'INDEPENDENT_FORMULA_CHECK_IN_NEW_OUTPUT (no new theorem, no experiment, no Lean/Wolfram)',
        'checks': {k: str(v) for k, v in checks.items()}, 'numerics': numerics,
        'passed': bool(all(str(v) == '0' for v in checks.values()) and invariants),
        'limitations': saved['limitations'],
        'source_reference': 'R/code/theory_execution_v1.py (reference only, not imported)',
    }
    return {'branch': branch, 'status': summary['status'],
            'artifacts': {'theory_baseline.json': summary},
            'summary': {'comparator_matched': sum(1 for r in rp.comparators if r['branch'] == branch and r['matched']),
                        'comparator_total': sum(1 for r in rp.comparators if r['branch'] == branch),
                        'checks_passed': sum(1 for c in rp.checks if c['branch'] == branch and c['passed']),
                        'checks_total': sum(1 for c in rp.checks if c['branch'] == branch)}}


# --------------------------------------------------------------------------
# branch: bos (P2-B-006 BOS / cloud-size sequential decomposition)
# --------------------------------------------------------------------------
BOS_STATS = {
    'step1_anchor': ('step1_anchor',),
    'step2_eff_rank': ('step2_channels_doubleBOS', 'eff_rank'),
    'step2_participation_ratio': ('step2_channels_doubleBOS', 'participation_ratio'),
    'step2_g5no_chi': ('step2_channels_doubleBOS', 'g5no_chi'),
    'step2_noeffrank': ('step2_channels_doubleBOS', 'composite_minus_effrank'),
    'step3_composite': ('step3_bos_swap', 'composite_singleBOS'),
    'step3_effrank': ('step3_bos_swap', 'eff_rank_singleBOS'),
    'step3_PR': ('step3_bos_swap', 'PR_singleBOS'),
    'step3_chi': ('step3_bos_swap', 'g5no_chi_singleBOS'),
    'step4_effrank': ('step4_rarefaction', 'eff_rank_rarefied'),
    'step4_PR': ('step4_rarefaction', 'PR_rarefied'),
    'step4_composite': ('step4_rarefaction', 'composite_rarefied'),
    'step4b_chi': ('step4b_chi_rarefaction', 'chi_rarefied'),
    'step4b_composite': ('step4b_chi_rarefaction', 'composite_all_rarefied'),
}


def branch_bos(rp):
    branch = 'bos'
    R = rp.R
    bos_dir = R / 'runs/study1_reuse_v1/bos'
    hashes = rp.load_json(bos_dir / 'artifact_hashes.json', 'bos_artifact_hashes', 'hash_declaration')

    def declared(name):
        return hashes.get(name)

    old = rp.load_json(bos_dir / 'results.json', 'accepted_producer_summary', 'comparator_only',
                       declared('results.json'))
    z = rp.load_npz(bos_dir / 'per_model_inputs.npz', 'bos_per_model_inputs', 'primary_cache_input',
                    declared('per_model_inputs.npz'))
    dz = rp.load_npz(bos_dir / 'bootstrap_draws.npz', 'bos_bootstrap_draws', 'primary_cache_input',
                     declared('bootstrap_draws.npz'))
    rare = rp.load_npz(bos_dir / 'rarefied_spectral_inputs.npz', 'bos_rarefied_spectral_inputs',
                       'secondary_linkage_input', declared('rarefied_spectral_inputs.npz'))
    rp.load_json(bos_dir / 'input_contract.json', 'bos_input_contract_snapshot', 'hash_declaration',
                 declared('input_contract.json'))
    rp.load_json(bos_dir / 'rarefaction_draws.json', 'bos_rarefaction_draws', 'secondary_linkage_input',
                 declared('rarefaction_draws.json'))
    rp.load_json(R / 'reports/study1_reuse_additional_contract.json', 'frozen_rule_binding',
                 'frozen_rule_config')
    rp.register(R / 'code/study1_reuse_bos.py', 'source_code_reference', 'reference_code_only')
    ver = rp.load_json(R / 'verifier/V_NUM_BOS_G_BRANCH.json', 'verifier_report', 'comparator_only')

    tags = [str(t) for t in z['model_order']]
    rp.check(branch, 'model_order_identity', tags == [str(t) for t in old['tags']] and len(tags) == 50
             and len(set(tags)) == 50,
             detail='50 unique math-panel model tags identical to the accepted producer list', exact=True)
    y = z['theta'].astype(float)
    c = z['controls'].astype(float)
    fam = np.asarray([str(v) for v in z['family']])
    fd = dz['family_draws']
    rp.check(branch, 'family_index_identity',
             fd.shape == (2000, 13) and int(fd.min()) >= 0 and int(fd.max()) < 13
             and len(sorted(set(fam.tolist()))) == 13,
             detail='2000 family-block draws over the 13 sorted math families', exact=True)
    rp.check(branch, 'rarefied_spectral_linkage',
             bool(np.array_equal(rare['eff_rank'].astype(float), z['step4_effrank'].astype(float)))
             and bool(np.array_equal(rare['participation_ratio'].astype(float), z['step4_PR'].astype(float))),
             detail='rarefied spectral inputs equal the saved step4 member arrays', exact=True)

    kernels = {
        'step1_anchor': pc1({'eff_rank': z['step2_eff_rank'].astype(float),
                             'participation_ratio': z['step2_participation_ratio'].astype(float),
                             'g5no_chi': z['step2_g5no_chi'].astype(float)}),
        'step2_noeffrank': pc1_sorted_first({'participation_ratio': z['step2_participation_ratio'].astype(float),
                                             'g5no_chi': z['step2_g5no_chi'].astype(float)}),
        'step3_composite': pc1({'eff_rank': z['step3_effrank'].astype(float),
                                'participation_ratio': z['step3_PR'].astype(float),
                                'g5no_chi': z['step3_chi'].astype(float)}),
        'step4_composite': pc1({'eff_rank': z['step4_effrank'].astype(float),
                                'participation_ratio': z['step4_PR'].astype(float),
                                'g5no_chi': z['step3_chi'].astype(float)}),
        'step4b_composite': pc1({'eff_rank': z['step4_effrank'].astype(float),
                                 'participation_ratio': z['step4_PR'].astype(float),
                                 'g5no_chi': z['step4b_chi'].astype(float)}),
    }
    for key, rebuilt in kernels.items():
        dmax = float(np.max(np.abs(rebuilt - z[key].astype(float))))
        rp.check(branch, f'{key}_kernel_regeneration', dmax <= 1e-12,
                 detail='saved member arrays reproduce the saved composite through the frozen PC1 kernel',
                 pointer=f'/bos/{key}', max_abs_difference=dmax, tolerance=1e-12)

    stats = {}
    matched = total = 0
    for key, path in BOS_STATS.items():
        v = z[key].astype(float)
        draws = dz[key].astype(float)
        st = {'raw_rho': rho(v, y), 'partial_rho': rho(v, y, c),
              'n_models': 50, 'n_families': 13, 'ci_n_valid': int(np.isfinite(draws).sum())}
        st.update(ci_from_draws(draws))
        st['cluster_ci'] = st['ci']
        ref = jget(old, '/' + '/'.join(path))
        for field, exact in (('n_models', True), ('raw_rho', False), ('partial_rho', False),
                             ('cluster_ci', False), ('ci_excludes_zero', True),
                             ('ci_n_valid', True), ('n_families', True)):
            total += 1
            matched += rp.compare(branch, f'/bos/{key}/{field}', st[field], ref[field],
                                  'R/runs/study1_reuse_v1/bos/results.json', exact=exact)
        stats[key] = st
    total += 1
    matched += rp.compare(branch, '/bos/step3_bos_swap/delta_composite_partial',
                          stats['step3_composite']['partial_rho'] - stats['step1_anchor']['partial_rho'],
                          old['step3_bos_swap']['delta_composite_partial'],
                          'R/runs/study1_reuse_v1/bos/results.json')
    seq = {'anchor': stats['step1_anchor']['partial_rho'],
           'after_BOS': stats['step3_composite']['partial_rho'],
           'after_spectral_rarefaction': stats['step4_composite']['partial_rho'],
           'anchor_minus_BOS': stats['step1_anchor']['partial_rho'] - stats['step3_composite']['partial_rho'],
           'BOS_minus_spectral_rarefaction': stats['step3_composite']['partial_rho']
                                              - stats['step4_composite']['partial_rho'],
           'after_all_rarefaction': stats['step4b_composite']['partial_rho']}
    total += 1
    matched += rp.compare(branch, '/bos/sequential_differences', seq, old['sequential_differences'],
                          'R/runs/study1_reuse_v1/bos/results.json')

    ver_rows = {row['key']: row for row in ver['bos']['statistics']}
    for key, row in ver_rows.items():
        total += 2
        matched += rp.compare(branch, f'/verifier/bos/{key}/partial_rho', stats[key]['partial_rho'],
                              row['partial_rho'], 'R/verifier/V_NUM_BOS_G_BRANCH.json')
        matched += rp.compare(branch, f'/verifier/bos/{key}/ci', stats[key]['cluster_ci'],
                              row['ci'], 'R/verifier/V_NUM_BOS_G_BRANCH.json')

    regen = []
    for key in ('step1_anchor', 'step3_composite', 'step4b_composite'):
        v = z[key].astype(float)
        saved = dz[key].astype(float)
        for row in (0, 1, 999, 1999):
            fresh = boot_stat(v, y, c, fam, fd[row])
            regen.append({'key': key, 'row': row, 'fresh': fresh, 'saved': float(saved[row]),
                          'abs_difference': abs(fresh - float(saved[row]))})
    worst = max(r['abs_difference'] for r in regen)
    rp.check(branch, 'representative_bootstrap_regeneration', worst <= 1e-9,
             detail=f'{len(regen)} saved family-bootstrap draws regenerated from per-model inputs and saved indices',
             max_abs_difference=worst, tolerance=1e-9)

    summary = {
        'schema': SCHEMA + '/bos', 'branch': 'bos', 'status': 'CACHED_BASELINE_REPLAY_PRODUCED',
        'replay_level': 'CACHED_BASELINE_REPLAY (saved single-BOS chi and rarefied spectral inputs reused; '
                        'no scoring, extraction or L0 rarefaction rerun)',
        'ids': ['P2-B-006'], 'n_models': 50, 'n_families': 13, 'b_boot': 2000,
        'statistics': stats, 'sequential_differences': seq,
        'interpretation': old['interpretation'],
        'limitations': ver['bos']['limitations'],
        'representative_regeneration': regen,
        'cache_boundary': old['cache_boundary'],
    }
    return {'branch': branch, 'status': summary['status'],
            'artifacts': {'bos_baseline.json': summary},
            'summary': {'comparator_matched': matched, 'comparator_total': total,
                        'checks_passed': sum(1 for c_ in rp.checks if c_['branch'] == branch and c_['passed']),
                        'checks_total': sum(1 for c_ in rp.checks if c_['branch'] == branch)}}


# --------------------------------------------------------------------------
# branch: retention (P2-B-007 disjoint-item signal retention)
# --------------------------------------------------------------------------
def branch_retention(rp):
    branch = 'retention'
    R = rp.R
    ret_dir = R / 'runs/study1_reuse_v1/retention'
    old = rp.load_json(ret_dir / 'results.json', 'accepted_producer_summary', 'comparator_only')
    contract = rp.load_json(R / 'reports/study1_reuse_retention_contract.json', 'frozen_rule_binding',
                            'frozen_rule_config', old['contract_sha256'])
    z = rp.load_npz(ret_dir / 'per_model_statistical_inputs.npz', 'retention_per_model_inputs',
                    'primary_cache_input')
    sz = rp.load_npz(ret_dir / 'seed0_sampling_statistics.npz', 'retention_seed0_draws',
                     'primary_cache_input')
    ret_contract = rp.load_json(ret_dir / 'input_contract.json', 'retention_input_contract_snapshot',
                                'hash_declaration')
    rp.register(R / 'code/study1_reuse_retention.py', 'source_code_reference', 'reference_code_only')
    ver = rp.load_json(R / 'verifier/V_NUM_RETENTION_FINAL_INDEPENDENT.json', 'verifier_report',
                       'comparator_only')
    rp.load_json(R / 'verifier/V_RETENTION_TRACKB_RESIDUAL_SCREEN.json', 'verifier_residual_screen',
                 'comparator_only')

    fd = z['family_bootstrap_draws']
    rp.check(branch, 'family_index_identity', fd.shape == (10000, 12)
             and int(fd.min()) >= 0 and int(fd.max()) < 12,
             detail='10000 family-block draws over the 12 sorted retention families', exact=True)
    seeds = [int(s) for s in old['seeds']]
    matched = total = 0
    domains_out = {}
    for dom in ('code', 'math'):
        tags = [str(t) for t in z[f'{dom}_model_order']]
        fam = np.asarray([str(v) for v in z[f'{dom}_families']])
        C = z[f'{dom}_C'].astype(float)
        rp.check(branch, f'{dom}_model_identity', len(tags) == 31 and len(set(tags)) == 31
                 and len(C) == 31,
                 detail='31 unique retained-protocol model tags with aligned covariates', exact=True)
        cells = {}
        for seed in seeds:
            for direction in ('geoA_thB', 'geoB_thA'):
                prefix = f'{dom}_{seed}_{direction}'
                key = f'{seed}|{direction}'
                mask = z[prefix + '_mask'].astype(bool)
                x = z[prefix + '_x'].astype(float)
                y = z[prefix + '_theta'].astype(float)
                members = {k: z[prefix + '_' + k].astype(float)
                           for k in ('eff_rank', 'participation_ratio', 'g5no_chi')}
                rebuilt = pc1({k: v[mask] for k, v in members.items()})
                dmax = float(np.max(np.abs(rebuilt - x)))
                rp.check(branch, f'{prefix}_kernel_regeneration', dmax <= 1e-12,
                         detail='saved member halves reproduce the saved disjoint composite through the frozen PC1 kernel',
                         pointer=f'/retention/{key}/x', max_abs_difference=dmax, tolerance=1e-12)
                cc = C[mask]
                cell = {'raw': rho(x, y), 'partial': rho(x, y, cc),
                        'n_models': int(mask.sum()),
                        'dropped': [t for t, ok in zip(tags, mask.tolist()) if not ok]}
                ref = old['domains'][dom]['cells'][key]
                for field, exact in (('raw', False), ('partial', False), ('n_models', True),
                                     ('dropped', True)):
                    total += 1
                    matched += rp.compare(branch, f'/retention/{dom}/{key}/{field}', cell[field],
                                          ref[field],
                                          'R/runs/study1_reuse_v1/retention/results.json', exact=exact)
                cells[key] = cell
        med = float(np.median([v['partial'] for v in cells.values()]))
        ref_value = float(contract['statistics']['retention_reference'][dom])
        retention = abs(med) / abs(ref_value)
        out = {'median_partial': med, 'full_run_partial_ref': ref_value, 'retention': retention,
               'retention_percent': 100 * retention,
               'same_sign': bool(np.sign(med) == np.sign(ref_value) and med != 0),
               'n_fits': len(cells), 'n_direction_cells': len(cells)}
        for field, exact in (('median_partial', False), ('full_run_partial_ref', False),
                             ('retention', False), ('retention_percent', False), ('same_sign', True)):
            total += 1
            matched += rp.compare(branch, f'/retention/{dom}/{field}', out[field],
                                  old['domains'][dom][field],
                                  'R/runs/study1_reuse_v1/retention/results.json', exact=exact)
        for field in ('median_partial', 'retention', 'n_fits', 'n_direction_cells'):
            total += 1
            matched += rp.compare(branch, f'/verifier/retention/{dom}/{field}', out[field],
                                  ver['results'][dom][field],
                                  'R/verifier/V_NUM_RETENTION_FINAL_INDEPENDENT.json',
                                  exact=field in ('n_fits', 'n_direction_cells'))
        seed0 = {}
        for direction in ('geoA_thB', 'geoB_thA'):
            prefix = f'{dom}_{seeds[0]}_{direction}'
            mask = z[prefix + '_mask'].astype(bool)
            x = z[prefix + '_x'].astype(float)
            y = z[prefix + '_theta'].astype(float)
            cc = C[mask]
            fam_m = fam[mask]
            bare = ci_from_draws(sz[prefix + '_bare'])
            bare['rho'] = rho(x, y, cc[:, 2])
            ctrl = ci_from_draws(sz[prefix + '_controlled'])
            ctrl['rho'] = rho(x, y, cc)
            perms = z[prefix + '_perm_indices']
            perm = perm_from_draws(ctrl['rho'], sz[prefix + '_permutation'])
            rp.check(branch, f'{prefix}_permutation_rows',
                     perms.shape[1] == int(mask.sum())
                     and bool(np.all(np.sort(perms, axis=1) == np.arange(int(mask.sum()))[None, :])),
                     detail='saved permutation rows are unrestricted permutations of the retained models', exact=True)
            entry = {'raw_rho': rho(x, y), 'partial_rho_controlling_C': ctrl['rho'],
                     'bare_cluster_ci': bare['ci'], 'controlled_cluster_ci': ctrl['ci'],
                     'bare_ci_excludes_zero': bare['ci_excludes_zero'],
                     'controlled_ci_excludes_zero': ctrl['ci_excludes_zero'],
                     'permutation_p': perm['p'], 'n_models': int(mask.sum()),
                     'bootstrap_valid_counts': [bare['n_valid'], ctrl['n_valid']],
                     'permutation_valid_count': perm['n_valid'],
                     'permutation_exceed_count': perm['exceed_count']}
            entry['verdict'] = historical_verdict(entry['bare_ci_excludes_zero'],
                                                  entry['controlled_ci_excludes_zero'],
                                                  perm['p'], len(y), 1000)
            ref = old['domains'][dom]['seed0_full'][direction]
            for field in ('raw_rho', 'partial_rho_controlling_C', 'bare_cluster_ci',
                          'controlled_cluster_ci', 'bare_ci_excludes_zero',
                          'controlled_ci_excludes_zero', 'permutation_p', 'verdict', 'n_models',
                          'bootstrap_valid_counts', 'permutation_valid_count',
                          'permutation_exceed_count'):
                total += 1
                matched += rp.compare(branch, f'/retention/{dom}/seed0/{direction}/{field}',
                                      entry[field], ref[field],
                                      'R/runs/study1_reuse_v1/retention/results.json',
                                      exact=field in ('bare_ci_excludes_zero',
                                                      'controlled_ci_excludes_zero', 'verdict',
                                                      'n_models', 'bootstrap_valid_counts',
                                                      'permutation_valid_count',
                                                      'permutation_exceed_count'))
            regen = []
            for row in (0, 1, 5000, 9999):
                fr = boot_stat(x, y, cc[:, 2], fam_m, fd[row])
                regen.append({'stream': 'bare', 'row': row, 'abs_difference':
                              abs(fr - float(sz[prefix + '_bare'][row]))})
                fc = boot_stat(x, y, cc, fam_m, fd[row])
                regen.append({'stream': 'controlled', 'row': row, 'abs_difference':
                              abs(fc - float(sz[prefix + '_controlled'][row]))})
            for row in (0, 1, 9999):
                fp = rho(x, y[perms[row]], cc)
                regen.append({'stream': 'permutation', 'row': row, 'abs_difference':
                              abs(fp - float(sz[prefix + '_permutation'][row]))})
            worst = max(r_['abs_difference'] for r_ in regen)
            rp.check(branch, f'{prefix}_seed0_representative_regeneration', worst <= 1e-9,
                     detail=f'{len(regen)} saved seed0 draws regenerated from per-model arrays and saved indices',
                     max_abs_difference=worst, tolerance=1e-9)
            entry['representative_regeneration'] = regen
            seed0[direction] = entry
        out['seed0_full'] = seed0
        out['seed0_verdicts_concordant'] = seed0['geoA_thB']['verdict'] == seed0['geoB_thA']['verdict']
        total += 1
        matched += rp.compare(branch, f'/retention/{dom}/seed0_verdicts_concordant',
                              out['seed0_verdicts_concordant'],
                              old['domains'][dom]['seed0_verdicts_concordant'],
                              'R/runs/study1_reuse_v1/retention/results.json', exact=True)
        domains_out[dom] = out

    summary = {
        'schema': SCHEMA + '/retention', 'branch': 'retention', 'status': 'CACHED_BASELINE_REPLAY_PRODUCED',
        'replay_level': 'CACHED_BASELINE_REPLAY (saved disjoint item splits and saved seed0 draws reused; '
                        'no scoring, extraction or fit rerun)',
        'ids': ['P2-B-007-CODE', 'P2-B-007-MATH'], 'seeds': seeds,
        'domains': domains_out,
        'interpretation': old['interpretation'],
        'limitations': ver['limitations'],
        'source_level': 'saved per-half geometry summaries (eff_rank/PR/chi) plus saved seed0 bootstrap and '
                        'permutation draws; the raw activation-level rebuild was already executed by the accepted '
                        'retention producer with its own cache diagnostics, not repeated here',
        'underlying_binding_contract': {'path': 'R/runs/study1_reuse_v1/retention/input_contract.json',
                                        'domains': sorted(ret_contract.get('domains', {})),
                                        'n_observed_npz_bindings': len(ret_contract.get('observed_NPZ_bindings', {}))},
    }
    return {'branch': branch, 'status': summary['status'],
            'artifacts': {'retention_baseline.json': summary},
            'summary': {'comparator_matched': matched, 'comparator_total': total,
                        'checks_passed': sum(1 for c_ in rp.checks if c_['branch'] == branch and c_['passed']),
                        'checks_total': sum(1 for c_ in rp.checks if c_['branch'] == branch)}}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='C17 cached-baseline replay (C10b)')
    ap.add_argument('--analysis-root', default=None,
                    help='analysis root R (default: repository containing this script)')
    ap.add_argument('--out', required=True, help='fresh (empty/nonexistent) output directory')
    ap.add_argument('--branch', choices=('all',) + BRANCHES, default='all')
    ap.add_argument('--expected-manifest', default=None,
                    help='optional pinned source manifest (produced source_manifest.json / C17_CACHE_BASELINE_v1.json '
                         'or a flat {path: sha256} object): every consumed source must be pinned with a matching sha256')
    args = ap.parse_args()
    root = Path(args.analysis_root).resolve() if args.analysis_root else Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'refusing to write into non-empty output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    pins = None
    if args.expected_manifest:
        pin_doc = json.loads(Path(args.expected_manifest).read_text())
        if isinstance(pin_doc, dict) and 'entries' in pin_doc:
            pins = {e['path']: e['sha256'] for e in pin_doc['entries']}
        elif isinstance(pin_doc, dict) and 'source_manifest' in pin_doc:
            pins = {e['path']: e['sha256'] for e in pin_doc['source_manifest']['entries']}
        elif isinstance(pin_doc, dict):
            pins = {str(k): str(v) for k, v in pin_doc.items()}
        else:
            raise SystemExit('unsupported --expected-manifest structure')
    rp = Replay(root, out, pins)
    todo = BRANCHES if args.branch == 'all' else (args.branch,)
    for name in todo:
        rp.branch_results[name] = globals()[f'branch_{name}'](rp)
    rp.verify_manifest()
    elapsed = time.time() - rp.t0

    for name, res in rp.branch_results.items():
        for filename, payload in res['artifacts'].items():
            rp.dump(out / name / filename, payload)
    by_branch = {}
    for name, res in rp.branch_results.items():
        rows = [c for c in rp.comparators if c['branch'] == name]
        checks = [c for c in rp.checks if c['branch'] == name]
        by_branch[name] = {**res['summary'],
                           'comparator_deviations': [c for c in rows if not c['matched']][:25],
                           'checks_failed': [c for c in checks if not c['passed']]}
    summary = {'schema': SCHEMA, 'status': 'CACHED_BASELINE_REPLAY_PRODUCED_PENDING_ROOT_VERIFICATION',
               'branches': {k: v['status'] for k, v in rp.branch_results.items()},
               'branch_summary': by_branch,
               'comparator_total': len(rp.comparators),
               'comparator_matched': sum(1 for c in rp.comparators if c['matched']),
               'checks_total': len(rp.checks),
               'checks_passed': sum(1 for c in rp.checks if c['passed']),
               'manifest_entries': len(rp.manifest),
               'gaps': rp.gaps,
               'replay_level': 'CACHED_BASELINE_REPLAY',
               'scope_note': 'recomputes retained frozen statistics from saved artifacts; not a fresh production run',
               'expected_manifest': str(Path(args.expected_manifest).resolve()) if args.expected_manifest else None,
               'pinned_entries': len(pins) if pins else None,
               'pinned_entries_not_consumed_by_this_run': (len(set(pins) - set(rp.manifest)) if pins else None),
               'not_computed_here': ['AN-B/C4/C6 simulation caches (other worker)',
                                     'C2 new28 grid / 6-window / G-probe observations (other worker)',
                                     'C10o fresh medical associations (other worker)',
                                     'full activation/scoring replay (baseline replay starts at saved per-fit artifacts by design)',
                                     'code shadow-only recurrence P2-C-003 (audited: requires the study2 code feature panel, '
                                     'not part of the C10b baseline branches)']}
    rp.dump(out / 'summary.json', summary)
    manifest = {'schema': SCHEMA + '/source-manifest', 'analysis_root': str(rp.R), 'project_root': str(rp.P),
                'entry_count': len(rp.manifest),
                'entries': sorted(rp.manifest.values(), key=lambda e: e['path']),
                'roles': sorted(set(e['role'] for e in rp.manifest.values())),
                'levels': sorted(set(e['level'] for e in rp.manifest.values())),
                'source_mutation_check': True,
                'comparator_only_declaration': 'accepted producer summaries are read after fresh numbers exist and are '
                                               'used exclusively for comparator rows'}
    rp.dump(out / 'source_manifest.json', manifest)
    rp.dump(out / 'comparators.json', {'schema': SCHEMA + '/comparators', 'rule': TOL_RULE,
                                       'counts_exact': True, 'rows': rp.comparators,
                                       'matched': sum(1 for c in rp.comparators if c['matched']),
                                       'total': len(rp.comparators)})
    rp.dump(out / 'checks.json', {'schema': SCHEMA + '/checks', 'rows': rp.checks,
                                  'passed': sum(1 for c in rp.checks if c['passed']), 'total': len(rp.checks)})
    receipt = {'schema': SCHEMA + '/receipt', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'producer': rp.rel(Path(__file__)), 'producer_sha256': dig(__file__),
               'analysis_root': str(rp.R), 'output_dir': str(out),
               'commands': [{'argv': sys.argv, 'cwd': str(Path.cwd()), 'seconds': elapsed}],
               'environment': {'python': platform.python_version(), 'executable': sys.executable,
                               'numpy': np.__version__, 'scipy': scipy.__version__,
                               'threads': {k: os.environ.get(k) for k in
                                           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                               'platform': platform.platform()},
               'source_manifest_sha256': dig(out / 'source_manifest.json'),
               'expected_manifest': str(Path(args.expected_manifest).resolve()) if args.expected_manifest else None,
               'pinned_entries': len(pins) if pins else None,
               'pinned_entries_not_consumed_by_this_run': (len(set(pins) - set(rp.manifest)) if pins else None),
               'no_writeback': 'all consumed files re-hashed after computation; outputs written only under --out',
               'validation_scope': 'cached numerical comparison; no independent scientific review'}
    rp.dump(out / 'receipt.json', receipt)
    (out / 'commands.log').write_text(
        '\n'.join([f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] cwd={Path.cwd()} argv='
                   + ' '.join(sys.argv)]) + '\n')
    print(json.dumps({'out': str(out), 'branches': summary['branches'],
                      'comparator': f"{summary['comparator_matched']}/{summary['comparator_total']}",
                      'checks': f"{summary['checks_passed']}/{summary['checks_total']}",
                      'manifest_entries': len(rp.manifest), 'seconds': round(elapsed, 2)}))
    if summary['checks_passed'] != summary['checks_total']:
        raise SystemExit('self-check failure: see checks.json')


if __name__ == '__main__':
    main()
