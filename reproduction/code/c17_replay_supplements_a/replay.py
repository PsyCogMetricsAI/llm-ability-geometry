#!/usr/bin/env python3
"""C10u retained existing-data replay supplements A.

Fresh replay of retained (not historical-reproduction) quantities that were
missing from the accepted cache drivers:

  * P2-A-005  girth vs mirt rank agreement on the saved 32-model thetas
  * P2-A-007  true-theta item split-half reliability (four saved fit-parameter
              sets, direct 41-node Gauss-Legendre EAP integral, no refit)
  * P2-C-NBIAS-* (all 7)  empirical cloud-size dependence from the nine real
              H4A cells: every stored draw index is re-evaluated on the raw L0
              activation arrays and the median-|relative-bias| statistics are
              recomputed and re-classified
  * P2-C-003  Study2 code shadow-only recurrence, recomputed from the stored
              per-draw arrays (x, theta, controls, families, family-bootstrap
              index draws, permutation index draws) of the code-domain cells

Every fresh number starts from raw vectors, saved fitted parameters or saved
per-draw arrays.  Terminal aggregate JSONs (accepted producer summaries,
independent-review reports and the C17 evidence ledger) are read only *after*
the fresh numbers exist and are used exclusively as comparators/pointers.

Usage:
    <pinned-python> code/c17_replay_supplements_a/replay.py \
        --analysis-root <R> --out <EMPTY_NEW_DIR> [--expected-manifest <manifest>] \
        [--groups all|study1,nbias,code_shadow] [--emit-expected-manifest <path>]

The driver never writes into the analysis tree, never imports analysis code and
fails closed on a missing input, a declared/pinned hash mismatch, an
unexpected source-path shape, or an output directory that is not empty.
Paths recorded in accepted artifacts are re-rooted under the supplied
--analysis-root/--project-root, so a renamed copy of the whole tree (with the
original still present) is supported without reading the original.
"""
from __future__ import annotations

import os

# CPU-only single-thread BLAS policy (declared in the run receipt).
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.special import expit, roots_legendre

SCHEMA = 'C17-REPLAY-SUPPLEMENTS-A-v1'
TOL_RULE = 'abs<=1e-9+1e-6*abs(reference)'
STUDY1_RULE = 'abs<=1e-10 (source: verify_study1_additional_reuse.py scalar rule; stricter than default)'
FEATURE_RULE = 'abs<=1e-8+1e-6*abs(reference) (source: STUDY2_FEATURE_REUSE_V1.json numeric_atol/rtol)'
INFERENCE_RULE = 'abs<=1e-9 (source: verify_study2_inference_reuse.py scalar rule; stricter than config atol1e-8/rtol1e-6)'
DISPLAY_RULE = 'abs<=5e-5 (4-decimal historical display)'
GROUPS = ('study1', 'nbias', 'code_shadow')
METRICS = ['twoNN_id', 'eff_rank_pr', 'rankme', 'stable_rank', 'spectral_alpha', 'vn_entropy', 'isoscore']
NBIAS_CLAIM = {
    'twoNN_id': 'P2-C-NBIAS-TWONN_ID', 'eff_rank_pr': 'P2-C-NBIAS-EFF_RANK_PR',
    'rankme': 'P2-C-NBIAS-RANKME', 'stable_rank': 'P2-C-NBIAS-STABLE_RANK',
    'spectral_alpha': 'P2-C-NBIAS-SPECTRAL_ALPHA', 'vn_entropy': 'P2-C-NBIAS-VN_ENTROPY',
    'isoscore': 'P2-C-NBIAS-ISOSCORE',
}
CODE_METRICS = METRICS
DEFAULT_H4A_CELL_ORDER = [
    'gemma_2_27b_it|science', 'qwen3_8b|science', 'afm_4_5b|science',
    'gemma_2_27b_it|medical', 'qwen3_8b|medical', 'llama_3_1_8b_instruct|medical',
    'gemma_2_27b_it|code', 'qwen3_8b|code', 'afm_4_5b|code',
]
MARKERS = (('/recovered/', 'R'), ('/Imports/', 'P'), ('/runs/', 'R'),
           ('/reports/', 'R'), ('/verifier/', 'R'), ('/config/', 'R'), ('/code/', 'R'))


def dig(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    """JSON-safe copy; non-finite floats become null after comparisons ran."""
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


def close(a, b, atol=1e-9, rtol=1e-6):
    return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))


# --------------------------------------------------------------------------
# numeric kernels (re-implemented in this owned directory; no analysis imports)
# --------------------------------------------------------------------------
def rank_avg(values):
    """Average-tie ranks (scipy rankdata-equivalent, without scipy)."""
    v = np.asarray(values, dtype=float)
    _, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]


def pearson(x, y):
    a = np.asarray(x, dtype=float) - float(np.mean(x))
    b = np.asarray(y, dtype=float) - float(np.mean(y))
    den = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / den) if den else float('nan')


def spearman(x, y):
    return pearson(rank_avg(x), rank_avg(y))


def eap_theta(matrix, difficulty, discrimination):
    """Direct 41-node Gauss-Legendre EAP logit integral (items x persons)."""
    roots, weights = roots_legendre(41)
    nodes = 4.5 * (roots + 1) - 4.5
    weights = weights * 4.5
    prior = weights * np.exp(-nodes * nodes / 2.0) / np.sqrt(2 * np.pi)
    logistic = np.asarray(discrimination, dtype=float)[:, None] * (
        nodes[None, :] - np.asarray(difficulty, dtype=float)[:, None])
    values, denominators = [], []
    for response in np.asarray(matrix, dtype=float).T:
        probabilities = expit((2 * response[:, None] - 1) * logistic)
        weight = np.prod(probabilities, axis=0) * prior
        denominators.append(float(weight.sum()))
        with np.errstate(invalid='ignore', divide='ignore'):
            values.append(float((weight @ nodes) / weight.sum()))
    return np.array(values), np.array(denominators)


def h4a_metrics(x):
    """Independent SVD/pairwise implementation of the frozen 7 geometry metrics."""
    x = np.asarray(x, dtype=float)
    n, d = x.shape
    z = x - x.mean(0)
    s = np.linalg.svd(z, compute_uv=False)
    lam = s * s / (n - 1)
    total = lam.sum()
    pr = float(total ** 2 / (lam @ lam))
    p = s / (s.sum() + 1e-7)
    p = p[p > 0]
    rankme = float(np.exp(-(p * np.log(p)).sum()))
    p = lam / total
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum())
    pos = lam[lam > lam.max() * len(lam) * np.finfo(float).eps]
    if pos.size < 2:
        alpha = float('nan')
    else:
        lx = np.log(np.arange(1, len(pos) + 1))
        ly = np.log(pos)
        alpha = float(-np.dot(lx - lx.mean(), ly - ly.mean()) / np.dot(lx - lx.mean(), lx - lx.mean()))
    from scipy.spatial.distance import pdist, squareform
    dist = squareform(pdist(x))
    np.fill_diagonal(dist, np.inf)
    nearest = np.partition(dist, 1, axis=1)[:, :2]
    nearest.sort(axis=1)
    ratios = nearest[:, 1][nearest[:, 0] > 0] / nearest[:, 0][nearest[:, 0] > 0]
    ratios = np.sort(ratios[ratios > 1 + 1e-12])
    m = len(ratios)
    k = max(int(.9 * m), 2)
    lx = np.log(ratios[:k])
    ly = -np.log1p(-np.arange(1, k + 1) / m)
    two = float(lx @ ly / (lx @ lx))
    return [two, pr, rankme, float(total / lam[0]), alpha, ent, float((pr - 1) / (d - 1))]


def qbasis(controls):
    cc = np.asarray(controls, dtype=float)
    if cc.ndim == 1:
        cc = cc[:, None]
    design = np.column_stack([np.ones(len(cc))] + [rank_avg(v) for v in cc.T])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    return np.linalg.qr(design, mode='reduced')[0]


def residual(q, v):
    return v - q @ (q.T @ v)


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
        self.groups = {}
        self.claims = {}
        self.t0 = time.time()

    # -- path re-rooting (renamed project / analysis roots) ---------------
    def localize(self, recorded):
        s = str(recorded).replace('\\', '/')
        for root in (self.R, self.P):
            if s.startswith(str(root) + '/'):
                return Path(s)
        for marker, tag in MARKERS:
            i = s.rfind(marker)
            if i >= 0:
                base = self.R if tag == 'R' else self.P
                # Existence is enforced by register() so that a missing data
                # input fails with an explicit "missing required input" error
                # (declared-but-unconsumed assets are not required to exist).
                return base / s[i + 1:]
        raise ValueError(f'cannot re-root recorded path under renamed roots: {s}')

    # -- manifest ---------------------------------------------------------
    def rel(self, path):
        path = Path(path).resolve()
        for root, tag in ((self.R, 'R'), (self.P, 'P')):
            try:
                return f'{tag}/{path.relative_to(root)}'
            except ValueError:
                continue
        raise ValueError(f'consumed path outside analysis roots: {path}')

    def register(self, path, role, level, declared_sha256=None, consumed_as=None):
        path = Path(path)
        rel = self.rel(path)
        if not path.exists():
            raise FileNotFoundError(f'missing required input: {rel} ({path})')
        sha = dig(path)
        if self.pins is not None and rel not in self.pins:
            raise ValueError(f'--expected-manifest pin violation: consumed source not pinned: {rel}')
        if self.pins is not None and self.pins[rel] != sha:
            raise ValueError(
                f'--expected-manifest pin mismatch: {rel} pinned {self.pins[rel]} got {sha}')
        if declared_sha256 is not None and sha != declared_sha256:
            raise ValueError(
                f'declared hash mismatch: {rel} declared {declared_sha256} got {sha}')
        prev = self.manifest.get(rel)
        if prev is not None and prev['sha256'] != sha:
            raise ValueError(f'input mutated mid-run: {rel}')
        group = getattr(self, 'current_group', None)
        groups = list(prev['groups']) if prev and 'groups' in prev else ([prev['group']] if prev and prev.get('group') else [])
        if group and group not in groups:
            groups.append(group)
        self.manifest[rel] = {'path': rel, 'role': role, 'level': level, 'sha256': sha,
                              'bytes': path.stat().st_size, 'consumed_as': consumed_as or prev and prev.get('consumed_as') or role,
                              'group': group, 'groups': groups}
        self._abs[rel] = path
        return path

    def load_bytes(self, path, role, level, declared_sha256=None, consumed_as=None):
        return self.register(path, role, level, declared_sha256, consumed_as).read_bytes()

    def load_json(self, path, role, level, declared_sha256=None, consumed_as=None):
        return json.loads(self.load_bytes(path, role, level, declared_sha256, consumed_as).decode('utf-8'))

    def load_npy(self, path, role, level, declared_sha256=None, consumed_as=None):
        path = self.register(path, role, level, declared_sha256, consumed_as)
        return np.load(path, allow_pickle=False)

    def load_npz(self, path, role, level, declared_sha256=None, consumed_as=None):
        path = self.register(path, role, level, declared_sha256, consumed_as)
        return np.load(path, allow_pickle=False)

    def verify_manifest(self):
        mutated = [rel for rel, entry in self.manifest.items() if dig(self._abs[rel]) != entry['sha256']]
        if mutated:
            raise RuntimeError(f'source mutation detected (writeback guard): {mutated}')
        return True

    # -- evidence rows ----------------------------------------------------
    def check(self, group, name, passed, detail=None, exact=False):
        row = {'group': group, 'check': name, 'passed': bool(passed), 'exact_required': bool(exact)}
        if detail is not None:
            row['detail'] = safe(detail)
        self.checks.append(row)
        return row

    def compare(self, group, claim, pointer, new, reference, artifact, rule=None, exact=False, role=None):
        rule = rule or TOL_RULE
        if exact or isinstance(reference, (str, bool)) or reference is None \
                or isinstance(new, (str, bool)) or new is None:
            matched = (new == reference)
            diff = 0.0 if matched else None
        elif isinstance(reference, dict):
            matched = set(map(str, reference)) == set(map(str, new))
            if matched:
                for key in reference:
                    if not self.compare(group, claim, f'{pointer}/{key}', new[key], reference[key],
                                        artifact, rule=rule, role=role):
                        matched = False
            diff = None
        elif isinstance(reference, list):
            if not isinstance(new, (list, tuple)) or len(new) != len(reference):
                matched = False
                diff = None
            else:
                matched = True
                diffs = []
                for i, ref in enumerate(reference):
                    if not self.compare(group, claim, f'{pointer}/{i}', new[i], ref, artifact,
                                        rule=rule, role=role):
                        matched = False
                    if isinstance(ref, (int, float)) and not isinstance(ref, bool):
                        diffs.append(abs(float(new[i]) - float(ref)))
                diff = max(diffs) if diffs else 0.0
        elif isinstance(reference, (int, float, np.number)) and not isinstance(reference, bool):
            if rule.startswith('exact'):
                matched = (float(new) == float(reference))
            elif 'abs<=1e-8+1e-6' in rule:
                matched = close(new, reference, 1e-8, 1e-6)
            elif 'abs<=1e-10' in rule:
                matched = close(new, reference, 1e-10, 0.0)
            elif 'abs<=5e-5' in rule:
                matched = close(new, reference, 5e-5, 0.0)
            else:
                matched = close(new, reference)
            diff = abs(float(new) - float(reference))
        else:
            matched = (new == reference)
            diff = 0.0 if matched else None
        row = {'group': group, 'claim': claim, 'pointer': pointer,
               'reference_artifact': artifact, 'reference_role': role or 'terminal_aggregate_comparator',
               'rule': 'exact' if exact else rule, 'new': safe(new), 'reference': safe(reference),
               'max_abs_difference': diff, 'matched': bool(matched)}
        self.comparators.append(row)
        return bool(matched)

    def gap(self, group, item, note):
        self.gaps.append({'group': group, 'item': item, 'note': note})

    def dump(self, path, obj):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(safe(obj), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        return path

    # -- claims -----------------------------------------------------------
    def claim(self, claim_id, title, level, fields, config_ids, parameter_ids, extra_pointers=None):
        entry = {'claim_id': claim_id, 'title': title, 'evidence_level': level,
                 'fields': fields, 'config_ids': config_ids, 'parameter_ids': parameter_ids,
                 'field_pointers': {f'/fields{k}': {'artifact': 'claims.json',
                                                    'pointer': f'/{claim_id}/fields{k}'}
                                    for k in _leaf_pointers(fields)}}
        if extra_pointers:
            entry['extra_pointers'] = extra_pointers
        self.claims[claim_id] = entry
        return entry


def _leaf_pointers(obj, prefix=''):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_leaf_pointers(v, f'{prefix}/{k}'))
    else:
        out.append(prefix)
    return out


def read_json(path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------
# group 1: study1 additional replay (P2-A-005, P2-A-007)
# --------------------------------------------------------------------------
def group_study1(rp):
    group = 'study1'
    contract_path = rp.R / 'reports/study1_reuse_additional_contract.json'
    contract = rp.load_json(contract_path, 'declared_contract', 'hash_declaration',
                            consumed_as='declared asset hashes and leaf ids')
    declared = {}
    for asset in contract['assets']:
        local = rp.localize(asset['path'])
        declared[rp.rel(local)] = asset
    results_path = rp.R / 'runs/study1_reuse_v1/additional/results.json'
    vnum_path = rp.R / 'verifier/V_NUM_STUDY1_ADDITIONAL_REUSE.json'
    hist_path = rp.P / 'Imports/geometry/g5no_gates_50.json'

    def dec(path):
        rel = rp.rel(path)
        if rel not in declared:
            raise ValueError(f'consumed study1 asset not declared in contract: {rel}')
        return declared[rel]['sha256']

    # ---- P2-A-005: girth vs mirt on the saved 32-model vectors
    girth_path = rp.P / 'Imports/geometry/theta_hat_girth_20260607.json'
    mirt_path = rp.P / 'Imports/geometry/theta_hat_mirt.json'
    girth = rp.load_json(girth_path, 'raw_vector', 'primary_raw',
                         dec(girth_path), consumed_as='saved32 girth theta vectors')
    mirt = rp.load_json(mirt_path, 'raw_vector', 'primary_raw',
                        dec(mirt_path), consumed_as='saved32 mirt theta vectors')
    a005 = {}
    for dom in ('code', 'math'):
        gt, mt = girth[dom]['theta'], mirt[dom]
        tags = sorted(gt)
        rp.check(group, f'A005/{dom}/model_id_sets_equal', tags == sorted(mt),
                 {'n_models': len(tags)}, exact=True)
        if tags != sorted(mt):
            raise ValueError(f'A005 model id sets differ in {dom}')
        ga = np.array([gt[t] for t in tags], dtype=float)
        ma = np.array([mt[t] for t in tags], dtype=float)
        a005[dom] = {'n_models': len(tags), 'models': tags,
                     'girth_theta': ga.tolist(), 'mirt_theta': ma.tolist(),
                     'n_finite_girth': int(np.isfinite(ga).sum()),
                     'n_finite_mirt': int(np.isfinite(ma).sum()),
                     'rank_rho': spearman(ga, ma)}
    results = rp.load_json(results_path, 'terminal_aggregate_comparator', 'comparator_only',
                           consumed_as='accepted study1 additional producer summary')
    for dom in ('code', 'math'):
        ref = results['A005_saved32_girth_mirt'][dom]
        rp.compare(group, 'P2-A-005', f'/domains/{dom}/models', a005[dom]['models'], ref['models'],
                   'runs/study1_reuse_v1/additional/results.json:/A005_saved32_girth_mirt/' + dom + '/models',
                   exact=True, role='accepted_current_source')
        rp.compare(group, 'P2-A-005', f'/domains/{dom}/girth_theta', a005[dom]['girth_theta'],
                   ref['girth_theta'],
                   'runs/study1_reuse_v1/additional/results.json:/A005_saved32_girth_mirt/' + dom + '/girth_theta',
                   exact=True, role='accepted_current_source')
        rp.compare(group, 'P2-A-005', f'/domains/{dom}/mirt_theta', a005[dom]['mirt_theta'],
                   ref['mirt_theta'],
                   'runs/study1_reuse_v1/additional/results.json:/A005_saved32_girth_mirt/' + dom + '/mirt_theta',
                   exact=True, role='accepted_current_source')
        rp.compare(group, 'P2-A-005', f'/domains/{dom}/rank_rho', a005[dom]['rank_rho'],
                   ref['rank_rho'],
                   'runs/study1_reuse_v1/additional/results.json:/A005_saved32_girth_mirt/' + dom + '/rank_rho',
                   rule=STUDY1_RULE, role='accepted_current_source')
    rp.check(group, 'A005/n_models_per_domain_32',
             all(a005[d]['n_models'] == 32 for d in a005),
             {d: a005[d]['n_models'] for d in a005}, exact=True)
    rp.claim('P2-A-005', 'Girth versus mirt rank agreement',
             'RETAINED_EXISTING_DATA_REPLAY: fresh Spearman from the two saved 32-model theta vectors',
             a005,
             {'contract': 'reports/study1_reuse_additional_contract.json',
              'leaf_id': 'A005_girth_mirt_saved32'},
             {'rank_rho': 'average-tie Spearman on sorted exact-intersection model order; no seed',
              'models_sha256': hashlib.sha256(json.dumps(a005['code']['models']).encode()).hexdigest()})

    # ---- P2-A-007: true-theta item split-half reliability
    a007 = {'domains': {}, 'n_models': 50, 'configured': {'split_seed': 20260605,
            'quadrature_n': 41, 'quadrature_bounds': [-4.5, 4.5]}}
    library_shas = set()
    for dom in ('code', 'math'):
        half_theta = {}
        for half in ('a', 'b'):
            csv_path = rp.P / f'Imports/geometry/theta_splithalf_in/resp_{dom}_half{half}.csv'
            fit_path = rp.R / f'runs/study1_reuse_v1/additional/fit_{dom}_{half}.json'
            raw = rp.load_bytes(csv_path, 'raw_vector', 'primary_raw', dec(csv_path),
                                consumed_as=f'{dom}/{half} item-half response CSV')
            import csv as _csv
            import io as _io
            rows = list(_csv.reader(_io.StringIO(raw.decode('utf-8'))))
            fit = rp.load_json(fit_path, 'saved_fit_parameters', 'saved_fit',
                               consumed_as=f'{dom}/{half} saved girth fit parameters and theta')
            library_shas.update(fit['library_sources'].values())
            model_order = [r[0] for r in rows[1:]]
            item_order = rows[0][1:]
            x = np.array([r[1:] for r in rows[1:]], dtype=int)
            keep = (x.sum(0) > 0) & (x.sum(0) < len(model_order))
            kept_items = [v for v, k in zip(item_order, keep) if k]
            if model_order != fit['model_order']:
                raise ValueError(f'A007 model order mismatch {dom}/{half}')
            if kept_items != fit['kept_item_order']:
                raise ValueError(f'A007 kept item mismatch {dom}/{half}')
            rp.compare(group, 'P2-A-007', f'/domains/{dom}/{half}/model_order', model_order,
                       fit['model_order'], f'runs/study1_reuse_v1/additional/fit_{dom}_{half}.json:/model_order',
                       exact=True, role='saved_fit_artifact')
            rp.compare(group, 'P2-A-007', f'/domains/{dom}/{half}/kept_item_order', kept_items,
                       fit['kept_item_order'], f'runs/study1_reuse_v1/additional/fit_{dom}_{half}.json:/kept_item_order',
                       exact=True, role='saved_fit_artifact')
            rp.check(group, f'A007/{dom}/{half}/n_models_50', len(model_order) == 50,
                     {'n_models': len(model_order), 'n_items': len(item_order),
                      'n_items_kept': int(keep.sum())}, exact=True)
            theta, den = eap_theta(x[:, keep].T, fit['Difficulty'], fit['Discrimination'])
            if not (np.isfinite(theta).all() and np.all(den > 0)):
                raise ValueError(f'A007 nonfinite EAP {dom}/{half}')
            rp.compare(group, 'P2-A-007', f'/domains/{dom}/{half}/theta_eap', theta.tolist(),
                       fit['theta'], f'runs/study1_reuse_v1/additional/fit_{dom}_{half}.json:/theta',
                       rule=STUDY1_RULE, role='saved_fit_artifact')
            half_theta[half] = {'theta': theta.tolist(),
                                'n_models': len(model_order), 'n_items': len(item_order),
                                'n_items_kept': int(keep.sum()), 'kept_items': kept_items,
                                'all_theta_finite': bool(np.isfinite(theta).all()),
                                'min_likelihood_denominator': float(den.min())}
        rho_half = spearman(half_theta['a']['theta'], half_theta['b']['theta'])
        sb = 2 * rho_half / (1 + rho_half)
        a007['domains'][dom] = {**half_theta, 'half_rank_rho': rho_half,
                                'spearman_brown_reliability': sb}
        ref = results['A007_half_theta']['domains'][dom]
        rp.compare(group, 'P2-A-007', f'/domains/{dom}/n_models', 50, ref['n_models'],
                   f'runs/study1_reuse_v1/additional/results.json:/A007_half_theta/domains/{dom}/n_models',
                   exact=True, role='accepted_current_source')
        rp.compare(group, 'P2-A-007', f'/domains/{dom}/kept_items',
                   [half_theta['a']['n_items_kept'], half_theta['b']['n_items_kept']], ref['kept_items'],
                   f'runs/study1_reuse_v1/additional/results.json:/A007_half_theta/domains/{dom}/kept_items',
                   exact=True, role='accepted_current_source')
        rp.compare(group, 'P2-A-007', f'/domains/{dom}/half_rank_rho', rho_half, ref['half_rank_rho'],
                   f'runs/study1_reuse_v1/additional/results.json:/A007_half_theta/domains/{dom}/half_rank_rho',
                   rule=STUDY1_RULE, role='accepted_current_source')
        rp.compare(group, 'P2-A-007', f'/domains/{dom}/spearman_brown_reliability', sb,
                   ref['spearman_brown_reliability'],
                   f'runs/study1_reuse_v1/additional/results.json:/A007_half_theta/domains/{dom}/spearman_brown_reliability',
                   rule=STUDY1_RULE, role='accepted_current_source')
    vnum = rp.load_json(vnum_path, 'independent_review_comparator', 'comparator_only',
                        consumed_as='accepted independent numeric review of study1 additional')
    for dom, value in vnum['A007_reliability'].items():
        rp.compare(group, 'P2-A-007', f'/domains/{dom}/spearman_brown_reliability',
                   a007['domains'][dom]['spearman_brown_reliability'], value,
                   'verifier/V_NUM_STUDY1_ADDITIONAL_REUSE.json:/A007_reliability/' + dom,
                   rule=STUDY1_RULE, role='independent_review_comparator')
    rp.compare(group, 'P2-A-007', '/independent_review/producer_code_sha256',
               results['code_sha256'], vnum['producer_code_sha256'],
               'verifier/V_NUM_STUDY1_ADDITIONAL_REUSE.json:/producer_code_sha256',
               exact=True, role='independent_review_comparator')
    rp.compare(group, 'P2-A-007', '/independent_review/verdict', 'PASS_INDEPENDENT_NUMERIC',
               vnum['verdict'], 'verifier/V_NUM_STUDY1_ADDITIONAL_REUSE.json:/verdict',
               exact=True, role='independent_review_comparator')
    hist = rp.load_json(hist_path, 'historical_comparator', 'comparator_only', dec(hist_path),
                        consumed_as='historical 4-decimal display comparator for A007')
    a007['historical_displayed'] = hist['theta_split_half_rel']
    a007['historical_raw_r'] = hist['theta_split_half_raw_r']
    a007['historical_seed'] = hist['seed']
    for dom in ('code', 'math'):
        rp.compare(group, 'P2-A-007', f'/domains/{dom}/spearman_brown_reliability',
                   a007['domains'][dom]['spearman_brown_reliability'], hist['theta_split_half_rel'][dom],
                   'Imports/geometry/g5no_gates_50.json:/theta_split_half_rel/' + dom,
                   rule=DISPLAY_RULE, role='historical_comparator')
        rp.compare(group, 'P2-A-007', f'/domains/{dom}/half_rank_rho',
                   a007['domains'][dom]['half_rank_rho'], hist['theta_split_half_raw_r'][dom],
                   'Imports/geometry/g5no_gates_50.json:/theta_split_half_raw_r/' + dom,
                   rule=DISPLAY_RULE, role='historical_comparator')
    rp.compare(group, 'P2-A-007', '/historical_seed', hist['seed'], 20260605,
               'Imports/geometry/g5no_gates_50.json:/seed', exact=True, role='historical_comparator')
    rp.claim('P2-A-007', 'True theta item split-half reliability',
             'RETAINED_EXISTING_DATA_REPLAY: four saved response CSVs + saved girth fit parameters, '
             'direct 41-node EAP integral, Spearman-Brown, no refit',
             a007,
             {'contract': 'reports/study1_reuse_additional_contract.json',
              'leaf_id': 'A007_theta_split_half50'},
             {'split_seed': 20260605, 'quadrature_n': 41, 'quadrature_bounds': [-4.5, 4.5],
              'girth_library_sha256': sorted(library_shas)})
    return {'status': 'PRODUCED', 'claims': ['P2-A-005', 'P2-A-007']}


# --------------------------------------------------------------------------
# group 2: empirical cloud-size NBIAS (7 x P2-C-NBIAS-*)
# --------------------------------------------------------------------------
def group_nbias(rp):
    group = 'nbias'
    cfg_path = rp.R / 'config/STUDY2_FEATURE_REUSE_V1.json'
    cfg = rp.load_json(cfg_path, 'config', 'configuration',
                       consumed_as='H4A cells/grid/seed/reps declaration')
    h4a = cfg['h4a']
    cells = list(h4a['cells'])
    grid = [int(v) for v in h4a['grid']]
    reps = int(h4a['reps'])
    seed = int(h4a['seed'])
    rp.check(group, 'H4A/configured_cells_9', len(cells) == 9 and cells == DEFAULT_H4A_CELL_ORDER,
             {'cells': cells}, exact=True)
    rp.check(group, 'H4A/configured_grid', grid == [25, 50, 75, 100, 150, 200, 300, 400, 600, 900] and reps == 6,
             {'grid': grid, 'reps': reps, 'seed': seed}, exact=True)
    streams = {'seed': seed, 'grid': grid, 'reps': reps, 'cells': cells,
               'role': 'fixed configured draw-stream metadata; numeric statistics come from the stored per-draw values'}
    fresh_curve = {}
    feature_checks = {'n_feature_values_compared': 0, 'max_abs_difference': 0.0,
                      'max_source_rule_excess': 0.0,
                      'max_abs_difference_location': None, 'max_abs_difference_metric': None,
                      'max_abs_difference_cell': None, 'max_abs_difference_size': None}
    stat_checks = {'n_statistics_compared': 0, 'max_abs_difference': 0.0,
                   'max_source_rule_excess': 0.0, 'location': None}
    rng = np.random.default_rng(seed)
    for cell in cells:
        slug = cell.replace('|', '__')
        obs = rp.load_json(rp.R / f'runs/study2_reuse_v1/observations/{slug}.json',
                           'raw_vector_metadata', 'primary_raw',
                           consumed_as=f'{cell} L0 path/hash/n_rows declaration')
        l0_path = rp.localize(obs['l0_path'])
        x_raw = rp.load_npy(l0_path, 'raw_vector', 'primary_raw',
                            consumed_as=f'{cell} L0 activation cloud (raw vectors)')
        payload_sha = hashlib.sha256(np.asarray(x_raw, dtype='<f4', order='C').tobytes()).hexdigest()
        if payload_sha != obs['l0_payload_sha256']:
            raise ValueError(
                f'L0 payload hash mismatch for {cell}: declared {obs["l0_payload_sha256"]} got {payload_sha} '
                f'({rp.rel(l0_path)})')
        rp.check(group, f'H4A/{cell}/l0_payload_sha256_declared_match', True,
                 {'declared': obs['l0_payload_sha256'], 'recomputed': payload_sha}, exact=True)
        x = np.asarray(x_raw, dtype=np.float64)
        rp.check(group, f'H4A/{cell}/n_rows_and_finiteness',
                 len(x) == obs['n_rows'] and bool(np.isfinite(x).all()),
                 {'n_rows': len(x), 'declared': obs['n_rows'], 'd_model': x.shape[1]}, exact=True)
        draw = rp.load_json(rp.R / f'runs/study2_reuse_v1/features/h4a/{slug}.json',
                            'per_draw_values', 'saved_draw',
                            consumed_as=f'{cell} stored draw indices and per-rep feature values')
        expected_indices = {}
        for size in grid:
            if size > len(x) - 1:
                continue
            expected_indices[str(size)] = [rng.choice(len(x), size, replace=False).tolist() for _ in range(reps)]
        stored_sizes = sorted(int(k) for k in draw['curve'])
        rp.check(group, f'H4A/{cell}/grid_sizes_match_configured_stream', stored_sizes == sorted(int(k) for k in expected_indices),
                 {'sizes': stored_sizes}, exact=True)
        full_fresh = h4a_metrics(x)
        rp.compare(group, 'P2-C-NBIAS', f'/{cell}/full_feat', full_fresh,
                   [draw['full_feat'][m] for m in METRICS],
                   f'runs/study2_reuse_v1/features/h4a/{slug}.json:/full_feat', rule=FEATURE_RULE,
                   role='saved_draw_artifact')
        fresh_curve[cell] = {'n_full': len(x), 'd_model': int(x.shape[1]),
                             'full_feat': dict(zip(METRICS, full_fresh)), 'curve': {}}
        for size in stored_sizes:
            stored = draw['curve'][str(size)]
            rp.check(group, f'H4A/{cell}/{size}/stored_indices_equal_configured_stream',
                     [list(map(int, ix)) for ix in stored['indices']] == expected_indices[str(size)],
                     {'n_reps': len(stored['indices'])}, exact=True)
            per_rep = []
            for rep, ix in enumerate(stored['indices']):
                values = h4a_metrics(x[np.asarray(ix, dtype=int)])
                ref = [float(v) for v in stored['per_rep_features'][rep]]
                per_rep.append(values)
                for j, name in enumerate(METRICS):
                    d = abs(values[j] - ref[j])
                    excess = d - (1e-8 + 1e-6 * abs(ref[j]))
                    feature_checks['n_feature_values_compared'] += 1
                    feature_checks['max_source_rule_excess'] = max(feature_checks['max_source_rule_excess'], excess)
                    if d > feature_checks['max_abs_difference']:
                        feature_checks.update({'max_abs_difference': d,
                                               'max_abs_difference_location': f'{cell}/{size}/rep{rep}/{name}',
                                               'max_abs_difference_metric': name,
                                               'max_abs_difference_cell': cell,
                                               'max_abs_difference_size': size})
            values = np.asarray(per_rep)
            mean = values.mean(0)
            std = values.std(0, ddof=1)
            bias = (mean - np.asarray(full_fresh)) / np.abs(np.asarray(full_fresh))
            fresh_curve[cell]['curve'][str(size)] = {
                'per_rep_features': per_rep,
                'statistics': {name: {'mean': float(mean[j]), 'std': float(std[j]), 'relbias': float(bias[j])}
                               for j, name in enumerate(METRICS)}}
            for name in METRICS:
                for field, value in (('mean', mean[METRICS.index(name)]), ('std', std[METRICS.index(name)]),
                                     ('relbias', bias[METRICS.index(name)])):
                    ref_value = stored['statistics'][name][field]
                    d = abs(float(value) - float(ref_value))
                    excess = d - (1e-8 + 1e-6 * abs(float(ref_value)))
                    stat_checks['n_statistics_compared'] += 1
                    stat_checks['max_source_rule_excess'] = max(stat_checks['max_source_rule_excess'], excess)
                    if d > stat_checks['max_abs_difference']:
                        stat_checks.update({'max_abs_difference': d,
                                            'location': f'{cell}/{size}/{name}/{field}'})
    rp.check(group, 'H4A/per_rep_feature_values_within_source_rule',
             feature_checks['max_source_rule_excess'] <= 0.0, feature_checks, exact=True)
    rp.check(group, 'H4A/per_size_statistics_within_source_rule',
             stat_checks['max_source_rule_excess'] <= 0.0, stat_checks, exact=True)

    classification = {}
    for name in METRICS:
        grouped = {size: [abs(fresh_curve[c]['curve'][str(size)]['statistics'][name]['relbias'])
                          for c in cells if str(size) in fresh_curve[c]['curve']] for size in grid}
        a, b, dd = [float(np.median(grouped[n])) for n in (50, 150, max(n for n in grouped if grouped[n]))]
        nmax = max(n for n in grouped if grouped[n])
        shrink = a / dd
        flat, strong = b < .05, a > .15 and shrink > 2
        classification[name] = {
            'median_abs_relbias_N50': a, 'median_abs_relbias_N150': b,
            'median_abs_relbias_Nmax': dd, 'shrink_ratio_N50_over_Nmax': shrink,
            'flat_by_N150': flat, 'strong_monotonic_N_scaling': strong,
            'verdict': 'N_ROBUST' if flat and not strong else 'N_CONFOUNDED' if strong else 'BORDERLINE',
            'largest_populated_N': nmax, 'n_cells_at_largest_N': len(grouped[nmax])}
    aout = rp.load_json(rp.R / 'runs/study2_reuse_v1/features/H4A_RECOMPUTED.json',
                        'terminal_aggregate_comparator', 'comparator_only',
                        consumed_as='accepted H4A aggregate classification (comparator only)')
    for name in METRICS:
        ref = aout['classification'][name]
        for field in ('median_abs_relbias_N50', 'median_abs_relbias_N150', 'median_abs_relbias_Nmax',
                      'shrink_ratio_N50_over_Nmax'):
            rp.compare(group, NBIAS_CLAIM[name], f'/classification/{name}/{field}',
                       classification[name][field], ref[field],
                       f'runs/study2_reuse_v1/features/H4A_RECOMPUTED.json:/classification/{name}/{field}',
                       rule=FEATURE_RULE, role='accepted_current_source')
        for field in ('flat_by_N150', 'strong_monotonic_N_scaling', 'verdict'):
            rp.compare(group, NBIAS_CLAIM[name], f'/classification/{name}/{field}',
                       classification[name][field], ref[field],
                       f'runs/study2_reuse_v1/features/H4A_RECOMPUTED.json:/classification/{name}/{field}',
                       exact=True, role='accepted_current_source')
        rp.claim(NBIAS_CLAIM[name], f'Cloud-size sensitivity: {name}',
                 'RETAINED_EXISTING_DATA_REPLAY: fresh per-draw metric values on the nine real L0 clouds '
                 'at every stored draw index; classification recomputed; not the C6 synthetic branch',
                 {'classification': classification[name], 'cells_used': cells,
                  'headline_relbias': {str(n): [fresh_curve[c]['curve'][str(n)]['statistics'][name]['relbias']
                                                for c in cells if str(n) in fresh_curve[c]['curve']]
                                       for n in (50, 150, classification[name]['largest_populated_N'])}},
                 {'config': 'config/STUDY2_FEATURE_REUSE_V1.json',
                  'h4a_section': h4a['classification']},
                 {'seed': seed, 'grid': grid, 'reps': reps,
                  'role': 'fixed configured draw-stream/classification constants (metadata), '
                          'numeric values recomputed from stored per-draw arrays'},
                 extra_pointers={'full_per_draw_curves':
                                 {'artifact': 'groups/nbias_h4a.json',
                                  'pointer': f'/fresh_cell_curves/*/curve/*/statistics/{name}'}})
    vnum = rp.load_json(rp.R / 'verifier/V_NUM_STUDY2_FEATURES_REUSE.json',
                        'independent_review_comparator', 'comparator_only',
                        consumed_as='accepted independent review of H4A curves/classifications')
    n_match = 0
    for name, ref in vnum['classification'].items():
        mine = classification[name]
        ok = (mine['verdict'] == ref['verdict']
              and mine['largest_populated_N'] == ref['largest_populated_N']
              and mine['n_cells_at_largest_N'] == ref['n_cells_at_largest_N'])
        n_match += int(ok)
        rp.compare(group, NBIAS_CLAIM[name], f'/classification/{name}/verdict', mine['verdict'],
                   ref['verdict'], 'verifier/V_NUM_STUDY2_FEATURES_REUSE.json:/classification/' + name + '/verdict',
                   exact=True, role='independent_review_comparator')
        rp.compare(group, NBIAS_CLAIM[name], f'/classification/{name}/largest_populated_N',
                   mine['largest_populated_N'], ref['largest_populated_N'],
                   'verifier/V_NUM_STUDY2_FEATURES_REUSE.json:/classification/' + name + '/largest_populated_N',
                   exact=True, role='independent_review_comparator')
        rp.compare(group, NBIAS_CLAIM[name], f'/classification/{name}/n_cells_at_largest_N',
                   mine['n_cells_at_largest_N'], ref['n_cells_at_largest_N'],
                   'verifier/V_NUM_STUDY2_FEATURES_REUSE.json:/classification/' + name + '/n_cells_at_largest_N',
                   exact=True, role='independent_review_comparator')
    rp.compare(group, 'P2-C-NBIAS', '/independent_review/classification_agreement', n_match,
               len(vnum['classification']), 'verifier/V_NUM_STUDY2_FEATURES_REUSE.json:/classification',
               exact=True, role='independent_review_comparator')
    rp.compare(group, 'P2-C-NBIAS', '/independent_review/verdict', 'PASS_INDEPENDENT_INTERMEDIATE_VALIDATION',
               vnum['verdict'], 'verifier/V_NUM_STUDY2_FEATURES_REUSE.json:/verdict',
               exact=True, role='independent_review_comparator')
    ledger = _ledger(rp)
    for name in METRICS:
        rec = ledger['records_by_id'][NBIAS_CLAIM[name]]
        fields = rec['historical_manifest_references']['historical_outputs'][0]['fields']
        ref = fields['/classification/' + name]
        for field in ('median_abs_relbias_N50', 'median_abs_relbias_N150', 'median_abs_relbias_Nmax',
                      'shrink_ratio_N50_over_Nmax'):
            rp.compare(group, NBIAS_CLAIM[name], f'/classification/{name}/{field}',
                       classification[name][field], ref[field],
                       'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json:/records/'
                       f'{rec["_index"]}/historical_manifest_references/historical_outputs/0/fields/classification/{name}/{field}',
                       rule=TOL_RULE, role='historical_comparator')
        for field in ('flat_by_N150', 'strong_monotonic_N_scaling', 'verdict'):
            rp.compare(group, NBIAS_CLAIM[name], f'/classification/{name}/{field}',
                       classification[name][field], ref[field],
                       'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json:/records/'
                       f'{rec["_index"]}/historical_manifest_references/historical_outputs/0/fields/classification/{name}/{field}',
                       exact=True, role='historical_comparator')
        rp.compare(group, NBIAS_CLAIM[name], '/seed', seed, fields['/seed'],
                   'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json:/records/'
                   f'{rec["_index"]}/historical_manifest_references/historical_outputs/0/fields/seed',
                   exact=True, role='historical_comparator')
    rp.gap(group, 'H4A_sampling_scope',
           'nine saved cells only; sample-size sensitivity classification is limited to these clouds and tested sizes; '
           'drawn from the stored per-draw index streams (draw stream re-derived from the configured seed and compared exactly)')
    return {'status': 'PRODUCED', 'n_metrics': len(METRICS), 'claims': [NBIAS_CLAIM[m] for m in METRICS],
            'fresh_curve': fresh_curve, 'feature_checks': feature_checks, 'stat_checks': stat_checks}


def _ledger(rp):
    if getattr(rp, '_ledger_cache', None) is None:
        doc = rp.load_json(rp.R / 'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json',
                           'claim_ledger_pointer_source', 'comparator_only',
                           consumed_as='retained claim ids, historical field pointers and accepted evidence paths')
        by_id = {}
        for i, rec in enumerate(doc['records']):
            rec = dict(rec)
            rec['_index'] = i
            by_id[rec['result_id']] = rec
        rp._ledger_cache = {'doc': doc, 'records_by_id': by_id}
    return rp._ledger_cache


# --------------------------------------------------------------------------
# group 3: Study2 code shadow-only recurrence (P2-C-003)
# --------------------------------------------------------------------------
CODE_KEYS = ['h3__code__eff_rank_pr'] + [f'{"native" if p == "native" else p}__code__{m}'
                                         for p in ('native', 'rarefied') for m in CODE_METRICS]


def group_code_shadow(rp):
    group = 'code_shadow'
    cfg_path = rp.R / 'config/STUDY2_INFERENCE_REUSE_V1.json'
    cfg = rp.load_json(cfg_path, 'config', 'configuration',
                       consumed_as='study2 inference bootstrap/permutation declarations')
    boot = cfg['inference']['bootstrap']
    perm = cfg['inference']['permutation']
    n_boot, seed_boot = int(boot['replicates']), int(boot['seed'])
    n_perm, seed_perm = int(perm['replicates']), int(perm['seed'])
    ledger = _ledger(rp)
    historical = ledger['records_by_id']['P2-C-003']['historical_manifest_references']['historical_outputs'][0]['fields']
    fresh = {}
    for key in CODE_KEYS:
        draw_path = rp.R / f'runs/study2_reuse_v1/inference/draws/{key}.npz'
        z = rp.load_npz(draw_path, 'per_draw_values', 'saved_draw',
                        consumed_as=f'{key} per-draw x/theta/controls/families and resample index draws')
        cell = rp.load_json(rp.R / f'runs/study2_reuse_v1/inference/cells/{key}.json',
                            'terminal_aggregate_comparator', 'comparator_only',
                            consumed_as=f'{key} accepted inference summary (comparator only)')
        x = np.asarray(z['x'], dtype=float)
        y = np.asarray(z['theta'], dtype=float)
        controls = np.asarray(z['controls'], dtype=float)
        families = np.asarray(z['families'])
        fam_draws = np.asarray(z['family_draws'], dtype=int)
        perm_idx = np.asarray(z['permutation_indices'], dtype=int)
        rp.check(group, f'{key}/counts',
                 len(x) == cell['n_models'] == 48 and len(set(families.tolist())) == cell['n_families'] == 13
                 and fam_draws.shape == (n_boot, len(set(families.tolist())))
                 and perm_idx.shape == (n_perm, len(x)) and bool(np.isfinite(x).all())
                 and bool(np.isfinite(y).all()) and bool(np.isfinite(controls).all()),
                 {'n_models': len(x), 'n_families': len(set(families.tolist())),
                  'family_draws_shape': list(fam_draws.shape), 'perm_shape': list(perm_idx.shape)}, exact=True)
        # Re-derive the resample streams from the configured seeds and require exact agreement.
        rng_b = np.random.default_rng(seed_boot)
        nf = len(set(families.tolist()))
        rp.check(group, f'{key}/family_draws_from_configured_seed',
                 np.array_equal(rng_b.integers(0, nf, size=(n_boot, nf)), fam_draws), {'seed': seed_boot}, exact=True)
        rng_p = np.random.default_rng(seed_perm)
        rp.check(group, f'{key}/permutation_draws_from_configured_seed',
                 np.array_equal(np.stack([rng_p.permutation(len(y)) for _ in range(n_perm)]), perm_idx),
                 {'seed': seed_perm}, exact=True)
        # Fresh statistics: own rank + QR residualization, mirrored from the accepted
        # verifier formulas (numeric_reuse_utils.rho / verify_study2_inference_reuse.stats).
        q = qbasis(controls)
        xr = residual(q, rank_avg(x))
        yr = residual(q, rank_avg(y))
        raw_rho = pearson(rank_avg(x), rank_avg(y))
        point = pearson(xr, yr)
        blocks = [np.flatnonzero(families == f) for f in sorted(set(families.tolist()))]

        def boot_for(control_values):
            out = np.full(len(fam_draws), np.nan)
            for i, row in enumerate(fam_draws):
                ix = np.concatenate([blocks[k] for k in row])
                cc = np.asarray(control_values)[ix]
                qq = qbasis(cc)
                if qq is None:
                    continue
                xx = residual(qq, rank_avg(x[ix]))
                yy = residual(qq, rank_avg(y[ix]))
                if np.unique(x[ix]).size < 2:
                    continue
                xx = xx - xx.mean()
                yy = yy - yy.mean()
                den = np.linalg.norm(xx) * np.linalg.norm(yy)
                if den:
                    out[i] = float((xx @ yy) / den)
            valid = out[np.isfinite(out)]
            ci = np.percentile(valid, [2.5, 97.5])
            return {'rho': point if control_values is controls else float('nan'), 'ci': ci.tolist(),
                    'n_valid': int(len(valid)), 'n_invalid': int(len(out) - len(valid)),
                    'ci_excludes_zero': bool(ci[0] > 0 or ci[1] < 0)}

        labels = {f: i for i, f in enumerate(sorted(set(families.tolist())))}
        bare_indicator = np.array([labels[f] for f in families], dtype=float)
        bare = boot_for(bare_indicator)
        bare['rho'] = pearson(residual(qbasis(bare_indicator), rank_avg(x)),
                              residual(qbasis(bare_indicator), rank_avg(y)))
        controlled = boot_for(controls)
        controlled['rho'] = point
        yranks = rank_avg(y)[perm_idx]
        yp = yranks - (yranks @ q) @ q.T
        yp = yp - yp.mean(1, keepdims=True)
        yp = yp / np.linalg.norm(yp, axis=1, keepdims=True)
        xc = xr - xr.mean()
        perms = yp @ (xc / np.linalg.norm(xc))
        tails = int(np.count_nonzero(np.abs(perms) >= abs(point) - 1e-12))
        p_value = (tails + 1) / (len(perms) + 1)
        verdict = ('CONFIRM' if controlled['ci_excludes_zero'] and bare['ci_excludes_zero'] and p_value < .05
                   else 'SHADOW-ONLY' if bare['ci_excludes_zero'] and not controlled['ci_excludes_zero']
                   else 'REFUTE')
        fresh[key] = {'domain': 'code', 'feature': cell['feature'], 'kind': cell['kind'],
                      'n_models': len(x), 'n_families': nf, 'raw_rho': raw_rho,
                      'partial_rho': point, 'bare_bootstrap': bare, 'controlled_bootstrap': controlled,
                      'controlled_ci': controlled['ci'], 'permutation_p': p_value,
                      'permutation': {'p': p_value, 'exceed_count': tails, 'n_valid': len(perms), 'n_invalid': 0},
                      'verdict': verdict}
        for field, ref in (('raw_rho', cell['raw_rho']), ('partial_rho', cell['partial_rho']),
                           ('controlled_ci', cell['controlled_ci']), ('permutation_p', cell['permutation_p']),
                           ('verdict', cell['verdict'])):
            rp.compare(group, 'P2-C-003', f'/{key}/{field}', fresh[key][field], ref,
                       f'runs/study2_reuse_v1/inference/cells/{key}.json:/{field}',
                       rule=INFERENCE_RULE, exact=(field == 'verdict'), role='accepted_current_source')
        for kind in ('bare', 'controlled'):
            src = cell[f'{kind}_bootstrap']
            for field in ('rho', 'ci', 'n_valid', 'n_invalid', 'ci_excludes_zero'):
                rp.compare(group, 'P2-C-003', f'/{key}/{kind}_bootstrap/{field}',
                           fresh[key][f'{kind}_bootstrap'][field], src[field],
                           f'runs/study2_reuse_v1/inference/cells/{key}.json:/{kind}_bootstrap/{field}',
                           rule=INFERENCE_RULE, exact=isinstance(src[field], (bool, int)),
                           role='accepted_current_source')
        rp.compare(group, 'P2-C-003', f'/{key}/permutation/exceed_count',
                   fresh[key]['permutation']['exceed_count'], cell['permutation']['exceed_count'],
                   f'runs/study2_reuse_v1/inference/cells/{key}.json:/permutation/exceed_count',
                   exact=True, role='accepted_current_source')
        rp.compare(group, 'P2-C-003', f'/{key}/n_models', len(x), cell['n_models'], 'cell n_models',
                   exact=True, role='accepted_current_source')
    primary = ['h3__code__eff_rank_pr', 'native__code__eff_rank_pr', 'rarefied__code__eff_rank_pr']
    # The historical raw_rho refers to the panel value that also recurs in the
    # native H4 panel; the rarefied panel has a genuinely different raw value.
    for key in ('h3__code__eff_rank_pr', 'native__code__eff_rank_pr'):
        rp.compare(group, 'P2-C-003', f'/{key}/raw_rho', fresh[key]['raw_rho'], historical['/raw_rho'],
                   'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json:/records/'
                   f'{ledger["records_by_id"]["P2-C-003"]["_index"]}/historical_manifest_references/'
                   'historical_outputs/0/fields/raw_rho', rule=INFERENCE_RULE, role='historical_comparator')
    rp.compare(group, 'P2-C-003', '/h3__code__eff_rank_pr/partial_rho',
               fresh['h3__code__eff_rank_pr']['partial_rho'], historical['/partial_rho_controlling_C'],
               'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json:/records/'
               f'{ledger["records_by_id"]["P2-C-003"]["_index"]}/historical_manifest_references/'
               'historical_outputs/0/fields/partial_rho_controlling_C', rule=INFERENCE_RULE,
               role='historical_comparator')
    rp.compare(group, 'P2-C-003', '/h3__code__eff_rank_pr/controlled_ci',
               fresh['h3__code__eff_rank_pr']['controlled_ci'], historical['/controlled_cluster_ci'],
               'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json:/records/'
               f'{ledger["records_by_id"]["P2-C-003"]["_index"]}/historical_manifest_references/'
               'historical_outputs/0/fields/controlled_cluster_ci', rule=INFERENCE_RULE,
               role='historical_comparator')
    rp.compare(group, 'P2-C-003', '/h3__code__eff_rank_pr/permutation_p',
               fresh['h3__code__eff_rank_pr']['permutation_p'], historical['/permutation_p'],
               'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json:/records/'
               f'{ledger["records_by_id"]["P2-C-003"]["_index"]}/historical_manifest_references/'
               'historical_outputs/0/fields/permutation_p', rule=INFERENCE_RULE,
               role='historical_comparator')
    rp.compare(group, 'P2-C-003', '/h3__code__eff_rank_pr/verdict',
               fresh['h3__code__eff_rank_pr']['verdict'], historical['/verdict'],
               'reports/C17_FINAL_EVIDENCE_LEDGER_v1.json:/records/'
               f'{ledger["records_by_id"]["P2-C-003"]["_index"]}/historical_manifest_references/'
               'historical_outputs/0/fields/verdict', exact=True, role='historical_comparator')
    # Recurrence: the shadow-only pattern must repeat across the independent code panels.
    recurrence = {
        'verdicts': {k: fresh[k]['verdict'] for k in primary},
        'verdicts_all_shadow_only': all(fresh[k]['verdict'] == 'SHADOW-ONLY' for k in primary),
        'bare_ci_excludes_zero': {k: fresh[k]['bare_bootstrap']['ci_excludes_zero'] for k in primary},
        'controlled_ci_excludes_zero': {k: fresh[k]['controlled_bootstrap']['ci_excludes_zero'] for k in primary},
        'partial_rho_h3': fresh['h3__code__eff_rank_pr']['partial_rho'],
        'partial_rho_native': fresh['native__code__eff_rank_pr']['partial_rho'],
        'partial_rho_rarefied': fresh['rarefied__code__eff_rank_pr']['partial_rho'],
        'abs_delta_h3_vs_native': abs(fresh['h3__code__eff_rank_pr']['partial_rho']
                                      - fresh['native__code__eff_rank_pr']['partial_rho']),
        'abs_delta_native_vs_rarefied': abs(fresh['native__code__eff_rank_pr']['partial_rho']
                                            - fresh['rarefied__code__eff_rank_pr']['partial_rho']),
        'other_code_cells_S1': {k: fresh[k]['verdict'] for k in CODE_KEYS
                                if k not in primary and fresh[k]['verdict'] == 'SHADOW-ONLY'},
        'other_code_cells_S2': {k: fresh[k]['verdict'] for k in CODE_KEYS
                                if k not in primary and fresh[k]['verdict'] != 'SHADOW-ONLY'},
    }
    rp.check(group, 'P2-C-003/recurrence_verdicts_shadow_only',
             recurrence['verdicts_all_shadow_only']
             and all(recurrence['bare_ci_excludes_zero'].values())
             and not any(recurrence['controlled_ci_excludes_zero'].values()),
             recurrence['verdicts'], exact=True)
    rp.check(group, 'P2-C-003/recurrence_h3_equals_native_partial',
             recurrence['abs_delta_h3_vs_native'] <= 1e-12,
             {'abs_delta': recurrence['abs_delta_h3_vs_native'],
              'rule': 'abs<=1e-12 panel-recurrence consistency (identical stored x stream)'}, exact=True)
    vnum = rp.load_json(rp.R / 'verifier/V_NUM_STUDY2_INFERENCE_REUSE.json',
                        'independent_review_comparator', 'comparator_only',
                        consumed_as='accepted all-draw independent review of the study2 inference cells')
    rp.compare(group, 'P2-C-003', '/independent_review/verdict',
               'PASS_INDEPENDENT_ALL_DRAW_VERIFICATION', vnum['verdict'],
               'verifier/V_NUM_STUDY2_INFERENCE_REUSE.json:/verdict', exact=True,
               role='independent_review_comparator')
    rp.compare(group, 'P2-C-003', '/independent_review/h3_code_partial_rho',
               fresh['h3__code__eff_rank_pr']['partial_rho'],
               vnum['computed']['h3__code__eff_rank_pr']['controlled']['rho'],
               'verifier/V_NUM_STUDY2_INFERENCE_REUSE.json:/computed/h3__code__eff_rank_pr/controlled/rho',
               rule=INFERENCE_RULE, role='independent_review_comparator')
    rp.claim('P2-C-003', 'Study2 code shadow-only recurrence',
             'RETAINED_EXISTING_DATA_REPLAY: fresh statistics from the stored per-draw arrays of the code-domain '
             'feature-panel/native/rarefied cells (rank + QR residualization, family bootstrap, label permutation)',
             {'cells': {k: fresh[k] for k in CODE_KEYS}, 'recurrence': recurrence,
              'primary_cells': primary},
             {'config': 'config/STUDY2_INFERENCE_REUSE_V1.json',
              'partial': cfg['inference']['partial'], 'verdict_rule': cfg['inference']['legacy_verdict']},
             {'bootstrap_replicates': n_boot, 'bootstrap_seed': seed_boot,
              'permutation_replicates': n_perm, 'permutation_seed': seed_perm,
              'role': 'fixed configured resample seeds/sizes; resample index streams re-derived from the seeds '
                      'and required to equal the stored draws exactly'})
    rp.gap(group, 'legacy_verdict_label',
           'SHADOW-ONLY/REFUTE are the preserved historical computational labels, not proof of power, absence '
           'of effect, preregistration, causation or construct realism; medical cells are out of this claim scope')
    return {'status': 'PRODUCED', 'claims': ['P2-C-003'], 'fresh': fresh, 'recurrence': recurrence}


GROUPS_FN = {'study1': group_study1, 'nbias': group_nbias, 'code_shadow': group_code_shadow}


def main():
    ap = argparse.ArgumentParser(description='C10u retained existing-data replay supplements A')
    ap.add_argument('--analysis-root', default=None,
                    help='analysis root R (default: repository containing this script)')
    ap.add_argument('--out', required=True, help='fresh (empty/nonexistent) output directory')
    ap.add_argument('--groups', default='all',
                    help='all or comma-separated subset of study1,nbias,code_shadow')
    ap.add_argument('--expected-manifest', default=None,
                    help='optional pinned source manifest (source_manifest.json / report document / flat {path: sha256})')
    ap.add_argument('--emit-expected-manifest', default=None,
                    help='also write the consumed-source pin manifest to this path (clean environment evidence)')
    args = ap.parse_args()
    root = Path(args.analysis_root).resolve() if args.analysis_root else Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        print(f'[output-error] refusing to write into non-empty output directory: {out}', file=sys.stderr)
        sys.exit(2)
    out.mkdir(parents=True, exist_ok=True)
    pins = None
    if args.expected_manifest:
        pin_doc = json.loads(Path(args.expected_manifest).read_text())
        if isinstance(pin_doc, dict) and 'sources' in pin_doc:
            pins = {e['path']: e['sha256'] for e in pin_doc['sources']}
        elif isinstance(pin_doc, dict) and 'entries' in pin_doc:
            pins = {e['path']: e['sha256'] for e in pin_doc['entries']}
        elif isinstance(pin_doc, dict) and 'source_manifest' in pin_doc:
            pins = {e['path']: e['sha256'] for e in pin_doc['source_manifest']['entries']}
        elif isinstance(pin_doc, dict):
            pins = {str(k): str(v) for k, v in pin_doc.items()}
        else:
            raise SystemExit('unsupported --expected-manifest structure')
    todo = GROUPS if args.groups == 'all' else tuple(g.strip() for g in args.groups.split(',') if g.strip())
    for name in todo:
        if name not in GROUPS_FN:
            print(f'[cli-error] unknown group: {name}', file=sys.stderr)
            sys.exit(2)
    rp = Replay(root, out, pins)
    try:
        for name in todo:
            rp.current_group = name
            rp.groups[name] = GROUPS_FN[name](rp)
        rp.current_group = None
        rp.verify_manifest()
    except FileNotFoundError as exc:
        print(f'[source-error] {exc}', file=sys.stderr)
        sys.exit(3)
    except ValueError as exc:
        print(f'[source-error] {exc}', file=sys.stderr)
        sys.exit(3)
    elapsed = time.time() - rp.t0

    for name, res in rp.groups.items():
        if name == 'nbias':
            rp.dump(out / 'groups/nbias_h4a.json',
                    {'schema': SCHEMA + '/nbias-h4a', 'fresh_cell_curves': res['fresh_curve'],
                     'feature_checks': res['feature_checks'], 'statistics_checks': res['stat_checks']})
        elif name == 'code_shadow':
            rp.dump(out / 'groups/code_shadow.json',
                    {'schema': SCHEMA + '/code-shadow', 'fresh_cells': res['fresh'],
                     'recurrence': res['recurrence']})
        else:
            rp.dump(out / 'groups/study1_additional.json',
                    {'schema': SCHEMA + '/study1-additional',
                     'fresh': {c: rp.claims[c]['fields'] for c in ('P2-A-005', 'P2-A-007')}})
    rp.dump(out / 'claims.json', {'schema': SCHEMA + '/claims', 'claims': rp.claims})

    by_group = {}
    for name in rp.groups:
        rows = [c for c in rp.comparators if c['group'] == name]
        checks = [c for c in rp.checks if c['group'] == name]
        by_group[name] = {'comparator_matched': sum(1 for c in rows if c['matched']),
                          'comparator_total': len(rows),
                          'comparator_deviations': [c for c in rows if not c['matched']][:25],
                          'checks_passed': sum(1 for c in checks if c['passed']),
                          'checks_total': len(checks),
                          'checks_failed': [c for c in checks if not c['passed']]}
    summary = {'schema': SCHEMA, 'status': 'SUPPLEMENTS_A_REPLAY_PRODUCED_PENDING_ROOT_VERIFICATION',
               'groups': {k: v['status'] for k, v in rp.groups.items()},
               'group_summary': by_group,
               'claims': sorted(rp.claims),
               'comparator_total': len(rp.comparators),
               'comparator_matched': sum(1 for c in rp.comparators if c['matched']),
               'checks_total': len(rp.checks),
               'checks_passed': sum(1 for c in rp.checks if c['passed']),
               'manifest_entries': len(rp.manifest),
               'replay_level': 'RETAINED_EXISTING_DATA_REPLAY (no refits, no forward passes, no new studies)',
               'gaps': rp.gaps,
               'expected_manifest': str(Path(args.expected_manifest).resolve()) if args.expected_manifest else None,
               'pinned_entries': len(pins) if pins else None,
               'pinned_entries_not_consumed_by_this_run': (len(set(pins) - set(rp.manifest)) if pins else None),
               'not_computed_here': ['C6 synthetic geometry branch (separate production; never a substitute for empirical NBIAS)',
                                     'medical-domain inference cells (blocked nonfinite inputs; preserved as limitations)',
                                     'G/pooling/prior16 detail and BOS/retention (supplements B, separate node)']}
    rp.dump(out / 'summary.json', summary)
    manifest = {'schema': SCHEMA + '/source-manifest', 'analysis_root': str(rp.R), 'project_root': str(rp.P),
                'entry_count': len(rp.manifest),
                'entries': sorted(rp.manifest.values(), key=lambda e: e['path']),
                'roles': sorted(set(e['role'] for e in rp.manifest.values())),
                'levels': sorted(set(e['level'] for e in rp.manifest.values())),
                'groups': sorted(set(g for e in rp.manifest.values() for g in e.get('groups', []))),
                'source_mutation_check': True,
                'comparator_only_declaration':
                    'terminal aggregate JSONs, independent-review reports and the C17 evidence ledger are read only '
                    'after the fresh numbers exist and are used exclusively as comparator/pointer sources'}
    rp.dump(out / 'source_manifest.json', manifest)
    rp.dump(out / 'comparators.json', {'schema': SCHEMA + '/comparators', 'rule': TOL_RULE,
                                       'counts_ids_and_null_masks_exact': True, 'rows': rp.comparators,
                                       'matched': sum(1 for c in rp.comparators if c['matched']),
                                       'total': len(rp.comparators)})
    rp.dump(out / 'checks.json', {'schema': SCHEMA + '/checks', 'rows': rp.checks,
                                  'passed': sum(1 for c in rp.checks if c['passed']), 'total': len(rp.checks)})
    receipt = {'schema': SCHEMA + '/receipt',
               'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'producer': rp.rel(Path(__file__)), 'producer_sha256': dig(__file__),
               'analysis_root': str(rp.R), 'project_root': str(rp.P), 'output_dir': str(out),
               'groups': list(rp.groups),
               'commands': [{'argv': sys.argv, 'cwd': str(Path.cwd()), 'seconds': elapsed}],
               'environment': {'python': platform.python_version(), 'executable': sys.executable,
                               'numpy': np.__version__, 'scipy': scipy.__version__,
                               'threads': {k: os.environ.get(k) for k in
                                           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                               'platform': platform.platform()},
               'source_manifest_sha256': dig(out / 'source_manifest.json'),
               'claims_sha256': dig(out / 'claims.json'),
               'expected_manifest': str(Path(args.expected_manifest).resolve()) if args.expected_manifest else None,
               'pinned_entries': len(pins) if pins else None,
               'no_writeback': 'all consumed files re-hashed after computation; outputs written only under --out',
               'exit_code_convention': {'2': 'CLI/output-directory error', '3': 'source/input error',
                                        '4': 'internal self-check failure'},
               'validation_scope': 'cached numerical comparison; no independent scientific review'}
    rp.dump(out / 'receipt.json', receipt)
    (out / 'commands.log').write_text(
        '\n'.join([f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] cwd={Path.cwd()} argv='
                   + ' '.join(sys.argv)]) + '\n')
    if args.emit_expected_manifest:
        expected = {'schema': SCHEMA + '/expected-manifest',
                    'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'analysis_root': str(rp.R), 'n_sources': len(rp.manifest),
                    'path_convention': 'paths are labelled R/... (analysis root) or P/... (project root)',
                    'sources': [{'path': e['path'], 'sha256': e['sha256'], 'role': e['role'], 'level': e['level']}
                                for e in sorted(rp.manifest.values(), key=lambda e: e['path'])]}
        rp.dump(args.emit_expected_manifest, expected)
    print(json.dumps({'out': str(out), 'groups': summary['groups'],
                      'comparator': f"{summary['comparator_matched']}/{summary['comparator_total']}",
                      'checks': f"{summary['checks_passed']}/{summary['checks_total']}",
                      'manifest_entries': len(rp.manifest), 'seconds': round(elapsed, 2)}))
    if summary['checks_passed'] != summary['checks_total']:
        print('[check-error] self-check failure: see checks.json', file=sys.stderr)
        sys.exit(4)
    if summary['comparator_matched'] != summary['comparator_total']:
        print('[check-error] comparator deviation: see comparators.json', file=sys.stderr)
        sys.exit(4)


if __name__ == '__main__':
    main()
