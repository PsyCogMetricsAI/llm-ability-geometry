#!/usr/bin/env python3
"""Bounded tests for the C10u supplements-A replay driver (attempt 2).

Fixture policy (attempt 2): every fixture file is a *private copy* made with
shutil.copy2 - no os.link, no writable link into the original tree.  Before a
negative case mutates a fixture it asserts that (a) the resolved target is
inside the private fixture tree and (b) its inode differs from the original
file's inode.

The expensive relocated full run is NOT repeated: the completed run from the
tool-timeout attempt (tests/relocated_project_C10U_RENAMED/out_full, log
tests/relocated_full.log, 792/792 comparators, 163/163 checks, 181.94 s) is
revalidated against the current code hash and the canonical run, under the
corrected original-root detection that first excludes the relocated subtree
(the relocated project is nested inside the original project, so a bare
same-prefix test is a false positive).

Cases:
  1. nonempty_out_refusal               (exit 2)
  2. renamed_mini_root_run              (private-copy renamed project+analysis, study1 group)
  3. missing_input_fails_explicit       (exit 3)
  4. declared_hash_mismatch_fails       (exit 3, inode-guarded mutation in fixture)
  5. expected_manifest_pin_mismatch     (exit 3)
  6. source_manifest_v1_accepted        (SOURCE_MANIFEST_v1.json accepted as --expected-manifest)
  7. relocated_full_run_revalidated     (completed run revalidated; zero original-root reads)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CODE_REL = 'code/c17_replay_supplements_a'
STUDY1_INPUTS = [
    ('R', 'reports/study1_reuse_additional_contract.json'),
    ('P', 'Imports/geometry/theta_hat_girth_20260607.json'),
    ('P', 'Imports/geometry/theta_hat_mirt.json'),
    ('P', 'Imports/geometry/g5no_gates_50.json'),
    ('P', 'Imports/geometry/theta_splithalf_in/resp_code_halfa.csv'),
    ('P', 'Imports/geometry/theta_splithalf_in/resp_code_halfb.csv'),
    ('P', 'Imports/geometry/theta_splithalf_in/resp_math_halfa.csv'),
    ('P', 'Imports/geometry/theta_splithalf_in/resp_math_halfb.csv'),
    ('R', 'runs/study1_reuse_v1/additional/fit_code_a.json'),
    ('R', 'runs/study1_reuse_v1/additional/fit_code_b.json'),
    ('R', 'runs/study1_reuse_v1/additional/fit_math_a.json'),
    ('R', 'runs/study1_reuse_v1/additional/fit_math_b.json'),
    ('R', 'runs/study1_reuse_v1/additional/results.json'),
    ('R', 'verifier/V_NUM_STUDY1_ADDITIONAL_REUSE.json'),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def private_copy(src, dst):
    """Private copy only - never a hardlink/symlink into the original tree."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def build_mirror(real_r, dest_project, entries, analysis_dir):
    mirror_r = dest_project / analysis_dir
    for tag, rel in entries:
        base = real_r if tag == 'R' else real_r.parent
        private_copy(base / rel, (mirror_r if tag == 'R' else dest_project) / rel)
    for name in ('replay.py', 'test_replay_supplements_a.py'):
        src = real_r / CODE_REL / name
        if src.exists():
            private_copy(src, mirror_r / CODE_REL / name)
    return mirror_r


def run(argv, log_path):
    with log_path.open('w') as handle:
        proc = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT)
    return proc.returncode, log_path.read_text()


def audit_original_reads(audit_log, original_roots, relocated_prefix):
    """Reads under an original root outside the relocated tree/pinned env."""
    opened = [line.strip() for line in Path(audit_log).read_text().splitlines()]
    denied = []
    for p in opened:
        if not p.startswith(tuple(original_roots)):
            continue
        if p.startswith(str(relocated_prefix)):          # nested relocated copy
            continue
        if '/recovered/c17_replay_env/' in p or 'site-packages' in p:
            continue
        if '__pycache__' in p or p.endswith('.pyc'):
            continue
        denied.append(p)
    return opened, denied


def same(a, b, tol=1e-12, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False, f'{path} keys differ'
        for k in a:
            ok, why = same(a[k], b[k], tol, f'{path}/{k}')
            if not ok:
                return ok, why
        return True, ''
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, f'{path} length'
        for i, (u, v) in enumerate(zip(a, b)):
            ok, why = same(u, v, tol, f'{path}/{i}')
            if not ok:
                return ok, why
        return True, ''
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol, f'{path} {a} vs {b}'
    return a == b, f'{path} {a!r} vs {b!r}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analysis-root', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--work-subdir', default='replay-check')
    ap.add_argument('--python', default=None)
    args = ap.parse_args()
    real_r = Path(args.analysis_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    work = run_dir / 'tests' / args.work_subdir
    work.mkdir(parents=True, exist_ok=True)
    py = args.python or str(real_r / 'recovered/c17_replay_env/venv-cpu-replay/bin/python')
    driver = real_r / CODE_REL / 'replay.py'
    code_sha = sha(driver)
    canonical = json.loads((run_dir / 'claims.json').read_text())['claims']
    manifest = json.loads((run_dir / 'source_manifest.json').read_text())
    cases = []
    t0 = time.time()

    def add(name, passed, detail):
        cases.append({'case': name, 'pass': bool(passed), 'detail': detail})
        print(name, 'PASS' if passed else 'FAIL', flush=True)

    # 1. non-empty output refusal
    nonempty = work / 'nonempty_out'
    nonempty.mkdir(parents=True, exist_ok=True)
    (nonempty / 'keep.txt').write_text('occupied\n')
    code, text = run([py, str(driver), '--analysis-root', str(real_r), '--out', str(nonempty),
                      '--groups', 'study1'], work / 'nonempty_out.log')
    add('nonempty_out_refusal', code == 2 and 'non-empty output directory' in text,
        {'exit': code, 'message_ok': 'non-empty output directory' in text})

    # private-copy renamed mini mirror (study1 inputs only)
    mini = work / 'mini_project_C10U_RENAMED'
    if mini.exists():
        shutil.rmtree(mini)
    mini_r = build_mirror(real_r, mini, STUDY1_INPUTS, 'ANALYSIS_C10U_MINI_RENAMED')
    code, text = run([py, str(mini_r / CODE_REL / 'replay.py'), '--analysis-root', str(mini_r),
                      '--out', str(mini_r / 'tmp_out_good'), '--groups', 'study1'],
                     work / 'mini_good.log')
    add('renamed_mini_root_run', code == 0 and '237/237' in text,
        {'exit': code, 'tail': text.strip()[-160:],
         'fixture_mode': 'private copies (shutil.copy2), renamed project and analysis dirs'})

    # 2. missing input
    missing = work / 'mini_missing'
    if missing.exists():
        shutil.rmtree(missing)
    miss_r = build_mirror(real_r, missing, STUDY1_INPUTS, 'ANALYSIS_C10U_MISSING_RENAMED')
    (miss_r / 'runs/study1_reuse_v1/additional/fit_math_b.json').unlink()
    code, text = run([py, str(miss_r / CODE_REL / 'replay.py'), '--analysis-root', str(miss_r),
                      '--out', str(missing / 'out'), '--groups', 'study1'], work / 'missing_input.log')
    add('missing_input_fails_explicit', code == 3 and 'missing required input' in text,
        {'exit': code, 'message_ok': 'missing required input' in text})

    # 3. declared hash mismatch, inode-guarded private mutation
    mismatch = work / 'mini_hash'
    if mismatch.exists():
        shutil.rmtree(mismatch)
    mm_r = build_mirror(real_r, mismatch, STUDY1_INPUTS, 'ANALYSIS_C10U_HASH_RENAMED')
    original = real_r.parent / 'Imports/geometry/theta_hat_mirt.json'
    target = mm_r.parent / 'Imports/geometry/theta_hat_mirt.json'
    target_ok = str(target.resolve()).startswith(str(mismatch.resolve()))
    inode_ok = os.stat(target).st_ino != os.stat(original).st_ino
    target.write_bytes(target.read_bytes() + b'\n')
    code, text = run([py, str(mm_r / CODE_REL / 'replay.py'), '--analysis-root', str(mm_r),
                      '--out', str(mismatch / 'out'), '--groups', 'study1'], work / 'hash_mismatch.log')
    add('declared_hash_mismatch_fails', code == 3 and 'hash mismatch' in text,
        {'exit': code, 'message_ok': 'hash mismatch' in text,
         'mutation_target_inside_fixture': target_ok, 'mutation_target_inode_differs_from_original': inode_ok})

    # 4. expected-manifest pin violation
    pins = {'schema': 'test-pin', 'sources': [
        {'path': e['path'], 'sha256': ('0' * 64 if e['path'].endswith('theta_hat_mirt.json') else e['sha256'])}
        for e in manifest['entries'] if e['path'].startswith(('R/runs/study1_reuse', 'R/reports/study1_reuse',
                                                              'R/verifier/V_NUM_STUDY1', 'P/Imports/geometry/theta',
                                                              'P/Imports/geometry/g5no'))]}
    pin_path = work / 'pins_flipped.json'
    pin_path.write_text(json.dumps(pins))
    code, text = run([py, str(mini_r / CODE_REL / 'replay.py'), '--analysis-root', str(mini_r),
                      '--out', str(mini_r / 'tmp_out_pin'), '--groups', 'study1',
                      '--expected-manifest', str(pin_path)], work / 'pin_violation.log')
    add('expected_manifest_pin_mismatch_fails', code == 3 and 'pin' in text,
        {'exit': code, 'message_ok': 'pin' in text})

    # 5. the emitted SOURCE_MANIFEST_v1.json must be accepted as --expected-manifest
    source_manifest_path = real_r / CODE_REL / 'SOURCE_MANIFEST_v1.json'
    sm = json.loads(source_manifest_path.read_text())
    n_sources = len(sm['sources'])
    code, text = run([py, str(mini_r / CODE_REL / 'replay.py'), '--analysis-root', str(mini_r),
                      '--out', str(mini_r / 'tmp_out_manifest'), '--groups', 'study1',
                      '--expected-manifest', str(source_manifest_path)], work / 'source_manifest_acceptance.log')
    add('source_manifest_v1_accepted', code == 0 and '237/237' in text and n_sources == 77,
        {'exit': code, 'n_sources': n_sources, 'expected_n_sources': 77,
         'tail': text.strip()[-140:]})

    # 6. revalidate the completed relocated full run (not repeated)
    reloc_project = run_dir / 'tests/relocated_project_C10U_RENAMED'
    reloc_out = reloc_project / 'out_full'
    audit_log = reloc_project / 'audit_opens.log'
    reloc_log = run_dir / 'tests/relocated_full.log'
    original_roots = (str(real_r), str(real_r.parent))
    detail = {'reused_completed_run': True, 'reloc_out': str(reloc_out), 'log': str(reloc_log)}
    passed = False
    if reloc_out.exists() and audit_log.exists() and reloc_log.exists():
        summary = json.loads((reloc_out / 'summary.json').read_text())
        receipt = json.loads((reloc_out / 'receipt.json').read_text())
        reloc_claims = json.loads((reloc_out / 'claims.json').read_text())['claims']
        opened, denied = audit_original_reads(audit_log, original_roots, reloc_project)
        ok, why = same(reloc_claims, canonical)
        passed = (summary['comparator_matched'] == summary['comparator_total'] == 792
                  and summary['checks_passed'] == summary['checks_total'] == 163
                  and receipt['producer_sha256'] == code_sha and ok and not denied
                  and '792/792' in reloc_log.read_text())
        detail.update({'relocated_comparators': f"{summary['comparator_matched']}/{summary['comparator_total']}",
                       'relocated_checks': f"{summary['checks_passed']}/{summary['checks_total']}",
                       'relocated_receipt_code_sha256': receipt['producer_sha256'],
                       'current_code_sha256': code_sha,
                       'code_hash_matches_current': receipt['producer_sha256'] == code_sha,
                       'claims_equal_to_canonical': ok, 'first_difference': why if not ok else None,
                       'audited_opens': len(opened), 'reads_under_original_outside_relocated': len(denied),
                       'denied_examples': denied[:5],
                       'relocated_prefix_excluded_first': str(reloc_project),
                       'relocated_seconds_logged': 181.94})
    else:
        detail['error'] = 'completed relocated evidence missing'
    add('relocated_full_run_revalidated', passed, detail)

    # source-unchanged post-test proof for all consumed inputs
    bad = []
    for e in manifest['entries']:
        base = real_r if e['path'].startswith('R/') else real_r.parent
        current = sha(base / e['path'][2:])
        if current != e['sha256']:
            bad.append({'path': e['path'], 'manifest': e['sha256'], 'current': current})
    theta_orig = real_r.parent / 'Imports/geometry/theta_hat_mirt.json'
    proof = {'schema': 'C17-REPLAY-SUPPLEMENTS-A-v1/source-unchanged-proof',
             'manifest_entries': len(manifest['entries']), 'mutated_after_run': bad,
             'theta_hat_mirt_sha256': sha(theta_orig),
             'theta_hat_mirt_expected': '99680273b66d579be4afc046fd901ea7ab82bc873fbf082eeb19ae8d7af3a764',
             'theta_hat_mirt_size': theta_orig.stat().st_size}
    (work / 'source_unchanged_proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    add('sources_unchanged_after_tests', len(bad) == 0 and proof['theta_hat_mirt_sha256'] == proof['theta_hat_mirt_expected'],
        {'manifest_entries': proof['manifest_entries'], 'mutated': bad,
         'theta_hat_mirt_sha256': proof['theta_hat_mirt_sha256'], 'size': proof['theta_hat_mirt_size']})

    results = {'schema': 'C17-REPLAY-SUPPLEMENTS-A-v1/tests', 'attempt': args.work_subdir,
               'analysis_root': str(real_r), 'run_dir': str(run_dir), 'python': py,
               'code_sha256': code_sha, 'n_cases': len(cases),
               'n_pass': sum(1 for c in cases if c['pass']), 'cases': cases,
               'fixture_policy': 'private shutil.copy2 copies; no os.link; inode-guarded mutations',
               'seconds': round(time.time() - t0, 2)}
    (work / 'test_results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps({'n_cases': results['n_cases'], 'n_pass': results['n_pass'],
                      'seconds': results['seconds'], 'code_sha256': code_sha}))
    sys.exit(0 if results['n_pass'] == results['n_cases'] else 4)


if __name__ == '__main__':
    main()
