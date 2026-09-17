#!/usr/bin/env python3
"""Export paper tables from accepted saved summaries, without re-estimation.

This is a result-inspection tool. It does not refit IRT, rerun inference, or
regenerate bootstrap intervals. See docs/REPRODUCING.md for cached replay.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'results/sources'


def read(name):
    return json.loads((SOURCES / name).read_text())


def interval(value):
    if value is None:
        return 'Undefined'
    if isinstance(value, dict):
        value = [value['lo'], value['hi']]
    return f'[{value[0]:.3f}, {value[1]:.3f}]'


def write_table(out, name, columns, rows):
    with (out / (name + '.csv')).open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    text = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    text.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    (out / (name + '.md')).write_text('\n'.join(text) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/paper_tables')
    args = parser.parse_args()
    # Fail on altered source data instead of silently exporting a different release.
    mapping = json.loads((ROOT / 'results/SOURCE_MAP.json').read_text())
    for rel, info in mapping.items():
        if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != info['sha256']:
            raise SystemExit('Source hash mismatch: ' + rel)
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = read('study1_baseline.json')['domains']
    accuracy = read('study1_accuracy_v2.json')['domains']
    rows = []
    for domain, title in [('code', 'Code / composite'), ('math', 'Mathematics / composite')]:
        d = baseline[domain]
        a = accuracy[domain]['q5_rho_g_t__a_C']
        rows.append([title, d['n_models'], f"{d['theta_accuracy_rho']:.3f}", f"{d['raw_rho']:.3f}",
                     f"{d['partial_rho_controlling_C']:.3f}", interval(d['controlled_cluster_ci']),
                     f"{a['rho']:.3f}", interval(a['ci'])])
    science = read('science/F1_MAP__native__science__eff_rank_pr.json')
    legacy = next(c for c in read('science_accuracy.json')['cells'] if c['domain'] == 'science' and c['feature'] == 'eff_rank_pr')
    a = legacy['acc_plus_C']
    rows.append(['Science / PR (exploratory)', science['n_models'], f"{legacy['spearman_theta_accuracy']:.3f}",
                 f"{science['raw_rho']:.3f}", f"{science['partial_rho']:.3f}", interval(science['ci_controlled']),
                 f"{a['partial_rho']:.3f}", interval(a['bootstrap']['ci'])])
    write_table(args.output, 'table1_correspondence', ['Domain / indicator', 'n', 'theta_accuracy_rho', 'raw_rho', 'rho_Z', 'CI_Z', 'rho_A_Z', 'CI_A_Z'], rows)
    features = [('eff_rank_pr', 'PR'), ('rankme', 'RankMe'), ('stable_rank', 'Stable rank'), ('spectral_alpha', 'Spectral decay'), ('vn_entropy', 'VN entropy'), ('isoscore', 'IsoScore'), ('twoNN_id', 'TwoNN')]
    rows = []
    for feature, label in features:
        d = read(f'science/F1_MAP__native__science__{feature}.json')
        rows.append([label, f"{d['raw_rho']:.3f}", f"{d['partial_rho']:.3f}", interval(d['ci_controlled'])])
    write_table(args.output, 'table2_science_native', ['Indicator', 'raw_rho', 'rho_Z', 'CI_Z'], rows)
    medical = read('medical_associations.json')['rows']
    rows = []
    for mode, label in [('raw', 'Raw'), ('partial4C', 'Given Z'), ('partial4C_plus_fitaccuracy', 'Given A,Z')]:
        row = [label]
        for score in ('MAP', 'Rasch'):
            d = next(x for x in medical if x['panel'] == 'A' and x['feature'] == 'native_eff_rank_pr' and x['score'] == score and x['mode'] == mode)
            row.extend(['Undefined' if d['point'] is None else f"{d['point']:.3f}", interval(d['CI95'])])
        rows.append(row)
    write_table(args.output, 'supplement_medical', ['Comparison', 'Regularized 2PL', 'CI_2PL', 'Rasch', 'CI_Rasch'], rows)
    print(json.dumps({'status': 'PASS', 'tables': 3, 'scope': 'Export of accepted summaries; no statistical re-estimation', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
