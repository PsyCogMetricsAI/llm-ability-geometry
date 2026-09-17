#!/usr/bin/env python3
"""Saved pooled clouds to original chi statistic; no model invocation."""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
R=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--run',default='window_smol_new_v1');args=a.parse_args();d=R/'runs'/args.run;run=json.loads((d/'run_200.json').read_text());used=[r for r in run['records']if 'skip'not in r];K=10
if len(used)<=K:raise ValueError('Too few retained items for k10')
outs={}
for w in [8,32,128]:
 X=np.stack([np.load(d/f"item_{r['id']}.npz")[f'W{w}'] for r in used]);neighbors=[]
 for j in range(X.shape[1]):
  D=cdist(X[:,j].astype('float64'),X[:,j].astype('float64'));np.fill_diagonal(D,np.inf);neighbors.append(np.argsort(D,axis=1)[:,:K])
 pairs=[float(np.mean([len(set(neighbors[j][i])&set(neighbors[j+1][i]))/K for i in range(len(used))]))for j in range(len(neighbors)-1)]
 outs[str(w)]={'chi':float(np.mean(pairs)),'pair_chis':pairs,'shape':list(X.shape)}
vals=[v['chi']for v in outs.values()];reld=max(abs(a-b)/max(abs(a),abs(b),1e-12)for a in vals for b in vals)
out={'producer':'scientific-analysis','classification':'NEW_PINNED_SINGLE_MODEL_WINDOW_RESULT_PENDING_INDEPENDENT_REVIEW','n_used':len(used),'used_ids':[r['id']for r in used],'windows':outs,'max_pairwise_rel_diff':reld,'per_model_pass':reld<=.2,'panel_rank_not_computed':'Only one of six original models regenerated; no six-model gate verdict.','bindings':{'run_200.json':hashlib.sha256((d/'run_200.json').read_bytes()).hexdigest(),'code':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
(d/'metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
