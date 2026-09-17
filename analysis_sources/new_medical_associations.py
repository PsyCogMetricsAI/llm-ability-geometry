"""New alternative-medical-model descriptive associations; no old p/BH restoration."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,sys,collections
import numpy as np
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[1];O=R/'runs/new_medical_associations';CF=R/'config/CT_MEDICAL_ASSOCIATIONS.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False))
STATUS={0:'valid',1:'constant_input',2:'rank_deficient_design',3:'zero_residual_rank',4:'zero_residual_norm'}

def stats(matrix,nfeat):
 # Columns Rasch-ranked input,MAP,feature1..k,C1..4,fitaccuracy.
 r=rankdata(matrix,axis=0,method='average');v=r[:,:2+nfeat];vals=np.full((3,2,nfeat),np.nan);statuses=np.zeros((3,2,nfeat),dtype=np.int8)
 for mode in range(3):
  D=np.ones((len(r),1)) if mode==0 else np.column_stack([np.ones(len(r)),r[:,2+nfeat:2+nfeat+4+(mode==2)]])
  rd=np.linalg.matrix_rank(D)
  if rd<D.shape[1]:statuses[mode]=2;continue
  Q=np.linalg.qr(D,mode='reduced')[0];res=v-Q@(Q.T@v);res-=res.mean(0);norm=np.linalg.norm(res,axis=0);state=np.zeros(v.shape[1],dtype=np.int8)
  for j in range(v.shape[1]):
   if np.unique(v[:,j]).size<2:state[j]=1
   elif np.linalg.matrix_rank(np.column_stack([D,v[:,j]]))==rd:state[j]=3
   elif norm[j]<=100*np.finfo(float).eps*max(D.shape)*max(np.linalg.norm(v[:,j]),1):state[j]=4
  for a in range(2):
   for j in range(nfeat):
    reason=state[a] or state[j+2];statuses[mode,a,j]=reason
    if not reason:vals[mode,a,j]=(res[:,a]@res[:,j+2])/(norm[a]*norm[j+2])
 return vals,statuses

def prepare(panel,names):
 if panel=='A':
  data=json.loads((R/'runs/study2_reuse_v1/inference/inputs/medical.json').read_text());tags=data['tags'];geom=np.column_stack([data['native'],data['rarefied']]);C=np.array(data['C']);family=np.array(data['families']);available=np.array(data['accuracy']);features=['native_'+n for n in names]+['rarefied_'+n for n in names]
 else:
  data=np.load(R/'runs/extension_reuse_v1/trackb_analysis_v1/diagnostics/HISTORICAL_NAN_RANK/medical/aligned_inputs.npz');tags=data['tags'].tolist();geom=data['pr_cached'][:,None];C=data['controls'];family=data['families'];available=data['accuracy_available'];features=['trackB_eff_rank_pr']
 base=json.loads((R/'runs/new_medical_repair/summary.json').read_text())['variants']['medical_'+panel];ref=json.loads((R/'runs/new_medical_repair_refined/summary.json').read_text())['variants']['medical_'+panel];assert base['tags']==ref['tags'];assert set(tags)==set(base['tags']) and len(tags)==len(set(tags));idx=[base['tags'].index(t) for t in tags]
 ras=np.load(R/f'runs/new_medical_repair/medical_{panel}_rasch_1.0.npz');mp=np.load(R/f'runs/new_medical_repair_refined/medical_{panel}_map_1.0.npz');assert np.array_equal(ras['Y'],mp['Y']);theta_r=ras['theta'][idx];theta_m=mp['theta'][idx];Y=ras['Y'][:,idx];total=Y.sum(0);acc=Y.mean(0)
 for source,kind in [(base,'rasch'),(ref,'map')]:assert next(f for f in source['fits'] if f['kind']==kind)['score_acceptance_candidate']
 levels=np.unique(total);means=np.array([theta_r[total==n].mean() for n in levels]);spreads=np.array([np.ptp(theta_r[total==n]) for n in levels]);assert np.all(np.diff(means)>0) and spreads.max()<1e-8
 matrix=np.column_stack([total,theta_m,geom,C,acc]);assert np.isfinite(matrix).all();return tags,features,family,matrix,theta_r,theta_m,available,{'max_EAP_within_total_spread':float(spreads.max()),'unique_totals':len(levels),'exact_total_ranking_used':True,'source_EAP_vs_total_rho':float(np.corrcoef(rankdata(theta_r),rankdata(total))[0,1])}

def main():
 cfg=json.loads(CF.read_text());approval=json.loads((R/'verifier/CT_MEDICAL_ASSOCIATIONS_APPROVED.json').read_text());assert approval['contract_sha256']==sha(CF)
 for p in cfg['inputs']:assert sha(p['path'])==p['sha256'],p['path']
 O.mkdir(exist_ok=False);allrows=[];meta={}
 for panel,seed in [('A',20260918),('B',20260919)]:
  out=O/panel;out.mkdir();tags,features,family,matrix,tr,tm,available,canon=prepare(panel,cfg['scope']['feature_names']);k=len(features);np.savez_compressed(out/'inputs.npz',tags=np.array(tags),features=np.array(features),family=family,matrix=matrix,rasch_EAP=tr,map_EAP=tm,available_accuracy=available);u=sorted(set(family));blocks=[np.flatnonzero(family==f) for f in u];rng=np.random.default_rng(seed);draws=rng.choice(len(u),size=(10000,len(u)),replace=True);indices=np.full((10000,len(u)*max(map(len,blocks))),-1,dtype=np.int16);lengths=np.empty(10000,dtype=np.int16);values=np.full((10000,3,2,k),np.nan);status=np.zeros(values.shape,dtype=np.int8);point,ps=stats(matrix,k)
  for b,chosen in enumerate(draws):
   idx=np.concatenate([blocks[i] for i in chosen]);indices[b,:len(idx)]=idx;lengths[b]=len(idx);values[b],status[b]=stats(matrix[idx],k)
   if (b+1)%1000==0:print(panel,b+1,'/10000',flush=True)
  np.savez_compressed(out/'bootstrap.npz',family_draws=draws,family_order=np.array(u),indices_padded=indices,lengths=lengths,statistics=values,status=status,point=point,point_status=ps)
  for mode,label in enumerate(['raw','partial4C','partial4C_plus_fitaccuracy']):
   for a,score in enumerate(['Rasch','MAP']):
    for j,feature in enumerate(features):
     v=values[:,mode,a,j];good=np.isfinite(v);counts=collections.Counter(status[:,mode,a,j].tolist());row={'panel':panel,'score':score,'mode':label,'feature':feature,'point':float(point[mode,a,j]) if np.isfinite(point[mode,a,j]) else None,'point_status':STATUS[int(ps[mode,a,j])],'valid':int(good.sum()),'B':10000,'CI95':np.percentile(v[good],[2.5,97.5]).tolist() if good.sum()>=9000 else None,'bootstrap_status':{STATUS[int(s)]:n for s,n in counts.items()},'interpretation':'descriptive sensitivity only; no calibrated significance or negative-control verdict'};allrows.append(row)
  meta[panel]={'n':len(tags),'features':features,'family_order':u,'family_sizes':[len(b) for b in blocks],'canonicalization':canon,'inputs_sha256':sha(out/'inputs.npz'),'bootstrap_sha256':sha(out/'bootstrap.npz')}
 result={'producer':'scientific-analysis','status':'PRODUCER_COMPLETE_PENDING_INDEPENDENT_REVIEW','classification':cfg['classification'],'contract_sha256':sha(CF),'code_sha256':sha(__file__),'panels':meta,'status_codes':STATUS,'rows':allrows,'limits':['Cached geometry start; no GPU re-extraction.','A has14columns;B isPR-only.','Rasch exact total-rank equivalence enforced; accuracy-conditioned residual undefined.','No permutationp,BH,negativecontrol verdict; descriptiveCI uncalibrated small-family inference.','No model chosen as primary from association results.']};dump(O/'report.json',result);dump(R/'reports/NEW_MEDICAL_ASSOCIATIONS.json',result);print('COMPLETE',len(allrows),flush=True)
if __name__=='__main__':main()
