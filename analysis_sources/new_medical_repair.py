"""New log-stable MML/MAP fits, separate from historical girth replay."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,time,sys
import numpy as np
from scipy.special import expit,logsumexp,roots_legendre,logit
from scipy.optimize import minimize
from scipy.stats import spearmanr
R=Path(__file__).resolve().parents[1]; C=R/'config/CT_MEDICAL_REPAIR.json';OUT=R/'runs/new_medical_repair'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def grid(q):
 t,w=roots_legendre(q);t*=6;w*=6
 lw=np.log(w)-t*t/2-.5*np.log(2*np.pi)
 return t,lw
class Objective:
 def __init__(self,y,kind,q=161):self.y=y;self.kind=kind;self.t,self.lw=grid(q);self.I=len(y);self.calls=0
 def unpack(self,z):
  if self.kind=='rasch':return np.ones(self.I),z
  return np.exp(z[:self.I]),z[self.I:]
 def calc(self,z,full=False):
  a,b=self.unpack(z);lin=a[:,None]*(self.t[None,:]-b[:,None]);lp=-np.logaddexp(0,-lin);ln=-np.logaddexp(0,lin)
  ll=self.y.T@lp+(1-self.y).T@ln;norm=logsumexp(ll+self.lw,axis=1);post=np.exp(ll+self.lw-norm[:,None])
  err=self.y@post-expit(lin)*post.sum(0)[None,:]
  gb=a*err.sum(1);ga=-a*np.sum(err*(self.t[None,:]-b[:,None]),axis=1)
  f=-norm.sum();g=gb if self.kind=='rasch' else np.r_[ga,gb]
  if self.kind=='map':f+=.5*np.sum((z[:self.I]/.5)**2)+.5*np.sum((b/3)**2);g+=np.r_[z[:self.I]/.25,b/9]
  self.calls+=1
  if full:return {'a':a,'b':b,'logML':norm,'posterior':post,'theta':post@self.t,'var':post@(self.t*self.t)-(post@self.t)**2,'objective':float(f),'gradient':g}
  return float(f),g

def projected(g,z,bounds):
 g=g.copy()
 for j,(lo,hi) in enumerate(bounds):
  if z[j]<=lo+1e-7 and g[j]>0:g[j]=0
  if z[j]>=hi-1e-7 and g[j]<0:g[j]=0
 return g

def fit(y,kind,start,label):
 obj=Objective(y,kind);I=len(y);b0=-logit((y.sum(1)+.5)/(y.shape[1]+1));z=b0.copy() if kind=='rasch' else np.r_[np.full(I,np.log(start)),b0]
 bounds=[(-10,10)]*I if kind=='rasch' else [(np.log(.2),np.log(5))]*I+[(-10,10)]*I
 # Deterministic coordinate central differences on initial objective.
 _,grad=obj.calc(z);coords=np.unique(np.linspace(0,len(z)-1,12,dtype=int));errors=[]
 for j in coords:
  zp=z.copy();zm=z.copy();zp[j]+=1e-5;zm[j]-=1e-5
  fd=(obj.calc(zp)[0]-obj.calc(zm)[0])/2e-5;errors.append(abs(fd-grad[j]))
 assert max(errors)<1e-4,(label,max(errors))
 trace=[]
 def cb(z):
  f,g=obj.calc(z);trace.append([f,float(np.max(abs(projected(g,z,bounds))))])
 t=time.monotonic();res=minimize(obj.calc,z,jac=True,method='L-BFGS-B',bounds=bounds,callback=cb,options={'maxiter':1200,'maxls':50,'ftol':1e-12,'gtol':1e-5,'maxcor':20})
 val=obj.calc(res.x,True);fine=Objective(y,kind,321).calc(res.x,True);pg=float(np.max(abs(projected(val['gradient'],res.x,bounds))))
 hit=(abs(np.log(val['a'])-np.log(.2))<1e-4)|(abs(np.log(val['a'])-np.log(5))<1e-4) if kind!='rasch' else np.zeros(I,dtype=bool)
 b_hit=abs(val['b'])>10-1e-4
 dt=float(np.max(abs(val['theta']-fine['theta'])));dl=float(np.max(abs(val['logML']-fine['logML'])))
 numerical=bool(res.success and pg<=.01 and np.isfinite(val['theta']).all() and dt<=.02 and dl<=.02 and not b_hit.any() and (kind=='rasch' or hit.mean()<=.1))
 np.savez_compressed(OUT/(label+'.npz'),Y=y,parameters=res.x,discrimination=val['a'],difficulty=val['b'],theta=val['theta'],variance=val['var'],posterior=val['posterior'],logML=val['logML'],theta321=fine['theta'],logML321=fine['logML'],gradient=val['gradient'],trace=np.array(trace),a_boundary=hit,b_boundary=b_hit)
 report={'label':label,'kind':kind,'start_a':start,'seconds':time.monotonic()-t,'optimizer_success':bool(res.success),'message':str(res.message),'nit':int(res.nit),'nfev':int(res.nfev),'projected_gradient_max':pg,'finite_theta':bool(np.isfinite(val['theta']).all()),'slope_boundary_count':int(hit.sum()),'difficulty_boundary_count':int(b_hit.sum()),'n_items':I,'logML':float(val['logML'].sum()),'objective':val['objective'],'quadrature_theta_maxdiff':dt,'quadrature_logML_person_maxdiff':dl,'numeric_screen_pass':numerical,'theta_accuracy_rho':float(spearmanr(val['theta'],y.mean(0)).statistic),'posterior_reliability':float(np.var(val['theta'],ddof=1)/(np.var(val['theta'],ddof=1)+val['var'].mean())),'gradient_check_maxabs':max(errors),'scientific_limit':'Numerical screen only; no construct validity or geometry inference release','arrays_sha256':sha(OUT/(label+'.npz'))}
 (OUT/(label+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return report

def main():
 c=json.loads(C.read_text());approval=json.loads((R/'verifier/CT_MEDICAL_REPAIR_APPROVED.json').read_text());assert approval['contract_sha256']==sha(C)
 OUT.mkdir(exist_ok=False);summary={'producer':'/root','contract_sha256':sha(C),'code_sha256':sha(__file__),'variants':{},'status':'PRODUCTION_PENDING_INDEPENDENT_REVIEW'}
 for name,info in c['inputs'].items():
  for k,h in info['hashes'].items():assert sha(R/info[k])==h
  m=np.load(R/info['matrix'],allow_pickle=False);y=m['matrix'][m['fit_mask']].astype(float);meta=json.loads((R/info['metadata']).read_text());assert len(meta['tags'])==y.shape[1]
  records=[]
  for kind,start in [('2pl',1.),('2pl',.7),('rasch',1.),('map',1.)]:records.append(fit(y,kind,start,name+'_'+kind+'_'+str(start)))
  one=np.load(OUT/(name+'_2pl_1.0.npz'));two=np.load(OUT/(name+'_2pl_0.7.npz'))
  stability={'theta_maxdiff':float(np.max(abs(one['theta']-two['theta']))),'logML_perperson_difference':float(abs(one['logML'].sum()-two['logML'].sum())/y.shape[1])}
  stability['passes']=stability['theta_maxdiff']<=.05 and stability['logML_perperson_difference']<=.02
  for rec in records:rec['score_acceptance_candidate']=bool(rec['numeric_screen_pass'] and (rec['kind']!='2pl' or stability['passes']))
  summary['variants'][name]={'tags':meta['tags'],'matrix_path':info['matrix'],'fit_mask_hash':hashlib.sha256(m['fit_mask'].tobytes()).hexdigest(),'missing_cells_total':int((~m['present']).sum()),'shape':list(y.shape),'fits':records,'unpenalized_start_stability':stability}
  (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (R/'reports/NEW_MEDICAL_REPAIR.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
