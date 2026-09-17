#!/usr/bin/env python3
"""Run a missing historical CPU IRT fit, preserving matrices and diagnostics.

Matches H3's girth call and separate posterior-reliability calculation. A
read-only trace records termination state; it does not change the fit algorithm.
"""
import argparse
from collections import Counter
import hashlib
import inspect
import importlib.metadata
import json
import platform
import sys
import warnings
import traceback
from pathlib import Path

import numpy as np
import scipy
from girth import ability_eap, grm_mml, twopl_mml
from girth.utilities import default_options
from scipy.stats import spearmanr

R = Path(__file__).resolve().parents[1]
CONTRACT = R/"config/STUDY2_FIT_REUSE_V1.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finite_json(value):
    if isinstance(value,np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value,dict):
        return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):
        return int(value)
    if isinstance(value,np.bool_):
        return bool(value)
    return value


def save(path,value):
    path.write_text(json.dumps(finite_json(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def posterior_stats(matrix,b,a):
    nodes = np.linspace(-5,5,61)
    weights = np.exp(-.5*nodes**2)
    weights /= weights.sum()
    logits = a[:,None]*(nodes[None,:]-b[:,None])
    p1,p0 = -np.logaddexp(0,-logits),-np.logaddexp(0,logits)
    means,varis = [],[]
    for j in range(matrix.shape[1]):
        ll = np.where(matrix[:,j,None]==1,p1,p0).sum(0)
        posterior = np.log(weights+1e-300)+ll
        posterior -= posterior.max()
        posterior = np.exp(posterior)
        posterior /= posterior.sum()
        mean = float(posterior@nodes)
        means.append(mean)
        varis.append(float(posterior@((nodes-mean)**2)))
    return np.array(means),np.array(varis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant",choices=["science_A","medical_A","science_drop3"],required=True)
    args = ap.parse_args()
    c = json.loads(CONTRACT.read_text())
    for desc in c["inputs"].values():
        if sha(desc["path"]) != desc["sha256"]:
            raise ValueError("Input/source changed: "+desc["path"])
    actual_environment = {"python":platform.python_version(),"numpy":np.__version__,
                          "scipy":scipy.__version__,"girth":importlib.metadata.version("girth")}
    for key,value in actual_environment.items():
        if value != c["environment"][key]:
            raise ValueError("Environment changed: "+key)
    domain = args.variant.split("_")[0]
    tags = sorted(c["panels"][domain])
    if args.variant == "science_drop3":
        tags = [t for t in tags if t not in c["drop3_tags"]]
    observation_receipt = json.loads(Path(c["inputs"]["observations"]["path"]).read_text())
    locked_cells = {row["cell"]:row for row in observation_receipt["cells"]}
    rows = []
    for tag in tags:
        key = tag+"|"+domain
        path = R/f"runs/study2_reuse_v1/observations/{tag}__{domain}.json"
        bound = locked_cells[key]
        if path.resolve() != Path(bound["output"]).resolve() or sha(path) != bound["sha256"]:
            raise ValueError("Per-cell observation binding changed: "+key)
        row = json.loads(path.read_text())
        if row["cell"] != key:
            raise ValueError("Cell identity changed: "+key)
        rows.append(row)
    ids = sorted(set(i for row in rows for i in row["item_ids"]))
    maps = [dict(zip(row["item_ids"],row["response_binary"])) for row in rows]
    matrix = np.array([[values.get(i,0) for values in maps] for i in ids],dtype=int)
    present = np.array([[i in values for values in maps] for i in ids],dtype=bool)
    keep = (matrix.sum(1)>0)&(matrix.sum(1)<len(tags))
    fit = matrix[keep]
    out = R/f"runs/study2_reuse_v1/fits/{args.variant}"
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use new version path; existing fit will not be overwritten")
    out.mkdir(parents=True,exist_ok=True)
    np.savez(out/"response_matrix.npz",matrix=matrix,present=present,fit_mask=keep)
    save(out/"inputs.json",{"tags":tags,"item_ids":ids,"fit_item_ids":[i for i,k in zip(ids,keep) if k],
                             "n_items_total":len(ids),"n_items_fit":int(keep.sum()),
                             "observation_hashes":{t:sha(R/f"runs/study2_reuse_v1/observations/{t}__{domain}.json") for t in tags},
                             "accuracy_available_mean":[r["accuracy_available_mean"] for r in rows],
                             "matrix_mean_accuracy":matrix.mean(0),"missing_per_model":(~present).sum(0)})
    defaults = default_options()
    distribution = defaults["distribution"].__self__
    if distribution.dist.name != "norm" or distribution.mean() != 0.0 or distribution.std() != 1.0:
        raise ValueError("Effective distribution is not the expected standard normal")
    for name,expected in c["defaults"].items():
        if name == "distribution":
            continue
        actual = defaults[name]
        if isinstance(actual,tuple): actual=list(actual)
        if actual != expected:
            raise ValueError("Effective girth default changed: "+name)
    termination = {}
    def trace(frame,event,arg):
        if frame.f_code is grm_mml.__code__:
            if event == "return":
                loc = frame.f_locals
                required = {"previous_discrimination","discrimination","iteration"}
                if required.issubset(loc):
                    try:
                        delta = np.abs(loc["previous_discrimination"]-loc["discrimination"]).max()
                        termination.update({"iterations":int(loc["iteration"])+1,
                                            "last_max_discrimination_delta":float(delta),
                                            "delta_below_1e_3":bool(delta<1e-3),
                                            "max_iteration_limit":int(defaults["max_iteration"]),
                                            "trace_complete":True})
                    except Exception as trace_error:
                        termination.update({"trace_complete":False,"trace_error":repr(trace_error)})
                else:
                    termination.update({"trace_complete":False,"missing_locals":sorted(required-set(loc))})
            return trace
        return None
    print(json.dumps({"fit_started":args.variant,"shape":list(fit.shape)}),flush=True)
    captured = []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            previous_trace = sys.gettrace()
            try:
                sys.settrace(trace)
                est = twopl_mml(fit)
            finally:
                sys.settrace(previous_trace)
            theta = ability_eap(fit,est["Difficulty"],est["Discrimination"])
            means,varis = posterior_stats(fit,est["Difficulty"],est["Discrimination"])
    except Exception as error:
        failure = {"variant":args.variant,"status":"FAILED_FIT_EXCEPTION","error":repr(error),
                   "traceback":traceback.format_exc(),"termination":termination,
                   "warnings":dict(Counter(type(w.message).__name__+": "+str(w.message) for w in captured)),
                   "environment":actual_environment,"contract_sha256":sha(CONTRACT),"code_sha256":sha(__file__)}
        save(out/"fit.json",failure)
        save(out/"receipt.json",{**failure,"outputs":{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
        raise
    var_mean = np.var(means,ddof=1)
    rel = float(var_mean/(var_mean+np.mean(varis)))
    np.savez(out/"fit_arrays.npz",theta=theta,difficulty=est["Difficulty"],discrimination=est["Discrimination"],
             posterior_mean_grid61=means,posterior_var_grid61=varis,theta_finite=np.isfinite(theta))
    p1 = json.loads(Path(c["inputs"]["p1_comparison"]["path"]).read_text())
    old = {r["tag"]:r for r in p1["cells_primary"]+p1["cells_medical"] if r["domain"]==domain}
    historical_compare = None
    if args.variant.endswith("_A") and all(isinstance(old.get(t,{}).get("theta"),(int,float)) and np.isfinite(old[t]["theta"]) for t in tags):
        historic = np.array([old[t]["theta"] for t in tags])
        historical_compare = {"full_model_set_equal":set(old)==set(tags),"max_abs_theta_difference":float(np.max(np.abs(theta-historic))),
                              "within_atol":bool(np.allclose(theta,historic,rtol=0,atol=1e-8))}
    acc = np.array([r["accuracy_available_mean"] for r in rows])
    report = {"variant":args.variant,"status":"FINITE_FIT" if np.isfinite(theta).all() else "FAILED_CRITERION_NONFINITE_THETA",
              "tags":tags,"theta":theta,"rel_th":rel,"n_items_fit":int(keep.sum()),"n_items_total":len(ids),
              "nonfinite_theta_count":int((~np.isfinite(theta)).sum()),"termination":termination,
              "theta_accuracy_rho":float(spearmanr(theta,acc).statistic) if np.isfinite(theta).all() else None,
              "historical_saved_theta_comparison":historical_compare,
              "warnings":dict(Counter(type(w.message).__name__+": "+str(w.message) for w in captured)),
              "contract_sha256":sha(CONTRACT),"code_sha256":sha(__file__),
              "environment":actual_environment,
              "algorithm_files":{str(inspect.getfile(fn)):sha(inspect.getfile(fn)) for fn in [grm_mml,twopl_mml,ability_eap]},
              "effective_defaults":c["defaults"],"science_acceptance":"NOT_RUN",
              "nan_serialization":"JSON null records nonfinite scalars; original IEEE values/masks preserved in fit_arrays.npz"}
    save(out/"fit.json",report)
    save(out/"receipt.json",{"variant":args.variant,"contract_sha256":sha(CONTRACT),"code_sha256":sha(__file__),
                             "outputs":{f.name:sha(f) for f in out.iterdir() if f.is_file()}})
    print(json.dumps(finite_json({k:report[k] for k in ["variant","status","n_items_fit","nonfinite_theta_count","rel_th","termination","theta_accuracy_rho","historical_saved_theta_comparison"]})))


if __name__ == "__main__":
    main()
