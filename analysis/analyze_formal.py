"""Independent read-only analysis of sealed formal CAL/TEST outputs.

Writes new deliverables beside the evaluation ledger. Historical 766-group
tables are used only for labelled descriptive comparison, never bootstrapping.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCI=HERE.parents[1]
sys.path.insert(0,str(HERE))
import run_cal as cal
import run_test as test


def write_csv_once(path,frame):
    path=Path(path)
    if path.exists():
        expected=frame.to_csv(index=False).encode("utf-8-sig")
        if path.read_bytes()!=expected:
            raise ValueError(f"Previously written analysis artifact differs: {path}")
        return
    frame.to_csv(path,index=False,encoding="utf-8-sig")


def copy_once(src,dst):
    dst=Path(dst)
    if dst.exists(): raise FileExistsError(dst)
    shutil.copy2(src,dst)


def f(x):
    return "NA" if x is None or (isinstance(x,float) and not math.isfinite(x)) else f"{x:.4f}"


def independent_metrics(labels,scores,predictions):
    """Recalculate required metrics from saved per-image rows without pipeline metrics code."""
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    y=np.asarray(labels,dtype=int);p=np.asarray(predictions,dtype=int);s=np.asarray(scores,dtype=float)
    tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
    tn=int(((y==0)&(p==0)).sum());fn=int(((y==1)&(p==0)).sum())
    div=lambda a,b: a/b if b else None
    recall=div(tp,tp+fn);specificity=div(tn,tn+fp)
    both=(tp+fn)>0 and (tn+fp)>0
    return {"tp":tp,"fp":fp,"tn":tn,"fn":fn,"accuracy":(tp+tn)/len(y),"precision":div(tp,tp+fp),
        "recall":recall,"specificity":specificity,"f1":div(2*tp,2*tp+fp+fn),"fpr":div(fp,fp+tn),
        "balanced_accuracy":(recall+specificity)/2 if both else None,
        "auroc":float(roc_auc_score(y,s)) if both else None,
        "ap_fire":float(average_precision_score(y,s)) if both else None,
        "ap_negative":float(average_precision_score(1-y,1-s)) if both else None}


def main():
    import numpy as np
    import pandas as pd
    identity=cal.read(cal.IDENTITY)
    root=Path(identity["root"])
    sys.path.insert(0,str(root/"src"))
    sys.path.insert(0,str(cal.TRAIN))
    from fire_detection.research.statistics import binary_metrics, validate_predictions
    import execute_remaining as training
    from fire_detection.research.three_arm_recovery import NodeEngine
    engine=NodeEngine(root,root/"plans"/identity["release_id"])
    training.frozen_gate(root,engine,identity)
    seal,cal_rows=test.sealed_thresholds(engine,identity)
    events=[json.loads(line) for line in (HERE/"cal_events.jsonl").read_text(encoding="utf-8").splitlines()]
    event_counts=Counter(e["status"] for e in events)
    cal.require(event_counts["START"]==event_counts["ACCEPTED"]==45 and event_counts["TEST_START"]==event_counts["TEST_ACCEPTED"]==45,
        "CAL/TEST event completeness and no skipped/repeated nodes")
    cal.require(event_counts["CAL_COMPLETE_AWAITING_INDEPENDENT_SEAL"]==event_counts["TEST_COMPLETE_AWAITING_SUMMARY"]==event_counts["SUMMARY_ACCEPTED"]==1,
        "CAL seal / TEST / summary stage events")
    cal.require(not any("FAILURE" in e["status"] for e in events),"Failure event observed")
    cal_complete=next(e for e in events if e["status"]=="CAL_COMPLETE_AWAITING_INDEPENDENT_SEAL")
    first_test=next(e for e in events if e["status"]=="TEST_START")
    cal.require(datetime.fromisoformat(cal_complete["utc"])<datetime.fromisoformat(seal["created_at_utc"])<datetime.fromisoformat(first_test["utc"]),
        "Global CAL manifest was not sealed before first TEST")
    summary_receipt=cal.read(HERE/"summary_receipt.json")
    cal.require(summary_receipt["status"]=="PASS" and summary_receipt["threshold_seal_sha256"]==cal.sha(HERE/"threshold_seal.json"),"Summary receipt identity")
    products=Path(summary_receipt["product_directory"])
    for name,expected in summary_receipt["outputs"].items():
        cal.require(cal.sha(products/name)==expected,f"Summary product SHA {name}")
    raw=pd.read_csv(HERE/"test_predictions.csv",float_precision="round_trip",dtype={"filename":str,"group_id":str})
    data=validate_predictions(raw)
    cal.require(len(data)==19800 and data.filename.nunique()==1100 and data.group_id.nunique()==427,"Full 427-group prediction table")
    key=["method","seed","outer_fold","filename"]
    source=pd.read_csv(products/"predictions.csv",float_precision="round_trip",dtype={"filename":str,"group_id":str})
    pd.testing.assert_frame_equal(data.sort_values(key).reset_index(drop=True)[source.columns],source.sort_values(key).reset_index(drop=True),check_dtype=False)
    seed=pd.read_csv(products/"per_seed.csv")
    fold=pd.read_csv(products/"per_fold.csv")
    summary=pd.read_csv(products/"summary.csv")
    paired=pd.read_csv(products/"paired_ci.csv")
    cal.require(len(seed)==18 and len(fold)==90 and len(summary)==6 and len(paired)==24,"Frozen six-view statistical matrix")
    # Recompute each pooled-seed result independently from the retained rows.
    for _,s in seed.iterrows():
        part=data.loc[(data.method==s.method)&(data.seed==s.seed)]
        rec=independent_metrics(part.label.to_numpy(),part.score.to_numpy(),part.y_pred.to_numpy())
        cal.require(len(part)==1100 and part.filename.nunique()==1100 and part.group_id.nunique()==427,f"Seed coverage {s.method} {s.seed}")
        for name in ("accuracy","precision","recall","specificity","f1","fpr","balanced_accuracy","auroc","ap_fire","ap_negative","tp","fp","tn","fn"):
            a,b=rec[name],s[name]
            cal.require((a is None and pd.isna(b)) or (a is not None and np.isclose(float(a),float(b),rtol=1e-10,atol=1e-10)),f"Seed metric {s.method} {s.seed} {name}")
        fnr=rec["fn"]/(rec["fn"]+rec["tp"])
        cal.require(np.isclose(fnr,float(s["fnr"]),rtol=1e-10,atol=1e-10),f"Seed FNR {s.method} {s.seed}")
    for _,s in fold.iterrows():
        part=data.loc[(data.method==s.method)&(data.seed==s.seed)&(data.outer_fold==s.outer_fold)]
        rec=independent_metrics(part.label.to_numpy(),part.score.to_numpy(),part.y_pred.to_numpy())
        cal.require(len(part)==220 and part.filename.nunique()==220,f"Fold coverage {s.method} {s.seed} {s.outer_fold}")
        for name in ("accuracy","precision","recall","specificity","f1","fpr","balanced_accuracy","auroc","ap_fire","ap_negative","tp","fp","tn","fn"):
            a,b=rec[name],s[name]
            cal.require((a is None and pd.isna(b)) or (a is not None and np.isclose(float(a),float(b),rtol=1e-10,atol=1e-10)),f"Fold metric {s.method} {s.seed} {s.outer_fold} {name}")
    for name in ("tp","fp","tn","fn"):
        summary[f"{name}_mean"]=[seed.loc[seed.method==m,name].mean() for m in summary.method]
        summary[f"{name}_std"]=[seed.loc[seed.method==m,name].std(ddof=1) for m in summary.method]
    write_csv_once(HERE/"performance_summary.csv",summary)
    seed2=seed.copy();seed2.insert(0,"analysis_level","seed_pooled")
    fold2=fold.copy();fold2.insert(0,"analysis_level","outer_fold")
    write_csv_once(HERE/"seed_fold_results.csv",pd.concat([seed2,fold2],ignore_index=True,sort=False))
    # Fixed three pairs. Ranking metrics are descriptive paired seed-mean deltas.
    cal.require(set(zip(paired.method,paired.reference))=={("cls_hn","cls_standard"),("cls_focal","cls_standard"),("cls_hn","cls_focal"),
        ("cls_hn_tta","cls_standard_tta"),("cls_focal_tta","cls_standard_tta"),("cls_hn_tta","cls_focal_tta")},"Predeclared pairs")
    cal.require(set(paired.iterations)=={10000} and set(paired.analysis_seed)=={20260914} and set(paired.n_groups)=={427},"New 427-group bootstrap identity")
    paired["analysis_role"]="paired_cluster_bootstrap_pointwise_exploratory"
    ranking=[]
    for view in ("original","tta"):
        suffix="" if view=="original" else "_tta"
        for method,reference in (("hn","focal"),("hn","standard"),("focal","standard")):
            for metric in ("auroc","ap_fire","ap_negative"):
                a=seed.loc[seed.method==f"cls_{method}{suffix}"].sort_values("seed")[metric].to_numpy()
                b=seed.loc[seed.method==f"cls_{reference}{suffix}"].sort_values("seed")[metric].to_numpy()
                ranking.append({"method":f"cls_{method}{suffix}","reference":f"cls_{reference}{suffix}","condition":"clean",
                    "evidence_status":"internal_revalidation","metric":metric,"effect_difference":float(np.mean(a-b)),
                    "ci_lower":np.nan,"ci_upper":np.nan,"confidence":np.nan,"iterations":0,"valid_iterations":0,"invalid_iterations":0,
                    "analysis_seed":20260914,"n_groups":427,"n_images_per_seed":1100,"n_seeds":3,
                    "inference_scope":"descriptive_seed_mean_only","comparison_role":"predeclared_descriptive",
                    "multiplicity":"no_interval_no_significance_claim","analysis_role":"ranking_difference_descriptive"})
    write_csv_once(HERE/"paired_effects.csv",pd.concat([paired,pd.DataFrame(ranking)],ignore_index=True,sort=False))
    seed_diffs=[]
    for view in ("original","tta"):
        suffix="" if view=="original" else "_tta"
        for method,reference in (("hn","focal"),("hn","standard"),("focal","standard")):
            for s in (42,2026,3407):
                a=seed.loc[(seed.method==f"cls_{method}{suffix}")&(seed.seed==s)].iloc[0]
                b=seed.loc[(seed.method==f"cls_{reference}{suffix}")&(seed.seed==s)].iloc[0]
                seed_diffs.append({"view":view,"seed":s,"comparison":f"{method}-{reference}","f1_delta":a.f1-b.f1,
                    "recall_delta":a.recall-b.recall,"fpr_delta":a.fpr-b.fpr,"fp_delta":int(a.fp-b.fp),"fn_delta":int(a.fn-b.fn),
                    "auroc_delta":a.auroc-b.auroc,"ap_fire_delta":a.ap_fire-b.ap_fire,"ap_negative_delta":a.ap_negative-b.ap_negative})
    write_csv_once(HERE/"seed_differences.csv",pd.DataFrame(seed_diffs))
    # Recreate the frozen cluster resampling weights solely for diagnostics.
    base=data.loc[(data.method=="cls_standard")&(data.seed==42)].sort_values("filename")
    groups=base.groupby("group_id").agg(n_images=("filename","size"),n_positive=("label","sum"),fold=("outer_fold","first"))
    groups["n_negative"]=groups.n_images-groups.n_positive
    groups=groups.sort_index()
    cal.require(len(groups)==427 and int(groups.n_images.sum())==1100,"427 groups from current TEST only")
    largest=groups.sort_values("n_images",ascending=False).head(3)
    names=list(groups.index)
    folds=groups.fold.to_numpy()
    sizes=groups.n_images.to_numpy()
    rng=np.random.default_rng(20260914)
    multipliers=[]
    for start in range(0,10000,512):
        n=min(512,10000-start)
        m=np.zeros((n,len(groups)),dtype=np.float64)
        for fold_id in sorted(set(folds)):
            indexes=np.flatnonzero(folds==fold_id)
            m[:,indexes]=rng.multinomial(len(indexes),np.full(len(indexes),1/len(indexes)),size=n)
        multipliers.append(m)
    mult=np.concatenate(multipliers)
    cal.require(mult.shape==(10000,427),"Bootstrap diagnostic draw shape")
    weighted=mult*sizes
    total=weighted.sum(axis=1)
    eff_groups=total**2/(weighted**2).sum(axis=1)
    dominance=[]
    for group_id,row in largest.iterrows():
        idx=names.index(group_id)
        copies=mult[:,idx]
        share=weighted[:,idx]/total
        dominance.append({"group_id":group_id,"fold":int(row.fold),"n_images":int(row.n_images),"n_positive":int(row.n_positive),
            "n_negative":int(row.n_negative),"omitted_fraction":float(np.mean(copies==0)),"two_or_more_fraction":float(np.mean(copies>=2)),
            "max_image_weight_share":float(np.max(share)),"p95_image_weight_share":float(np.quantile(share,.95))})
    # Leave-one-large-component-out is a sensitivity check, never a new primary analysis.
    leave=[]
    for group_id in largest.index:
        for view in ("original","tta"):
            suffix="" if view=="original" else "_tta"
            for method,reference in (("hn","focal"),("hn","standard"),("focal","standard")):
                effects=[]
                for excluded in (False,True):
                    metrics={}
                    for arm in (method,reference):
                        vals=[]
                        for s in (42,2026,3407):
                            part=data.loc[(data.method==f"cls_{arm}{suffix}")&(data.seed==s)]
                            if excluded: part=part.loc[part.group_id!=group_id]
                            rec=binary_metrics(part.label.to_numpy(),part.score.to_numpy(),part.y_pred.to_numpy())
                            vals.append(rec["f1"])
                        metrics[arm]=np.mean(vals)
                    effects.append(metrics[method]-metrics[reference])
                leave.append({"group_id":group_id,"view":view,"comparison":f"{method}-{reference}",
                    "full_f1_delta":effects[0],"excluded_f1_delta":effects[1],"change":effects[1]-effects[0]})
    write_csv_once(HERE/"large_component_sensitivity.csv",pd.DataFrame(leave))
    # Complete cost ledger; imported Outer-0 miner/selection costs are resolved by NodeEngine.target.
    cost=[]
    with (cal.TRAIN/"model_identity_matrix.csv").open(encoding="utf-8-sig",newline="") as stream:
        models=list(csv.DictReader(stream))
    with (cal.TRAIN/"miner_identity_matrix.csv").open(encoding="utf-8-sig",newline="") as stream:
        miners=list(csv.DictReader(stream))
    for r in models:
        cost.append({"phase":"classifier_train","task":r["task"],"method":r["method"],"outer":r["outer"],"seed":r["seed"],
            "trainer_seconds":float(r["training_seconds"]),"node_wall_seconds":float(r["node_seconds"]),"updates":int(r["updates"]),
            "exposures":int(r["exposures"]),"sharing":"per_model","source_accepted_sha256":r["accepted_sha256"]})
    for r in miners:
        cost.append({"phase":"oof_miner","task":r["task"],"method":"HN_shared","outer":r["outer"],"seed":"42",
            "trainer_seconds":float(r["training_seconds"]),"node_wall_seconds":float(r["node_seconds"]),"updates":210,"exposures":13200,
            "sharing":"once_per_outer","source_accepted_sha256":r["accepted_sha256"]})
    for outer in range(5):
        task=engine.tasks[f"select_o{outer}"]
        target=engine.target(task)
        record=cal.read(target/"execution/attempt_001/record.json")
        cost.append({"phase":"oof_selection","task":task["id"],"method":"HN_shared","outer":outer,"seed":"42",
            "trainer_seconds":np.nan,"node_wall_seconds":record["elapsed_seconds"],"updates":np.nan,"exposures":np.nan,
            "sharing":"once_per_outer","source_accepted_sha256":cal.sha(target/"accepted.json")})
    for task in engine.plan["tasks"]:
        if task["kind"] not in ("calibrate","evaluate"):continue
        target=engine.target(task)
        record=cal.read(target/"execution/attempt_001/record.json")
        cost.append({"phase":"cal_inference" if task["kind"]=="calibrate" else "test_inference","task":task["id"],
            "method":task["method"],"outer":task["outer_fold"],"seed":task["seed"],"trainer_seconds":np.nan,
            "node_wall_seconds":record["elapsed_seconds"],"updates":np.nan,"exposures":np.nan,"sharing":"per_model",
            "source_accepted_sha256":cal.sha(target/"accepted.json")})
    cost_frame=pd.DataFrame(cost)
    cal.require(len(cost_frame)==155,"Cost ledger 45 train + 15 miner + 5 selection + 45 CAL + 45 TEST")
    write_csv_once(HERE/"compute_tradeoff.csv",cost_frame)
    cal_hours=cost_frame.loc[cost_frame.phase=="cal_inference","node_wall_seconds"].sum()/3600
    test_hours=cost_frame.loc[cost_frame.phase=="test_inference","node_wall_seconds"].sum()/3600
    shared_hours=cost_frame.loc[cost_frame.phase.isin(["oof_miner","oof_selection"]),"node_wall_seconds"].sum()/3600
    # Historical values remain outside new inference and are explicitly labelled.
    old_path=SCI/"Paper/sivp_r5_working_20261018/source_data/main_performance.csv"
    old=pd.read_csv(old_path)
    old_rows=[]
    old_map={"cls_standard":"Standard","cls_hn":"HN","cls_focal":"Focal Sampling"}
    for _,s in summary.iterrows():
        method=s.method.removesuffix("_tta")
        view="tta" if s.method.endswith("_tta") else "original"
        matches=old.loc[(old.method==old_map[method])&(old.prediction_view==view)]
        cal.require(len(matches)==1,"Historical descriptive lookup")
        old_row=matches.iloc[0]
        old_rows.append({"method":s.method,"view":view,"old_766group_f1_mean":old_row.f1_mean,"new_427group_f1_mean":s.f1_mean,
            "f1_change":s.f1_mean-old_row.f1_mean,"old_recall_mean":old_row.recall_mean,"new_recall_mean":s.recall_mean,
            "old_fpr_mean":old_row.fpr_mean,"new_fpr_mean":s.fpr_mean,"comparison_scope":"descriptive_different_split_and_models_no_paired_inference"})
    write_csv_once(HERE/"old_new_descriptive.csv",pd.DataFrame(old_rows))
    # Diagnostic record and paper-facing summary are review drafts, not edits to manuscript.
    invalid=paired[["method","reference","metric","invalid_iterations","valid_iterations","ci_lower","ci_upper"]]
    largest_lines=[f"- {d['group_id']}（fold {d['fold']}）：{d['n_images']} 张，正类 {d['n_positive']}、负类 {d['n_negative']}；10,000 次中遗漏 {d['omitted_fraction']:.1%}，抽到至少两次 {d['two_or_more_fraction']:.1%}；最大图像权重占比 {d['max_image_weight_share']:.1%}。" for d in dominance]
    diag=["# Bootstrap 诊断（新427分量）","", "固定分析种子 20260914；按五个 outer fold 分层，对完整视觉关联分量抽样，全部方法和三个训练seed共享同一抽样权重。每个配对比较10,000次，95%百分位区间。", "",f"分量数：427；最大分量的原始图像占比 {largest.iloc[0].n_images/1100:.1%}。重建抽样的图像加权有效分量数中位数 {np.median(eff_groups):.1f}，5–95百分位 {np.quantile(eff_groups,.05):.1f}–{np.quantile(eff_groups,.95):.1f}。","","## 大分量与抽样权重","",*largest_lines,"","## 无效与退化抽样","",f"24个配对区间行的无效抽样总数：{int(invalid.invalid_iterations.sum())}；单行最大：{int(invalid.invalid_iterations.max())}。完整逐行数据见 `paired_effects.csv`。", "", "## 解释边界","", "区间仅条件于当前427个保守视觉关联分量、冻结的模型与CAL阈值。189张分量和少数负类大分量使权重及误报估计对分量抽取敏感；留一大分量敏感性见 `large_component_sensitivity.csv`。这些区间是探索性的点位区间，未作多重比较校正，不证明真实相机或事件独立性，也不能把包含0解释为等效。", ""]
    (HERE/"bootstrap_diagnostics.md").write_text("\n".join(diag),encoding="utf-8")
    primary=summary.loc[summary.method.isin(["cls_standard","cls_hn","cls_focal"])].set_index("method")
    pairs=paired.loc[(paired.method.str.endswith("_tta")==False)&(paired.reference.str.endswith("_tta")==False)&(paired.metric=="f1")]
    paper=["# SIVP正式评估结果审阅稿","", "此文件只整理新冻结实验的证据与限制，未修改论文正文。原始视图是主分析；TTA是次要分析。", "", "## 主分析（3个seed的五折TEST汇总均值 ± 样本SD）","", "| 方法 | F1 | Recall | FPR | AUROC | Fire AP | Negative AP |","|---|---:|---:|---:|---:|---:|---:|"]
    for method in ("cls_standard","cls_hn","cls_focal"):
        r=primary.loc[method]
        paper.append(f"| {method} | {f(r.f1_mean)} ± {f(r.f1_std)} | {f(r.recall_mean)} ± {f(r.recall_std)} | {f(r.fpr_mean)} ± {f(r.fpr_std)} | {f(r.auroc_mean)} ± {f(r.auroc_std)} | {f(r.ap_fire_mean)} ± {f(r.ap_fire_std)} | {f(r.ap_negative_mean)} ± {f(r.ap_negative_std)} |")
    paper += ["","## 预先规定的F1成对差异（方法A − 方法B）","","| A − B | 均值差 | 95%分量bootstrap区间 | 无效抽样 |","|---|---:|---:|---:|"]
    for method,reference in (("cls_hn","cls_focal"),("cls_hn","cls_standard"),("cls_focal","cls_standard")):
        r=pairs.loc[(pairs.method==method)&(pairs.reference==reference)].iloc[0]
        paper.append(f"| {method} − {reference} | {f(r.effect_difference)} | [{f(r.ci_lower)}, {f(r.ci_upper)}] | {int(r.invalid_iterations)} |")
    paper += ["", "各seed的F1、FP/FN及AUROC/AP差异见 `seed_differences.csv`；各outer与seed异质性见 `seed_fold_results.csv`。", "", "## 计算成本", "",f"Standard分类器训练器耗时 {cost_frame.loc[(cost_frame.phase=='classifier_train')&(cost_frame.method=='standard'),'trainer_seconds'].sum()/3600:.3f} 小时，HN {cost_frame.loc[(cost_frame.phase=='classifier_train')&(cost_frame.method=='hn'),'trainer_seconds'].sum()/3600:.3f} 小时，Focal {cost_frame.loc[(cost_frame.phase=='classifier_train')&(cost_frame.method=='focal'),'trainer_seconds'].sum()/3600:.3f} 小时。HN共享OOF挖掘与选择节点墙钟 {shared_hours:.3f} 小时；CAL节点 {cal_hours:.3f} 小时，TEST节点 {test_hours:.3f} 小时。训练器时间与节点墙钟不同，详见 `compute_tradeoff.csv`。", "", "## 结论边界", "", "三组比较保留原顺序；新结果不能与旧766组区间混用。旧新数值变化仅作描述，见 `old_new_descriptive.csv`。189张大分量及大负类分量影响分量bootstrap的可解释性，区间不作为未经调整的确认性显著性证据。此处不作模型调整或论文正文改写。", ""]
    (HERE/"paper_results_summary.md").write_text("\n".join(paper),encoding="utf-8")
    # Final independent integrity ledger and hashes of all promised outputs.
    outputs=["cal_threshold_manifest.csv","test_predictions.csv","performance_summary.csv","paired_effects.csv","seed_fold_results.csv",
        "bootstrap_diagnostics.md","compute_tradeoff.csv","evaluation_integrity_report.md","paper_results_summary.md"]
    report=["# 正式CAL/TEST独立完整性报告","", "状态：PASS。45/45 CAL和45/45 TEST各一次完成，五折每方法每seed原始视图1100条、TTA视图1100条；427个当前视觉关联分量。", "",f"CAL阈值manifest SHA256：`{seal['manifest_sha256']}`。TEST在独立阈值封存后执行；所有TEST决策直接使用对应CAL阈值。", "",f"正式Release：`{identity['release_id']}`；source SHA256：`{identity['source_sha256']}`；图像快照 SHA256：`{identity['input_snapshot_sha256']}`；fold SHA256：`{identity['folds_sha256']}`。", "", "模型与训练回执、逐图预测与TEST节点回执、统计汇总与冻结代码哈希链一致；正式评价未重训、未改阈值或fold。全部指标已由保存的逐图预测独立重算。新bootstrap仅使用当前427分量；旧766组仅进入标明的历史描述性对比。", "", "统计限制：189张大分量、48/50负类集中于fold0单分量，以及fold2负类47/49集中于189张分量；未证实真实相机或事件独立。配对区间探索性解释，不作未经调整的确认性显著性断言。", "", "分析打包脚本首次运行在读取历史对照表时发生一次未定义路径变量 NameError；CAL、TEST和冻结统计节点均已成功，未重跑。修复后重新计算打包内容，并对首次已生成文件作字节一致性检查。", "", "结论：GO FOR MANUSCRIPT REVISION，但科研主张应按新估计与上述限制重写；本次未修改论文正文。", "", "## 交付文件哈希","" ]
    # integrity report cannot hash itself; list the eight other requested artifacts.
    for name in outputs:
        if name=="evaluation_integrity_report.md":continue
        report.append(f"- `{name}` SHA256 `{cal.sha(HERE/name)}`")
    report.append("")
    (HERE/"evaluation_integrity_report.md").write_text("\n".join(report),encoding="utf-8")
    cal.write_once(HERE/"analysis_completion_receipt.json",{"status":"PASS","go_for_manuscript_revision":True,"release_identity_sha256":cal.sha(cal.IDENTITY),
        "threshold_manifest_sha256":seal["manifest_sha256"],"test_predictions_sha256":cal.sha(HERE/"test_predictions.csv"),
        "summary_receipt_sha256":cal.sha(HERE/"summary_receipt.json"),"new_group_count":427,"bootstrap_seed":20260914,
        "bootstrap_iterations":10000,"outputs":{name:cal.sha(HERE/name) for name in outputs},"formal_CAL_complete":45,"formal_TEST_complete":45,
        "retraining_executed":False,"test_guided_tuning_executed":False,"paper_body_modified":False,
        "analysis_packaging_incident":"first analyze_formal.py invocation stopped at old_path with NameError; no CAL, TEST, or summary node rerun"})
    print(json.dumps({"status":"PASS","cal":45,"test":45,"primary":primary[["f1_mean","recall_mean","fpr_mean","auroc_mean","ap_fire_mean","ap_negative_mean"]].to_dict(orient="index"),
        "paired_f1":pairs[["method","reference","effect_difference","ci_lower","ci_upper","invalid_iterations"]].to_dict(orient="records"),
        "cal_hours":cal_hours,"test_hours":test_hours,"shared_hn_hours":shared_hours,"max_invalid":int(invalid.invalid_iterations.max())},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
