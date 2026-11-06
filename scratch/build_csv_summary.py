import json, csv, pathlib

results_dir = pathlib.Path("d:/PROJECT/MLNC/experiments/results")

comp_path = results_dir / "comparison_rms_vs_no_rms.json"
r3_path = results_dir / "run3_scores.json"

comp = json.loads(comp_path.read_text(encoding="utf-8")) if comp_path.exists() else {}
r1 = comp.get("run1_with_rms", {})
r2 = comp.get("run2_no_rms", {})
r3 = json.loads(r3_path.read_text(encoding="utf-8")) if r3_path.exists() else {}

s1 = json.loads((results_dir / "run1_with_rms_summary.json").read_text(encoding="utf-8"))
s2 = json.loads((results_dir / "run2_no_rms_summary.json").read_text(encoding="utf-8"))
s3 = json.loads((results_dir / "run3_split_10_15_summary.json").read_text(encoding="utf-8"))

rows = [
    {
        "Run_ID": "Run_1",
        "Experiment_Name": "Dual-CED-Tiny (Original Champion, With RMS)",
        "Architecture": "Dual-Student MoE (2 independent CED-Tiny)",
        "Student_Params_M": 10.84,
        "Total_MoE_Params_M": 191.52,
        "RMS_Norm": "0.07925 (Enabled)",
        "Low_Expert_SNR": "All SNRs (-5..20 dB)",
        "High_Expert_SNR": "20 dB Only",
        "Router_Threshold_dB": 18.0,
        "Low_Student_Best_Macro_Acc_%": round(s1["low"]["best_score"] * 100, 2),
        "Low_Best_Epoch": s1["low"]["best_epoch"],
        "High_Student_Best_Top1_Acc_%": round(s1["high"]["best_score"] * 100, 2),
        "High_Best_Epoch": s1["high"]["best_epoch"],
        "Pipeline_Overall_Top1_Acc_%": round(r1.get("top1_accuracy", 0.0) * 100, 2),
        "Pipeline_Overall_Top3_Acc_%": round(r1.get("top3_accuracy", 0.0) * 100, 2),
        "Pipeline_Balanced_Acc_%": round(r1.get("balanced_accuracy", 0.0) * 100, 2),
        "Pipeline_Macro_Precision_%": round(r1.get("precision_macro", 0.0) * 100, 2),
        "Pipeline_Macro_Recall_%": round(r1.get("recall_macro", 0.0) * 100, 2),
        "Pipeline_Macro_F1_%": round(r1.get("macro_f1", 0.0) * 100, 2),
        "Pipeline_Micro_F1_%": round(r1.get("micro_f1", 0.0) * 100, 2),
        "Pipeline_Macro_mAP_%": round(r1.get("mAP", 0.0) * 100, 2),
        "Pipeline_Macro_ROC_AUC_%": round(r1.get("macro_auc", 0.0) * 100, 2),
        "Acc_SNR_-5dB_%": round(r1.get("acc_snr_-5", 0.0) * 100, 2),
        "Acc_SNR_0dB_%": round(r1.get("acc_snr_0", 0.0) * 100, 2),
        "Acc_SNR_5dB_%": round(r1.get("acc_snr_5", 0.0) * 100, 2),
        "Acc_SNR_10dB_%": round(r1.get("acc_snr_10", 0.0) * 100, 2),
        "Acc_SNR_15dB_%": round(r1.get("acc_snr_15", 0.0) * 100, 2),
        "Acc_SNR_20dB_%": round(r1.get("acc_snr_20", 0.0) * 100, 2),
    },
    {
        "Run_ID": "Run_2",
        "Experiment_Name": "Dual-CED-Tiny (No RMS Ablation)",
        "Architecture": "Dual-Student MoE (2 independent CED-Tiny)",
        "Student_Params_M": 10.84,
        "Total_MoE_Params_M": 191.52,
        "RMS_Norm": "0.0 (Bypassed)",
        "Low_Expert_SNR": "All SNRs (-5..20 dB)",
        "High_Expert_SNR": "20 dB Only",
        "Router_Threshold_dB": 18.0,
        "Low_Student_Best_Macro_Acc_%": round(s2["low"]["best_score"] * 100, 2),
        "Low_Best_Epoch": s2["low"]["best_epoch"],
        "High_Student_Best_Top1_Acc_%": round(s2["high"]["best_score"] * 100, 2),
        "High_Best_Epoch": s2["high"]["best_epoch"],
        "Pipeline_Overall_Top1_Acc_%": round(r2.get("top1_accuracy", 0.0) * 100, 2),
        "Pipeline_Overall_Top3_Acc_%": round(r2.get("top3_accuracy", 0.0) * 100, 2),
        "Pipeline_Balanced_Acc_%": round(r2.get("balanced_accuracy", 0.0) * 100, 2),
        "Pipeline_Macro_Precision_%": round(r2.get("precision_macro", 0.0) * 100, 2),
        "Pipeline_Macro_Recall_%": round(r2.get("recall_macro", 0.0) * 100, 2),
        "Pipeline_Macro_F1_%": round(r2.get("macro_f1", 0.0) * 100, 2),
        "Pipeline_Micro_F1_%": round(r2.get("micro_f1", 0.0) * 100, 2),
        "Pipeline_Macro_mAP_%": round(r2.get("mAP", 0.0) * 100, 2),
        "Pipeline_Macro_ROC_AUC_%": round(r2.get("macro_auc", 0.0) * 100, 2),
        "Acc_SNR_-5dB_%": round(r2.get("acc_snr_-5", 0.0) * 100, 2),
        "Acc_SNR_0dB_%": round(r2.get("acc_snr_0", 0.0) * 100, 2),
        "Acc_SNR_5dB_%": round(r2.get("acc_snr_5", 0.0) * 100, 2),
        "Acc_SNR_10dB_%": round(r2.get("acc_snr_10", 0.0) * 100, 2),
        "Acc_SNR_15dB_%": round(r2.get("acc_snr_15", 0.0) * 100, 2),
        "Acc_SNR_20dB_%": round(r2.get("acc_snr_20", 0.0) * 100, 2),
    },
    {
        "Run_ID": "Run_3",
        "Experiment_Name": "Dual-CED-Tiny (New Partitioning: Low <=10 dB, High >=15 dB)",
        "Architecture": "Dual-Student MoE (2 independent CED-Tiny)",
        "Student_Params_M": 10.84,
        "Total_MoE_Params_M": 191.52,
        "RMS_Norm": "0.07925 (Enabled)",
        "Low_Expert_SNR": "-5, 0, 5, 10 dB",
        "High_Expert_SNR": "15, 20 dB",
        "Router_Threshold_dB": 12.5,
        "Low_Student_Best_Macro_Acc_%": round(s3["low"]["best_score"] * 100, 2),
        "Low_Best_Epoch": s3["low"]["best_epoch"],
        "High_Student_Best_Top1_Acc_%": round(s3["high"]["best_score"] * 100, 2),
        "High_Best_Epoch": s3["high"]["best_epoch"],
        "Pipeline_Overall_Top1_Acc_%": round(r3.get("top1_accuracy", 0.0) * 100, 2),
        "Pipeline_Overall_Top3_Acc_%": round(r3.get("top3_accuracy", 0.0) * 100, 2),
        "Pipeline_Balanced_Acc_%": round(r3.get("balanced_accuracy", 0.0) * 100, 2),
        "Pipeline_Macro_Precision_%": round(r3.get("precision_macro", 0.0) * 100, 2),
        "Pipeline_Macro_Recall_%": round(r3.get("recall_macro", 0.0) * 100, 2),
        "Pipeline_Macro_F1_%": round(r3.get("macro_f1", 0.0) * 100, 2),
        "Pipeline_Micro_F1_%": round(r3.get("micro_f1", 0.0) * 100, 2),
        "Pipeline_Macro_mAP_%": round(r3.get("mAP", 0.0) * 100, 2),
        "Pipeline_Macro_ROC_AUC_%": round(r3.get("macro_auc", 0.0) * 100, 2),
        "Acc_SNR_-5dB_%": round(r3.get("acc_snr_-5", 0.0) * 100, 2),
        "Acc_SNR_0dB_%": round(r3.get("acc_snr_0", 0.0) * 100, 2),
        "Acc_SNR_5dB_%": round(r3.get("acc_snr_5", 0.0) * 100, 2),
        "Acc_SNR_10dB_%": round(r3.get("acc_snr_10", 0.0) * 100, 2),
        "Acc_SNR_15dB_%": round(r3.get("acc_snr_15", 0.0) * 100, 2),
        "Acc_SNR_20dB_%": round(r3.get("acc_snr_20", 0.0) * 100, 2),
    },
]

out_csv = results_dir / "benchmark_comparison_summary.csv"
with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"SUCCESS: Wrote {len(rows)} experiment rows to {out_csv}")
