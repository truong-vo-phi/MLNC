import csv, pathlib

results_dir = pathlib.Path("d:/PROJECT/MLNC/experiments/results")
in_csv = results_dir / "benchmark_comparison_summary.csv"
out_csv = results_dir / "benchmark_comparison_transposed.csv"

with open(in_csv, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

fields = list(rows[0].keys())

transposed_rows = []
for field in fields:
    row_dict = {"Metric / Configuration": field}
    for r in rows:
        run_name = r["Run_ID"] + " (" + r["Experiment_Name"].split("(")[0].strip() + ")"
        row_dict[run_name] = r[field]
    transposed_rows.append(row_dict)

with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(transposed_rows[0].keys()))
    writer.writeheader()
    writer.writerows(transposed_rows)

print(f"SUCCESS: Wrote transposed CSV to {out_csv}")
