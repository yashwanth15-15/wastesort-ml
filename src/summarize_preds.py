# src/summarize_preds.py
import csv, collections

fn = "predictions_test.csv"
rows = []
with open(fn, newline='', encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        rows.append(row)

total = len(rows)
mismatches = [r for r in rows if r['true_label'] != r['pred_label']]
count_mis = len(mismatches)

by_true = collections.Counter(r['true_label'] for r in rows)
by_pred = collections.Counter(r['pred_label'] for r in rows)
confusions = collections.Counter((r['true_label'], r['pred_label']) for r in rows)

print(f"Total test images: {total}")
print("Counts by true label:", dict(by_true))
print("Counts by predicted label:", dict(by_pred))
print(f"Total mismatches: {count_mis}  ({count_mis/total:.2%})\n")

print("Top 15 confusions (true -> pred : count):")
for (t,p),c in confusions.most_common(15):
    if t != p:
        print(f"  {t} -> {p} : {c}")

# print 10 example mismatches
print("\nExamples (true, pred, confidence, path):")
for r in mismatches[:10]:
    print(r['true_label'], r['pred_label'], r['pred_confidence'], r['path'])
