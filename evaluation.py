import argparse
import ast
import re
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def parse_list_col(val):
  """Convert string representations of lists/arrays from CSV into true numerical lists."""
  if isinstance(val, list):
    return val
  if pd.isna(val):
    return []

  val_str = str(val).strip()
  # Remove leading and trailing brackets
  val_str = val_str.strip("[]")
  if not val_str:
    return []

  # Handle comma-separated or space-separated formats
  if "," in val_str:
    items = val_str.split(",")
  else:
    items = val_str.split()

  return [float(x.strip()) for x in items if x.strip()]


def main():
  parser = argparse.ArgumentParser(
      description=(
          "Calculate protein-level and global evaluation metrics from prediction results."
      )
  )
  parser.add_argument(
      "--input_csv",
      type=str,
      default="esm2_t30_predictions.csv",
      help="Path to the input CSV file containing predictions.",
  )
  parser.add_argument(
      "--output_csv",
      type=str,
      default="esm2_t30_metrics.csv",
      help="Path to save the output protein-level metrics CSV file.",
  )
  args = parser.parse_args()

  print(f"Reading prediction results from {args.input_csv}...")
  df = pd.read_csv(args.input_csv)

  metrics_rows = []

  all_y_true = []
  all_y_scores = []

  print(
      "Calculating evaluation metrics for each protein (AUC, Accuracy, F1,"
      " Recall, Precision, Specificity)..."
  )

  for idx, row in df.iterrows():
    p_id = row["ID"]
    length = row["Length"]

    # Parse numerical arrays
    y_true = np.array(parse_list_col(row["True_label"]), dtype=int)
    y_pred = np.array(parse_list_col(row["Predicted_label"]), dtype=int)
    y_scores = np.array(parse_list_col(row["Prediction_score"]), dtype=float)

    # Collect global data for concatenated AUC
    if len(y_true) > 0 and len(y_scores) > 0:
      all_y_true.extend(y_true.tolist())
      all_y_scores.extend(y_scores.tolist())

    # 1. Calculate per-protein AUC 
    try:
      if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_scores)
      else:
        auc = np.nan
    except Exception:
      auc = np.nan

    # 2. Calculate confusion matrix components
    try:
      tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except Exception:
      tn, fp, fn, tp = 0, 0, 0, 0

    # 3. Calculate evaluation metrics
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Recall
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity

    # Store results for the current protein
    metrics_rows.append({
        "ID": p_id,
        "Length": length,
        "AUC": round(auc, 3) if pd.notna(auc) else "NaN",
        "Accuracy": round(accuracy, 3),
        "F1 Score": round(f1, 3),
        "Recall": round(recall, 3),
        "Precision": round(precision, 3),
        "Specificity": round(specificity, 3),
    })

  # Calculate global concatenated AUC
  global_auc = np.nan
  try:
    arr_true = np.array(all_y_true, dtype=int)
    arr_scores = np.array(all_y_scores, dtype=float)
    if len(np.unique(arr_true)) > 1:
      global_auc = roc_auc_score(arr_true, arr_scores)
  except Exception as e:
    print(f"Error calculating global AUC: {e}")

  print(
      "\n"
      + "=" * 40
      + f"\n>>> Global Concatenated AUC: {round(global_auc, 4) if pd.notna(global_auc) else 'NaN'}"
      + "\n"
      + "=" * 40
  )

  df_metrics = pd.DataFrame(metrics_rows)
  # Save to the output CSV file
  df_metrics.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
  print(f"Calculation completed! Protein metrics saved to: {args.output_csv}")

if __name__ == "__main__":
  main()