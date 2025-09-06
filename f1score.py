import pandas as pd
import ast

def parse_errors(errors_str):
    """Parses a string representation of a tuple or list of tuples into a Python list of tuples."""
    if pd.isna(errors_str) or not isinstance(errors_str, str):
        return []
    try:
        clean_str = errors_str.strip()
        if not clean_str:
            return []

        # Handle multiple tuples without a comma and space in between
        if ')' in clean_str and '(' in clean_str:
            clean_str = clean_str.replace(')(', '), (')
            clean_str = clean_str.replace(') (', '), (')

        if not clean_str.startswith('['):
            clean_str = '[' + clean_str + ']'

        parsed_result = ast.literal_eval(clean_str)
        if isinstance(parsed_result, tuple):
            return [parsed_result]
        elif isinstance(parsed_result, list):
            return parsed_result
        else:
            return []
    except (ValueError, SyntaxError):
        return []

def get_total_duration(intervals):
    """Calculates the total duration of a list of intervals."""
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])

    merged_intervals = []
    if intervals:
        current_start, current_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                merged_intervals.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_intervals.append((current_start, current_end))

    total_duration = sum(end - start for start, end in merged_intervals)
    return total_duration

def calculate_file_metrics(gt_intervals, pred_intervals):
    """
    Calculates TP, FP, and FN durations for a single file.
    gt_intervals: ground truth intervals
    pred_intervals: predicted intervals
    """
    # Merge the ground truth intervals
    merged_gt = []
    if gt_intervals:
        gt_intervals = sorted(gt_intervals, key=lambda x: x[0])
        current_start, current_end = gt_intervals[0]
        for next_start, next_end in gt_intervals[1:]:
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                merged_gt.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_gt.append((current_start, current_end))

    # Merge the predicted intervals
    merged_pred = []
    if pred_intervals:
        pred_intervals = sorted(pred_intervals, key=lambda x: x[0])
        current_start, current_end = pred_intervals[0]
        for next_start, next_end in pred_intervals[1:]:
            if next_start <= current_end:
                current_end = max(current_end, next_end)
            else:
                merged_pred.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_pred.append((current_start, current_end))

    tp_duration = 0
    for gt_start, gt_end in merged_gt:
        for pred_start, pred_end in merged_pred:
            intersection_start = max(gt_start, pred_start)
            intersection_end = min(gt_end, pred_end)
            if intersection_end > intersection_start:
                tp_duration += (intersection_end - intersection_start)

    fp_duration = get_total_duration(pred_intervals) - tp_duration
    fn_duration = get_total_duration(gt_intervals) - tp_duration

    return tp_duration, fp_duration, fn_duration

# --- Main Script ---

try:
    df_gt = pd.read_csv('metadata_freq - Sheet1.csv')
    df_pred = pd.read_csv('dtw_final - Sheet1.csv')
except FileNotFoundError:
    print("One or both CSV files were not found. Please check the filenames.")
    exit()

df_gt.set_index('s_file', inplace=True)
df_pred.set_index('s_file', inplace=True)

common_files = pd.merge(df_gt, df_pred, on='s_file', suffixes=('_gt', '_pred'))

total_tp = 0
total_fp = 0
total_fn = 0

for name, row in common_files.iterrows():
    gt_errors = parse_errors(row['Errors_gt'])
    pred_errors = parse_errors(row['Errors_pred'])

    # Filter for tuples with two elements before converting to float
    gt_intervals = [(float(t[0]), float(t[1])) for t in gt_errors if isinstance(t, tuple) and len(t) == 2]
    pred_intervals = [(float(t[0]), float(t[1])) for t in pred_errors if isinstance(t, tuple) and len(t) == 2]

    tp, fp, fn = calculate_file_metrics(gt_intervals, pred_intervals)

    total_tp += tp
    total_fp += fp
    total_fn += fn

precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"Total True Positive Duration: {total_tp:.2f} seconds")
print(f"Total False Positive Duration: {total_fp:.2f} seconds")
print(f"Total False Negative Duration: {total_fn:.2f} seconds")
print("-" * 30)
print(f"Overall F1 Score: {f1_score:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")