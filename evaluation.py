import pandas as pd
import ast
from pitchmistakes import process_metadata_csv, DTWAnalyzer

def parse_ground_truth(gt_str):
    intervals = []
    try:
        gt_str = gt_str.replace('nan', "'nan'")  # Patch: make nan a string
        gt = ast.literal_eval(gt_str)
        for entry in gt:
            if len(entry) != 3:
                continue
            start, end, label = entry
            if label != 'F':
                continue
            try:
                start_f = float(start)
                end_f = float(end)
                intervals.append((start_f, end_f))
            except Exception:
                continue
    except Exception as e:
        print(f"Error parsing ground truth: {e}")
    return intervals

def evaluate_pair(gt_intervals, detected_times):
    """
    Evaluate detection for a single pair.
    - gt_intervals: list of (start, end) for 'F' mistakes
    - detected_times: list of detected mistake times (float)
    Returns: TP, FP, FN
    """
    matched_gt = set()
    TP = 0
    for t in detected_times:
        for idx, (start, end) in enumerate(gt_intervals):
            if start <= t <= end:
                TP += 1
                matched_gt.add(idx)
                break
    FP = len(detected_times) - TP
    FN = len(gt_intervals) - len(matched_gt)
    return TP, FP, FN

def main(metadata_csv='metadata.csv', threshold=0.3):
    # Step 1: Load pitch data and run DTW analysis
    pitch_data = process_metadata_csv(metadata_csv, normalization_method='semitones')
    dtw_analyzer = DTWAnalyzer(pitch_data)
    analysis_results = dtw_analyzer.run_full_analysis()

    # Step 2: Load metadata for ground truth
    meta = pd.read_csv(metadata_csv)
    summary = []

    for idx, row in meta.iterrows():
        pair_id = f"pair_{idx}"
        if pair_id not in analysis_results:
            continue
        gt_intervals = parse_ground_truth(row['ground_truth'])
        if not gt_intervals:
            continue

        result = analysis_results[pair_id]
        # Extract note correspondences and mistakes
        student_times = dtw_analyzer.pitch_data[pair_id]['student']['times']
        student_notes = result['note_correspondences']['student_notes']
        teacher_notes = result['note_correspondences']['teacher_notes']
        path = result['optimal_path']
        cost_matrix = result['cost_matrix']

        mistakes = dtw_analyzer.extract_pitch_mistakes(
            path=path,
            cost_matrix=cost_matrix,
            student_times=student_times,
            student_notes=student_notes,
            teacher_notes=teacher_notes,
            threshold=threshold
        )
        detected_times = [m['time'] for m in mistakes]

        TP, FP, FN = evaluate_pair(gt_intervals, detected_times)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        summary.append({
            'pair_id': pair_id,
            'TP': TP, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall, 'f1': f1
        })

    df = pd.DataFrame(summary)
    print("\nPer-pair evaluation:")
    print(df)
    
    print("\nOverall mean metrics:")
    print(df[['precision', 'recall', 'f1']].mean())

if __name__ == "__main__":
    main()