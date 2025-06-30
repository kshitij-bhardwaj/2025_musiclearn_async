# import pandas as pd
# import ast
# from pitchmistakes import process_metadata_csv, DTWAnalyzer

# def parse_ground_truth(gt_str):
#     intervals = []
#     try:
#         gt_str = gt_str.replace('nan', "'nan'")  # Patch: make nan a string
#         gt = ast.literal_eval(gt_str)
#         for entry in gt:
#             if len(entry) != 3:
#                 continue
#             start, end, label = entry
#             if label != 'F':
#                 continue
#             try:
#                 start_f = float(start)
#                 end_f = float(end)
#                 intervals.append((start_f, end_f))
#             except Exception:
#                 continue
#     except Exception as e:
#         print(f"Error parsing ground truth: {e}")
#     return intervals

# def evaluate_pair(gt_intervals, detected_times):
#     """
#     Evaluate detection for a single pair.
#     - gt_intervals: list of (start, end) for 'F' mistakes
#     - detected_times: list of detected mistake times (float)
#     Returns: TP, FP, FN
#     """
#     matched_gt = set()
#     TP = 0
#     for t in detected_times:
#         for idx, (start, end) in enumerate(gt_intervals):
#             if start <= t <= end:
#                 TP += 1
#                 matched_gt.add(idx)
#                 break
#     FP = len(detected_times) - TP
#     FN = len(gt_intervals) - len(matched_gt)
#     return TP, FP, FN

# def main(metadata_csv='metadata.csv', threshold=0.3):
#     # Step 1: Load pitch data and run DTW analysis
#     pitch_data = process_metadata_csv(metadata_csv, normalization_method='semitones')
#     dtw_analyzer = DTWAnalyzer(pitch_data)
#     analysis_results = dtw_analyzer.run_full_analysis()

#     # Step 2: Load metadata for ground truth
#     meta = pd.read_csv(metadata_csv)
#     summary = []

#     for idx, row in meta.iterrows():
#         pair_id = f"pair_{idx}"
#         if pair_id not in analysis_results:
#             continue
#         gt_intervals = parse_ground_truth(row['ground_truth'])
#         if not gt_intervals:
#             continue

#         result = analysis_results[pair_id]
#         # Extract note correspondences and mistakes
#         student_times = dtw_analyzer.pitch_data[pair_id]['student']['times']
#         student_notes = result['note_correspondences']['student_notes']
#         teacher_notes = result['note_correspondences']['teacher_notes']
#         path = result['optimal_path']
#         cost_matrix = result['cost_matrix']

#         mistakes = dtw_analyzer.extract_pitch_mistakes(
#             path=path,
#             cost_matrix=cost_matrix,
#             student_times=student_times,
#             student_notes=student_notes,
#             teacher_notes=teacher_notes,
#             threshold=threshold
#         )
#         detected_times = [m['time'] for m in mistakes]

#         TP, FP, FN = evaluate_pair(gt_intervals, detected_times)
#         precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#         recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#         f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

#         summary.append({
#             'pair_id': pair_id,
#             'TP': TP, 'FP': FP, 'FN': FN,
#             'precision': precision, 'recall': recall, 'f1': f1
#         })

#     df = pd.DataFrame(summary)
#     print("\nPer-pair evaluation:")
#     print(df)
    
#     print("\nOverall mean metrics:")
#     print(df[['precision', 'recall', 'f1']].mean())

# if __name__ == "__main__":
#     main()

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

def debug_detection_stats(metadata_csv='metadata.csv', threshold=2.5):
    """
    Debug function to show detection statistics before evaluation.
    """
    print("=== DETECTION STATISTICS DEBUG ===")
    
    # Load data
    pitch_data = process_metadata_csv(metadata_csv, normalization_method='semitones')
    dtw_analyzer = DTWAnalyzer(pitch_data)
    analysis_results = dtw_analyzer.run_full_analysis()
    
    meta = pd.read_csv(metadata_csv)
    
    total_gt = 0
    total_detected = 0
    
    for idx, row in meta.iterrows():
        pair_id = f"pair_{idx}"
        if pair_id not in analysis_results:
            continue
            
        # Count ground truth
        gt_intervals = parse_ground_truth(row['ground_truth'])
        total_gt += len(gt_intervals)
        
        # Count detections
        if len(gt_intervals) > 0:  # Only check pairs with GT
            result = analysis_results[pair_id]
            mistakes = dtw_analyzer.extract_pitch_mistakes(
                path=result['optimal_path'],
                cost_matrix=result['cost_matrix'],
                student_times=dtw_analyzer.pitch_data[pair_id]['student']['times'],
                student_notes=result['note_correspondences']['student_notes'],
                teacher_notes=result['note_correspondences']['teacher_notes'],
                threshold=threshold
            )
            detected_count = len(mistakes)
            total_detected += detected_count
            
            print(f"Pair {idx}: GT={len(gt_intervals)}, Detected={detected_count}")
    
    print(f"\nTOTAL: GT={total_gt}, Detected={total_detected}")
    print(f"Detection Rate: {total_detected/total_gt:.2f}x ground truth")
    
    return total_gt, total_detected



def apply_collar(intervals, collar_duration):
    """
    Apply collar (dilation) to time intervals.
    
    Parameters:
    - intervals: list of (start, end) tuples
    - collar_duration: duration to extend on both sides (Tc)
    
    Returns:
    - list of dilated (start, end) tuples
    """
    dilated = []
    for start, end in intervals:
        dilated_start = max(0, start - collar_duration)  # Don't go below 0
        dilated_end = end + collar_duration
        dilated.append((dilated_start, dilated_end))
    return dilated

def intervals_overlap(interval1, interval2):
    """
    Check if two intervals have non-zero overlap.
    
    Parameters:
    - interval1, interval2: (start, end) tuples
    
    Returns:
    - bool: True if intervals overlap
    """
    start1, end1 = interval1
    start2, end2 = interval2
    return not (end1 <= start2 or end2 <= start1)

def evaluate_pair_with_collar(gt_intervals, detected_times, collar_duration=0.5):
    """
    Evaluate detection for a single pair using collar-based soft boundaries.
    
    Parameters:
    - gt_intervals: list of (start, end) for 'F' mistakes
    - detected_times: list of detected mistake times (float)
    - collar_duration: collar duration Tc in seconds
    
    Returns: TP, FP, FN
    """
    if not detected_times:
        return 0, 0, len(gt_intervals)
    
    if not gt_intervals:
        return 0, len(detected_times), 0
    
    # Convert detected times to point intervals (assuming minimal duration)
    # You might want to adjust this based on your detection method
    detection_duration = 0.1  # Assume 100ms duration for point detections
    detected_intervals = [(t - detection_duration/2, t + detection_duration/2) 
                         for t in detected_times]
    
    # Apply collar to detected intervals
    dilated_detected = apply_collar(detected_intervals, collar_duration)
    
    # Find matches between dilated detections and ground truth
    matched_gt = set()
    matched_detected = set()
    
    for det_idx, det_interval in enumerate(dilated_detected):
        for gt_idx, gt_interval in enumerate(gt_intervals):
            if intervals_overlap(det_interval, gt_interval):
                matched_gt.add(gt_idx)
                matched_detected.add(det_idx)
                break  # One detection can only match one GT interval
    
    TP = len(matched_detected)
    FP = len(detected_intervals) - len(matched_detected)
    FN = len(gt_intervals) - len(matched_gt)
    
    return TP, FP, FN

def evaluate_pair_collar_bidirectional(gt_intervals, detected_times, collar_duration=0.5):
    """
    Alternative implementation: Apply collar to both GT and detected intervals.
    This gives more lenient matching.
    
    Parameters:
    - gt_intervals: list of (start, end) for 'F' mistakes
    - detected_times: list of detected mistake times (float)
    - collar_duration: collar duration Tc in seconds
    
    Returns: TP, FP, FN
    """
    if not detected_times:
        return 0, 0, len(gt_intervals)
    
    if not gt_intervals:
        return 0, len(detected_times), 0
    
    # Convert detected times to intervals and apply collar
    detection_duration = 0.1
    detected_intervals = [(t - detection_duration/2, t + detection_duration/2) 
                         for t in detected_times]
    dilated_detected = apply_collar(detected_intervals, collar_duration)
    
    # Apply collar to ground truth intervals as well
    dilated_gt = apply_collar(gt_intervals, collar_duration)
    
    # Find matches
    matched_gt = set()
    matched_detected = set()
    
    for det_idx, det_interval in enumerate(dilated_detected):
        for gt_idx, gt_interval in enumerate(dilated_gt):
            if intervals_overlap(det_interval, gt_interval):
                matched_gt.add(gt_idx)
                matched_detected.add(det_idx)
                break
    
    TP = len(matched_detected)
    FP = len(detected_intervals) - len(matched_detected)
    FN = len(gt_intervals) - len(matched_gt)
    
    return TP, FP, FN

def main(metadata_csv='metadata.csv', threshold=1.5, collar_duration=0.5, 
         evaluation_mode='standard'):
    """
    Main evaluation function with collar support.
    
    Parameters:
    - metadata_csv: path to metadata file
    - threshold: threshold for mistake detection
    - collar_duration: collar duration Tc in seconds
    - evaluation_mode: 'standard', 'collar', or 'collar_bidirectional'
    """
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

        # Choose evaluation method
        if evaluation_mode == 'collar':
            TP, FP, FN = evaluate_pair_with_collar(gt_intervals, detected_times, collar_duration)
        elif evaluation_mode == 'collar_bidirectional':
            TP, FP, FN = evaluate_pair_collar_bidirectional(gt_intervals, detected_times, collar_duration)
        else:  # standard
            TP, FP, FN = evaluate_pair(gt_intervals, detected_times)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        summary.append({
            'pair_id': pair_id,
            'TP': TP, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall, 'f1': f1,
            'gt_count': len(gt_intervals),
            'detected_count': len(detected_times)
        })

    df = pd.DataFrame(summary)
    print(f"\nPer-pair evaluation (mode: {evaluation_mode}, collar: {collar_duration}s):")
    print(df)
    
    print(f"\nOverall mean metrics:")
    print(df[['precision', 'recall', 'f1']].mean())
    
    # Overall aggregate metrics
    total_TP = df['TP'].sum()
    total_FP = df['FP'].sum()
    total_FN = df['FN'].sum()
    
    overall_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    overall_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"\nOverall aggregate metrics:")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1: {overall_f1:.3f}")
    
    return df

def evaluate_pair(gt_intervals, detected_times):
    """
    Original evaluation without collar (for comparison).
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


if __name__ == "__main__":
    # First debug the detection statistics
    print("=== DEBUGGING DETECTION COUNTS ===")
    debug_detection_stats(threshold=2.5)
    
    # Run evaluation with conservative settings
    print("\n=== Standard Evaluation (High Threshold) ===")
    main(evaluation_mode='standard', threshold=2.5)
    
    print("\n=== Collar Evaluation (0.1s, High Threshold) ===")
    main(evaluation_mode='collar', collar_duration=0.1, threshold=2.5)
    
    print("\n=== Comparison: Lower Threshold ===")
    main(evaluation_mode='standard', threshold=1.0)