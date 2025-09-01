# Complete DTW Pipeline for note segmentation and note wise mistake detection
import pandas as pd
import numpy as np
import parselmouth
import librosa
from pathlib import Path
import matplotlib.pyplot as plt


def _safe_to_float_array(x):
    """Coerce sequence-like x to a 1D float numpy array, converting non-numeric to np.nan."""
    arr = np.asarray(x)
    if arr.dtype.kind in ("U", "S", "O"):
        out = []
        for v in arr.ravel():
            try:
                # handle strings like 'nan', "'0.0'", etc.
                s = str(v).strip()
                if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
                    s = s[1:-1].strip()
                out.append(float(s))
            except Exception:
                out.append(np.nan)
        return np.asarray(out, dtype=float).reshape(arr.shape)
    else:
        return arr.astype(float, copy=False)


def extract_and_normalize_pitch(audio_file, pitch_floor=50, pitch_ceiling=800, normalization_method='semitones'):
    """
    Extract pitch contour from audio file and apply normalization.

    Returns:
    - time_points: array of time values
    - normalized_pitch: normalized pitch values
    - raw_voiced_pitch: raw voiced pitch in Hz (same length as time_points)
    - stats: dictionary with pitch statistics
    """
    # Load sound and extract pitch
    sound = parselmouth.Sound(audio_file)
    pitch = sound.to_pitch(time_step = 0.015,pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)

    # Get raw pitch values and time points
    time_points = np.asarray(pitch.xs(), dtype=float)
    raw_pitch = pitch.selected_array['frequency']
    raw_pitch = _safe_to_float_array(raw_pitch)

    # Keep only voiced frames (>0)
    voiced_mask = np.isfinite(raw_pitch) & (raw_pitch > 0)
    voiced_times = time_points[voiced_mask]
    voiced_pitch = raw_pitch[voiced_mask]  # raw voiced Hz

    # If no voiced frames, return empties
    if voiced_pitch.size == 0:
        stats = {'mean_f0': np.nan, 'std_f0': np.nan, 'min_f0': np.nan, 'max_f0': np.nan, 'voicing_percentage': 0.0}
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float), stats

    # Stats on raw voiced Hz
    stats = {
        'mean_f0': np.nanmean(voiced_pitch),
        'std_f0': np.nanstd(voiced_pitch),
        'min_f0': np.nanmin(voiced_pitch),
        'max_f0': np.nanmax(voiced_pitch),
        'voicing_percentage': (len(voiced_pitch) / max(len(raw_pitch), 1)) * 100
    }

    # Normalization (operate on voiced_pitch (Hz))
    if normalization_method == 'raw':
        normalized_pitch = voiced_pitch.copy()
    elif normalization_method == 'zscore':
        if stats['std_f0'] == 0 or np.isnan(stats['std_f0']):
            normalized_pitch = voiced_pitch - stats['mean_f0']
        else:
            normalized_pitch = (voiced_pitch - stats['mean_f0']) / stats['std_f0']
    elif normalization_method == 'semitones':
        if stats['mean_f0'] <= 0 or np.isnan(stats['mean_f0']):
            normalized_pitch = np.full_like(voiced_pitch, np.nan)
        else:
            normalized_pitch = 12.0 * np.log2(voiced_pitch / stats['mean_f0'])
    elif normalization_method == 'centered':
        normalized_pitch = voiced_pitch - stats['mean_f0']
    else:
        raise ValueError("normalization_method must be 'raw', 'zscore', 'semitones', or 'centered'")

    return voiced_times, normalized_pitch, voiced_pitch, stats



def process_metadata_csv(metadata_file, normalization_method='semitones'):
    """
    Process all audio files from metadata CSV and extract normalized pitch contours.
    Returns results dictionary directly without CSV storage.
    """
    
    # Read metadata
    df = pd.read_csv(metadata_file, sep='\t' if '\t' in Path(metadata_file).read_text(encoding='utf-8') else ',')
    results = {}
    
    for idx, row in df.iterrows():
        # Construct file paths
        student_path = f"audio_subset/student/{row['s_file']}"
        teacher_path = f"audio_subset/teacher/{row['t_file']}"
        
        try:
            # Extract pitch for student (now returns raw voiced Hz as well)
            s_times, s_norm_pitch, s_raw_pitch, s_stats = extract_and_normalize_pitch(
                student_path, normalization_method=normalization_method
            )
            
            # Extract pitch for teacher  
            t_times, t_norm_pitch, t_raw_pitch, t_stats = extract_and_normalize_pitch(
                teacher_path, normalization_method=normalization_method
            )
            
            # Store results (include raw Hz for mapping to sargam)
            results[f"pair_{idx}"] = {
                'metadata': row.to_dict(),
                'student': {
                    'times': s_times,
                    'pitch': s_norm_pitch,
                    'raw_pitch': s_raw_pitch,
                    'stats': s_stats,
                    'file': row['s_file']
                },
                'teacher': {
                    'times': t_times,
                    'pitch': t_norm_pitch,
                    'raw_pitch': t_raw_pitch,
                    'stats': t_stats,
                    'file': row['t_file']
                }
            }
            
            print(f"Processed pair {idx}: {row['s_file']} & {row['t_file']}")
            
        except Exception as e:
            print(f"Error processing pair {idx}: {e}")
            continue
    
    return results


def get_sargam_boundaries(sa_freq):
    """
    Given the tonic frequency (Sa), return swara names and frequency boundaries for one octave.
    """
    #print("Sa_Frequency is:",sa_freq,"\n\n")


    swaras = [
        "Sa","Re","Ga", "Ma",
        "Pa","Dha","Ni"
    ]

    ratios = {
        'Sa': 1.0, 'Re': 9/8, 'Ga': 5/4, 'Ma': 4/3,
        'Pa': 3/2, 'Dha': 5/3, 'Ni': 15/8
    }

    # Calculate frequencies for each swara
    swara_freqs = np.array([ratios[s]*sa_freq for s in swaras])
    #print("Swara_Frequencies are:",swara_freqs,"\n\n")
    # Boundaries: midpoints between swaras
    boundaries = [(swara_freqs[i] + swara_freqs[i+1]) / 2 for i in range(len(swara_freqs)-1)]
    #print("Boundaries are:",boundaries,"\n\n")
    # Add min/max for outer boundaries
    boundaries = [swara_freqs[0] - (boundaries[0] - swara_freqs[0])] + boundaries + [swara_freqs[-1] + (swara_freqs[-1] - boundaries[-1])]
    return swaras, boundaries, swara_freqs

def segment_pitch_by_sargam(pitch, boundaries):
    """Assign each pitch value to a swara index based on sargam boundaries."""
    return np.digitize(pitch, boundaries)


class DTWAnalyzer:
    def __init__(self, pitch_data):
        """
        Initialize DTW analyzer with pitch data from process_metadata_csv
        
        Parameters:
        - pitch_data: dictionary returned by process_metadata_csv
        """
        self.pitch_data = pitch_data
        
    @staticmethod
    def log_distance(f1, f2):
        """
        Log-scale distance function for DTW cost calculation
        """
        try:
            a = float(f1)
            b = float(f2)
        except Exception:
            return 1.0
        if a > 0 and b > 0:
            return abs(np.log(abs(a) + 1e-10) - np.log(abs(b) + 1e-10))
        else:
            return 1.0
    
    def compute_dtw_cost_matrix(self, student_pitch, teacher_pitch):
        """
        Compute DTW cost matrix using log-scale distance
        """
        n, m = len(student_pitch), len(teacher_pitch)
        #print(n,m,"\n\n")
        cost_matrix = np.zeros((n, m))
        
        #print("Student Pitch is:",student_pitch,"\n\n")
        #print("Teacher Pitch is:",teacher_pitch,"\n\n") 

        # Fill cost matrix with log-scale distances
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = self.log_distance(student_pitch[i], teacher_pitch[j])
        
        #print("Cost Matrix is:",cost_matrix,"\n\n")
        # plt.figsize(8, 6)
        # plt.imshow(cost_matrix, origin='lower', cmap='viridis', aspect='auto')
        # plt.show()
        return cost_matrix
    
    def find_optimal_dtw_path(self, cost_matrix):
        """
        Find optimal DTW path using dynamic programming
        """
        n, m = cost_matrix.shape
        
        # Initialize accumulated cost matrix
        acc_cost = np.zeros((n, m))
        acc_cost[0, 0] = cost_matrix[0, 0]
        
        # Fill first row and column
        for i in range(1, n):
            acc_cost[i, 0] = acc_cost[i-1, 0] + cost_matrix[i, 0]
        for j in range(1, m):
            acc_cost[0, j] = acc_cost[0, j-1] + cost_matrix[0, j]
        
        # Fill the rest of the matrix
        for i in range(1, n):
            for j in range(1, m):
                acc_cost[i, j] = cost_matrix[i, j] + min(
                    acc_cost[i-1, j],      # Insertion
                    acc_cost[i, j-1],      # Deletion  
                    acc_cost[i-1, j-1]     # Match
                )
        
        # Backtrack to find optimal path
        path = []
        i, j = n-1, m-1
        
        while i > 0 or j > 0:
            path.append((i, j))
            
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                # Choose the direction with minimum cost
                min_cost = min(acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1])
                
                if acc_cost[i-1, j-1] == min_cost:
                    i -= 1
                    j -= 1
                elif acc_cost[i-1, j] == min_cost:
                    i -= 1
                else:
                    j -= 1
        
        path.append((0, 0))
        path.reverse()
        #print("The path is-",path,"\n\n")
        x_vals = [p[0] for p in path]
        y_vals = [p[1] for p in path]

        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(x_vals, y_vals, marker=".", linestyle="-", color="b")
        plt.xlabel("X values")
        plt.ylabel("Y values")
        plt.title("Plot of Given Points")
        plt.grid(True)
        plt.show()
        
        return acc_cost, path
    
    # This section is to be verified
    def map_to_scale_note(self, frequency, scale_str):
        """
        Map frequency to nearest sargam note given a scale string (e.g. 'G3').
        Handles numeric and string scale inputs and protects against invalid frequencies.
        """
        # treat non-positive / nan / None as silence
        try:
            fval = float(frequency)
        except Exception:
            return 'Silence'
        if not np.isfinite(fval) or fval <= 0:
            return 'Silence'
    
        # convert scale string (e.g. 'G3') to tonic frequency (Hz)
        try:
            # allow numeric tonic as well
            if isinstance(scale_str, (int, float)):
                tonic_hz = float(scale_str)
            else:
                tonic_hz = librosa.note_to_hz(str(scale_str))
        except Exception:
            # fallback: cannot determine tonic -> return Unknown
            return 'Unknown'
    
        # get swara names and frequencies
        swaras,boundaries, swara_freqs = get_sargam_boundaries(tonic_hz)
    
        # choose closest swara by frequency
        idx = int(np.argmin(np.abs(swara_freqs - abs(fval))))
        return swaras[idx]
    
    
    def analyze_note_correspondences(self, pair_data, cost_matrix, path):
        # Use raw voiced Hz for mapping to sargam boundaries (not normalized semitones)
        student_times = pair_data['student']['times']
        teacher_times = pair_data['teacher']['times']
        student_raw = pair_data['student'].get('raw_pitch', np.array([]))
        teacher_raw = pair_data['teacher'].get('raw_pitch', np.array([]))
        s_scale = pair_data['metadata']['s_scale']
        t_scale = pair_data['metadata']['t_scale']
        #print(teacher_raw,"\n\n")
        #print(student_raw,"\n\n")
        # Map raw frequencies to swaras (silence for non-finite or <=0)
        student_notes = [self.map_to_scale_note(f, s_scale) for f in student_raw]
        teacher_notes = [self.map_to_scale_note(f, t_scale) for f in teacher_raw]
        #print("Student Notes are:",student_notes,"\n\n")
        #print("Teacher Notes are:",teacher_notes,"\n\n")


        student_duration = student_times[-1] - student_times[0] if len(student_times) > 1 else 0
        print("Student Duration is:",student_duration,"\n\n")
        unique_note_order = []
        # initialize note_pair_costs for same-note pairs
        for i in range(len(teacher_notes)-1):
            if (teacher_notes[i] != teacher_notes[i+1] ):
                unique_note_order.append(teacher_notes[i])
        #unique_notes = set(teacher_notes)
        note_pair_costs = { (n, n): [] for n in unique_note_order if n not in ["Silence", "Unknown"] }
        #print(unique_note_order,"\n\n")

        # Traverse DTW path (indices refer to normalized arrays lengths -> which match raw voiced lengths)
        for i, j in path:
            s_note = student_notes[i] if i < len(student_notes) else 'Silence'
            t_note = teacher_notes[j] if j < len(teacher_notes) else 'Silence'
            cost = cost_matrix[i, j]
            if s_note == t_note and (s_note, t_note) in note_pair_costs:
                note_pair_costs[(s_note, t_note)].append(cost)

            #     print({'student_notes': student_notes,
            # 'teacher_notes': teacher_notes,
            # 'student_duration': student_duration,
            # 'note_pair_costs': note_pair_costs})

        return {
            'student_notes': student_notes,
            'teacher_notes': teacher_notes,
            'student_duration': student_duration,
            'note_pair_costs': note_pair_costs
        }


    
   
    def aggregate_costs(self, note_pair_costs):
        """
        Aggregate costs using average and max methods
        """
        aggregated = {
            'average': {},
            'max': {}
        }
        
        for note_pair, costs in note_pair_costs.items():
            if costs:  # Only process if there are costs
                aggregated['average'][note_pair] = np.mean(costs)
                aggregated['max'][note_pair] = np.max(costs)
        
        return aggregated
    
    def analyze_single_pair(self, pair_id):
        """
        Complete DTW analysis for a single pair
        """
        if pair_id not in self.pitch_data:
            raise ValueError(f"Pair {pair_id} not found in data")
        
        pair_data = self.pitch_data[pair_id]
        #print("Pair Data is:",pair_data,"\n\n")
        
        # Get pitch contours
        student_pitch = pair_data['student']['pitch']
        teacher_pitch = pair_data['teacher']['pitch']
        
        # print(pair_id, "student pitch dtype:", np.asarray(student_pitch).dtype, "sample:", np.asarray(student_pitch)[:5])
        # print(pair_id, "teacher pitch dtype:", np.asarray(teacher_pitch).dtype, "sample:", np.asarray(teacher_pitch)[:5])

        # Compute DTW
        cost_matrix = self.compute_dtw_cost_matrix(student_pitch, teacher_pitch)
        acc_cost, optimal_path = self.find_optimal_dtw_path(cost_matrix)
        
        # Analyze note correspondences
        note_analysis = self.analyze_note_correspondences(pair_data, cost_matrix, optimal_path)
        
        # Aggregate costs
        cost_aggregation = self.aggregate_costs(note_analysis['note_pair_costs'])
        
        return {
            'pair_id': pair_id,
            'cost_matrix': cost_matrix,
            'accumulated_cost': acc_cost,
            'optimal_path': optimal_path,
            'total_dtw_cost': acc_cost[-1, -1],
            'path_length': len(optimal_path),
            'student_duration': note_analysis['student_duration'],
            'note_correspondences': note_analysis,
            'cost_aggregation': cost_aggregation
        }
    
    def run_full_analysis(self):
        """
        Run DTW analysis for all pairs in the dataset
        """
        all_results = {}
        
        for pair_id in self.pitch_data.keys():
            print(f"Analyzing {pair_id}...")
            try:
                result = self.analyze_single_pair(pair_id)
                all_results[pair_id] = result
                
                print(f"  Total DTW cost: {result['total_dtw_cost']:.4f}")
                print(f"  Path length: {result['path_length']} alignments")
                print(f"  Student duration: {result['student_duration']:.2f}s")
                
            except Exception as e:
                print(f"Error analyzing {pair_id}: {e}")
                continue
        
        return all_results

"""
Note detection and segmentation and saving it to RESULTS.csv
This part is seperate from the mistake detection part above
These parts have been integrated on a temporary basis for ease of use but better modularity is needed
"""   
# Constants for A3 scale (Sa at 220Hz)
SA_FREQ = 220.0
MAX_FREQ = 440.0
FREQ_TOLERANCE = 0.05
MIN_NOTE_DURATION = 0.15

def freq_to_note(freq):
    ratios = {
        'Sa': 1.0, 'Re': 9/8, 'Ga': 5/4, 'Ma': 4/3,
        'Pa': 3/2, 'Dha': 5/3, 'Ni': 15/8
    }
    if freq < SA_FREQ or freq > MAX_FREQ:
        return None
    closest_note = min(ratios.keys(),
                      key=lambda note: abs(freq - SA_FREQ*ratios[note]))
    return closest_note


def detect_notes(audio_path, merge_gap=0.5):
    y, sr = librosa.load(audio_path)
    f0, _, _ = librosa.pyin(y, fmin=SA_FREQ*0.9, fmax=MAX_FREQ*1.1, sr=sr)
    
    times = librosa.times_like(f0, sr=sr)
    notes = [freq_to_note(f) if not np.isnan(f) else None for f in f0]
    
    note_segments = {}
    current_note = None
    start_time = None
    
    for i, (time, note) in enumerate(zip(times, notes)):
        if note != current_note:
            # close previous note segment
            if current_note is not None and start_time is not None:
                end_time = time
                duration = end_time - start_time
                if duration >= MIN_NOTE_DURATION:
                    note_segments.setdefault(current_note, []).append((start_time, end_time))
            # start new note segment (only if voiced)
            current_note = note
            start_time = time if note is not None else None
        # otherwise continue the current segment
    
    # close last open segment at end of file
    if current_note is not None and start_time is not None:
        end_time = times[-1]
        duration = end_time - start_time
        if duration >= MIN_NOTE_DURATION:
            note_segments.setdefault(current_note, []).append((start_time, end_time))
    
    # helper to merge close segments for the same note
    def _merge_close_segments(segments, max_gap):
        if not segments:
            return []
        segments = sorted(segments, key=lambda s: s[0])
        merged = [segments[0]]
        for s_start, s_end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if s_start - prev_end <= max_gap:
                # join segments
                merged[-1] = (prev_start, s_end)
            else:
                merged.append((s_start, s_end))
        return merged

    # merge same-note segments separated by small gaps and re-check min duration
    for note, segs in list(note_segments.items()):
        merged = _merge_close_segments(segs, merge_gap)
        # filter out any that became too short after merging
        merged = [seg for seg in merged if (seg[1] - seg[0]) >= MIN_NOTE_DURATION]
        if merged:
            note_segments[note] = merged
        else:
            del note_segments[note]
    
    return note_segments, f0, times, y, sr



# plot_results(note_segments, f0, times, y, sr)

def process_student_folder(student_folder, limit=None):
    
    student_folder = Path(student_folder)
    wav_files = sorted(student_folder.glob("*.wav"))
    if limit:
        wav_files = wav_files[:limit]

    results = {}
    for wav in wav_files:
        try:
            print(f"Processing {wav.name}...")
            note_segments, f0, times, y, sr = detect_notes(str(wav))
            results[wav.name] = note_segments

        except Exception as e:
            print(f"  Error processing {wav.name}: {e}")

    return results

def save_results_csv(results, out_csv_path):
   
    rows = []
    for fname in sorted(results.keys()):
        notes = results.get(fname) or {}
        if not notes:
            rows.append({
                'student_file': fname,
                'note': '',
                'segment_index': '',
                'time_duration': '',
                'duration': '',
                'total_note_duration': ''
            })
            continue
        totals = {note: sum((e - s) for s, e in segs) for note, segs in notes.items()}
        for note, segs in notes.items():
            for idx, (s, e) in enumerate(segs):
                time_duration = f'({s:.3f} , {e:.3f})'
                rows.append({
                    'student_file': fname,
                    'note': note,
                    'segment_index': idx,
                    'time_duration': time_duration,
                    'duration': round(float(e - s), 3),
                    'total_note_duration': round(float(totals.get(note, 0.0)), 3)
                })

    df = pd.DataFrame(rows, columns=['student_file', 'note', 'segment_index', 'time_duration', 'duration', 'total_note_duration'])
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

# Main execution function
def main():
    """
    Main function to run the complete DTW pipeline
    """
    # Step 1: Extract normalized pitch data from audio files
    print("Step 1: Extracting normalized pitch contours...")
    metadata_file = r"C:\Users\abhin\OneDrive\Pictures\JAVASCRIPT\BCI_Challenge\Groove\2025_musiclearn_async\metasam.csv" # Your metadata file
    normalization_method = 'semitones'  # Use semitone normalization
    
    # Extract pitch data directly without CSV storage
    pitch_data = process_metadata_csv(metadata_file, normalization_method)
    
    if not pitch_data:
        print("No data extracted. Please check your file paths and metadata.csv")
        return
    
    print(f"Successfully extracted data for {len(pitch_data)} pairs")
    
    # Step 2: Initialize DTW analyzer
    print("\nStep 2: Initializing DTW analyzer...")
    dtw_analyzer = DTWAnalyzer(pitch_data)
    
    # Step 3: Run full DTW analysis
    print("\nStep 3: Running DTW analysis for all pairs...")
    analysis_results = dtw_analyzer.run_full_analysis()
    
     
    # Step 6: Save detailed results to CSV
    print("\nStep 6: Saving detailed results...")
    detailed_results = []
    
    for pair_id, result in analysis_results.items():
        # Extract metadata from original pitch data
        metadata = dtw_analyzer.pitch_data[pair_id]['metadata']
        
        row = {
            'pair_id': pair_id,
            'student_file': metadata['s_file'],
            'teacher_file': metadata['t_file'],
            'student_bpm': metadata['s_bpm'],
            'teacher_bpm': metadata['t_bpm'],
            'student_scale': metadata['s_scale'],
            'teacher_scale': metadata['t_scale'],
            'total_dtw_cost': result['total_dtw_cost'],
            'path_length': result['path_length'],
            'student_duration': result['student_duration']
        }
        
        # print("All note pairs in aggregation:")
        # print(list(result['cost_aggregation']['average'].keys()))


        # Add average cost aggregation
        for note_pair, cost in result['cost_aggregation']['average'].items():
            s_note, t_note = note_pair
            if s_note == t_note:  # keep only same-note comparisons
                row[f'{s_note}_to_{t_note}'] = cost

            
        # Add max cost aggregation  
        # for note_pair, cost in result['cost_aggregation']['max'].items():
        #     row[f'max_cost_{note_pair[1]}_to_{note_pair[1]}'] = cost
            
        detailed_results.append(row)
    
    # Save to CSV
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv('dtw_final.csv', index=False)
    print("Detailed results saved to 'dtw_final.csv'")

    #Step 7: Classify as mistake or not based on threshold
    print("\nStep 7: Classifying performances based on DTW cost threshold...")
    df = pd.read_csv('dtw_final.csv')

    note_cost_cols = [
    c for c in df.columns
    if "_to_" in c and c.split("_to_")[0].strip() == c.split("_to_")[1].strip()
]

    id_cols = ['pair_id', 'student_file', 'teacher_file', 'student_bpm', 'teacher_bpm',
            'student_scale', 'teacher_scale', 'total_dtw_cost', 'path_length', 'student_duration']
    filtered_df = df[id_cols + note_cost_cols]

    threshold = 0.99
    classified_df = filtered_df.copy()
    for col in note_cost_cols:
        classified_df[col] = classified_df[col].apply(lambda x: "1" if pd.notna(x) and x > threshold else ("0" if pd.notna(x) else None))

    output_path = 'dtw_final.csv'
    classified_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(output_path, classified_df.head())
    
        
    return analysis_results

def _base_swara(name: str) -> str:
    """
    Map 'Komal Re'/'Shuddh Re'/'Re' -> 'Re', 'Sa (next)' -> 'Sa', keep 'Sa', 'Ma', etc.
    """
    if not isinstance(name, str):
        return name
    n = name.strip()
    if n.startswith("Sa"):
        return "Sa"
    parts = n.replace("(", " ").replace(")", " ").split()
    return parts[-1] if parts else n

 
def replace_mistakes_with_durations(dtw_path='dtw_final.csv', results_path='RESULTS.csv', out_path='dtw_final_with_times.csv'):
    """
    Replace '1' mistake markers in '*_to_*' columns of dtw_path with the corresponding
    time_duration string from results_path (format: "(start , end)").
    Writes a filtered CSV that contains only the student file column and the note_to_note columns.
    """
    dtw_df = pd.read_csv(dtw_path, dtype=object)
    dtw_df.columns = dtw_df.columns.str.strip()

    res_df = pd.read_csv(results_path, dtype=str)
    res_df['student_file'] = res_df['student_file'].astype(str).str.strip()
    res_df['note'] = res_df['note'].astype(str).str.strip().map(_base_swara)
    res_df['time_duration'] = res_df['time_duration'].astype(str).str.strip()
    mapping = res_df.groupby(['student_file', 'note'])['time_duration'].apply(list).to_dict()


    # detect note comparison columns
    note_cols = [c for c in dtw_df.columns if '_to_' in c]

    # determine student file column in dtw_df
    stu_col_candidates = ['student_file', 's_file', 'student', 'student_filename', 'file']
    stu_col = next((c for c in stu_col_candidates if c in dtw_df.columns), None)
    if stu_col is None:
        # fallback to first column if none of the candidates found
        stu_col = dtw_df.columns[0]

    # iterate rows and replace '1' markers with time_duration strings
    for idx, row in dtw_df.iterrows():
        student_file = str(row.get(stu_col, '')).strip()
        if student_file == '' or pd.isna(student_file):
            continue

        for col in note_cols:
            val = row.get(col)
            # detect numeric or string '1' as mistake marker
            is_one = False
            try:
                is_one = float(val) == 1.0
            except Exception:
                is_one = str(val).strip() == '1'

            if not is_one:
                continue

            student_note = col.split('_to_')[0].strip()

            # try exact mapping
            durations = mapping.get((student_file, student_note))

            # fallback: case-insensitive note match for same student_file
            if durations is None:
                for (sf, note), dur_list in mapping.items():
                    if sf == student_file and note.lower() == student_note.lower():
                        durations = dur_list
                        break

            # replace with first time_duration string if available
            if durations and len(durations) > 0:
                dtw_df.at[idx, col] = durations[0]
            # otherwise leave the value as-is

    # keep only student file column and note_to_note columns
    keep_cols = [stu_col] + note_cols
    filtered = dtw_df.loc[:, keep_cols]

    if out_path is None:
        out_path = str(Path(dtw_path).with_name(Path(dtw_path).stem + "_with_times_filtered.csv"))

    filtered.to_csv(out_path, index=False)
    print(f"Saved updated & filtered DTW CSV with time ranges to: {out_path}")
    return out_path


if __name__ == "__main__":
    # Run the complete pipeline
    results = main()

    student_dir = Path(r"C:\Users\abhin\OneDrive\Pictures\JAVASCRIPT\BCI_Challenge\Groove\audio_subset\student")  
    results = process_student_folder(student_dir)
    out_csv = 'RESULTS.csv'
    save_results_csv(results, out_csv)
    print(f"Saved detected notes to: {out_csv}")

    out = replace_mistakes_with_durations(dtw_path='dtw_final.csv', results_path=out_csv)
    print(f"Final DTW results with durations saved to: {out}")

