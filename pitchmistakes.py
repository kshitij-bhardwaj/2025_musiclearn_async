# Complete DTW Pipeline using Direct Pitch Extraction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
import os
from pathlib import Path

# Function to trim leading and trailing silence from pitch contours
def trim_silence(time_array, pitch_array):
    """
    Trim leading and trailing unvoiced (zero pitch) segments.
    
    Returns:
    - Trimmed time and pitch arrays
    """
    voiced_mask = pitch_array > 0
    if not np.any(voiced_mask):
        return np.array([]), np.array([])
    
    start = np.argmax(voiced_mask)
    end = len(pitch_array) - np.argmax(voiced_mask[::-1])
    
    return time_array[start:end], pitch_array[start:end]


def extract_and_normalize_pitch(audio_file, pitch_floor=50, pitch_ceiling=800, normalization_method='semitones'):
    """
    Extract pitch contour from audio file and apply normalization.
    
    Parameters:
    - audio_file: path to .wav file
    - pitch_floor: minimum pitch for analysis (Hz)
    - pitch_ceiling: maximum pitch for analysis (Hz) 
    - normalization_method: 'raw', 'zscore', 'semitones', 'centered'
    
    Returns:
    - time_points: array of time values
    - pitch_values: normalized pitch values
    - stats: dictionary with pitch statistics
    """
    
    # Load sound and extract pitch
    sound = parselmouth.Sound(audio_file)
    pitch = sound.to_pitch(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    
    # Get raw pitch values and time points
    time_points = pitch.xs()
    raw_pitch = pitch.selected_array['frequency']
    
    # Remove unvoiced segments (0 Hz values)
    voiced_mask = raw_pitch > 0
    voiced_pitch = raw_pitch[voiced_mask]
    voiced_times = time_points[voiced_mask]
    
    # Calculate statistics
    stats = {
        'mean_f0': np.mean(voiced_pitch),
        'std_f0': np.std(voiced_pitch),
        'min_f0': np.min(voiced_pitch),
        'max_f0': np.max(voiced_pitch),
        'voicing_percentage': (len(voiced_pitch) / len(raw_pitch)) * 100
    }
    
    # Apply normalization
    if normalization_method == 'raw':
        normalized_pitch = voiced_pitch
        
    elif normalization_method == 'zscore':
        normalized_pitch = (voiced_pitch - stats['mean_f0']) / stats['std_f0']
        
    elif normalization_method == 'semitones':
        # Convert to semitones relative to speaker mean
        normalized_pitch = 12 * np.log2(voiced_pitch / stats['mean_f0'])
        
    elif normalization_method == 'centered':
        # Mean-centered Hz
        normalized_pitch = voiced_pitch - stats['mean_f0']
        
    else:
        raise ValueError("normalization_method must be 'raw', 'zscore', 'semitones', or 'centered'")
    #to update the code to trim silence
    voiced_times, normalized_pitch = trim_silence(voiced_times, normalized_pitch)


    return voiced_times, normalized_pitch, stats

def process_metadata_csv(metadata_file, normalization_method='semitones'):
    """
    Process all audio files from metadata CSV and extract normalized pitch contours.
    Returns results dictionary directly without CSV storage.
    """
    
    # Read metadata
    df = pd.read_csv(metadata_file)
    results = {}
    
    for idx, row in df.iterrows():
        # Construct file paths
        student_path = f"audio_subset/student/{row['s_file']}"
        teacher_path = f"audio_subset/teacher/{row['t_file']}"
        
        try:
            # Extract pitch for student
            s_times, s_pitch, s_stats = extract_and_normalize_pitch(
                student_path, normalization_method=normalization_method
            )
            
            # Extract pitch for teacher  
            t_times, t_pitch, t_stats = extract_and_normalize_pitch(
                teacher_path, normalization_method=normalization_method
            )
            
            # Store results
            results[f"pair_{idx}"] = {
                'metadata': row.to_dict(),
                'student': {
                    'times': s_times,
                    'pitch': s_pitch,
                    'stats': s_stats,
                    'file': row['s_file']
                },
                'teacher': {
                    'times': t_times,
                    'pitch': t_pitch,
                    'stats': t_stats,
                    'file': row['t_file']
                }
            }
            
            print(f"Processed pair {idx}: {row['s_file']} & {row['t_file']}")
            
        except Exception as e:
            print(f"Error processing pair {idx}: {e}")
            continue
    
    return results

import librosa

def semitone_to_hz(semitone, tonic_hz):
    """Convert semitone-normalized value back to Hz using the tonic."""
    return tonic_hz * (2 ** (semitone / 12))

def get_sargam_boundaries(sa_freq):
    swaras = [
        "Sa", "Komal Re", "Shuddh Re", "Komal Ga", "Shuddh Ga", "Ma", "Tivra Ma",
        "Pa", "Komal Dha", "Shuddh Dha", "Komal Ni", "Shuddh Ni", "Sa (next)"
    ]
    offsets = np.arange(13)
    swara_freqs = sa_freq * (2 ** (offsets / 12))
    boundaries = [(swara_freqs[i] + swara_freqs[i+1]) / 2 for i in range(len(swara_freqs)-1)]
    boundaries = [swara_freqs[0] - (boundaries[0] - swara_freqs[0])] + boundaries + [swara_freqs[-1] + (swara_freqs[-1] - boundaries[-1])]
    return swaras, boundaries, swara_freqs

def map_to_sargam(frequency, sa_freq):
    """Map frequency (Hz) to nearest Sargam note using tonic."""
    if frequency <= 0:
        return 'Silence'
    swaras, boundaries, _ = get_sargam_boundaries(sa_freq)
    idx = np.digitize([frequency], boundaries)[0]
    return swaras[idx] if idx < len(swaras) else swaras[-1]

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
        if f1 > 0 and f2 > 0:
            return abs(np.log(abs(f1) + 1e-10) - np.log(abs(f2) + 1e-10))
        else:
            return 1.0  # Penalty for unvoiced segments
    
    def compute_dtw_cost_matrix(self, student_pitch, teacher_pitch):
        """
        Compute DTW cost matrix using log-scale distance
        """
        n, m = len(student_pitch), len(teacher_pitch)
        cost_matrix = np.zeros((n, m))
        
        # Fill cost matrix with log-scale distances
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = self.log_distance(student_pitch[i], teacher_pitch[j])
        
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
        
        return acc_cost, path

     #this section should is the replacement for the find_optimal_dtw_path method  
    def find_subsequence_dtw_path(self, cost_matrix):
      """
       Find DTW path where student sequence (x-axis) matches a subsequence of teacher (y-axis).

     Returns:
      - acc_cost: accumulated cost matrix
      - best_path: optimal path (as list of (i, j))
      - best_end_j: end index on teacher
     """
      n, m = cost_matrix.shape
      acc_cost = np.full((n, m), np.inf)
      acc_cost[0, :] = cost_matrix[0, :]  # Allow student to start anywhere on teacher

     # Fill rest of matrix
      for i in range(1, n):
        for j in range(m):
            min_prev = np.inf
            if j > 0:
                min_prev = min(acc_cost[i-1, j], acc_cost[i, j-1], acc_cost[i-1, j-1])
            acc_cost[i, j] = cost_matrix[i, j] + min_prev

      # Find best ending point on teacher (anywhere along last student frame)
      end_j = np.argmin(acc_cost[-1, :])
      min_cost = acc_cost[-1, end_j]

      # Backtrack from (n-1, end_j)
      path = []
      i, j = n-1, end_j
      while i > 0:
        path.append((i, j))
        choices = []
        if j > 0:
            choices = [acc_cost[i-1, j-1], acc_cost[i-1, j], acc_cost[i, j-1]]
        else:
            choices = [np.inf, acc_cost[i-1, j], np.inf]

        move = np.argmin(choices)
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
      path.append((0, j))
      path.reverse()

      return acc_cost, path

    
    # This section is to be verified
    def map_to_sargam(frequency, sa_freq):
        """
        Map frequency (Hz) to nearest Sargam note using the given tonic (Sa frequency).
        """
        if frequency <= 0:
            return 'Silence'
        swaras, boundaries, _ = get_sargam_boundaries(sa_freq)
        idx = np.digitize([frequency], boundaries)[0]
        return swaras[idx] if idx < len(swaras) else swaras[-1]
    
    def analyze_note_correspondences(self, pair_data, cost_matrix, path):
        student_pitch = pair_data['student']['pitch']
        teacher_pitch = pair_data['teacher']['pitch']
        student_times = pair_data['student']['times']
        s_scale = pair_data['metadata']['s_scale']
        t_scale = pair_data['metadata']['t_scale']

        # Get tonic frequencies
        s_sa_freq = librosa.note_to_hz(s_scale.split()[0])
        t_sa_freq = librosa.note_to_hz(t_scale.split()[0])

        # Convert semitone-normalized pitch back to Hz
        student_pitch_hz = [semitone_to_hz(st, s_sa_freq) for st in student_pitch]
        teacher_pitch_hz = [semitone_to_hz(tt, t_sa_freq) for tt in teacher_pitch]

        # Map to Sargam using the correct tonic
        student_notes = [map_to_sargam(f, s_sa_freq) for f in student_pitch_hz]
        teacher_notes = [map_to_sargam(f, t_sa_freq) for f in teacher_pitch_hz]

        student_duration = student_times[-1] - student_times[0] if len(student_times) > 1 else 0

        note_pair_costs = {}
        for i, j in path:
            s_note = student_notes[i] if i < len(student_notes) else 'Silence'
            t_note = teacher_notes[j] if j < len(teacher_notes) else 'Silence'
            note_pair = (s_note, t_note)
            cost = cost_matrix[i, j]
            if note_pair not in note_pair_costs:
                note_pair_costs[note_pair] = []
            note_pair_costs[note_pair].append(cost)

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
        
        # Get pitch contours
        student_pitch = pair_data['student']['pitch']
        teacher_pitch = pair_data['teacher']['pitch']
        
        # Compute DTW
        cost_matrix = self.compute_dtw_cost_matrix(student_pitch, teacher_pitch)
        #updated the code to use the new method of finding optimal path
        acc_cost, optimal_path = self.find_subsequence_dtw_path(cost_matrix)

        
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
    
    def visualize_cost_matrix(self, analysis_result, save_path=None):
        """
        Visualize DTW cost matrix with optimal path
        """
        cost_matrix = analysis_result['cost_matrix']
        path = analysis_result['optimal_path']
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cost_matrix.T, origin='lower', cmap='viridis', aspect='auto')
        
        # Plot optimal path
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'r-', linewidth=3, label='Optimal DTW Path')
        
        plt.colorbar(label='Log Distance Cost')
        plt.title(f"DTW Cost Matrix - {analysis_result['pair_id']}")
        plt.xlabel("Student Frames")
        plt.ylabel("Teacher Frames")
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cost matrix plot saved to {save_path}")
        
        plt.show()
    
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

    def print_summary_results(self, all_results):
        """
        Print comprehensive summary of DTW analysis results
        """
        print("\n" + "="*60)
        print("DTW ANALYSIS SUMMARY")
        print("="*60)
        
        for pair_id, result in all_results.items():
            print(f"\n{pair_id.upper()}:")
            print(f"  Total DTW Cost: {result['total_dtw_cost']:.4f}")
            print(f"  Path Length: {result['path_length']} alignments")
            print(f"  Student Duration: {result['student_duration']:.2f}s")
            
            # Print average cost aggregation
            print(f"  Average Cost Aggregation:")
            avg_costs = result['cost_aggregation']['average']
            for note_pair, cost in sorted(avg_costs.items()):
                print(f"    {note_pair[0]}→{note_pair[1]}: {cost:.4f}")
            
            # Print max cost aggregation
            print(f"  Maximum Cost Aggregation:")
            max_costs = result['cost_aggregation']['max']
            for note_pair, cost in sorted(max_costs.items()):
                print(f"    {note_pair[0]}→{note_pair[1]}: {cost:.4f}")

    def extract_pitch_mistakes(self, path, cost_matrix, student_times, student_notes, teacher_notes, threshold=0.3):
     """
      Identify pitch mistakes from the DTW path using a threshold on cost.

      Parameters:
     - path: list of (i, j) tuples (DTW alignment path)
     - cost_matrix: computed DTW cost matrix
     - student_times: time values for each student pitch frame
     - student_notes: list of mapped student notes (e.g., ['Sa', 'Re', ...])
     - teacher_notes: list of mapped teacher notes
     - threshold: cost above which a pitch difference is considered a mistake

     Returns:
     - List of dictionaries with mistake information
     """
    # Calculate costs for all path points
     all_costs = [cost_matrix[i, j] for (i, j) in path if i < len(student_times)]
    
     if not all_costs:
        return []
    
    # Use percentile-based threshold (only top 10% as mistakes)
     percentile_threshold = np.percentile(all_costs, 90)
     effective_threshold = max(threshold, percentile_threshold)
    
     mistakes = []
     for (i, j) in path:
         if i >= len(student_times):
             continue

         cost = cost_matrix[i, j]
         if cost > effective_threshold:
             mistake = {
                 'time': student_times[i],
                 'cost': cost,
                 'student_note': student_notes[i] if i < len(student_notes) else 'Silence',
                 'teacher_note': teacher_notes[j] if j < len(teacher_notes) else 'Silence'
             }
             mistakes.append(mistake)

     return mistakes           

    def plot_pitch_mistakes_with_teacher_comparison(self,pair_id, result, save_dir="mistake_plots", threshold=0.3):
        """
        Generate a plot comparing student and teacher pitch with mistake highlights and DTW alignment.
    
        Parameters:
        - pair_id: ID string of the pair (e.g., "pair_0")
        - result: dict returned by DTWAnalyzer.analyze_single_pair()
        - save_dir: directory where plot image will be saved
        - threshold: cost above which to flag a pitch mistake
        """
        os.makedirs(save_dir, exist_ok=True)

        # Extract data
        path = result['optimal_path']
        cost_matrix = result['cost_matrix']
        student_pitch = result['note_correspondences']['student_notes']
        teacher_pitch = result['note_correspondences']['teacher_notes']
        student_times = result['note_correspondences']['student_duration']
        student_times = result['note_correspondences']['student_notes']
        teacher_times = result['note_correspondences']['teacher_notes']

        # Get arrays again to retrieve time and pitch
        student_times_arr = result['pair_id_data']['student']['times']
        teacher_times_arr = result['pair_id_data']['teacher']['times']
        student_pitch_arr = result['pair_id_data']['student']['pitch']
        teacher_pitch_arr = result['pair_id_data']['teacher']['pitch']

        plt.figure(figsize=(14, 7))

        # Plot raw pitch contours
        plt.plot(student_times_arr, student_pitch_arr, color='blue', label='Student Pitch')
        plt.plot(teacher_times_arr, teacher_pitch_arr, color='green', label='Teacher Pitch')

        # Plot all DTW alignment connections as light gray dotted lines
        for (i, j) in path:
            if i < len(student_times_arr) and j < len(teacher_times_arr):
                plt.plot(
                    [student_times_arr[i], teacher_times_arr[j]],
                    [student_pitch_arr[i], teacher_pitch_arr[j]],
                    color='gray', linestyle='dotted', linewidth=0.5, alpha=0.4
                )

        # Now plot mistakes as red dots (student) and black Xs (teacher)
        for (i, j) in path:
            if i >= len(student_times_arr) or j >= len(teacher_times_arr):
                continue

            cost = cost_matrix[i, j]
            if cost > threshold:
                plt.plot(student_times_arr[i], student_pitch_arr[i], 'ro', label='Student Mistake' if 'Student Mistake' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.plot(teacher_times_arr[j], teacher_pitch_arr[j], 'kx', label='Expected Teacher Pitch' if 'Expected Teacher Pitch' not in plt.gca().get_legend_handles_labels()[1] else "")
                plt.plot(
                    [student_times_arr[i], teacher_times_arr[j]],
                    [student_pitch_arr[i], teacher_pitch_arr[j]],
                    'r--', linewidth=1, alpha=0.8
                )

        plt.xlabel("Time (s)")
        plt.ylabel("Pitch (Normalized)")
        plt.title(f"Pitch Mistakes Compared with Teacher - {pair_id}")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(save_dir, f"pitch_mistakes_{pair_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved mistake plot to: {save_path}")
        plt.close()


# Main execution function
def main():
    """
    Main function to run the complete DTW pipeline
    """
    # Step 1: Extract normalized pitch data from audio files
    print("Step 1: Extracting normalized pitch contours...")
    metadata_file = 'metadata.csv'  # Your metadata file
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
    
    # Step 4: Display comprehensive results
    print("\nStep 4: Displaying results...")
    dtw_analyzer.print_summary_results(analysis_results)
    
    
    # Step 5: Visualize cost matrix for all pairs and save to 'plots/' directory
    print("\nStep 5: Visualizing cost matrices for all pairs...")

    #Create 'plots' directory if it doesn't exist
    os.makedirs("new_dtw_plots", exist_ok=True)

    for pair_id, result in analysis_results.items():
        print(f"  Generating cost matrix plot for {pair_id}...")
        dtw_analyzer.visualize_cost_matrix(
        result,
        save_path=os.path.join("new_dtw_plots", f"dtw_cost_matrix_{pair_id}.png")
    )

    #Step 6: Visualize pitch mistakes
    result['pair_id_data'] = dtw_analyzer.pitch_data[pair_id]
    
    for pair_id, result in analysis_results.items():
        result['pair_id_data'] = dtw_analyzer.pitch_data[pair_id]  # Attach original data
        dtw_analyzer.plot_pitch_mistakes_with_teacher_comparison(pair_id, result)


    # Step 7: Save detailed results to CSV
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
        
        # Add average cost aggregation
        for note_pair, cost in result['cost_aggregation']['average'].items():
            row[f'avg_cost_{note_pair[0]}_to_{note_pair[1]}'] = cost
            
        # Add max cost aggregation  
        for note_pair, cost in result['cost_aggregation']['max'].items():
            row[f'max_cost_{note_pair[0]}_to_{note_pair[1]}'] = cost
            
        detailed_results.append(row)
    
    # Save to CSV
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv('dtw_analysis_results.csv', index=False)
    print("Detailed results saved to 'dtw_analysis_results.csv'")
    
    return analysis_results

if __name__ == "__main__":
    # Run the complete pipeline
    results = main()


