# Complete DTW Pipeline using Direct Pitch Extraction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
import os
from pathlib import Path
import librosa

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

def get_sargam_boundaries(sa_freq):
    """
    Given the tonic frequency (Sa), return swara names and frequency boundaries for one octave.
    """
    swaras = [
        "Sa", "Komal Re", "Shuddh Re", "Komal Ga", "Shuddh Ga", "Ma", "Tivra Ma",
        "Pa", "Komal Dha", "Shuddh Dha", "Komal Ni", "Shuddh Ni", "Sa (next)"
    ]
    # Semitone offsets for each swara
    offsets = np.arange(13)
    # Calculate frequencies for each swara
    swara_freqs = sa_freq * (2 ** (offsets / 12))
    # Boundaries: midpoints between swaras
    boundaries = [(swara_freqs[i] + swara_freqs[i+1]) / 2 for i in range(len(swara_freqs)-1)]
    # Add min/max for outer boundaries
    boundaries = [swara_freqs[0] - (boundaries[0] - swara_freqs[0])] + boundaries + [swara_freqs[-1] + (swara_freqs[-1] - boundaries[-1])]
    return swaras, boundaries, swara_freqs

def segment_pitch_by_sargam(pitch, boundaries):
    """Assign each pitch value to a swara index based on sargam boundaries."""
    return np.digitize(pitch, boundaries)

def plot_pitch_with_sargam_segments(times, pitch, sa_freq, title, color, ax):
    swaras, boundaries, swara_freqs = get_sargam_boundaries(sa_freq)
    swara_indices = segment_pitch_by_sargam(pitch, boundaries)
    ax.plot(times, pitch, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True, alpha=0.3)
    # Overlay swara segments
    for i, swara in enumerate(swaras):
        mask = swara_indices == i
        if np.any(mask):
            ax.hlines(np.mean(pitch[mask]), times[mask][0], times[mask][-1], colors='k', linestyles='dashed', alpha=0.5)
            ax.text(times[mask][0], np.mean(pitch[mask]), swara, verticalalignment='bottom', fontsize=9, color='k')
    ax.legend([title])

def plot_normalized_pitch_with_sargam_segments(times, norm_pitch, raw_pitch_hz, sa_freq, title, color, ax):
    swaras, boundaries, swara_freqs = get_sargam_boundaries(sa_freq)
    swara_indices = segment_pitch_by_sargam(raw_pitch_hz, boundaries)
    ax.plot(times, norm_pitch, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel('Normalized Pitch')
    ax.grid(True, alpha=0.3)
    for i, swara in enumerate(swaras):
        mask = swara_indices == i
        if np.any(mask):
            ax.hlines(np.mean(norm_pitch[mask]), times[mask][0], times[mask][-1], colors='k', linestyles='dashed', alpha=0.5)
            ax.text(times[mask][0], np.mean(norm_pitch[mask]), swara, verticalalignment='bottom', fontsize=9, color='k')
    ax.legend([title])

def plot_normalized_pitch_with_sargam_yaxis(times, norm_pitch, raw_pitch_hz, sa_freq, title, color, ax):
    """
    Plot normalized pitch with y-axis as Sargam (swara) note names.
    """
    swaras, boundaries, swara_freqs = get_sargam_boundaries(sa_freq)
    swara_indices = segment_pitch_by_sargam(raw_pitch_hz, boundaries)
    ax.plot(times, norm_pitch, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_ylabel('Sargam (Swara)')
    ax.grid(True, alpha=0.3)
    # Set y-ticks at the mean normalized pitch for each swara
    yticks = []
    yticklabels = []
    for i, swara in enumerate(swaras):
        mask = swara_indices == i
        if np.any(mask):
            mean_val = np.mean(norm_pitch[mask])
            yticks.append(mean_val)
            yticklabels.append(swara)
            # Optionally, annotate on plot
            ax.text(times[mask][0], mean_val, swara, verticalalignment='bottom', fontsize=9, color='k')
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.legend([title])

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
    
    def map_to_scale_note(self, frequency, scale_str):
        if frequency <= 0:
            return 'Silence'
        sa_freq = librosa.note_to_hz(scale_str.split()[0])
        notes, boundaries, _ = get_sargam_boundaries(sa_freq)
        idx = np.digitize([frequency], boundaries)[0]
        return notes[idx] if idx < len(notes) else notes[-1]
    
    def analyze_note_correspondences(self, pair_data, cost_matrix, path):
        """
        Analyze note correspondences and calculate durations using scale notes.
        """
        student_pitch = pair_data['student']['pitch']
        teacher_pitch = pair_data['teacher']['pitch']
        student_times = pair_data['student']['times']
        s_scale = pair_data['metadata']['s_scale']
        t_scale = pair_data['metadata']['t_scale']

        # Map frequencies to scale notes
        student_notes = [self.map_to_scale_note(f, s_scale) for f in student_pitch]
        teacher_notes = [self.map_to_scale_note(f, t_scale) for f in teacher_pitch]

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
    
    def plot_sargam_segments(self, pair_id, save_dir="output/pitch_plots_segments_sargam"):
        """
        Plot student and teacher pitch contours with Sargam (swara) segments using raw Hz pitch.
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        if pair_id not in self.pitch_data:
            print(f"Pair {pair_id} not found.")
            return
        data = self.pitch_data[pair_id]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # Student
        s_sa_freq = librosa.note_to_hz(data['metadata']['s_scale'].split()[0])
        s_times = data['student']['times']
        s_raw_pitch = data['student'].get('raw_pitch_hz')
        if s_raw_pitch is None:
            s_file = f"audio_subset/student/{data['student']['file']}"
            s_times, s_raw_pitch, _ = extract_and_normalize_pitch(s_file, normalization_method='raw')
        plot_pitch_with_sargam_segments(
            s_times,
            s_raw_pitch,
            s_sa_freq,
            f"Student Pitch Contour (Sargam) - {data['student']['file']}",
            'b',
            ax1
        )
        # Teacher
        t_sa_freq = librosa.note_to_hz(data['metadata']['t_scale'].split()[0])
        t_times = data['teacher']['times']
        t_raw_pitch = data['teacher'].get('raw_pitch_hz')
        if t_raw_pitch is None:
            t_file = f"audio_subset/teacher/{data['teacher']['file']}"
            t_times, t_raw_pitch, _ = extract_and_normalize_pitch(t_file, normalization_method='raw')
        plot_pitch_with_sargam_segments(
            t_times,
            t_raw_pitch,
            t_sa_freq,
            f"Teacher Pitch Contour (Sargam) - {data['teacher']['file']}",
            'r',
            ax2
        )
        ax2.set_xlabel('Time (s)')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sargam_segments_{pair_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Sargam segment plot saved to {save_path}")

    def plot_normalized_sargam_segments(self, pair_id, save_dir="output/pitch_plots_segments_sargam_norm"):
        """
        Plot student and teacher normalized pitch contours with Sargam (swara) segments (segmentation by Hz).
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        if pair_id not in self.pitch_data:
            print(f"Pair {pair_id} not found.")
            return
        data = self.pitch_data[pair_id]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # Student
        s_sa_freq = librosa.note_to_hz(data['metadata']['s_scale'].split()[0])
        s_times = data['student']['times']
        s_norm_pitch = data['student']['pitch']
        s_raw_pitch = data['student'].get('raw_pitch_hz')
        if s_raw_pitch is None:
            s_file = f"audio_subset/student/{data['student']['file']}"
            s_times, s_raw_pitch, _ = extract_and_normalize_pitch(s_file, normalization_method='raw')
        plot_normalized_pitch_with_sargam_segments(
            s_times,
            s_norm_pitch,
            s_raw_pitch,
            s_sa_freq,
            f"Student Normalized Pitch (Sargam) - {data['student']['file']}",
            'b',
            ax1
        )
        # Teacher
        t_sa_freq = librosa.note_to_hz(data['metadata']['t_scale'].split()[0])
        t_times = data['teacher']['times']
        t_norm_pitch = data['teacher']['pitch']
        t_raw_pitch = data['teacher'].get('raw_pitch_hz')
        if t_raw_pitch is None:
            t_file = f"audio_subset/teacher/{data['teacher']['file']}"
            t_times, t_raw_pitch, _ = extract_and_normalize_pitch(t_file, normalization_method='raw')
        plot_normalized_pitch_with_sargam_segments(
            t_times,
            t_norm_pitch,
            t_raw_pitch,
            t_sa_freq,
            f"Teacher Normalized Pitch (Sargam) - {data['teacher']['file']}",
            'r',
            ax2
        )
        ax2.set_xlabel('Time (s)')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sargam_segments_normalized_{pair_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Normalized Sargam segment plot saved to {save_path}")
    
    def plot_normalized_sargam_yaxis(self, pair_id, save_dir="output/pitch_plots_segments_sargam_yaxis"):
        import os
        os.makedirs(save_dir, exist_ok=True)
        if pair_id not in self.pitch_data:
            print(f"Pair {pair_id} not found.")
            return
        data = self.pitch_data[pair_id]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # Student
        s_sa_freq = librosa.note_to_hz(data['metadata']['s_scale'].split()[0])
        s_times = data['student']['times']
        s_norm_pitch = data['student']['pitch']
        s_raw_pitch = data['student'].get('raw_pitch_hz')
        if s_raw_pitch is None:
            s_file = f"audio_subset/student/{data['student']['file']}"
            s_times, s_raw_pitch, _ = extract_and_normalize_pitch(s_file, normalization_method='raw')
        plot_normalized_pitch_with_sargam_yaxis(
            s_times,
            s_norm_pitch,
            s_raw_pitch,
            s_sa_freq,
            f"Student Normalized Pitch (Sargam Y-axis) - {data['student']['file']}",
            'b',
            ax1
        )
        # Teacher
        t_sa_freq = librosa.note_to_hz(data['metadata']['t_scale'].split()[0])
        t_times = data['teacher']['times']
        t_norm_pitch = data['teacher']['pitch']
        t_raw_pitch = data['teacher'].get('raw_pitch_hz')
        if t_raw_pitch is None:
            t_file = f"audio_subset/teacher/{data['teacher']['file']}"
            t_times, t_raw_pitch, _ = extract_and_normalize_pitch(t_file, normalization_method='raw')
        plot_normalized_pitch_with_sargam_yaxis(
            t_times,
            t_norm_pitch,
            t_raw_pitch,
            t_sa_freq,
            f"Teacher Normalized Pitch (Sargam Y-axis) - {data['teacher']['file']}",
            'r',
            ax2
        )
        ax2.set_xlabel('Time (s)')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sargam_yaxis_{pair_id}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Sargam y-axis plot saved to {save_path}")


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


def save_full_cost_aggregation_csv(all_results, pitch_data, filename="dtw_all_cost_aggregation.csv"):
    import pandas as pd
    rows = []
    for pair_id, result in all_results.items():
        metadata = pitch_data[pair_id]['metadata']
        row = {
            'pair_id': pair_id,
            'student_file': metadata['s_file'],
            'teacher_file': metadata['t_file'],
            'total_dtw_cost': result['total_dtw_cost'],
            'path_length': result['path_length'],
            'student_duration': result['student_duration']
        }
        # Add all avg costs
        for note_pair, cost in result['cost_aggregation']['average'].items():
            row[f'avg_cost_{note_pair[0]}_to_{note_pair[1]}'] = cost
        # Add all max costs
        for note_pair, cost in result['cost_aggregation']['max'].items():
            row[f'max_cost_{note_pair[0]}_to_{note_pair[1]}'] = cost

        # Add perfect match costs
        avg_costs = result['cost_aggregation']['average']
        max_costs = result['cost_aggregation']['max']
        perfect_matches = [(k, v) for k, v in avg_costs.items() if k[0] == k[1]]
        if perfect_matches:
            perfect_avg = np.mean([v for k, v in perfect_matches])
            perfect_max = np.mean([max_costs[k] for k, v in perfect_matches if k in max_costs])
            row['perfect_match_avg_cost'] = perfect_avg
            row['perfect_match_max_cost'] = perfect_max

        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"✓ Full cost aggregation saved to '{filename}'")

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
    
    # Step 5: Visualize cost matrix for first pair
    if analysis_results:
        first_pair = list(analysis_results.keys())[0]
        print(f"\nStep 5: Visualizing cost matrix for {first_pair}...")
        dtw_analyzer.visualize_cost_matrix(
            analysis_results[first_pair], 
            save_path=f'dtw_cost_matrix_{first_pair}.png'
        )
    
    # Step 6: Save detailed results to CSV
    print("\nStep 6: Saving detailed results...")
    detailed_results = []
    
    for pair_id, result in analysis_results.items():
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

        # Add perfect match costs (Sa→Sa, Re→Re, etc.)
        avg_costs = result['cost_aggregation']['average']
        max_costs = result['cost_aggregation']['max']
        perfect_matches = [(k, v) for k, v in avg_costs.items() if k[0] == k[1]]
        if perfect_matches:
            perfect_avg = np.mean([v for k, v in perfect_matches])
            perfect_max = np.mean([max_costs[k] for k, v in perfect_matches if k in max_costs])
            row['perfect_match_avg_cost'] = perfect_avg
            row['perfect_match_max_cost'] = perfect_max

        detailed_results.append(row)
    
    # Save to CSV
    results_df = pd.DataFrame(detailed_results)
    results_df.to_csv('dtw_analysis_results.csv', index=False)
    print("Detailed results saved to 'dtw_analysis_results.csv'")
    # After analysis_results = dtw_analyzer.run_full_analysis()
    save_full_cost_aggregation_csv(analysis_results, pitch_data, filename="dtw_all_cost_aggregation.csv")

    print("\nStep 7: Saving Sargam pitch segment plots...")
    for pair_id in pitch_data.keys():
        dtw_analyzer.plot_sargam_segments(pair_id)

    print("\nStep 8: Saving Normalized Sargam pitch segment plots...")
    for pair_id in pitch_data.keys():
        dtw_analyzer.plot_normalized_sargam_segments(pair_id)

    print("\nStep 9: Saving Normalized Sargam pitch segment plots (y-axis_renamed)...")
    for pair_id in pitch_data.keys():
        dtw_analyzer.plot_normalized_sargam_yaxis(pair_id)
        
    print("Number of pitch data pairs:", len(pitch_data))
    print("Number of analysis results:", len(analysis_results))
    for pair_id, result in analysis_results.items():
        print(pair_id, "avg:", result['cost_aggregation']['average'])
        print(pair_id, "max:", result['cost_aggregation']['max'])
    return analysis_results

if __name__ == "__main__":
    # Run the complete pipeline
    results = main()