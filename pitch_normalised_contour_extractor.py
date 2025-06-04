# Pitch Contour Extraction and Normalization - Complete Implementation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
import os
from pathlib import Path

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
    
    Parameters:
    - metadata_file: path to metadata.csv
    - normalization_method: normalization to apply
    
    Returns:
    - results: dictionary with pitch data for each file pair
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

def plot_pitch_contours(results, pair_id, save_path=None, plot=False):
    """
    Plot student and teacher pitch contours for a given pair.
    
    Parameters:
    - results: output from process_metadata_csv
    - pair_id: which pair to plot (e.g., "pair_0")
    - save_path: optional path to save plot
    """
    
    if pair_id not in results:
        print(f"Pair {pair_id} not found in results")
        return
    
    data = results[pair_id]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Student plot
    ax1.plot(data['student']['times'], data['student']['pitch'], 'b-', linewidth=2, label='Student')
    ax1.set_title(f"Student Pitch Contour - {data['student']['file']}")
    ax1.set_ylabel('Normalized Pitch')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add statistics text
    stats_text = f"Mean F0: {data['student']['stats']['mean_f0']:.1f} Hz\n"
    stats_text += f"Voicing: {data['student']['stats']['voicing_percentage']:.1f}%"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Teacher plot
    ax2.plot(data['teacher']['times'], data['teacher']['pitch'], 'r-', linewidth=2, label='Teacher')
    ax2.set_title(f"Teacher Pitch Contour - {data['teacher']['file']}")
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Normalized Pitch')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add statistics text
    stats_text = f"Mean F0: {data['teacher']['stats']['mean_f0']:.1f} Hz\n"
    stats_text += f"Voicing: {data['teacher']['stats']['voicing_percentage']:.1f}%"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if plot: 
        plt.show()

# Example usage:
if __name__ == "__main__":
    # Process all files from metadata
    print("Processing metadata.csv...")
    results = process_metadata_csv('metadata.csv', normalization_method='semitones')
    
    if results:
        for pair_id in results:
            save_file = f'output/pitch_plots_normalised/pitch_contours_{pair_id}.png'
            plot_pitch_contours(results, pair_id, save_path=save_file)

    
    # Save summary statistics
    summary_data = []
    for pair_id, data in results.items():
        summary_data.append({
            'pair_id': pair_id,
            'student_file': data['student']['file'],
            'teacher_file': data['teacher']['file'],
            'student_mean_f0': data['student']['stats']['mean_f0'],
            'teacher_mean_f0': data['teacher']['stats']['mean_f0'],
            'student_voicing_pct': data['student']['stats']['voicing_percentage'],
            'teacher_voicing_pct': data['teacher']['stats']['voicing_percentage'],
            's_bpm': data['metadata']['s_bpm'],
            't_bpm': data['metadata']['t_bpm'],
            's_scale': data['metadata']['s_scale'],
            't_scale': data['metadata']['t_scale']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('output/pitch_normalised_analysis_summary.csv', index=False)
    print("Summary saved to pitch_normalised_analysis_summary.csv")