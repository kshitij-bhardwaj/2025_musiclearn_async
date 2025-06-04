import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json

import parselmouth
from parselmouth.praat import call

def extract_pitch_contour(audio_file_path, pitch_floor=75, pitch_ceiling=600, time_step=0.01):
    """
    Extract pitch contour from an audio file using Parselmouth/Praat
    
    Parameters:
    audio_file_path (str): Path to the audio file
    pitch_floor (float): Minimum pitch to consider (Hz)
    pitch_ceiling (float): Maximum pitch to consider (Hz) 
    time_step (float): Time step for analysis (seconds)
    
    Returns:
    tuple: (times, pitch_values) where times is array of time points and 
           pitch_values is array of pitch values (0 for unvoiced frames)
    """
    try:
        # Load the sound file
        sound = parselmouth.Sound(audio_file_path)
        
        # Extract pitch using Praat's algorithm
        pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
        
        # Get the pitch values and times
        pitch_values = pitch.selected_array['frequency']
        times = pitch.xs()
        
        return times, pitch_values
        
    except Exception as e:
        print(f"Error processing {audio_file_path}: {str(e)}")
        return None, None

def read_metadata_csv(csv_file_path):
    """
    Read metadata CSV file
    
    Parameters:
    csv_file_path (str): Path to the metadata CSV file
    
    Returns:
    pandas.DataFrame: DataFrame containing metadata
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded metadata with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        return None
    
def process_audio_files(metadata_df, student_dir="audio_subset/student/", teacher_dir="audio_subset/teacher/"):
    """
    Process audio files and extract pitch contours
    
    Parameters:
    metadata_df (pandas.DataFrame): DataFrame containing metadata
    student_dir (str): Directory containing student audio files
    teacher_dir (str): Directory containing teacher audio files
    
    Returns:
    list: List of dictionaries containing processed data
    """
    results = []
    
    for idx, row in metadata_df.iterrows():
        print(f"\nProcessing pair {idx+1}/{len(metadata_df)}")
        
        # Construct full file paths
        student_file = os.path.join(student_dir, row['s_file'])
        teacher_file = os.path.join(teacher_dir, row['t_file'])
        
        print(f"Student file: {student_file}")
        print(f"Teacher file: {teacher_file}")
        
        # Extract pitch contours
        print("Extracting pitch contours...")
        student_times, student_pitch = extract_pitch_contour(student_file)
        teacher_times, teacher_pitch = extract_pitch_contour(teacher_file)
        
        if student_times is None or teacher_times is None:
            print("Error: Failed to extract pitch contours")
            continue
        
        # Store results
        result = {
            'index': idx,
            'student_file': row['s_file'],
            'teacher_file': row['t_file'],
            'student_bpm': row['s_bpm'],
            'teacher_bpm': row['t_bpm'],
            'student_scale': row['s_scale'],
            'teacher_scale': row['t_scale'],
            'ground_truth': row['ground_truth'],
            'student_times': student_times,
            'student_pitch': student_pitch,
            'teacher_times': teacher_times,
            'teacher_pitch': teacher_pitch
        }
        
        results.append(result)
        print(f"Successfully processed audio pair {idx+1}")
    
    return results

def plot_pitch_contours(results, output_dir="output/pitch_plots/", show_plots=False):
    """
    Plot pitch contours for student and teacher files separately
    
    Parameters:
    results (list): List of dictionaries containing pitch data
    output_dir (str): Directory to save plots
    show_plots (bool): Whether to display plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        print(f"Plotting contours for student: {result['student_file']} and teacher: {result['teacher_file']}")
        
        # Create subplot for student and teacher
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot student pitch contour
        student_pitch_clean = result['student_pitch'].copy()
        student_pitch_clean[student_pitch_clean == 0] = np.nan  # Replace 0 with NaN for better plotting
        
        ax1.plot(result['student_times'], student_pitch_clean, 'b-', linewidth=1.5, label='Student')
        ax1.set_title(f"Student: {result['student_file']} (BPM: {result['student_bpm']}, Scale: {result['student_scale']})")
        ax1.set_ylabel('Pitch (Hz)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot teacher pitch contour  
        teacher_pitch_clean = result['teacher_pitch'].copy()
        teacher_pitch_clean[teacher_pitch_clean == 0] = np.nan  # Replace 0 with NaN for better plotting
        
        ax2.plot(result['teacher_times'], teacher_pitch_clean, 'r-', linewidth=1.5, label='Teacher')
        ax2.set_title(f"Teacher: {result['teacher_file']} (BPM: {result['teacher_bpm']}, Scale: {result['teacher_scale']})")
        ax2.set_ylabel('Pitch (Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add ground truth information
        try:
            ground_truth = json.loads(result['ground_truth'].replace("'", '"'))
            ground_truth_text = "Ground Truth: " + ", ".join([f"{gt[0]}-{gt[1]} {gt[2]}" for gt in ground_truth])
            fig.text(0.5, 0.01, ground_truth_text, ha='center', fontsize=10)
        except:
            print("Could not parse ground truth information")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save plot
        plot_filename = f"pitch_comparison_s_{result['student_file'][:-4]}.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

def save_pitch_data_csv(results, output_file="output/pitch_data_extracted.csv"):
    """
    Save extracted pitch data to CSV file
    
    Parameters:
    results (list): List of dictionaries containing pitch data
    output_file (str): Path to output CSV file
    """
    data_rows = []
    
    for result in results:
        if result['student_times'] is None or result['teacher_times'] is None:
            continue
            
        # Calculate statistics for student and teacher
        student_pitch_voiced = result['student_pitch'][result['student_pitch'] > 0]
        teacher_pitch_voiced = result['teacher_pitch'][result['teacher_pitch'] > 0]
        
        row = {
            'student_file': result['student_file'],
            'teacher_file': result['teacher_file'],
            'student_bpm': result['student_bpm'],
            'teacher_bpm': result['teacher_bpm'],
            'student_scale': result['student_scale'],
            'teacher_scale': result['teacher_scale'],
            'student_mean_pitch': np.mean(student_pitch_voiced) if len(student_pitch_voiced) > 0 else np.nan,
            'student_std_pitch': np.std(student_pitch_voiced) if len(student_pitch_voiced) > 0 else np.nan,
            'student_min_pitch': np.min(student_pitch_voiced) if len(student_pitch_voiced) > 0 else np.nan,
            'student_max_pitch': np.max(student_pitch_voiced) if len(student_pitch_voiced) > 0 else np.nan,
            'teacher_mean_pitch': np.mean(teacher_pitch_voiced) if len(teacher_pitch_voiced) > 0 else np.nan,
            'teacher_std_pitch': np.std(teacher_pitch_voiced) if len(teacher_pitch_voiced) > 0 else np.nan,
            'teacher_min_pitch': np.min(teacher_pitch_voiced) if len(teacher_pitch_voiced) > 0 else np.nan,
            'teacher_max_pitch': np.max(teacher_pitch_voiced) if len(teacher_pitch_voiced) > 0 else np.nan,
            'ground_truth': result['ground_truth']
        }
        
        data_rows.append(row)
    
    # Create DataFrame and save
    df_output = pd.DataFrame(data_rows)
    df_output.to_csv(output_file, index=False)
    print(f"Saved pitch statistics to: {output_file}")
    
    return df_output

def main():
    """
    Main function to process audio files and extract pitch contours
    """
    # Configuration
    csv_file = "metadata.csv"
    student_directory = "audio_subset/student/"
    teacher_directory = "audio_subset/teacher/"
    
    print("=== Pitch Contour Extraction with Parselmouth ===")
    print(f"Processing metadata from: {csv_file}")
    print(f"Student audio directory: {student_directory}")
    print(f"Teacher audio directory: {teacher_directory}")
    print()
    
    # Read metadata
    metadata_df = read_metadata_csv(csv_file)
    
    if metadata_df is None:
        print("Failed to read metadata. Exiting.")
        return
    
    # Process the audio files and extract pitch contours
    results = process_audio_files(metadata_df, student_directory, teacher_directory)
    
    if not results:
        print("No results generated. Exiting.")
        return
    
    print(f"\nSuccessfully processed {len(results)} audio file pairs")
    
    # Plot the pitch contours
    print("\nCreating pitch contour plots...")
    plot_pitch_contours(results, show_plots=False)  # Set to True to display plots
    
    # Save pitch statistics to CSV
    print("\nSaving pitch statistics...")
    pitch_stats = save_pitch_data_csv(results)
    
    print("\n=== Processing Complete ===")
    print(f"Generated {len(results)} pitch comparison plots")
    print("Check the 'pitch_plots' directory for individual plots")
    print("Check 'pitch_data_extracted.csv' for pitch statistics")

if __name__ == "__main__":
    main()
