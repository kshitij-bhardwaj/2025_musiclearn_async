# !pip install librosa soundfile pyaudio
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyaudio as pa
import wave
import sounddevice as sd
import soundfile as sf
from collections import Counter

# Advanced silence removal function
def remove_silence_advanced(y, sr, top_db=40):
    """
    Remove silence with energy-based detection
    """
    # Calculate RMS energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    # Find non-silent frames
    rms_db = librosa.amplitude_to_db(rms)
    non_silent = rms_db > (rms_db.max() - top_db)
    # Convert frame indices to sample indices
    frames = np.where(non_silent)[0]
    if len(frames) == 0:
        return y  # Return original if all frames are silent
    start_frame = frames[0]
    end_frame = frames[-1] + 1
    start_sample = librosa.frames_to_samples(start_frame, hop_length=512)
    end_sample = librosa.frames_to_samples(end_frame, hop_length=512)
    return y[start_sample:end_sample]

def set_tonic_frequency(filename, chunk, channels, sample_format, fs):
    p = pa.PyAudio()
    ans = input("Do you want to set tonic frequency? (y/n): ")
    if ans.lower() == 'y':
        print("Set your tonic frequency (Sa)")
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True,)
        frames = []
        for i in range(0, int(fs / chunk * 3)):
            data = stream.read(chunk)
            frames.append(data)
        stream.stop_stream()
        stream.close()
        print('Finished recording tonic frequency')
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()
        data, fs = sf.read(filename, dtype='float32')
        sd.play(data, fs)
        status = sd.wait()
        tonic_audio, _ = librosa.load(filename, sr=44100)
        tonic_frequencies = librosa.yin(tonic_audio, fmin=75, fmax=600)
        tonic_freq = np.median(tonic_frequencies)
    else:
        tonic_freq = 207.65
    print("Tonic frequency is set to:", tonic_freq)
    return tonic_freq

def play_teacher_audio(teacher_audio_path):
    input("Press Enter when you are ready to play the teacher audio ")
    print("Now playing the teacher audio")
    data, fs = sf.read(teacher_audio_path, dtype='float32')
    sd.play(data, fs)
    status = sd.wait()
    print("Teacher audio finished playing")

def start_recording(filename, channels, sample_format, fs, chunk, seconds):
    input("Press Enter when you are ready to record your audio ")
    print("Now record your audio")
    print('Recording')
    p = pa.PyAudio()
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    frames = []
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('Finished recording')
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def play_recorded_audio(filename):
    input("Press Enter when you are ready to play your recorded audio ")
    print("Now playing your recorded audio")
    data, fs = sf.read(filename, dtype='float32')
    sd.play(data, fs)
    status = sd.wait()

def plot_pitch_and_svara(freq, notes, sr, title):
    """
    Plots the pitch contour with svara labels.
    """
    hop_length = 512
    time_frames = librosa.frames_to_time(np.arange(len(freq)), sr=sr, hop_length=hop_length)
    valid_freq = freq.copy()
    valid_freq[valid_freq == 0] = np.nan
    valid_freq[np.isnan(freq)] = np.nan
    plt.figure(figsize=(15, 8))
    plt.plot(time_frames, valid_freq, 'b-', linewidth=2, alpha=0.7)
    plt.fill_between(time_frames, valid_freq, alpha=0.3)
    for i in range(0, len(time_frames), max(1, len(time_frames)//15)):
        if not np.isnan(valid_freq[i]):
            plt.annotate(notes[i],
                        (time_frames[i], valid_freq[i]),
                        xytext=(0, 15),
                        textcoords='offset points',
                        fontsize=12,
                        ha='center',
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    plt.title(title)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.ylim(75, 600)
    plt.tight_layout()
    plt.show()

## DTW Implementation - Core Logic
def log_distance(f1, f2):
    """
    Log-scale distance function for DTW cost calculation
    """
    try:
        a = float(f1)
        b = float(f2)
    except (ValueError, TypeError):
        return 1.0
    if a > 0 and b > 0:
        return abs(np.log(abs(a) + 1e-10) - np.log(abs(b) + 1e-10))
    else:
        return 1.0

def compute_dtw_cost_matrix(student_pitch, teacher_pitch):
    """
    Compute DTW cost matrix using log-scale distance
    """
    n, m = len(student_pitch), len(teacher_pitch)
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = log_distance(student_pitch[i], teacher_pitch[j])
    return cost_matrix

def find_optimal_dtw_path(cost_matrix):
    """
    Find optimal DTW path using dynamic programming
    """
    n, m = cost_matrix.shape
    acc_cost = np.zeros((n, m))
    acc_cost[0, 0] = cost_matrix[0, 0]
    for i in range(1, n):
        acc_cost[i, 0] = acc_cost[i-1, 0] + cost_matrix[i, 0]
    for j in range(1, m):
        acc_cost[0, j] = acc_cost[0, j-1] + cost_matrix[0, j]
    for i in range(1, n):
        for j in range(1, m):
            acc_cost[i, j] = cost_matrix[i, j] + min(
                acc_cost[i-1, j],
                acc_cost[i, j-1],
                acc_cost[i-1, j-1]
            )
    path = []
    i, j = n-1, m-1
    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
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

def note_durations(notes):
    """
    Segments teacher notes into continuous segments with start and end indices.
    """
    notes_duration = []
    # FIX: Check if the array is empty using .size
    if notes.size == 0:
        return notes_duration
    
    start_idx = 0
    end_idx = 0

    for i in range(len(notes)-1):
        if notes[i] == notes[i+1]:
            end_idx = i+1
        else:
            notes_duration.append((notes[i], start_idx , end_idx))
            start_idx = i+1
            end_idx = i+1

    notes_duration.append((notes[-1], start_idx , len(notes)-1))

    return notes_duration

def mistake_detection(notes_duration, teacher_freq, student_freq, cost_matrix, path):
    """
    Analyzes DTW results to identify mistakes in the student's performance.
    """
    normalized_teacher_freq = 12.0 * np.log2(teacher_freq / np.nanmean(teacher_freq))
    normalized_student_freq = 12.0 * np.log2(student_freq / np.nanmean(student_freq))
    student_mistakes = []
    for i in range(len(notes_duration)):
        student = []
        teacher = []
        cost = 0
        for j in range(notes_duration[i][1], notes_duration[i][2]+1):
            teacher.append(normalized_teacher_freq[j])
            for k in range(len(path)):
                if path[k][1] == j:
                    student.append(path[k][0])
                    cost += cost_matrix[path[k][0]][path[k][1]]
        if not student:
            continue
        student_mean = abs(np.nanmean(normalized_student_freq[student]))
        teacher_mean = abs(np.nanmean(teacher))
        if (student_mean + teacher_mean) == 0:
            mean_diff = 1.0
        else:
            mean_diff = (abs(student_mean - teacher_mean)) / (student_mean + teacher_mean)
        if cost * mean_diff > 0.05:  # Tunable threshold
            student_mistakes.append((student[0], student[-1]))
    student_mistakes = list(dict.fromkeys(student_mistakes))
    student_mistakes_duration = [idx for t in student_mistakes for idx in t]
    student_times = librosa.frames_to_time(student_mistakes_duration, sr=44100, hop_length=512)
    student_mistakes_times = []
    for i in range(0, len(student_mistakes_duration), 2):
        if student_times[i] < student_times[i+1]:
            student_mistakes_times.append((student_times[i], student_times[i+1]))
    print("\nDetected Mistakes:")
    for (i, j) in student_mistakes_times:
        print(f"Start Time: {i:.2f} s, End Time: {j:.2f} s")
    return student_mistakes, student_mistakes_times

def plot_student_pitch_with_mistakes(freq, notes, sr, title, mistake_times):
    """
    Plots the student's pitch contour with svara labels and highlights mistakes in red.
    """
    hop_length = 512
    time_frames = librosa.frames_to_time(np.arange(len(freq)), sr=sr, hop_length=hop_length)

    valid_freq = freq.copy()
    valid_freq[valid_freq == 0] = np.nan
    valid_freq[np.isnan(freq)] = np.nan

    plt.figure(figsize=(15, 8))
    plt.plot(time_frames, valid_freq, 'b-', linewidth=2, alpha=0.7, label='Student Pitch Contour')
    plt.fill_between(time_frames, valid_freq, alpha=0.3)

    # Highlight mistake segments in red
    for start_time, end_time in mistake_times:
        mask = (time_frames >= start_time) & (time_frames <= end_time)
        if np.any(mask):
            plt.plot(time_frames[mask], valid_freq[mask], 'r-', linewidth=3, alpha=0.8)
            plt.fill_between(time_frames[mask], valid_freq[mask], color='red', alpha=0.2)
    # Add a legend entry for the mistake segments
    plt.plot([], [], 'r-', linewidth=3, alpha=0.8, label='Mistake Segments')

    # Add svara labels along the pitch contour
    for i in range(0, len(time_frames), max(1, len(time_frames)//15)):
        if not np.isnan(valid_freq[i]):
            plt.annotate(notes[i],
                        (time_frames[i], valid_freq[i]),
                        xytext=(0, 15),
                        textcoords='offset points',
                        fontsize=12,
                        ha='center',
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    plt.title(title)
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.ylim(75, 600)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Audio settings
    chunk = 1024
    sample_format = pa.paInt16
    channels = 1
    fs = 44100
    seconds = 15
    filename = "output.wav"
    teacher_audio_path = "/Users/aashviagarwal/Documents/GROOVE_Eclub/audio_subset/teacher/saregamagaresa.wav"
    student_audio_path = "output.wav"

    # STEP 1: Set the tonic frequency
    tonic_freq = set_tonic_frequency(filename, chunk, channels, sample_format, fs)

    # STEP 2: Play teacher audio
    play_teacher_audio(teacher_audio_path)

    # STEP 3: Record student audio
    start_recording(filename, channels, sample_format, fs, chunk, seconds)

    # STEP 4: Play back the recorded audio
    play_recorded_audio(filename)

    # STEP 5: Process and identify notes for both student and teacher
    print("\n--- Starting Note Identification and Plotting ---")
    
    # Load and process teacher audio
    teacher_audio, sr = librosa.load(teacher_audio_path, sr=fs)
    teacher_trimmed = remove_silence_advanced(teacher_audio, sr, top_db=40)
    teacher_trimmed, _ = librosa.effects.trim(teacher_trimmed, top_db=30)
    teacher_freq = librosa.yin(teacher_trimmed, fmin=75, fmax=300)
    teacher_notes = librosa.hz_to_svara_h(teacher_freq, Sa=220.0, octave=3, abbr=False)
    
    print("\nDetected Teacher Notes:")
    for note in teacher_notes:
        print(note, end=" ")
    print(f"\nTotal notes detected: {len(teacher_notes)}\n")
    
    # Plot teacher's notes with the original function
    plot_pitch_and_svara(teacher_freq, teacher_notes, sr, 'Teacher Pitch Contour with Svara Labels')

    # Load and process student audio
    student_audio, sr = librosa.load(student_audio_path, sr=fs)
    student_trimmed = remove_silence_advanced(student_audio, sr, top_db=40)
    student_trimmed, _ = librosa.effects.trim(student_trimmed, top_db=30)
    student_freq = librosa.yin(student_trimmed, fmin=75, fmax=300)
    student_notes = librosa.hz_to_svara_h(student_freq, Sa=tonic_freq, octave=3, abbr=False)

    print("\nDetected Student Notes:")
    for note in student_notes:
        print(note, end=" ")
    print(f"\nTotal notes detected: {len(student_notes)}")
    
    # STEP 6: DTW Analysis and Mistake Detection
    print("\n--- Starting DTW Analysis ---")
    
    notes_duration = note_durations(teacher_notes)
    cost_matrix = compute_dtw_cost_matrix(student_freq, teacher_freq)
    acc_cost, path = find_optimal_dtw_path(cost_matrix)
    student_mistakes, student_mistake_times = mistake_detection(notes_duration, teacher_freq, student_freq, cost_matrix, path)
    
    # STEP 7: Save the final results and plot the student's chart with mistakes
    df = pd.DataFrame(student_mistake_times, columns=['Start Time (s)', 'End Time (s)'])
    df.to_csv('RESULTS.csv', index=False)
    print("\nMistake analysis saved to RESULTS.csv")

    # Read and print the CSV content
    results_df = pd.read_csv('RESULTS.csv')
    print("\n--- Contents of RESULTS.csv ---")
    print(results_df)

    # Plot student's notes with highlighted mistakes
    plot_student_pitch_with_mistakes(student_freq, student_notes, sr, 'Student Pitch Contour with Svara Labels', mistake_times=student_mistake_times)

if __name__ == "_main_":
    main()
