!pip install librosa numpy matplotlib
import librosa
import numpy as np
import matplotlib.pyplot as plt

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

def detect_notes(audio_path):
    y, sr = librosa.load(audio_path)
    f0, _, _ = librosa.pyin(y, fmin=SA_FREQ*0.9, fmax=MAX_FREQ*1.1, sr=sr)
    
    times = librosa.times_like(f0, sr=sr)
    notes = [freq_to_note(f) if not np.isnan(f) else None for f in f0]
    
    note_segments = {}
    current_note = None
    start_time = 0
    
    for i, (time, note) in enumerate(zip(times, notes)):
        if note != current_note:
            if current_note:
                duration = time - start_time
                if duration >= MIN_NOTE_DURATION:
                    if current_note not in note_segments:
                        note_segments[current_note] = []
                    note_segments[current_note].append((start_time, time))
            current_note = note
            start_time = time
    
    return note_segments, f0, times, y, sr

def plot_results(note_segments, f0, times, y, sr):
    plt.figure(figsize=(14, 6))
    
    # Create main waveform plot
    ax1 = plt.gca()
    librosa.display.waveshow(y, sr=sr, alpha=0.4, ax=ax1)
    
    # Add pitch contour on secondary Y-axis
    ax2 = ax1.twinx()
    ax2.plot(times, f0, color='cyan', alpha=0.8, linewidth=2, label='Pitch')
    
    # Set frequency range and labels
    ax2.set_ylim(SA_FREQ*0.9, MAX_FREQ*1.1)
    ax2.set_ylabel('Frequency (Hz)', color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan')
    
    # Add horizontal lines for each note's ideal frequency
    ratios = {'Sa':1.0, 'Re':9/8, 'Ga':5/4, 'Ma':4/3,
             'Pa':3/2, 'Dha':5/3, 'Ni':15/8}
    colors = {'Sa':'red', 'Re':'orange', 'Ga':'yellow', 
             'Ma':'green', 'Pa':'blue', 'Dha':'indigo', 'Ni':'violet'}
    
    for note, ratio in ratios.items():
        freq = SA_FREQ * ratio
        ax2.axhline(y=freq, color=colors[note], linestyle=':', 
                   alpha=0.7, label=f'{note} ({freq:.1f}Hz)')
    
    # Add note segments as shaded regions
    for note, segments in note_segments.items():
        for start, end in segments:
            ax1.axvspan(start, end, color=colors[note], alpha=0.2)
    
    ax1.set_title('Indian Classical Note Detection (Frequency on Y-axis)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Usage
audio_path = "sanidhapamagaresa.wav"
note_segments, f0, times, y, sr = detect_notes(audio_path)

print("Detected Notes:")
for note, segments in note_segments.items():
    for start, end in segments:
        print(f"{note}: {start:.2f}s to {end:.2f}s")

plot_results(note_segments, f0, times, y, sr)