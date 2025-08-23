# !pip install librosa numpy scipy --quiet

import librosa
import numpy as np
from scipy.signal import medfilt
# from google.colab import files


# Upload your audio file
print("Upload your WAV audio file")
uploaded = files.upload()
audio_file = list(uploaded.keys())[0]
print(f"Loaded file: {audio_file}")

NOTE_NAMES = ['Sa', 'Komal Re', 'Re', 'Komal Ga', 'Ga', 'Ma', 'Tivra Ma', 'Pa', 'Komal Dha', 'Dha', 'Komal Ni', 'Ni']

def freq_to_semitone(f, tonic_freq=440):
    if f is None or np.isnan(f) or f <= 0:
        return None
    return 12 * np.log2(f / tonic_freq)

def semitone_to_note_name(semi):
    if semi is None:
        return 'Silence'
    semi_rounded = int(np.round(semi)) % 12
    return NOTE_NAMES[semi_rounded]

def smooth_notes(notes, kernel_size=7):
    note_map = {n:i for i,n in enumerate(NOTE_NAMES + ['Silence'])}
    inv_note_map = {v:k for k,v in note_map.items()}
    numeric = [note_map.get(n, note_map['Silence']) for n in notes]
    filtered = medfilt(numeric, kernel_size=kernel_size)
    smoothed_notes = [inv_note_map[n] for n in filtered]
    return smoothed_notes

def segment_notes_with_threshold(note_sequence, times, min_duration=0.1):
    segments = []
    if len(note_sequence) == 0:
        return segments

    prev_note = note_sequence[0]
    start_time = times[0]

    for i in range(1, len(note_sequence)):
        if note_sequence[i] != prev_note:
            end_time = times[i]
            duration = end_time - start_time
            if duration >= min_duration:
                segments.append((prev_note, start_time, end_time))
            prev_note = note_sequence[i]
            start_time = times[i]

    # Last segment
    end_time = times[-1]
    duration = end_time - start_time
    if duration >= min_duration:
        segments.append((prev_note, start_time, end_time))

    return segments

def analyze_notes(audio_file, tonic_freq=440):
    y, sr = librosa.load(audio_file)
    print(f"Audio loaded: {audio_file}, Sample rate: {sr}, Duration: {len(y)/sr:.2f}s")

    # Pitch detection
    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=tonic_freq/2, fmax=tonic_freq*4, sr=sr)
    times = librosa.times_like(f0, sr=sr)

    # Map frequencies to notes
    notes = []
    for f in f0:
        semi = freq_to_semitone(f, tonic_freq)
        note = semitone_to_note_name(semi)
        notes.append(note)

    # Smooth notes to reduce jitter
    smoothed_notes = smooth_notes(notes, kernel_size=7)

    # Segment notes with minimum duration threshold
    segments = segment_notes_with_threshold(smoothed_notes, times, min_duration=0.2)

    print("\nDetected note segments:")
    for note, start, end in segments:
        print(f"{note}: {start:.2f}s to {end:.2f}s")

# Change tonic_freq=your tonic frequency (Hz), e.g., 220 or 440
analyze_notes(audio_file, tonic_freq=220)
f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=150, fmax=350, sr=sr)