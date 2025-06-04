# Pitch Contour Normalization Guide for Student-Teacher Audio Analysis

## Key Question: Are Praat/Parselmouth pitch contours normalized?

**Answer: NO** - Praat and Parselmouth output **raw fundamental frequency values in Hz (Hertz)**. These are absolute frequency measurements that reflect the actual acoustic properties of each speaker's voice.

## Why Normalization is Needed

### Raw Pitch Output Characteristics:
- **Student (female)**: Typically 180-300 Hz range
- **Teacher (male)**: Typically 80-180 Hz range  
- **Anatomical differences**: Vocal fold size creates systematic frequency differences
- **Comparison challenge**: Raw values cannot be directly compared across speakers

## Normalization Methods

### 1. Z-Score Normalization (Standardization)
```
z = (frequency - speaker_mean) / speaker_std_dev
```
- **Result**: Mean = 0, Standard deviation = 1
- **Use when**: Statistical analysis, machine learning
- **Advantage**: Both speakers on same scale
- **Caution**: Problems with flat contours (low std dev)

### 2. Semitone Conversion
```
semitones = 12 * log2(frequency / reference_frequency)
```
- **Reference options**: Speaker mean, 100 Hz, musical note
- **Result**: Perceptually meaningful intervals
- **Use when**: Musical analysis, cross-linguistic studies
- **Advantage**: Reflects human pitch perception

### 3. Mean-Centered Hz
```
centered = frequency - speaker_mean
```
- **Result**: Relative pitch movement in Hz
- **Use when**: Preserving Hz units is important
- **Advantage**: Simple, interpretable
- **Best for**: Student-teacher prosody comparison

### 4. Proportion of Range (POR)
```
POR = (frequency - speaker_min) / (speaker_max - speaker_min)
```
- **Result**: Values between 0 and 1
- **Use when**: Comparing pitch range utilization

## Recommendations for Your Dataset

### For 44.1kHz, 32-bit Audio Files:
1. **Extract raw pitch first** using Praat/Parselmouth
2. **Apply normalization before plotting/analysis**
3. **Choose method based on research goals**:
   - **Prosody comparison**: Mean-centered Hz or semitones
   - **Statistical modeling**: Z-score normalization
   - **Cross-speaker analysis**: Semitones relative to speaker mean

### Best Practice Workflow:
```python
# 1. Extract raw pitch (unnormalized)
pitch = sound.to_pitch()
pitch_values = pitch.selected_array['frequency']

# 2. Remove unvoiced frames (zeros)
voiced_frames = pitch_values[pitch_values > 0]

# 3. Apply normalization
speaker_mean = np.mean(voiced_frames)
normalized_pitch = voiced_frames - speaker_mean  # Mean-centered

# 4. Plot normalized contours
```

## Important Notes:
- **Raw output is NOT normalized** - you must normalize manually
- **Choose normalization method based on research questions**
- **Document which method used for reproducibility**
- **Consider voicing detection before normalization**