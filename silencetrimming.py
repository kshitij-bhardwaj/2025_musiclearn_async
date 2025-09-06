import librosa
import numpy as np
from google.colab import files
import IPython.display as ipd

def detect_speech_timeframe(audio_file_path, top_db=20):
    """
    Detect the start and end time of speech in an audio file by trimming silences.

    Parameters:
    - audio_file_path: path to the audio file
    - top_db: threshold for silence detection (higher = more aggressive trimming)

    Returns:
    - start_time: start time of speech in seconds
    - end_time: end time of speech in seconds
    - duration: duration of speech in seconds
    """

    # Load the audio file
    print("Loading audio file...")
    y, sr = librosa.load(audio_file_path, sr=None)

    # Get original duration
    original_duration = len(y) / sr
    print(f"Original audio duration: {original_duration:.2f} seconds")

    # Trim silence from beginning and end
    print("Detecting speech boundaries...")
    y_trimmed, index = librosa.effects.trim(y, top_db=top_db)

    # Calculate start and end times
    start_sample = index[0]
    end_sample = index[1]

    start_time = start_sample / sr
    end_time = end_sample / sr
    speech_duration = end_time - start_time

    return start_time, end_time, speech_duration, y, sr, y_trimmed

def main():
    print("=== Audio Speech Detection ===")
    print("Please upload your audio file (supported formats: wav, mp3, flac, etc.)")

    # Prompt user to upload file
    uploaded = files.upload()

    if not uploaded:
        print("No file uploaded. Exiting.")
        return

    # Get the uploaded file name
    file_name = list(uploaded.keys())[0]
    print(f"\nProcessing file: {file_name}")

    try:
        # Detect speech timeframe
        start_time, end_time, speech_duration, original_audio, sr, trimmed_audio = detect_speech_timeframe(file_name)

        # Display results
        print("\n" + "="*50)
        print("SPEECH DETECTION RESULTS")
        print("="*50)
        print(f"Speech starts at: {start_time:.3f} seconds")
        print(f"Speech ends at: {end_time:.3f} seconds")
        print(f"Speech duration: {speech_duration:.3f} seconds")
        print(f"Original duration: {len(original_audio)/sr:.3f} seconds")
        print(f"Silence trimmed: {len(original_audio)/sr - speech_duration:.3f} seconds")
        print("="*50)

        # Optional: Play original and trimmed audio for comparison
        print("\nOriginal audio:")
        ipd.display(ipd.Audio(original_audio, rate=sr))

        print("\nTrimmed audio (speech only):")
        ipd.display(ipd.Audio(trimmed_audio, rate=sr))

        # Return the timeframe for further use if needed
        return {
            'start_time': start_time,
            'end_time': end_time,
            'speech_duration': speech_duration,
            'sample_rate': sr
        }

    except Exception as e:
        print(f"Error processing audio file: {str(e)}")
        print("Make sure the file is a valid audio format and try again.")
        return None

# Additional function to extract speech segment and save it
def save_speech_segment(file_name, start_time, end_time, output_name="speech_only.wav"):
    """
    Extract and save just the speech segment to a new file.
    """
    y, sr = librosa.load(file_name, sr=None)

    # Convert times to sample indices
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract the speech segment
    speech_segment = y[start_sample:end_sample]

    # Save the trimmed audio
    import soundfile as sf
    sf.write(output_name, speech_segment, sr)
    print(f"Speech segment saved as: {output_name}")

if __name__ == "__main__":
    # Run the main detection
    result = main()

    # If you want to save the speech-only segment, uncomment the following lines:
    # if result:
    #     save_speech_segment(list(files.upload().keys())[0],
    #                        result['start_time'],
    #                        result['end_time'])