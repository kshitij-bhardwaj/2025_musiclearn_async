import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wave
import soundfile as sf
import streamlit as st
import streamlit.web.cli as stcli


#Configure Page
st.set_page_config(
    page_title="ASYNC Audio Mistake Detection",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

#DTW implementation

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
    
def compute_dtw_cost_matrix(student_pitch, teacher_pitch):
        """
        Compute DTW cost matrix using log-scale distance
        """
        n, m = len(student_pitch), len(teacher_pitch)
        cost_matrix = np.zeros((n, m))
        
        # Fill cost matrix with log-scale distances
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = log_distance(student_pitch[i], teacher_pitch[j])
        
        return cost_matrix
    
def find_optimal_dtw_path(cost_matrix):
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

def get_sargam_boundaries(sa_freq):
    """
    Given the tonic frequency (Sa), return swara names and frequency boundaries for one octave.
    """
  
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
   
    # Boundaries: midpoints between swaras
    boundaries = [(swara_freqs[i] + swara_freqs[i+1]) / 2 for i in range(len(swara_freqs)-1)]
    
    # Add min/max for outer boundaries
    boundaries = [swara_freqs[0] - (boundaries[0] - swara_freqs[0])] + boundaries + [swara_freqs[-1] + (swara_freqs[-1] - boundaries[-1])]
    return boundaries


def set_tonic_frequency(filename,chunk,channels,fs):
    
    # p = pa.PyAudio()

    # print("Set your tonic frequency (Sa)")

    # stream = p.open(format=sample_format,
    #                     channels=channels,
    #                     rate=fs,
    #                     frames_per_buffer=chunk,
    #                     input=True,)

    # frames = []

    # for i in range(0, int(fs / chunk * 3)):
    #         data = stream.read(chunk)
    #         frames.append(data)

    # stream.stop_stream()
    # stream.close()


    # print('Finished recording tonic frequency')

    # wf = wave.open(filename, 'wb')
    # wf.setnchannels(channels)   
    # wf.setsampwidth(p.get_sample_size(sample_format))
    # wf.setframerate(fs)
    # wf.writeframes(b''.join(frames))
    # wf.close()


    # data,fs = sf.read(filename,dtype='float32')
    # sd.play(data,fs)
    # status = sd.wait()  # Wait until file is done playing

    tonic_audio = librosa.load(filename,sr=44100)
    tonic_freqencies = librosa.yin(tonic_audio[0],fmin=75,fmax=600)
    tonic_freq = np.median(tonic_freqencies)

    #print("Tonic frequency is set to:",tonic_freq)

    return tonic_freq


# def play_teacher_audio(teacher_audio_path):
    
#     print("Now playing the teacher audio")

#     data,fs = sf.read(teacher_audio_path,dtype='float32')
#     sd.play(data,fs)
#     status = sd.wait()  # Wait until file is done playing

#     print("Teacher audio finished playing")

# def start_recording(filename,channels,sample_format,fs,chunk,seconds):
    
#     print("Now record your audio")

#     print('Recording')

#     p = pa.PyAudio()

#     stream = p.open(format=sample_format,
#                     channels=channels,
#                     rate=fs,
#                     frames_per_buffer=chunk,
#                     input=True)

#     frames = []

#     for i in range(0, int(fs / chunk * seconds)):
#         data = stream.read(chunk)
#         frames.append(data)

#     stream.stop_stream()
#     stream.close() 
#     p.terminate()

#     print('Finished recording')

#     wf = wave.open(filename, 'wb')
#     wf.setnchannels(channels)   
#     wf.setsampwidth(p.get_sample_size(sample_format))
#     wf.setframerate(fs)
#     wf.writeframes(b''.join(frames))
#     wf.close()
    

# def play_recorded_audio(filename):
    
#     print("Now playing your recorded audio")
#     data,fs = sf.read(filename,dtype='float32')
#     sd.play(data,fs)
#     status = sd.wait()  # Wait until file is done playing

def load_audio(student_audio_path,teacher_audio_path):
    student_audio, sr = librosa.load(student_audio_path, sr=44100)
    teacher_audio, sr = librosa.load(teacher_audio_path, sr=44100)

    teacher_trimmed, index = librosa.effects.trim(teacher_audio ,top_db=20, frame_length = 2048,hop_length=512)
    student_trimmed, index = librosa.effects.trim(student_audio ,top_db=20, frame_length = 2048,hop_length=512)  


    teacher_freq = librosa.yin(teacher_trimmed,fmin=75,fmax=600)
    student_freq = librosa.yin(student_trimmed,fmin=75,fmax=600)

    return teacher_freq,student_freq

def freq_to_note(teacher_freq, student_freq,tonic_freq):
    # Convert frequencies to notes using Sa as 220 Hz for teacher and 207 Hz for student
    # Sa is chosen as 220 Hz (A3) for teacher and 207 Hz (G#3) for student to account for pitch differences
    # Convert Hz -> ratio to tonic.
    # Ratio -> cents.
    # Cents -> nearest svara index.
    # Find deviation from exact equal-tempered svara.
    teacher_notes = librosa.hz_to_svara_h(teacher_freq, Sa=220.0, abbr = False)
    student_notes = librosa.hz_to_svara_h(student_freq, Sa=tonic_freq, abbr = False)

    return teacher_notes,student_notes

def DTW_Analysis(student_freq,teacher_freq):
    # Compute DTW cost matrix
    cost_matrix = compute_dtw_cost_matrix(student_freq, teacher_freq)
    # Find optimal DTW path
    acc_cost, path = find_optimal_dtw_path(cost_matrix)
    
    return cost_matrix,path

def note_durations(notes):
    # Segment teacher notes into continuous segments and store their start and end indices along with the note
    notes_duration = []
    start_idx = 0
    end_idx = 0

    for i in range(len(notes)-1):
        if notes[i] == notes[i+1]:
            end_idx = i+1
        else:
            duration = end_idx - start_idx + 1
            notes_duration.append((notes[i], start_idx , end_idx))
            start_idx = i+1
            end_idx = i+1    

    notes_duration.append((notes[-1], start_idx , len(notes)-1))

    return notes_duration

def mistake_detection(notes_duration,teacher_freq,student_freq,cost_matrix,path):

    normalized_teacher_freq = 12.0 * np.log2(teacher_freq / np.nanmean(teacher_freq))
    normalized_student_freq = 12.0 * np.log2(student_freq / np.nanmean(student_freq))
    # Note wise mistake analysis for student audio based on DTW path and cost matrix and mean frequency difference
    student_mistakes = []

    for i in range(len(notes_duration)):
        student = []
        teacher = []
        cost = 0
        for j in range(notes_duration[i][1],notes_duration[i][2]+1):
            teacher.append(normalized_teacher_freq[j])
            for k in range(len(path)):
                if path[k][1] == j:
                        # print("The path is-",path[k],"\n\n")
                        student.append(path[k][0]) 
                        cost += cost_matrix[path[k][0]][path[k][1]]   
        student_mean = abs(np.mean(normalized_student_freq[student]))
        teacher_mean = abs(np.mean(teacher))
        
        mean_diff = (abs(student_mean - teacher_mean))/(student_mean + teacher_mean)
        
        if cost*mean_diff > 0.05:   # Threshold has to be tuned 
            
            student_mistakes.append((student[0],student[-1]))

    student_mistakes = list(dict.fromkeys(student_mistakes))

    student_mistakes_duration = []

    for (i,j) in student_mistakes:
        student_mistakes_duration.append(i)
        student_mistakes_duration.append(j)

    student_times = librosa.frames_to_time(student_mistakes_duration, sr=44100, hop_length=512)

    student_mistakes_times = []

    for i in range(0,len(student_mistakes_duration),2):
        if student_times[i] < student_times[i+1]:   # to avoid negative durations
            student_mistakes_times.append((student_times[i],student_times[i+1]))

    # for (i,j) in student_mistakes_times:
    #     print("Start time:",i,"End time:",j)

    return student_mistakes, student_mistakes_times    

def plot_student_pitch_with_mistakes(freq, notes, sr, title, mistake_times):
    """
    Plots the student's pitch contour with svara labels and highlights mistakes in red,
    and renders the resulting figure to Streamlit (st.pyplot).
    """
    hop_length = 512
    time_frames = librosa.frames_to_time(np.arange(len(freq)), sr=sr, hop_length=hop_length)

    # ensure float array and mark unvoiced as NaN
    valid_freq = np.array(freq, dtype=float)
    valid_freq[valid_freq == 0] = np.nan

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(time_frames, valid_freq, 'b-', linewidth=2, alpha=0.7, label='Student Pitch Contour')
    ax.fill_between(time_frames, valid_freq, alpha=0.3)

    # Highlight mistake segments in red
    for start_time, end_time in mistake_times:
        mask = (time_frames >= start_time) & (time_frames <= end_time)
        if np.any(mask):
            ax.plot(time_frames[mask], valid_freq[mask], 'r-', linewidth=3, alpha=0.8)
            ax.fill_between(time_frames[mask], valid_freq[mask], color='red', alpha=0.2)

    # Add a legend entry for the mistake segments
    ax.plot([], [], 'r-', linewidth=3, alpha=0.8, label='Mistake Segments')

    # Add svara labels along the pitch contour (sparse)
    step = max(1, len(time_frames) // 15)
    for i in range(0, len(time_frames), step):
        if not np.isnan(valid_freq[i]):
            ax.annotate(notes[i],
                        (time_frames[i], valid_freq[i]),
                        xytext=(0, 15),
                        textcoords='offset points',
                        fontsize=12,
                        ha='center',
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    ax.set_title(title)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (seconds)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(75, 600)
    ax.legend()
    plt.tight_layout()

    # Render to Streamlit and close figure to avoid duplicate output
    st.pyplot(fig)
    plt.close(fig)


st.title("GROOVE", anchor=None, help=None, width="stretch")
st.divider()
st.header("Sing along and learn music on your own pace")
st.divider()
st.text("Record your audio after listening to the teacher audio and learn where you are making mistakes")

# STEP 1 - Set the tonic frequency for user input
chunk = 1024
# sample_format = pa.paInt16
channels = 1    
fs = 44100
seconds = 15  # Duration of recording
filename = "output.wav"

tonic_freq = 220.0

option = st.selectbox("How do you want to set your tonic?",("Sing","Manual"),index=None)
if (option == "Sing"):
    tonic_audio = st.audio_input("Record Tonic")
    if(tonic_audio):
        tonic_freq = set_tonic_frequency(tonic_audio,chunk,channels,fs)
        st.text("Tonic frequency is set to-")
        st.write(tonic_freq)
elif (option == "Manual"):
    tonic_freq = st.slider("Select your tonic-",60,500,220)
# STEP 2 - Playback the teachers Audio

teacher_audio_path = "20220330101307.wav"
st.text("Listen to the teacher audio and try to sing in a similar way.")
st.audio(teacher_audio_path,loop=True)
# STEP 3 - Record Student Audio

audio_value = st.audio_input("Record your Audio")

# STEP 4 - Playback the recorded Audio
if (audio_value):
    st.text("Play your recorded audio")
    st.audio(audio_value)

teacher_audio, sr = librosa.load(teacher_audio_path, sr=44100)
student_audio_path = audio_value

# STEP 5 - Analyze the Audio to detect mistakes

if(audio_value):
    if(st.button("Analyze my Audio",type="primary")):
        teacher_freq,  student_freq = load_audio(student_audio_path,teacher_audio_path)
        
        teacher_notes,student_notes = freq_to_note(teacher_freq,student_freq,tonic_freq)
    
        notes_duration = note_durations(teacher_notes)

        cost_matrix,path = DTW_Analysis(student_freq,teacher_freq)

        student_mistakes, student_mistake_times = mistake_detection(notes_duration,teacher_freq,student_freq,cost_matrix,path)

        teacher_mistakes = []

        if (student_mistake_times):
            plot_student_pitch_with_mistakes(student_freq,student_notes,fs,"Student Mistakes",mistake_times=student_mistake_times)
            plot_student_pitch_with_mistakes(teacher_freq,teacher_notes,fs,"Teacher Pitch Contour",teacher_mistakes)

        else:
            st.info("No mistakes detected to overlay on the plot.")
            plot_student_pitch_with_mistakes(student_freq,student_notes,fs,"Student Mistakes",mistake_times=student_mistake_times)

        fig, ax = plt.subplots(figsize=(12, 4))
        # time axis for student_freq (frames -> seconds)
        times = librosa.frames_to_time(np.arange(len(student_freq)), sr=44100, hop_length=512)

        # plot full student pitch contour
        ax.plot(times, student_freq, color='tab:blue', linewidth=1, label='Student Pitch')

        # overlay mistake spans in red using the start/end times from student_mistake_times
        if student_mistake_times:
            for (t_start, t_end) in student_mistake_times:
                # clamp to valid range
                t0 = max(0.0, float(t_start))
                t1 = max(t0, float(t_end))

                # mask frames that fall inside the mistake interval and overplot them in red
                mask = (times >= t0) & (times <= t1)
                if np.any(mask):
                    ax.plot(times[mask], student_freq[mask], color='red', linewidth=2)

                # draw faint vertical boundary lines at start and end
                ax.axvline(t0, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
                ax.axvline(t1, color='red', linestyle='--', linewidth=0.8, alpha=0.8)
        else:
            st.info("No mistakes detected to overlay on the plot.")

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Student Pitch with Mistake Segments')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize='small')
        # show plot in Streamlit
        st.pyplot(fig)

# STEP 6 - Display the final results 

# df = pd.DataFrame(student_mistake_times, columns=['Start Time (s)', 'End Time (s)'])
# df.to_csv('RESULTS.csv', index=False)












