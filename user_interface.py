import streamlit as st
import sounddevice as sd
import soundfile as sf


audio_path = "C:/Users/abhin/OneDrive/Pictures/JAVASCRIPT/BCI_Challenge/Groove/audio_subset/teacher/20220330101307.wav"

def play_teacher_audio(teacher_audio_path):
       
    print("Now playing the teacher audio")

    data,fs = sf.read(teacher_audio_path,dtype='float32')
    sd.play(data,fs)
    status = sd.wait()  # Wait until file is done playing

    print("Teacher audio finished playing")

st.title("GROOVE", anchor=None, help=None, width="stretch")
st.divider()
st.header("Sing along and learn music on your own pace")
st.divider()
st.text("Record your audio after listening to the teacher audio and learn where you are making mistakes")

if (st.button("Listen to teacher Audio",type="primary")):
    play_teacher_audio(audio_path)























