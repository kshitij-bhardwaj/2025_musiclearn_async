import librosa
import numpy as np
import pandas as pd
import crepe
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict


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
        #print(n,m,"\n\n")
        cost_matrix = np.zeros((n, m))
        
        #print("Student Pitch is:",student_pitch,"\n\n")
        #print("Teacher Pitch is:",teacher_pitch,"\n\n") 

        # Fill cost matrix with log-scale distances
        for i in range(n):
            for j in range(m):
                cost_matrix[i, j] = log_distance(student_pitch[i], teacher_pitch[j])
        
        #print("Cost Matrix is:",cost_matrix,"\n\n")
        # plt.figsize(8, 6)
        # plt.imshow(cost_matrix, origin='lower', cmap='viridis', aspect='auto')
        # plt.show()
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
        #print("The path is-",path,"\n\n")
        # x_vals = [p[0] for p in path]
        # y_vals = [p[1] for p in path]

        # #Plot
        # plt.figure(figsize=(12, 6))
        # plt.plot(x_vals, y_vals, marker=".", linestyle="-", color="b")
        # plt.xlabel("X values")
        # plt.ylabel("Y values")
        # plt.title("Plot of Given Points")
        # plt.grid(True)
        # plt.show()
        
        return acc_cost, path


#Audio preprocessing and pitch extraction

teacher_audio_path = "C:/Users/abhin/OneDrive/Pictures/JAVASCRIPT/BCI_Challenge/Groove/audio_subset/teacher/20220330101307.wav"
student_audio_path = "C:/Users/abhin/OneDrive/Pictures/JAVASCRIPT/BCI_Challenge/Groove/audio_subset/student/20220425105156.wav"

teacher_audio, sr = librosa.load(teacher_audio_path, sr=44100)
student_audio, sr = librosa.load(student_audio_path, sr=44100)

#print(len(teacher_audio),len(student_audio),"\n\n")

teacher_trimmed, index = librosa.effects.trim(teacher_audio ,top_db=20, frame_length = 2048,hop_length=512)
student_trimmed, index = librosa.effects.trim(student_audio ,top_db=20, frame_length = 2048,hop_length=512)  

#print(len(teacher_trimmed),len(student_trimmed),"\n\n")

teacher_freq = librosa.yin(teacher_trimmed,fmin=75,fmax=600)
student_freq = librosa.yin(student_trimmed,fmin=75,fmax=600)

#print(len(teacher_freq),len(student_freq),"\n\n")

# Convert frequencies to notes using Sa as 220 Hz for teacher and 207 Hz for student
# Sa is chosen as 220 Hz (A3) for teacher and 207 Hz (G#3) for student to account for pitch differences
teacher_notes = librosa.hz_to_svara_h(teacher_freq, Sa=220.0, abbr = False)
student_notes = librosa.hz_to_svara_h(student_freq, Sa=207.0, abbr = False)

# print(len(teacher_freq),len(student_freq),"\n")
# print(len(teacher_notes),len(student_notes))

# Compute DTW cost matrix
cost_matrix = compute_dtw_cost_matrix(student_freq, teacher_freq)
# Find optimal DTW path
acc_cost, path = find_optimal_dtw_path(cost_matrix)

#print(path)

# Segment teacher notes into continuous segments and store their start and end indices along with the note
teacher_notes_duration = []
start_idx = 0
end_idx = 0
print(len(teacher_notes))
for i in range(len(teacher_notes)-1):
     if teacher_notes[i] == teacher_notes[i+1]:
          end_idx = i+1
     else:
          duration = end_idx - start_idx + 1
          teacher_notes_duration.append((teacher_notes[i], start_idx , end_idx))
          start_idx = i+1
          end_idx = i+1    

teacher_notes_duration.append((teacher_notes[-1], start_idx , len(teacher_notes)-1))

# for note,start_idx,end_idx in teacher_notes_duration:
#      print(f"Note: {note}, Start_idx: {start_idx}, End_idx: {end_idx}")  

# Note wise mistake analysis for student audio based on DTW path and cost matrix and mean frequency difference
student_mistakes = []

for i in range(len(teacher_notes_duration)):
     student = []
     teacher = []
     cost = 0
     for j in range(teacher_notes_duration[i][1],teacher_notes_duration[i][2]+1):
          teacher.append(teacher_freq[j])
          for k in range(len(path)):
               if path[k][1] == j:
                    # print("The path is-",path[k],"\n\n")
                    student.append(path[k][0]) 
                    cost += cost_matrix[path[k][0]][path[k][1]]   
     student_mean = np.mean(student_freq[student])
     teacher_mean = np.mean(teacher)
     mean_diff = abs(student_mean - teacher_mean)
     
     if cost*mean_diff > 5:
        #print(cost*mean_diff,"\n\n")
        student_mistakes.append((student[0],student[-1]))

student_mistakes = list(dict.fromkeys(student_mistakes))

student_mistakes_duration = []

for (i,j) in student_mistakes:
     student_mistakes_duration.append((round(i*0.011,2), round((j+1)*0.011,2)))

print(student_mistakes_duration,"\n\n")

print("Student mistakes at indices:",student_mistakes)

df = pd.DataFrame(student_mistakes_duration, columns=['Start Time (s)', 'End Time (s)'])
df.to_csv('RESULTS.csv', index=False)


# New idea for mapping notes to indices and retrieving indices for a specific note 
def solve(v, w, s):
    m = defaultdict(list)   # string -> list of ints

    for pair in v:
        if len(pair) < 2:   # safety check
            continue
        idx = pair[1]
        if 0 <= idx < len(w):   # ensure index is valid
            m[w[idx]].append(pair[0])

    return m.get(s, [])  # return list if exists, else empty
    
          
array = solve(path, teacher_notes, 'Sa')

