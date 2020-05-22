import librosa,librosa.display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pickle
def augment(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def extract():
    df=pd.read_csv("esc50.csv")
    selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
    'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
     'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
      'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
    df=df.loc[df['category'].isin(selected)]
    path="audio/44100/"
    n_fft=2048
    hop_length=512
    X=[]
    sum=0
    for i in df.filename:
        #melspectrogram
        signal,rate=librosa.load(path+i,sr=44100)#load file in original rate
        signal=librosa.resample(signal,rate,22050)#resample to 22050

        #signal=augment(signal,0.005)#augment data to have more training samples
        #print(signal.shape)
        #signal=librosa.effects.pitch_shift(signal, 22050, n_steps=4)#)#pitch shift data to have more training samples

        signal_n= librosa.util.normalize(signal)#normalize
        stft=librosa.core.stft(signal_n,hop_length=hop_length,n_fft=n_fft)#calculate stft
        spectrogram=np.abs(stft)#get magnitude
        log_spectrogram=librosa.amplitude_to_db(spectrogram)#apply this to bring to logarithmic scale
        sum=sum+1
        if sum==1198:
            print("Reached 1st fase")
        X.append(signal)
    sum=0
    for i in df.filename:
        #melspectrogram
        signal,rate=librosa.load(path+i,sr=44100)#load file in original rate
        signal=librosa.resample(signal,rate,22050)#resample to 22050
        signal=augment(signal,0.005)#augment data to have more training samples
        #print(signal.shape)
        signal=librosa.effects.pitch_shift(signal, 22050, n_steps=4)#)#pitch shift data to have more training samples

        signal_n= librosa.util.normalize(signal)#normalize
        stft=librosa.core.stft(signal_n,hop_length=hop_length,n_fft=n_fft)#calculate stft
        spectrogram=np.abs(stft)#get magnitude
        log_spectrogram=librosa.amplitude_to_db(spectrogram)#apply this to bring to logarithmic scale
        X.append(log_spectrogram)
        sum=sum+1
        if sum==1198:
            print("Reached 2nd fase")
    return X

"""
df=pd.read_csv("esc50.csv")
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
 'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
  'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
df=df.loc[df['category'].isin(selected)]
path="audio/44100/"
n_fft=2048
hop_length=512
X=[]
for c in selected:
    wav_file=df[df.category==c].iloc[0,0]
    print(wav_file,c)
    #framing audio
    audio,rate=librosa.load('audio/44100/'+wav_file,sr=44100)
    audio=librosa.resample(audio,rate,22050)#resample to 22050
    #audio,index=librosa.effects.trim(audio, top_db=2,frame_length=40, hop_length=512)
    #signal_n= librosa.util.normalize(audio)#normalize
    #stft=librosa.core.stft(signal_n,hop_length=hop_length,n_fft=n_fft)#calculate stft
    #spectrogram=np.abs(stft)#get magnitude
    #log_spectrogram=librosa.amplitude_to_db(spectrogram)#

    audio,index=librosa.effects.trim(audio, top_db=2,frame_length=40, hop_length=512)
    mel=librosa.feature.melspectrogram(audio, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=1024)
    #print("Mel shape",mel.shape)
    #framed_audio=librosa.util.frame(mel, frame_length=41, hop_length=20, axis=-1)
    #print("framed",framed_audio.shape)
    #save as image

    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(mel,sr=22050,hop_length=hop_length, ax=ax, y_axis='log', x_axis='time')
    fig.savefig('classes/mel_trimed/mel_spec_tr_'+c+'.png')
    X.append(mel)
X=np.array(X)
print("X shape: ",X.shape)
print("  Spectrogram shape: ",X[0].shape)
print("  Spectrogram 1: ",X[0])
"""
df=pd.read_csv("esc50.csv")
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
    'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
     'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
      'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
df=df.loc[df['category'].isin(selected)]
path="audio/44100/"
n_fft=2048
hop_length=512
X=[]
Y=[]
sum=0
for i in df.filename:
    signal,rate=librosa.load(path+i,sr=44100)
    librosa.resample(signal,rate,22050)
    signal_n= librosa.util.normalize(signal)
    stft=librosa.core.stft(signal_n,hop_length=hop_length,n_fft=n_fft)
    spectrogram=np.abs(stft)
    log_spectrogram=librosa.amplitude_to_db(spectrogram)
    X.append(log_spectrogram)
    cl=df.loc[df['filename']==i,'category'].iloc[0]
    Y.append(cl)
    print("filename: ",i,"Class: ",cl)
    fig = plt.Figure()
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    p = librosa.display.specshow(log_spectrogram,sr=rate,hop_length=hop_length, ax=ax, y_axis='log', x_axis='time')
    fig.savefig('spectrograms/'+i+'.png')

X=np.array(X)
Y=np.array(Y)
print("X shape: ",X.shape)
print("Spectrogram shape: ",X[0].shape)
print("Spectrogram 1: ",X[0])
print("label shape: ",Y.shape)

pickle_out = open("y.pickle","wb")#saving the calculations
pickle.dump(Y, pickle_out)
pickle_out.close()
