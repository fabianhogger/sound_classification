import librosa
import numpy as np # linear algebra
import pandas as pd
from pydub import AudioSegment
import re
import pickle
import matplotlib.pyplot as plt # plotting


"""
X=[]
for i in range(50):
    newAudio,rate = librosa.load('audio/segmented100/1-977-A-39'+str(i)+'.wav',44100)
    mfcc=librosa.feature.mfcc(newAudio,sr=44100)
    X.append(newAudio)
X=np.array(X)
print(X[0].shape)

df=pd.read_csv("esc50.csv")
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
    'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
     'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
      'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
df=df.loc[df['category'].isin(selected)]
path='audio/44100/'

for filename in df.filename:
    t1 = 0#Works in milliseconds
    t2 =  20
    Audio = AudioSegment.from_wav(path+filename)
    for i in range(250):
        newAudio = Audio[t1:t2]
        t1=t1+20
        t2=t2+20
        name=re.sub(".wav","",filename)
        newAudio.export('audio/segmented10/'+name+str(i)+".wav", format="wav")

path='audio/44100/segmented10/'

aud,rate=librosa.load("",sr=44100)
aud=librosa.feature.mfcc(aud,rate)
fig = plt.Figure()
canvas = FigureCanvas(fig)
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
ax = fig.add_subplot(111)
p = librosa.display.specshow(aud,sr=44100,hop_length=512, ax=ax, y_axis='log', x_axis='time')
fig.savefig('classes/mel_framed/mel_spec_fr_'+c+'.png')
"""
