import librosa
import numpy as np # linear algebra
import pandas as pd
from pydub import AudioSegment
import re


"""
t1 = 0#Works in milliseconds
t2 =  1000
Audio = AudioSegment.from_wav("test.wav")
"""
df=pd.read_csv("esc50.csv")
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
    'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
     'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
      'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
df=df.loc[df['category'].isin(selected)]
path='audio/44100/'
for filename in df.filename:
    t1 = 0#Works in milliseconds
    t2 =  100
    Audio = AudioSegment.from_wav(path+filename)
    for i in range(50):
        newAudio = Audio[t1:t2]
        t1=t1+100
        t2=t2+100
        name=re.sub(".wav","",filename)
        newAudio.export('audio/segmented100/'+name+str(i)+".wav", format="wav")
