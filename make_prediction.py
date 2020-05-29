import pickle
from pydub import AudioSegment
from keras.models import load_model
import librosa
import numpy as np
import pandas as pd
import sys
# load model
model = load_model('model3.sav')
df=pd.read_csv("classes.csv")#loading the csv with the classes mapped to integers
audio=AudioSegment.from_wav(sys.argv[1])#opening file 
X=[]
t1=0
t2=100
duration=int(sys.argv[2])
for i in  range(duration):#cutting file to 100millisecond pieces
    newAudio = audio[t1:t2]
    t1=t1+100
    t2=t2+100
    newAudio.export('name'+str(i)+".wav", format="wav")
    tmp,rate=librosa.load('name'+str(i)+".wav",sr=44100)
    mf=librosa.feature.mfcc(tmp,rate)
    X.append(mf)
X=np.array(X)
#print("X shape: ",X.shape)
#print("X[0] shape: ",X[0].shape)
X=X.reshape(duration ,20,9,1)
pred=model.predict(X)
#print("pred arr",pred)
print("      ")
pred = pred.argmax(axis=1)
for i in range(pred.size):
    print("Class ",df.loc[df['val']==pred[i],'Name'].iloc[0])

def make_pred(name,duration):
    # load model
    model = load_model('model3.sav')
    df=pd.read_csv("classes.csv")
    audio=AudioSegment.from_wav("P:/"+name)
    X=[]
    t1=0
    t2=100
    duration=int(duration)
    for i in  range(duration):#cutting file to 100millisecond pieces
        newAudio = audio[t1:t2]
        t1=t1+100
        t2=t2+100
        newAudio.export('name'+str(i)+".wav", format="wav")
        tmp,rate=librosa.load('name'+str(i)+".wav",sr=44100)
        mf=librosa.feature.mfcc(tmp,rate)
        X.append(mf)
    X=np.array(X)
    #print("X shape: ",X.shape)
    #print("X[0] shape: ",X[0].shape)
    X=X.reshape(duration ,20,9,1)
    pred=model.predict(X)
    #print("pred arr",pred)
    print("      ")
    pred = pred.argmax(axis=1)
    for i in range(pred.size):
        print("Class ",df.loc[df['val']==pred[i],'Name'].iloc[0])
