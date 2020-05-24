import pickle
from pydub import AudioSegment
from keras.models import load_model
import librosa
import numpy as np
import pandas as pd
pickle_in=open("X6.pickle","rb")
X=pickle.load(pickle_in)
print("X shape: ",X.shape)
print("X[0] shape: ",X[0].shape)
pickle_in=open("Y6.pickle","rb")#loading them back
Y=pickle.load(pickle_in)
print("Y ",Y)
"""
y2=[]

j=0
for i in range(60000):
    if j==0:
        y2.append(Y[i])
    print(Y[i])
    j=j+1
    if j==50:
        j=0
print("y2 ",y2)
print("Y[0] shape: ",Y[0].shape)
df=pd.read_csv("esc50.csv")
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
 'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
  'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
df=df.loc[df['category'].isin(selected)]
y3=[]
for i in df.filename:
    y3.append(df.loc[df['filename']==i,'category'].iloc[0])
print("y3 ",y3)

dr = pd.DataFrame(list(zip(y3, y2)),
               columns =['Name', 'val'])
print(dr.drop_duplicates())
dt=dr.drop_duplicates()
dt.to_csv(r'classes.csv', index = False)
"""

# load model
model = load_model('model3.sav')
# summarize model.
model.summary()
audio=AudioSegment.from_wav('airland.wav')
X=[]
t1=0
t2=100
df=pd.read_csv("classes.csv")

for i in  range(9):
    newAudio = audio[t1:t2]
    t1=t1+100
    t2=t2+100
    newAudio.export('airland'+str(i)+".wav", format="wav")
    tmp,rate=librosa.load('airland'+str(i)+".wav",sr=44100)
    mf=librosa.feature.mfcc(tmp,rate)
    X.append(mf)
X=np.array(X)
print("X shape: ",X.shape)
print("X[0] shape: ",X[0].shape)
X=X.reshape(9,20,9,1)
pred=model.predict(X)
print("pred arr",pred)
print("      ")
pred = pred.argmax(axis=1)
for i in range(pred.size):
    print("Class ",df.loc[df['val']==pred[i],'Name'].iloc[0])
