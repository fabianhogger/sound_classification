from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd
import librosa.display
import librosa
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM
from sklearn.preprocessing import LabelEncoder
import pickle
import re
from python_speech_features import mfcc

"""
##DATA CREATION CODE ,UNCOMMENT THIS SECTION TO RECTREATE THE DATA

df=pd.read_csv("esc50.csv")#open csv containing filenames and classes
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
 'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
  'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']

df=df.loc[df['category'].isin(selected)]# selected only classes listed above


for filename in df.filename:
    t1 = 0    #Works in milliseconds ,move 100ms every time ,i goes up to 250 because training files are 5s long
    t2 =  100
    Audio = AudioSegment.from_wav(path+filename)
    for i in range(250):
        newAudio = Audio[t1:t2]
        t1=t1+100
        t2=t2+100
        name=re.sub(".wav","",filename)
        newAudio.export('audio/segmented10/'+name+str(i)+".wav", format="wav")

X=[]
Y=[]
_min=200000
rate=44100
_max=-1
step=int(rate/10)
rand1=0
for i in range(60000):
X=[]
Y=[]
for filename in df.filename:
    class_=df.loc[df['filename']==filename,'category'].iloc[0]#find  class of audio file
    name=re.sub(".wav","",filename) #remove wav part
    for i in range(50):
        tmp,rate=librosa.load("audio/segmented100/"+name+str(i)+'.wav',sr=44100) #load the 100ms segments
        mf=librosa.feature.mfcc(tmp,rate) #calculate mfcc
        X.append(mf)
        Y.append(class_)

print("X shape: ",X.shape)
print("X[0] shape: ",X[0].shape)

pickle_out=open("X6.pickle","wb") #save calculations
pickle.dump(X,pickle_out)
pickle_out.close()
Y=np.array(Y)
print("Y shape: ",Y.shape)
print(Y)

LabelEncoder=LabelEncoder()
Y=LabelEncoder.fit_transform(Y)# map the strings to int

pickle_out=open("Y6.pickle","wb") #save calculations
pickle.dump(Y,pickle_out)
pickle_out.close()
"""


pickle_in=open("Y6.pickle","rb")#loading them back
Y=pickle.load(pickle_in)

pickle_in=open("X6.pickle","rb")#loading them back
X=pickle.load(pickle_in)


print("label shape",Y.shape)
print("x shape",X.shape)
print("SAMPLE shape",X[0].shape)

print("X[0] : ",X[0])
X=X.reshape(60000,20,9,1) #reshape to 4d array


model=Sequential()

model.add(Conv2D(128,kernel_size=(3,3) ,activation="relu",input_shape=(20,9,1)))
model.add(Conv2D(64,(3,3),activation='relu',strides=(1,1),padding='same'))
model.add(Conv2D(128,(3,3),activation='relu',strides=(1,1),padding='same'))
model.add(Conv2D(256,(3,3),activation='relu',strides=(1,1),padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(30,activation='softmax'))

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, Y, batch_size = 1 ,epochs=10,validation_split=0.1)
model.save("model3.sav")# save model


