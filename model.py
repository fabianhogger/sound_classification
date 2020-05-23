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

#from tqdm import tqdm
"""
df=pd.read_csv("esc50.csv")
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
 'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
  'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
df=df.loc[df['category'].isin(selected)]
X=[]
Y=[]
_min=200000
rate=44100
_max=-1
step=int(rate/10)
rand1=0
for i in range(50000):
    filename=np.random.choice(df['filename'])
    class_=df.loc[df['filename']==filename,'category'].iloc[0]
    #name=re.sub(".wav","",filename)
    #rand1==np.random.randint(0,49)
    rate,wav=wavfile.read('audio/44100/'+filename)
    rand_index=np.random.randint(0,wav.shape[0]-step)
    sample=wav[rand_index:rand_index+step]#ra
    mf=mfcc(sample,samplerate=44100, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=1103, lowfreq=0, highfreq=None, preemph=0.97, ceplifter=22, appendEnergy=True)
    _min=min(np.amin(mf),_min)
    _max=max(np.amax(mf),_max)
    X.append(mf)
    Y.append(class_)
print(X)
X=np.array(X)
X=(X-_min)/(_max-_min)
print("X shape: ",X.shape)
print("X[0] shape: ",X[0].shape)

pickle_out=open("X8.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
Y=np.array(Y)


print("Y shape: ",Y.shape)
print(Y)

LabelEncoder=LabelEncoder()
Y=LabelEncoder.fit_transform(Y)

pickle_out=open("Y8.pickle","wb")
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
X=X.reshape(60000,20,9,1)
#print("reshaped: ",X.shape)
#print("X[0] after reshape : ",X[0])

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
# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training and Evaluation of the model
model.fit(X, Y, batch_size = 30 ,epochs=10,validation_split=0.1)
model.save("model3.sav")
