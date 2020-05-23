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
import play
from sklearn.preprocessing import LabelEncoder
import pickle
import re
#from tqdm import tqdm
df=pd.read_csv("esc50.csv")
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
 'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
  'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
df=df.loc[df['category'].isin(selected)]

path="audio/segmented100/"
n_fft=2048
hop_length=512
X=[]
Y=[]
for filename in df.filename:
    class_=df.loc[df['filename']==filename,'category'].iloc[0]
    name=re.sub(".wav","",filename)
    for i in range(50):
        tmp,rate=librosa.load(path+name+str(i)+'.wav',sr=44100)
        stft=librosa.core.stft(tmp,hop_length=hop_length,n_fft=n_fft)
        spectrogram=np.abs(stft)
        log_spectrogram=librosa.amplitude_to_db(spectrogram)
        X.append(log_spectrogram)
        Y.append(class_)
print(X)
X=np.array(X)
print("X shape: ",X.shape)
print("X[0] shape: ",X[0].shape)

pickle_out=open("X5.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()
Y=np.array(Y)



print("Y shape: ",Y.shape)
print(Y)

LabelEncoder=LabelEncoder()
Y=LabelEncoder.fit_transform(Y)

pickle_out=open("Y5.pickle","wb")
pickle.dump(Y,pickle_out)
pickle_out.close()
"""

X=np.array(X)
#print("X:",X[0].shape)
Y=np.concatenate((Y,Y))
LabelEncoder=LabelEncoder()
Y=LabelEncoder.fit_transform(Y)
print("label shape",Y.shape)
print("x shape",X.shape)
print("SAMPLE shape",X[0].shape)
split=1800
train,test = X[:split,:],X[split:,:]
train_y,test_y= Y[:split],Y[split:]
print("train shape ",train.shape)
print("test shape ",test.shape)
print("1 sample :",X[0])


#print("input_1,input_2",input_1,input_2)
#print("X shape",X.shape)
input_1=1025
input_2=431
train=train.reshape(1800,1025,431,1)
"""
print("X[0] : ",X[0])
X=X.reshape(1,60000,2,1)
print("reshaped: ",X.shape)
print("X[0] after reshape : ",X[0])
"""
model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3) ,activation="relu",input_shape=(input_1,input_2,1)))
model.add(Dropout(0,5))
model.add(Conv2D(64,kernel_size=(3,3) ,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,kernel_size=(3,3) ,activation="relu"))
model.add(Dropout(0,5))
model.add(Conv2D(128,kernel_size=(3,3) ,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128,kernel_size=(3,3) ,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation="softmax"))

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training and Evaluation of the model
model.fit(train, train_y, batch_size = 30 ,epochs=10,validation_split=0.1)


# save the model to disk
filename = 'model1.sav'
pickle.dump(model, open(filename, 'wb'))
pickle.close()
"""
