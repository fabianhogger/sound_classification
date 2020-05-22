from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd
import librosa.display
import librosa
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pickle
from tqdm import tqdm
from PIL import Image
from numpy import asarray
from keras.utils import to_categorical

df=pd.read_csv("esc50.csv")
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
 'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
  'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
df=df.loc[df['category'].isin(selected)]

path="audio/44100/"


#print(Y)

Y=[]
X=[]
x_path='spectrograms/'
for i in df.filename:
    cl=df.loc[df['filename']==i,'category'].iloc[0]
    Y.append(cl)
    img=Image.open(x_path+i+'.png')
    img_array=asarray(img)
    X.append(img_array)
for i in df.filename:
    cl=df.loc[df['filename']==i,'category'].iloc[0]
    Y.append(cl)
    img=Image.open(x_path+'morphed/'+i+'.png')
    img_array=asarray(img)
    X.append(img_array)


X=np.array(X)
print("X:",X[0].shape)

LabelEncoder=LabelEncoder()
Y=LabelEncoder.fit_transform(Y)
print("label shape",Y.shape)
#print("input_1,input_2",input_1,input_2)
print("X shape",X.shape)
train_x,test_x,train_y,test_y = train_test_split(X, Y, test_size = 0.1, random_state=5, shuffle = True)
input_1,input_2,input_3=X[0].shape

model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3) ,activation="relu",input_shape=(input_1,input_2,input_3)))
model.add(Dropout(0,2))
model.add(Conv2D(32,kernel_size=(3,3) ,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64,kernel_size=(3,3) ,activation="relu"))
model.add(Dropout(0,2))
model.add(Conv2D(64,kernel_size=(3,3) ,activation="relu"))
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
model.fit(train_x, train_y, batch_size = 32 ,epochs=30,validation_split=0.1)


# save the model to disk
filename = 'model1.sav'
pickle.dump(model, open(filename, 'wb'))
