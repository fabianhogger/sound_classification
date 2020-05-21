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
import extract_feat as ex
df=pd.read_csv("esc50.csv")
selected=['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
 'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
  'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
df=df.loc[df['category'].isin(selected)]
X=[]
path="audio/44100/"

X=ex.extract()

"""
model=Sequential()

model.add(Conv1D(32,kernel_size=(3,3) ,activation="relu",input_shape=()))
model.add(Dropout(0,2))
model.add(Conv1D(32,kernel_size=(3,3) ,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv1D(64,kernel_size=(3,3) ,activation="relu"))
model.add(Dropout(0,2))
model.add(Conv1D(64,kernel_size=(3,3) ,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv1D(128,kernel_size=(3,3) ,activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
"""
