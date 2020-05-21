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

def test_model(test_x,test_y):
    random_index = np.random.randint(0,len(test_x))
    audio = test_x[random_index]

    plt.figure(figsize=(20,5))
    plt.subplot(121)

    # Generating Spectrogram of audio file.
    spectrogram = librosa.feature.mfcc(audio,rate)
    #plt.title("Spectrogram")
    #librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')

    # Generating wave form of audio file.
    #plt.subplot(122)
    #plt.title("Wave")
    #librosa.display.waveplot(audio, sr=rate)
    #plt.ylabel('Amplitude')
    #  plt.show()

    spectrogram = np.reshape(spectrogram,(1,SPEC_H, SPEC_W,1))
    prediction = list(loaded_model.predict(spectrogram).flatten())
    print("Predicted category:",categories[prediction.index(max(prediction))//10])
    print("Actual category:",categories[test_y[random_index]//10])

    print('\nPredicted class:',classes[prediction.index(max(prediction))])
    print('Actual class:',classes[test_y[random_index]])


df=pd.read_csv('esc50.csv')

print(df.head())
print(df.shape)
selected=['airplane', 'breathing',  'car_horn']

'''
 ['airplane', 'breathing',  'car_horn', 'cat',  'chirping_birds', 'church_bells', 'clapping',
 'coughing',   'crickets','crying_baby', 'dog', 'door_wood_creaks', 'door_wood_knock',  'engine',
  'fireworks', 'footsteps',  'glass_breaking','hand_saw', 'helicopter',  'insects',  'laughing',
   'mouse_click',  'pouring_water', 'rain', 'rooster','siren', 'sneezing','thunderstorm',  'train','wind']
'''

df=df.loc[df['category'].isin(selected)]
categories = df.columns.tolist()
print("list of categories: ",categories)

classes =list(df[categories[0]].values)
classes.extend(list(df[categories[1]].values))
classes.extend(list(df[categories[2]].values))
print("list of classes: ",classes)
print(df.head())
#classes=list(np.unique(df.category))
#print(classes)
#fig,ax=plt.subplots()
#ax.pie(categories,labels=categories.index,autopct='%1.1f%%',shadow=False,startangle=90)
#plt.show()
#print(df['category'].unique())

#x,sr=librosa.load('clean/1-11687-A-47.wav',sr=44100)
#X = librosa.stft(x)
#Xdb = librosa.amplitude_to_db(abs(X))
#plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#plt.colorbar()
#plt.show()
#signals={}
fft={}
'''
for c in classes:
    wav_file=df[df.category==c].iloc[0,0]
    signal,rate=librosa.load('audio/44100/'+wav_file,sr=44100)
    m=librosa.feature.mfcc(signal,sr=44100,n_mfcc=20,dct_type=2,norm='ortho',lifter=0)
    print(m)
    librosa.display.specshow(m, x_axis='time')
    plt.colorbar()
    plt.title(c)
    plt.tight_layout()
    plt.show()
'''
audio_files=[]
Y=[]
for i in df.filename:
    y=df.loc[df['filename'] == i, 'category'].iloc[0]
    audio,rate=librosa.load('audio/'+i,sr=44100)
    audio_files.append(audio)
    Y.append(y)

Y=np.array(Y)
print(Y)
LabelEncoder=LabelEncoder()
integer_encoded=LabelEncoder.fit_transform(Y)
print("integer: ",integer_encoded)

audio_files= np.array(audio_files)
print(audio_files.shape)
X=list(audio_files)
Y=list(integer_encoded)

train_x,test_x,train_y,test_y = train_test_split(X, integer_encoded, test_size = 0.1, random_state=5, shuffle = True)
print("train x: ",train_x)
print("test x: ",test_x)
print("test y: ",test_y)

x = train_x
train_x = []
length = len(train_y)
for i in range(length):
    train_x.append(librosa.feature.melspectrogram(x[i], rate))
del x
SPEC_H, SPEC_W = train_x[0].shape
train_x = np.reshape(train_x,(length,SPEC_H, SPEC_W,1))
print("train x: ",train_x.shape)


print("train x: ",train_x.shape)
x = test_x
test_x = []
length = len(test_y)
for i in range(length):
    test_x.append(librosa.feature.melspectrogram(x[i], rate))
del x
SPEC_H, SPEC_W = test_x[0].shape
test_x = np.reshape(test_x,(length,SPEC_H, SPEC_W,1))

print("test x: ",test_x.shape)
print("test y: ",test_y.shape)


model = Sequential()

# add layers
model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(SPEC_H, SPEC_W, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=3, activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation="softmax"))

# compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training and Evaluation of the model
model.fit(train_x, train_y, batch_size = 30 ,epochs=30,validation_split=0.1)


# save the model to disk
filename = 'mfcc_model.sav'
pickle.dump(model, open(filename, 'wb'))



# some time later...

filename = 'mfcc_model.sav'
"""
filename = 'finalized_model.sav'

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model.summary())

"""
random_index =0
audio1 = test_x[random_index]
spectrogram = librosa.feature.melspectrogram(audio1,44100)
print("audio1 shape ",audio1.shape)
spectrogram = np.reshape(spectrogram,(1,SPEC_H, SPEC_W,1))
print("audio1 shape now ",audio1.shape)
prediction = list(loaded_model.predict(spectrogram).flatten())
print("prediction ",prediction)
print("prediction ",prediction.index(max(prediction))//10   )
#print("Actual category:",[test_y[random_index]])
"""

for i in range(5):
    test_model(test_x,test_y)
