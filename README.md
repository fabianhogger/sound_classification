# Enviromental_Sound_Classification

![Rasp](/IMG_20200525_193715.png)
 Raspberry Pi with Webcam

![dist](/class_distribution.png)


Dataset Class distribution
Dataset used : https://www.kaggle.com/mmoreaux/environmental-sound-classification-50<br/>
CNN with 8 layers.
Data extracted are mfccs for 1/10 of a second of audio.
## Mel Frequency Cepstral Coefficents 
Mfcc is a feature widely used in speech recognition.It consists of a complex series of steps that
have been established to help with sound classification.The thought process behind these steps is to 
mimic the way human hearing works.<br/>

Initially the audio is cut in 20-40ms frames where we assume that the frequency does not have huge changes.
On those frames we apply the SFFT to transfer the signal to the frequency zone and to get the power magnitude.
Afterwards we use the mel filterbank which is a series of low and high frequency filters that bring the signal into something
closer to what the human ear hears(the human ear is better adjusted to hearing the differences between low frequency sounds).<br/>
With the mel filterbank the signal also passes to a logarithmic scale, meaning that lowder noises are now much lowder, something that is true in human hearing aswell.<br/>
The last step is to find the delta coefficients of the signal to show the trajectory of the sound over time ,which as shown in the past increases the accuracy of neural networks.


![breath](/log_spec_breathing.png)

Nice articles related
* http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
* https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

## Code
To classify audio (wav),place it in the same directory with make_prediction.py<br/>
and model3.sav files and give as arguments filename and length in seconds.<br/>
**ex. python make_prediction.py filename.wav 7**
