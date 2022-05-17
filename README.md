
# Resume parser
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

Building a Note taking bot using keras version of deepspeech, Python and basic natural language processing techniques.

## Introduction 

#### Note taking bot
Note taking bot can make a participant more focus on meeting rather than taking notes side by side.
we have used Automatic speech recognition with a CTC loss to create the bot 

ref : https://keras.io/examples/audio/ctc_asr/. 

![alt text](https://raw.githubusercontent.com/vivekalex61/resume_ner/main/images/)
                
## Overview 
- Datasets and Data-Loading
- Data Preprocessing
- Model creation and training
- Recorder creation

### Datasets and Data-Loading
LJ speech is a  public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books. A transcription is provided for each clip. Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.

link : https://keithito.com/LJ-Speech-Dataset/


![alt text](https://raw.githubusercontent.com/vivekalex61/note_taking_bot/main/images/model.png)



### Data Preprocessing

The waveforms in the dataset are represented in the time domain. Next, we will transform the waveforms from the time-domain signals into the time-frequency-domain signals by computing the short-time Fourier transform (STFT) to convert the waveforms to as spectrograms, which show frequency changes over time and can be represented as 2D images. we will feed the spectrogram images into your neural network to train the model

The spectogram is then normalized to feed into the network

ref:

1)https://khareanu1612.medium.com/audio-signal-processing-with-spectrograms-and-librosa-b66a0a6bc5cc

2)http://mirlab.org/jang/books/audiosignalprocessing/audioBasicFeature.asp?title=3-2%20Basic%20Acoustic%20Features%20(%B0%F2%A5%BB%C1n%BE%C7%AFS%BCx)#:~:text=Frame%20step%20(or%20hop%20size,divided%20by%20the%20frame%20step.

3)https://distill.pub/2017/ctc/


### Model building and training

#### 1)Model
This demonstration shows how to combine a 2D CNN, RNN and a Connectionist Temporal Classification (CTC) loss to build an ASR. CTC is an algorithm used to train deep neural networks in speech recognition, handwriting recognition and other sequence problems. CTC is used when we don’t know how the input aligns with the output (how the characters in the transcript align to the audio). The model we create is similar to DeepSpeech2.
The model can be downloaded from  https://keras.io/examples/audio/ctc_asr/


![alt text](https://raw.githubusercontent.com/vivekalex61/note_taking_bot/main/images/model.png)


#### Training

1)Convert audio to spectogram

2)Train model  


## Results

Below are the results  got from trained transformer.



![alt text](https://raw.githubusercontent.com/vivekalex61/resume_ner/main/images/)

![alt text](https://raw.githubusercontent.com/vivekalex61/resume_ner/main/images/)


## How to use.

`install  requirements.txt`

Run ` python3 predict.py`

It will automatically start recording and result will be saved in a text file

## End Notes

The predictions are not upto the mark. The main reason is because of little data.
It will  perform well if we have atleast 50 training epochs with different accents and with different noises .
