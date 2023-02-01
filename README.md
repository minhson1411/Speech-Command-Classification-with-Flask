# Speech Command Classification

### ([Flask](https://github.com/pallets/flask) + SpeechCommandClassification+ [Recorderjs](https://github.com/mattdiamond/Recorderjs))

1) Flask is a micro web framework written in Python.
2) SpeechCommandClassification Your own speech command classification model, with support for converting speech command to text.
3) Recorderjs A plugin for recording/exporting the output of Web Audio API nodes 

## Project stucture
```
root
    |speech_command_classification.ipynb: Notebook for model experiments 
    |config: config for flask server and AI model
    |models: model architecture/checkpoints and inference code
    |static: js and css files for frontend
    |templates: html files for frontend
    |app.py: Flask app
    |Dockerfile
    |requirements.txt
```
## Detail about the experiments
[This notebook](https://github.com/gnvml/Speech-Command-Classification-with-Flask/blob/master/speech_command_classification.ipynb) implemented speech command recognition using convolutional neural networks trained on the Google SpeechCommand dataset.

M5, M18 layer architecture convolutional neural networks were implemented and the networks were trained on the time domain waveform inputs of the SpeechCommand dataset.

The dataset consists of 105 000 .wav files that was split into a training (105829 files) and testing (11005 files) datasets.

The files were pre-processed by resampling each to 8kHz which results in smaller files to enable faster processing.

An Adam optimiser was used, with weight decay set to 0.0001. Initial learning rate was set at 0.01, and a scheduler was used to decrease learning reate to 0.001 during training after 20 epochs

Networks were trained with increased 30 and 50 epochs with incremental improvements. The results and graphs are shown above in TensorBoard

*   30 Epoch network accuracy:  M5: 92%, M18: 95%
*   50 Epoch network accuracy: M5: 94%, M18: 98%

### Access [live notebook](https://colab.research.google.com/drive/1V-HQeaBzyZu7nyJNo7uTAdE7i3_w59-T?usp=sharing) result for more details

![LossAndAccuracy](https://github.com/gnvml/Speech-Command-Classification-with-Flask/blob/master/experiment/loss_acc.png)


### Build docker and run demo

```
cd path-to-this-project
docker build -t speech_command .
docker run --rm -p4999:5000 -v$PWD:/code speech_command
```
*Note: If run fail with docker, change [config](https://github.com/gnvml/Speech-Command-Classification-with-Flask/blob/master/config/speech_command_config.yaml) project with another port*

Access: http://localhost:4999/ to open demo website

![Homepage](https://github.com/gnvml/Speech-Command-Classification-with-Flask/blob/master/experiment/home.png)
![Demo](https://github.com/gnvml/Speech-Command-Classification-with-Flask/blob/master/experiment/demo.png)



## Further work
The challenges of different datasets recorded with different microphones/sample rates/files formats/speech lengths can be explored.

Preprocessing and transforming the audio into MFCC or mel-spectrogram arrays can be performed to investigate speed of training differences, alternative architectures and recognition accuracy. 

Audio data augmentation techniques (such as adding noise) to improve recognition robustness can be investigated.

Alternative architectures such as LSTM or RNN networks that encode memory can be investigated as the likehood of a current phoneme being pronounced is affected by previous phonemes that were present in the word structure.