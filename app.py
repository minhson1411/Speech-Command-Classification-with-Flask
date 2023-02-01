import os

import torchaudio
import yaml
from flask import Flask, render_template, request

from models.inference import SpeechCommand

# Load config
base_dir = os.path.dirname(os.path.realpath(__file__))
config_path = base_dir + '/config/speech_command_config.yaml'
with open(config_path, "r") as stream:
    try:
        dataConfig = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Init speech command
sc = SpeechCommand(dataConfig)

app = Flask(__name__)
app.secret_key = "BrianGiang"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/audio_to_text/')
def audio_to_text():
    return render_template('audio_to_text.html')

@app.route('/audio', methods=['POST'])
def audio():
    with open('upload/audio.wav', 'wb') as f:
        f.write(request.data)

    waveform, sample_rate = torchaudio.load('upload/audio.wav')
    text = sc.predict_command(waveform)
    print('Predict text: ',text)
    return_text = " Did you say : <br> "
    try:
        for num, texts in enumerate(text):
            return_text += str(num+1) +") " + texts  + " <br> "
    except:
        return_text = " Sorry!!!! Voice not Detected "
        
    return str(return_text)


if __name__ == "__main__":
    host = dataConfig['server_config']['host']
    port = dataConfig['server_config']['port']
    app.run(host, port, debug=True)
