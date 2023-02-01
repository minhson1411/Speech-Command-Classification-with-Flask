from models.model import M5
from models.utils import get_likely_index,index_to_label
import torch
import torchaudio


class SpeechCommand():
    def __init__(self, config) -> None:
        #Origin sample rate
        sample_rate = config['ai_model_config']['sample_rate']
        #New sample rate
        new_sample_rate = config['ai_model_config']['new_sample_rate']
        #Check device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Transformation
        self.transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        #Number of command labels
        with open(config['ai_model_config']['label_path'], 'r') as f:
            self.labels = f.readlines()
            self.labels = [label.replace('\n', '') for label in self.labels]

        #Load model
        self.model = M5(n_input=1, n_output=len(self.labels))
        self.model.load_state_dict(torch.load(config['ai_model_config']['checkpoint_path'], map_location=torch.device(self.device)))
        self.model.to(self.device)

    def predict_command(self, tensor):
        # Use the model to predict the label of the waveform
        tensor = tensor.to(self.device)
        tensor = self.transform(tensor)
        tensor = self.model(tensor.unsqueeze(0))
        tensor = get_likely_index(tensor)
        command = [index_to_label(index.squeeze(), self.labels) for index in tensor]
        return command


