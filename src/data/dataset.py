import torch
import torchaudio
from torch.utils.data import Dataset
import os
import json

class NeuralProxyDataset(Dataset):
    def __init__(self, effect_name, dataset_dir, sample_rate=44100, duration=2.0):
        self.effect_name = effect_name.lower()
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)

        metadata_path = os.path.join(dataset_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
        )
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        output_path = os.path.join(self.dataset_dir, item['output_file'])
        target_audio, _ = torchaudio.load(output_path)
        
        if self.effect_name == "operator":
            input_audio = torch.zeros_like(target_audio)
        else:
            input_path = os.path.join(self.dataset_dir, item['input_file'])
            input_audio, _ = torchaudio.load(input_path)
        
        params = [setting["value"] for setting in item["settings"] if "value" in setting]

        params_tensor = torch.tensor(params, dtype=torch.float32)
        return input_audio, params_tensor, target_audio