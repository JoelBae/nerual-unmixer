import torch
import torchaudio
from torch.utils.data import Dataset
import os
import json
import soundfile as sf
from tqdm import tqdm

class NeuralProxyDataset(Dataset):
    def __init__(self, effect_name, dataset_dir, sample_rate=44100, duration=2.0, split="train", preload=True):
        self.effect_name = effect_name.lower()
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.preload = preload

        metadata_path = os.path.join(dataset_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            full_metadata = json.load(f)
            
        # 80/20 Train/Val Split
        split_idx = int(len(full_metadata) * 0.8)
        if split == "train":
            self.metadata = full_metadata[:split_idx]
        elif split == "val":
            self.metadata = full_metadata[split_idx:]
        else:
            self.metadata = full_metadata
            
        self.data_cache = []
        if self.preload:
            print(f"--- Pre-loading {split} dataset into RAM ---")
            for item in tqdm(self.metadata, desc=f"Loading {split}"):
                input_audio, params, target_audio = self._load_item(item)
                self.data_cache.append((input_audio, params, target_audio))

    def _load_item(self, item):
        output_path = os.path.join(self.dataset_dir, item['output_file'])
        target_audio = self._load_audio(output_path)
        
        if self.effect_name == "operator":
            input_audio = torch.zeros_like(target_audio)
        else:
            input_path = os.path.join(self.dataset_dir, item['input_file'])
            input_audio = self._load_audio(input_path)
        
        params = [setting["value"] for setting in item["settings"] if "value" in setting]
        params_tensor = torch.tensor(params, dtype=torch.float32)
        
        return input_audio, params_tensor, target_audio

    def __len__(self):
        return len(self.metadata)

    def _load_audio(self, path):
        audio_array, sr = sf.read(path, dtype='float32')
        if audio_array.ndim == 1:
            tensor = torch.from_numpy(audio_array).unsqueeze(0)
        else:
            tensor = torch.from_numpy(audio_array).t()
        return tensor

    def __getitem__(self, idx):
        if self.preload:
            return self.data_cache[idx]
            
        item = self.metadata[idx]
        return self._load_item(item)