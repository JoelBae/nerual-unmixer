import torch
import torchaudio
from torch.utils.data import Dataset
import os
import json
import soundfile as sf
from tqdm import tqdm
from src.data.augment import AudioAugmentor

class NeuralProxyDataset(Dataset):
    def __init__(self, effect_name, dataset_dir, sample_rate=44100, duration=2.0, split="train", preload=True, augment=False):
        self.effect_name = effect_name.lower()
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.preload = preload
        self.augment = augment
        self.augmentor = AudioAugmentor(sample_rate=sample_rate) if augment else None

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
                self.data_cache.append(self._load_item(item))

    def _load_item(self, item):
        output_path = os.path.join(self.dataset_dir, item['output_file'])
        target_audio = self._load_audio(output_path)
        
        if self.effect_name == "operator":
            input_audio = torch.zeros_like(target_audio)
        elif 'input_file' in item and item['input_file'] is not None:
            input_path = os.path.join(self.dataset_dir, item['input_file'])
            input_audio = self._load_audio(input_path)
        else:
            # Fallback for older full_chain datasets that didn't save dry input
            # In this case, dry input is the same as the target audio before this effect
            # which for the full chain is the Operator's output. We can't know that here.
            # So, we'll return zeros and the chainer will need to handle it.
            input_audio = torch.zeros_like(target_audio)

        params = [setting["value"] for setting in item["settings"] if "value" in setting]
        params_tensor = torch.tensor(params, dtype=torch.float32)
        
        # Get order index for permutation prediction, default to 0 if not present
        order_idx = torch.tensor(item.get('order_idx', 0), dtype=torch.long)
        
        return input_audio, params_tensor, target_audio, order_idx

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
            input_audio, params_tensor, target_audio, order_idx = self.data_cache[idx]
        else:
            item = self.metadata[idx]
            input_audio, params_tensor, target_audio, order_idx = self._load_item(item)
            
        if self.augment and self.augmentor is not None:
            target_audio = self.augmentor(target_audio)
            
        return input_audio, params_tensor, target_audio, order_idx