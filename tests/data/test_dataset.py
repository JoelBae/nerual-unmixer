import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.dataset import NeuralProxyDataset

if __name__ == '__main__':
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../dataset/operator'))
    
    if os.path.exists(dataset_dir):
        print(f"Loading proxy dataset from {dataset_dir}")
        dataset = NeuralProxyDataset(effect_name="operator", dataset_dir=dataset_dir)
        print(f"Dataset Size: {len(dataset)}")
        
        if len(dataset) > 0:
            input_audio, params, target_audio = dataset[0]
            print(f"Input Audio Shape: {input_audio.shape}")
            print(f"Target Audio Shape: {target_audio.shape}")
            print(f"Parameters Shape: {params.shape}")
            print("First item loaded successfully.")
    else:
        print(f"Cannot test dataset: {dataset_dir} does not exist.")
