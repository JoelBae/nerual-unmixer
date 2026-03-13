import numpy as np
import soundfile as sf
import os
from src.data.pedalboard_engine import PedalboardEngine

def main():
    print("Initializing engine for Vital test...")
    engine = PedalboardEngine()
    sr = engine.sample_rate
    duration = 2.0
    
    print("Rendering through Vital (C4 note)...")
    # Using 261.63 Hz for C4
    vital_audio = engine.render("vital", {}, duration_sec=duration, note_freq=261.63)
    
    output_dir = "test_audio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vital_test.wav")
    
    print(f"Saving to {output_path}...")
    sf.write(output_path, vital_audio.T, sr)
    print("Done!")

if __name__ == "__main__":
    main()
