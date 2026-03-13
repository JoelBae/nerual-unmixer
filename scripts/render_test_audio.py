import numpy as np
import soundfile as sf
import os
from src.data.pedalboard_engine import PedalboardEngine

def generate_sawtooth(freq, duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sawtooth wave [0, 1] then [-1, 1]
    saw = 2 * ( (t * freq) % 1 ) - 1
    return np.tile(saw, (2, 1)).astype(np.float32)

def main():
    print("Initializing engine...")
    engine = PedalboardEngine()
    sr = engine.sample_rate
    duration = 2.0
    
    # 1. Create source audio (Sawtooth)
    print("Generating source audio...")
    freq = 110.0 # A2
    source_audio = generate_sawtooth(freq, duration, sr)
    
    # 2. Render through KH Filter with a sweep
    print("Rendering through KH Filter (Cutoff sweep)...")
    # We'll simulate a sweep by rendering in chunks or just a static low-pass for a quick test
    # Actually, PedalboardEngine.render sets static parameters. 
    # For a real sweep we'd need a more complex render loop, but let's just do a nice filtered sound.
    filtered_audio = engine.render("kh_filter", {"cutoff": 0.2, "q": 0.7, "type": 0.0}, duration_sec=duration, input_audio=source_audio)
    
    # 3. Render through OTT
    print("Processing through OTT...")
    final_audio = engine.render("ott", {"depth": 0.5, "out_gain": 0.5}, duration_sec=duration, input_audio=filtered_audio)
    
    # 4. Save to file
    output_dir = "test_audio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "kh_filter_ott_test.wav")
    
    print(f"Saving to {output_path}...")
    # Pedalboard process returns (channels, samples), soundfile expect (samples, channels)
    sf.write(output_path, final_audio.T, sr)
    print("Done!")

if __name__ == "__main__":
    main()
