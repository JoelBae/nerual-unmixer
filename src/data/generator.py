import sys
import argparse
import time
import os
import json
import numpy as np

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pythonosc import udp_client
from src.data.ableton_client import AbletonClient

class ParameterRandomizer:
    """
    Handles randomization of parameters based on a defined schema.
    """
    def __init__(self):
        # Define Schema for Ableton Devices
        # Indices must match what verify_setup.py reports
        self.schema = [
            # 0. OPERATOR (Device Index 0)
            # Focused on Subtractive Synthesis + Pitch/Amp Envelopes
            {
                "track_index": 0, "device_index": 0,
                # Device 0 is an INSTRUMENT RACK.
                # Parameters are MACROS mapped to the internal synth.
                # We target safe ranges for these macros.
                "params": [
                    # Macro 1: Transpose (Pitch) [0-127 MIDI Range]
                    # Midpoint is 64 (0st). +/- 6 semitones = 58-70.
                    {"index": 1, "name": "Transpose", "min": 58, "max": 70}, 
                    
                    # Macro 2: Osc-A Wave (Timbre) [0-127 MIDI Range]
                    # Safe to randomize fully.
                    {"index": 2, "name": "Osc-A Wave", "min": 0, "max": 127},
                    
                    # Macro 3: Filter Freq (Brightness) [0-127 MIDI Range]
                    # Min 20 (~300Hz) to ensure valid tone.
                    {"index": 3, "name": "Filter Freq", "min": 20, "max": 127},
                    
                    # Macro 4: Filter Res [0-127 MIDI Range]
                    # Max 90 to avoid extreme resonance spike.
                    {"index": 4, "name": "Filter Res", "min": 0, "max": 90},
                    
                    # Macro 5: Fe Amount (Filter Env) [0-127 MIDI Range]
                    # User Request: Don't set to -100 (which is 0 in bipolar MIDI).
                    # -50 corresponds to roughly 32 on 0-127 scale (assuming 64 is 0).
                    # Wait, if 0=-100 & 127=+100, then -50 is at 25% = 32.
                    # If 0=0 & 127=+100, then 0 is fine. Assuming Bipolar (-100 to +100).
                    {"index": 5, "name": "Fe Amount", "min": 32, "max": 127}, 
                    
                    # Macro 6-9: Filter Envelope ADSR [0-127 MIDI Range]
                    {"index": 6, "name": "Fe Attack", "min": 0, "max": 64}, 
                    {"index": 7, "name": "Fe Decay", "min": 10, "max": 127},
                    {"index": 8, "name": "Fe Sustain", "min": 0, "max": 127},
                    {"index": 9, "name": "Fe Release", "min": 10, "max": 127}, 
                    
                    # Macro 10-12: Pitch Envelope [0-127 MIDI Range]
                    # 64 max keeps modulation subtle (0 to +50% or -100 to 0 depending on map)
                    {"index": 10, "name": "Pe Amount", "min": 0, "max": 64}, 
                    {"index": 11, "name": "Pe Decay", "min": 10, "max": 100},
                    {"index": 12, "name": "Pe Peak", "min": 0, "max": 64},
                    
                    # Macro 13-16: Amp Envelope ADSR (CRITICAL) [0-127 MIDI Range]
                    # Decay OR Sustain must > 0. Enforce Decay min 30 (approx 200ms).
                    {"index": 13, "name": "Ae Attack", "min": 0, "max": 40},
                    {"index": 14, "name": "Ae Decay", "min": 30, "max": 127}, 
                    {"index": 15, "name": "Ae Sustain", "min": 0, "max": 127}, 
                    {"index": 16, "name": "Ae Release", "min": 15, "max": 127}, 
                ]
            },
            # 1. SATURATOR (Device Index 1)
            {
                "track_index": 0, "device_index": 1,
                "params": [
                    {"index": 1, "name": "Drive", "min": 0.0, "max": 1.0},
                    # Output (Index 10) removed to keep fixed at 0dB (User Request)
                    # Type (Index 3) is an Enum (0-6)
                    # 0=Analog, 1=Soft, 2=Med, 3=Hard, 4=Sinoid, 5=Digi, 6=Wave
                    {"index": 3, "name": "Type", "min": 0.0, "max": 6.0}, 
                    {"index": 15, "name": "WS Curve", "min": 0.0, "max": 1.0},
                    {"index": 18, "name": "WS Depth", "min": 0.0, "max": 1.0}
                ]
            },
            # 2. EQ EIGHT (Device Index 2)
            {
                "track_index": 0, "device_index": 2,
                "params": [
                    # Band 1 (Low Shelf/Cut)
                    {"index": 6, "name": "Bond 1 Freq", "min": 0.0, "max": 0.5},
                    {"index": 7, "name": "Band 1 Gain", "min": -15.0, "max": 15.0}, # Full +/- 15dB
                    
                    # Band 2 (Bell/Parametric)
                    {"index": 16, "name": "Band 2 Freq", "min": 0.1, "max": 0.8},
                    {"index": 17, "name": "Band 2 Gain", "min": -15.0, "max": 15.0}, # Full +/- 15dB
                    {"index": 18, "name": "Band 2 Q", "min": 0.1, "max": 0.9},

                    # Band 3 (Bell/Parametric)
                    {"index": 26, "name": "Band 3 Freq", "min": 0.3, "max": 0.9},
                    {"index": 27, "name": "Band 3 Gain", "min": -15.0, "max": 15.0}, # Full +/- 15dB
                    {"index": 28, "name": "Band 3 Q", "min": 0.1, "max": 0.9},

                    # Band 4 (High Shelf/Cut)
                    {"index": 36, "name": "Band 4 Freq", "min": 0.5, "max": 1.0},
                    {"index": 37, "name": "Band 4 Gain", "min": -15.0, "max": 15.0}, # Full +/- 15dB
                ]
            },
            # 3. OTT (Device Index 3) - Multiband Dynamics
            {
                "track_index": 0, "device_index": 3,
                "params": [
                    {"index": 6, "name": "Amount", "min": 0.0, "max": 1.0}, # Dry/Wet
                    
                    # Thresholds (Raw dB: -60 to 0)
                    {"index": 17, "name": "Abv Thresh L", "min": -60.0, "max": 0.0},
                    {"index": 18, "name": "Abv Thresh M", "min": -60.0, "max": 0.0},
                    {"index": 19, "name": "Abv Thresh H", "min": -60.0, "max": 0.0},
                    
                    {"index": 20, "name": "Blw Thresh L", "min": -60.0, "max": 0.0},
                    {"index": 21, "name": "Blw Thresh M", "min": -60.0, "max": 0.0},
                    {"index": 22, "name": "Blw Thresh H", "min": -60.0, "max": 0.0},
                ]
            },
             # 4. PHASER (Device Index 4)
            {
                "track_index": 0, "device_index": 4,
                "params": [
                    {"index": 3, "name": "Frequency", "min": 0.0, "max": 1.0},
                    {"index": 25, "name": "Feedback", "min": 0.0, "max": 0.95}, 
                    {"index": 1, "name": "Amount", "min": 0.0, "max": 1.0}
                ]
            },
            # 5. REVERB (Device Index 5)
            {
                "track_index": 0, "device_index": 5,
                "params": [
                    # Decay Time appears to be RAW Seconds (scaling non-linearly)
                    # Safe range: 0.3s (room) to 6.0s (hall)
                    {"index": 20, "name": "Decay Time", "min": 0.3, "max": 6.0}, 
                    {"index": 26, "name": "Size", "min": 0.0, "max": 1.0}, 
                    {"index": 32, "name": "Dry/Wet", "min": 0.0, "max": 0.5} 
                ]
            }
        ]
        
    def randomize(self, target_device_index=None):
        settings_log = []
        flat_params = []
        
        for device in self.schema:
            t_idx = device["track_index"]
            d_idx = device["device_index"]
            
            # If focusing on a single effect (e.g., Saturator), 
            # we MUST randomize the generator (Operator - d_idx 0) AND the target effect.
            # We skip other random effects to keep the signal chain pure.
            if target_device_index is not None:
                if d_idx != target_device_index and d_idx != 0: # Device 0 is the instrument
                    continue
            
            for param in device["params"]:
                val = np.random.uniform(param["min"], param["max"])
                flat_params.append(val)
                
                settings_log.append({
                    "track": t_idx,
                    "device": d_idx,
                    "param": param["index"],
                    "value": val,
                    "name": param["name"]
                })
                
        return flat_params, settings_log

    def get_device_index_by_name(self, name):
        name_map = {
            "saturator": 1,
            "eq8": 2,
            "ott": 3,
            "phaser": 4,
            "reverb": 5
        }
        return name_map.get(name.lower())

def record_audio(duration, sample_rate=44100):
    """
    Records audio from the default input device.
    Moved heavy imports inside to allow verification script to run light.
    """
    import sounddevice as sd
    
    # print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, blocking=True)
    return recording

def generate_dataset(num_samples: int, output_dir: str, duration: float = 2.0, sample_rate: int = 44100, effect_filter: str = None):
    """
    Main loop to generate the dataset.
    effect_filter: Name of the effect to collect data for (e.g., 'saturator').
    """
    # Import here
    import sounddevice as sd
    import soundfile as sf

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    client = AbletonClient()
    randomizer = ParameterRandomizer()
    
    # Check for Single Effect Mode
    target_d_idx = None
    if effect_filter:
        if effect_filter.lower() == "operator":
            target_d_idx = 0
            print(f"--- Single Effect Mode: Operator (Device 0) ---")
            print("--- Bypassing ALL effects (Device 1-5) ---")
            # Enable Operator (0), Disable ALL others (1-5)
            # Note: Device 0 (Rack) usually stays enabled, we just toggle its children or bypass the chain.
            # Here we assume we just bypass downstream effects.
            for d_idx in range(1, 6):
                # Track 0, Device d_idx, Param 0 (Device On), Value 0.0 (Off)
                client.set_track_parameter(0, d_idx, 0, 0.0)
            time.sleep(0.05)
        else:
            target_d_idx = randomizer.get_device_index_by_name(effect_filter)
            if target_d_idx is None:
                print(f"Error: Effect '{effect_filter}' not found in schema map.")
                return
            print(f"--- Single Effect Mode: {effect_filter} (Device {target_d_idx}) ---")
            
            # Setup: Enable Target, Disable Others (Best Effort)
            # We use Parameter 0 ("Device On") to toggle, as set_device_enabled might fail
            for d_idx in range(1, 6):
                is_target = (d_idx == target_d_idx)
                val = 1.0 if is_target else 0.0
                # Track 0, Device d_idx, Param 0 (Device On), Value
                client.set_track_parameter(0, d_idx, 0, val)
                time.sleep(0.05)
    
    
    # Check for existing samples to RESUME
    start_index = 0
    if os.path.exists(output_dir):
        existing_files = [f for f in os.listdir(output_dir) if f.startswith("output_") and f.endswith(".wav")]
        if existing_files:
            indices = []
            for f in existing_files:
                try:
                    # Extract index from 'output_00123.wav'
                    idx_str = f.split('_')[1].split('.')[0]
                    indices.append(int(idx_str))
                except (IndexError, ValueError):
                    pass
            
            if indices:
                start_index = max(indices) + 1
                print(f"--- Found {len(indices)} existing samples. Resuming from index {start_index} ---")

    if start_index >= num_samples:
        print(f"Target count {num_samples} already reached (current max index {start_index-1}). Skipping generation.")
        return

    print(f"Starting data generation for {num_samples - start_index} samples...")
    print(f"Saving to {output_dir}")
    
    metadata = []
    
    for i in range(start_index, num_samples):
        print(f"--- Generating Sample {i+1}/{num_samples} ---")
        
        # 1. Randomize Parameters
        flat_params, settings_log = randomizer.randomize(target_device_index=target_d_idx)
        
        # 2. Send to Ableton
        for setting in settings_log:
            client.set_track_parameter(
                setting["track"], 
                setting["device"], 
                setting["param"], 
                setting["value"]
            )
        
        time.sleep(0.1)

        # 3. Record INPUT (Dry)
        # Only needed if we are in Single Effect Mode and NOT in Operator Mode
        # (Operator Mode is already dry by definition)
        input_data = None
        if effect_filter and effect_filter.lower() != "operator" and target_d_idx is not None:
             print("   > Recording Input (Dry)...")
             # Bypass Target Effect
             # client.set_device_enabled(track_index=0, device_index=target_d_idx, enabled=False)
             client.set_track_parameter(0, target_d_idx, 0, 0.0) # Param 0, Value 0.0
             time.sleep(0.05)
             
             # Trigger Transport
             client.play()
             input_data = record_audio(duration, sample_rate)
             client.stop()
             
             settings_log.append({"input_dry": True})
             
             # Save Input
             filename_input = f"input_{i:05d}.wav"
             filepath_input = os.path.join(output_dir, filename_input)
             sf.write(filepath_input, input_data, sample_rate)
             
             # Re-enable Target Effect
             # client.set_device_enabled(track_index=0, device_index=target_d_idx, enabled=True)
             client.set_track_parameter(0, target_d_idx, 0, 1.0) # Param 0, Value 1.0
             time.sleep(0.05)

        # 4. Record OUTPUT (Wet)
        print("   > Recording Output (Wet)...")
        # Trigger Transport Again
        client.play()
        output_data = record_audio(duration, sample_rate)
        client.stop()
        
        # Save Output
        filename_output = f"output_{i:05d}.wav"
        filepath_output = os.path.join(output_dir, filename_output)
        sf.write(filepath_output, output_data, sample_rate)
        
        # Log metadata for *both* input and output filepaths
        meta = {
            "index": i,
            "settings": settings_log,
            "input_file": filename_input if effect_filter and effect_filter.lower() != "operator" and target_d_idx is not None else None,
            "output_file": filename_output,
            "effect_type": effect_filter if effect_filter else "full_chain"
        }
        metadata.append(meta)
        
    # Save Metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        
    print("Data generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Audio Dataset via Ableton")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="Output directory")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    parser.add_argument("--effect", type=str, default=None, help="Specific effect to train on (e.g., 'saturator', 'eq8')")

    args = parser.parse_args()
    
    # If effect is specified, append it to output dir
    if args.effect:
        args.output_dir = os.path.join(args.output_dir, args.effect)

    generate_dataset(args.num_samples, args.output_dir, args.duration, effect_filter=args.effect)
