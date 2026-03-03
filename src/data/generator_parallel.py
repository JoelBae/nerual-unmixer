"""
Parallel Data Generator — 4x Speed via BlackHole 16ch

Requires Ableton setup with 8 tracks:
  Track 0: Operator A (dry)     → Output: BlackHole 16ch (Ch 1-2)
  Track 1: Operator A + Effect  → Output: BlackHole 16ch (Ch 3-4)
  Track 2: Operator B (dry)     → Output: BlackHole 16ch (Ch 5-6)
  Track 3: Operator B + Effect  → Output: BlackHole 16ch (Ch 7-8)
  Track 4: Operator C (dry)     → Output: BlackHole 16ch (Ch 9-10)
  Track 5: Operator C + Effect  → Output: BlackHole 16ch (Ch 11-12)
  Track 6: Operator D (dry)     → Output: BlackHole 16ch (Ch 13-14)
  Track 7: Operator D + Effect  → Output: BlackHole 16ch (Ch 15-16)

Each dry track has: Operator (device 0) — no effects
Each wet track has: Operator (device 0) + target effect (device 1)

All 8 tracks receive the same MIDI clip, so they all play simultaneously.
The 4 Operators are randomized independently, giving 4 unique samples per play cycle.

Usage:
  PYTHONPATH=. python src/data/generator_parallel.py --effect ott --num_samples 3000
"""
import sys
import argparse
import time
import os
import json
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.ableton_client import AbletonClient

# === PARAMETER SCHEMAS ===
# Each lane has Operator on device 0 of both dry and wet tracks.
# The effect is on device 1 of the wet track only.

OPERATOR_PARAMS = [
    {"index": 1,  "name": "Transpose",   "min": 58,  "max": 70},
    {"index": 2,  "name": "Osc-A Wave",  "min": 0,   "max": 127},
    {"index": 3,  "name": "Filter Freq", "min": 20,  "max": 127},
    {"index": 4,  "name": "Filter Res",  "min": 0,   "max": 90},
    {"index": 5,  "name": "Fe Amount",   "min": 32,  "max": 127},
    {"index": 6,  "name": "Fe Attack",   "min": 0,   "max": 64},
    {"index": 7,  "name": "Fe Decay",    "min": 10,  "max": 127},
    {"index": 8,  "name": "Fe Sustain",  "min": 0,   "max": 127},
    {"index": 9,  "name": "Fe Release",  "min": 10,  "max": 127},
    {"index": 10, "name": "Pe Amount",   "min": 0,   "max": 64},
    {"index": 11, "name": "Pe Decay",    "min": 10,  "max": 100},
    {"index": 12, "name": "Pe Peak",     "min": 0,   "max": 64},
    {"index": 13, "name": "Ae Attack",   "min": 0,   "max": 40},
    {"index": 14, "name": "Ae Decay",    "min": 30,  "max": 127},
    {"index": 15, "name": "Ae Sustain",  "min": 0,   "max": 127},
    {"index": 16, "name": "Ae Release",  "min": 15,  "max": 127},
]

EFFECT_SCHEMAS = {
    "ott": [
        {"index": 6,  "name": "Amount",       "min": 0.5,   "max": 1.0,  "scale": "linear"},
        {"index": 17, "name": "Abv Thresh L", "min": -60.0, "max": 0.0,  "scale": "linear"},
        {"index": 18, "name": "Abv Thresh M", "min": -60.0, "max": 0.0,  "scale": "linear"},
        {"index": 19, "name": "Abv Thresh H", "min": -60.0, "max": 0.0,  "scale": "linear"},
        {"index": 20, "name": "Blw Thresh L", "min": -60.0, "max": 0.0,  "scale": "linear"},
        {"index": 21, "name": "Blw Thresh M", "min": -60.0, "max": 0.0,  "scale": "linear"},
        {"index": 22, "name": "Blw Thresh H", "min": -60.0, "max": 0.0,  "scale": "linear"},
    ],
    "saturator": [
        {"index": 1,  "name": "Drive",    "min": 0.0, "max": 1.0,  "scale": "linear"},
        {"index": 3,  "name": "Type",     "min": 0.0, "max": 6.0,  "scale": "discrete"},
        {"index": 15, "name": "WS Curve", "min": 0.0, "max": 1.0,  "scale": "linear"},
        {"index": 18, "name": "WS Depth", "min": 0.0, "max": 1.0,  "scale": "linear"},
    ],
    "reverb": [
        {"index": 20, "name": "Decay Time", "min": 0.05, "max": 0.4,  "scale": "linear"},
        {"index": 26, "name": "Size",       "min": 0.0,  "max": 1.0,  "scale": "linear"},
        {"index": 32, "name": "Dry/Wet",    "min": 0.0,  "max": 1.0,  "scale": "linear"},
    ],
    "eq8": [
        # EQ8 values in AbletonOSC behavior for this user:
        # Categorical (Type) and Gain often require RAW values.
        # Band 1
        {"index": 6, "name": "Band 1 Freq", "min": 20.0, "max": 2000.0, "scale": "log"}, 
        {"index": 7, "name": "Band 1 Gain", "min": -15.0, "max": 15.0, "scale": "raw"},
        {"index": 8, "name": "Band 1 Q",    "min": 0.1,  "max": 18.0, "scale": "linear"},
        {"index": 5, "name": "Band 1 Type", "min": 3,    "max": 3,    "scale": "raw"},
        
        # Band 2-7: Locked to Bell (3)
        {"index": 16, "name": "Band 2 Freq", "min": 100.0, "max": 5000.0, "scale": "log"},
        {"index": 17, "name": "Band 2 Gain", "min": -15.0, "max": 15.0, "scale": "raw"},
        {"index": 18, "name": "Band 2 Q",    "min": 0.1,  "max": 18.0, "scale": "linear"},
        {"index": 15, "name": "Band 2 Type", "min": 3,    "max": 3,    "scale": "raw"},

        {"index": 26, "name": "Band 3 Freq", "min": 200.0, "max": 8000.0, "scale": "log"},
        {"index": 27, "name": "Band 3 Gain", "min": -15.0, "max": 15.0, "scale": "raw"},
        {"index": 28, "name": "Band 3 Q",    "min": 0.1,  "max": 18.0, "scale": "linear"},
        {"index": 25, "name": "Band 3 Type", "min": 3,    "max": 3,    "scale": "raw"},

        {"index": 36, "name": "Band 4 Freq", "min": 500.0, "max": 15000.0, "scale": "log"},
        {"index": 37, "name": "Band 4 Gain", "min": -15.0, "max": 15.0, "scale": "raw"},
        {"index": 38, "name": "Band 4 Q",    "min": 0.1,  "max": 18.0, "scale": "linear"},
        {"index": 35, "name": "Band 4 Type", "min": 3,    "max": 3,    "scale": "raw"},

        {"index": 46, "name": "Band 5 Freq", "min": 500.0, "max": 15000.0, "scale": "log"},
        {"index": 47, "name": "Band 5 Gain", "min": -15.0, "max": 15.0, "scale": "raw"},
        {"index": 48, "name": "Band 5 Q",    "min": 0.1,  "max": 18.0, "scale": "linear"},
        {"index": 45, "name": "Band 5 Type", "min": 3,    "max": 3,    "scale": "raw"},

        {"index": 56, "name": "Band 6 Freq", "min": 500.0, "max": 15000.0, "scale": "log"},
        {"index": 57, "name": "Band 6 Gain", "min": -15.0, "max": 15.0, "scale": "raw"},
        {"index": 58, "name": "Band 6 Q",    "min": 0.1,  "max": 18.0, "scale": "linear"},
        {"index": 55, "name": "Band 6 Type", "min": 3,    "max": 3,    "scale": "raw"},

        {"index": 66, "name": "Band 7 Freq", "min": 500.0, "max": 15000.0, "scale": "log"},
        {"index": 67, "name": "Band 7 Gain", "min": -15.0, "max": 15.0, "scale": "raw"},
        {"index": 68, "name": "Band 7 Q",    "min": 0.1,  "max": 18.0, "scale": "linear"},
        {"index": 65, "name": "Band 7 Type", "min": 3,    "max": 3,    "scale": "raw"},

        # Band 8
        {"index": 76, "name": "Band 8 Freq", "min": 1000.0, "max": 20000.0, "scale": "log"},
        {"index": 77, "name": "Band 8 Gain", "min": -15.0, "max": 15.0, "scale": "raw"},
        {"index": 78, "name": "Band 8 Q",    "min": 0.1,  "max": 18.0, "scale": "linear"},
        {"index": 75, "name": "Band 8 Type", "min": 3,    "max": 3,    "scale": "raw"},
    ],
}

def scale_to_normalized(val, param):
    """
    Scale a raw value to 0.0-1.0 based on the parameter's range and scale type.
    Note: Ableton instruments/effects almost always expect 0-1 via OSC.
    However, some setups expect RAW values for specific parameters.
    """
    scale = param.get("scale", "linear")
    
    if scale == "raw":
        return float(val)
        
    # If device_min/max are provided, use them for absolute normalization
    # otherwise fall back to randomization min/max.
    p_min = param.get("device_min", param.get("min", 0.0))
    p_max = param.get("device_max", param.get("max", 1.0))
    
    if scale == "linear":
        return (val - p_min) / (p_max - p_min + 1e-8)
    elif scale == "log":
        # Log scaling for frequency: log(val/min) / log(max/min)
        # For log, we usually use the randomization min/max as the base
        r_min = param.get("min", 20.0)
        r_max = param.get("max", 20000.0)
        return np.log10(val / r_min) / (np.log10(r_max / r_min) + 1e-8)
    elif scale == "discrete":
        # Discrete values: normalize absolute value within the device range
        val_rounded = np.round(val)
        return (val_rounded - p_min) / (p_max - p_min + 1e-8)
    return val


def detect_blackhole(min_channels=4):
    """Auto-detect BlackHole and return (device_index, max_channels)."""
    import sounddevice as sd
    devices = sd.query_devices()
    
    # Prefer highest channel count
    candidates = []
    for i, dev in enumerate(devices):
        if "BlackHole" in dev["name"] and dev["max_input_channels"] >= min_channels:
            candidates.append((i, dev["max_input_channels"], dev["name"]))
    
    if not candidates:
        print("❌ No BlackHole device found with enough channels!")
        print("   Install from https://existential.audio/blackhole/")
        print("   Available input devices:")
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                print(f"     [{i}] {dev['name']} ({dev['max_input_channels']}ch)")
        sys.exit(1)
    
    best = max(candidates, key=lambda x: x[1])
    return best[0], best[1], best[2]


def record_multi_ch(num_channels, duration, sample_rate, device_idx):
    """Record N channels from BlackHole simultaneously."""
    import sounddevice as sd
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=num_channels,
        blocking=True,
        device=device_idx
    )
    return recording


def randomize_lane(client, lane_idx, effect_name, lane_tracks):
    """
    Randomize Operator + Effect parameters for one lane.
    Sets identical Operator params on both dry and wet tracks.
    Sets effect params on the wet track only.
    Returns (flat_params, settings_log) for metadata.
    """
    dry_track, wet_track = lane_tracks[lane_idx]
    settings_log = []
    flat_params = []
    
    # 1. Randomize Operator params (skip for EQ8 as requested)
    if effect_name.lower() != "eq8":
        for param in OPERATOR_PARAMS:
            val = np.random.uniform(param["min"], param["max"])
            flat_params.append(val)
            
            # Normalize to 0-1 for Ableton Macros (as they map to internal ranges)
            norm_val = val / 127.0
            
            # Set on dry track (device 0)
            client.set_track_parameter(dry_track, 0, param["index"], norm_val)
            time.sleep(0.002) # Tiny throttle
            # Set on wet track (device 0) — must match!
            client.set_track_parameter(wet_track, 0, param["index"], norm_val)
            time.sleep(0.002)
            
            settings_log.append({
                "track": dry_track,
                "device": 0,
                "param": param["index"],
                "raw_value": val,
                "norm_value": norm_val,
                "name": param["name"]
            })
    
    # 2. Randomize Effect params (wet track only, device 1)
    effect_params = EFFECT_SCHEMAS.get(effect_name.lower(), [])
    for param in effect_params:
        if param.get("scale") == "log":
             val = 10**np.random.uniform(np.log10(param["min"]), np.log10(param["max"]))
        else:
             val = np.random.uniform(param["min"], param["max"])
        flat_params.append(val)
        
        # Normalize for Ableton API
        norm_val = scale_to_normalized(val, param)
        
        client.set_track_parameter(wet_track, 1, param["index"], norm_val)
        time.sleep(0.002) # Tiny throttle
        
        settings_log.append({
            "track": wet_track,
            "device": 1,
            "param": param["index"],
            "raw_value": val,
            "norm_value": norm_val,
            "name": param["name"]
        })
    
    return flat_params, settings_log


def generate_parallel(num_samples, output_dir, effect_name, duration=2.0, sample_rate=44100, num_lanes=4):
    """Main parallel generation loop."""
    import soundfile as sf
    
    os.makedirs(output_dir, exist_ok=True)
    client = AbletonClient()
    
    print("--- Checking OSC Connection ---")
    if client.ping():
        print("✅ Ableton is responsive (Bidirectional).")
    else:
        print("⚠️  Warning: Ableton did NOT respond to handshake (11001).")
        print("   Proceeding in SEND-ONLY mode. Ensure your faders move!")

    actual_tracks = client.get_track_count()
    required_tracks = num_lanes * 2
    if actual_tracks is not None:
        print(f"✅ Found {actual_tracks} tracks in Ableton.")
        if actual_tracks < required_tracks:
            print(f"❌ ERROR: Parallel generator ({num_lanes} lanes) requires {required_tracks} tracks. You only have {actual_tracks}.")
            print("   Please add enough tracks to your session.")
            sys.exit(1)
    else:
        print("⚠️  Warning: Could not verify track count. Proceeding with caution...")
    
    # Auto-detect BlackHole
    required_channels = num_lanes * 4
    bh_idx, bh_channels, bh_name = detect_blackhole(min_channels=required_channels)
    print(f"🎧 Using {bh_name} ({bh_channels}ch) — device [{bh_idx}]")
    
    max_lanes = bh_channels // 4
    if num_lanes > max_lanes:
        print(f"⚠️  Requested {num_lanes} lanes but {bh_name} only supports {max_lanes}. Using {max_lanes}.")
        num_lanes = max_lanes
    
    # Build track layout for this lane count
    lane_tracks = [(i * 2, i * 2 + 1) for i in range(num_lanes)]
    
    # Ensure all EQ8 bands are ON for all wet tracks
    if effect_name.lower() == "eq8":
        print("🔧 Setting up EQ8: Turning on all 8 bands for all wet tracks...")
        for _, wet_track in lane_tracks:
            for band_idx in range(8):
                # Filter On A: 4, 14, 24, 34, 44, 54, 64, 74
                on_index = 4 + (band_idx * 10)
                client.set_track_parameter(wet_track, 1, on_index, 1.0)
            time.sleep(0.05) # Small throttle per track
        print("✅ EQ8 Setup complete.")
    
    # Check for existing samples to resume
    start_index = 0
    if os.path.exists(output_dir):
        # Fix: Filter for output_*.wav to avoid matching metadata.json or input_*.wav
        existing = [f for f in os.listdir(output_dir) if f.startswith("output_") and f.endswith(".wav")]
        indices = []
        for f in existing:
            try:
                # Expecting 'output_00000.wav'
                idx_str = f.split('_')[1].split('.')[0]
                indices.append(int(idx_str))
            except (IndexError, ValueError):
                pass
        if indices:
            start_index = max(indices) + 1
            print(f"--- Found {len(indices)} existing samples in {output_dir}. Resuming from index {start_index} ---")
    
    if start_index >= num_samples:
        print(f"Target {num_samples} already reached. Skipping.")
        return
    
    remaining = num_samples - start_index
    num_cycles = (remaining + num_lanes - 1) // num_lanes
    
    print(f"=== Parallel Generator ({num_lanes} lanes × 2s) ===")
    print(f"Effect: {effect_name}")
    print(f"Samples: {remaining} remaining ({num_cycles} play cycles)")
    print(f"Est. time: ~{num_cycles * (duration + 0.5) / 60:.1f} minutes")
    print(f"Ableton tracks needed: {num_lanes * 2} (check your setup!)")
    
    metadata = []
    sample_idx = start_index
    
    for cycle in range(num_cycles):
        # How many lanes to use this cycle (last cycle may be partial)
        lanes_this_cycle = min(num_lanes, num_samples - sample_idx)
        
        print(f"--- Cycle {cycle+1}/{num_cycles} | Samples {sample_idx}-{sample_idx + lanes_this_cycle - 1} ---")
        
        # 1. Randomize all lanes
        lane_data = []
        for lane in range(lanes_this_cycle):
            flat_params, settings_log = randomize_lane(client, lane, effect_name, lane_tracks)
            lane_data.append((flat_params, settings_log))
        
        time.sleep(0.1)
        
        # 2. Play and record all channels simultaneously
        client.play()
        recording = record_multi_ch(num_lanes * 4, duration, sample_rate, bh_idx)
        client.stop()
        
        # 3. Split the 16-channel recording into 4 stereo dry/wet pairs
        for lane in range(lanes_this_cycle):
            dry_ch_start = lane * 4          # Channels 0-1, 4-5, 8-9, 12-13
            wet_ch_start = lane * 4 + 2      # Channels 2-3, 6-7, 10-11, 14-15
            
            dry_audio = recording[:, dry_ch_start:dry_ch_start + 2]
            wet_audio = recording[:, wet_ch_start:wet_ch_start + 2]
            
            # Save files
            idx = sample_idx + lane
            input_file = f"input_{idx:05d}.wav"
            output_file = f"output_{idx:05d}.wav"
            
            sf.write(os.path.join(output_dir, input_file), dry_audio, sample_rate)
            sf.write(os.path.join(output_dir, output_file), wet_audio, sample_rate)
            
            flat_params, settings_log = lane_data[lane]
            metadata.append({
                "index": idx,
                "settings": settings_log,
                "input_file": input_file,
                "output_file": output_file,
                "effect_type": effect_name,
            })
        
        sample_idx += lanes_this_cycle
        
        # Save metadata periodically (every 10 cycles)
        if (cycle + 1) % 10 == 0 or cycle == num_cycles - 1:
            with open(os.path.join(output_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
    
    # Final metadata save
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Done! Generated {sample_idx - start_index} samples.")
    print(f"Metadata saved to {output_dir}/metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel Data Generator (4 lanes via BlackHole 16ch)")
    parser.add_argument("--effect", type=str, required=True, help="Effect name (ott, phaser, saturator, reverb)")
    parser.add_argument("--num_samples", type=int, default=3000, help="Total samples to generate")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration per sample in seconds")
    parser.add_argument("--lanes", type=int, default=4, help="Parallel lanes (4 for 16ch, 16 for 64ch)")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Base output directory")
    
    args = parser.parse_args()
    
    if args.effect.lower() not in EFFECT_SCHEMAS:
        print(f"❌ Unknown effect: {args.effect}. Available: {list(EFFECT_SCHEMAS.keys())}")
        sys.exit(1)
    
    output_dir = os.path.join(args.output_dir, args.effect.lower())
    
    generate_parallel(
        num_samples=args.num_samples,
        output_dir=output_dir,
        effect_name=args.effect.lower(),
        duration=args.duration,
        num_lanes=args.lanes,
    )
