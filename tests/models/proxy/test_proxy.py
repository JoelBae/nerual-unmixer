import os
import sys
import torch
import torchaudio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.models.proxy.ddsp_modules import OperatorProxy

if __name__ == "__main__":
    print("Testing the Complete Operator Proxy...")
    
    sample_rate = 44100
    # Our generated note will be held for 1.0 seconds, then released.
    # We'll generate 3 seconds of audio to let the release decay out.
    note_off_time = 0.1
    num_samples = int(4.0 * sample_rate) 
    
    proxy = OperatorProxy(sample_rate=sample_rate)
    
    # We need to simulate the batch output from our Dataset (batch=1, 16 params)
    # These are raw Ableton dial values (0 to 127)
    params = torch.zeros((1, 16), dtype=torch.float32)
    
    # 0: Transpose (64 = C3 Center) Let's test a lower note: -12 semitones = 48
    params[0, 0] = 48.0
    
    # 12: Attack (0-127 -> 0-20s). Let's do a fast attack: dial at 2 (0.3 seconds)
    params[0, 12] = 2.0
    
    # 13: Decay (0-127 -> 0-60s). Let's do dial at 2 (~1 second)
    params[0, 13] = 1.0
    
    # 14: Sustain (0-127 -> Volume). Let's hold at 40% volume: dial at 50
    params[0, 14] = 50.0
    
    # 15: Release (0-127 -> 0-60s). Let's do a long 2-second release tail: dial at 4
    params[0, 15] = 4.0
    
    # 9: Pe Amount (0-127 -> -1.0 to 1.0). Let's set it to max (127) for full effect
    params[0, 9] = 127.0
    
    # 10: Pe Decay (0-127 -> 0-20s). Fast drop. Dial at 2 (~0.3 seconds)
    params[0, 10] = 2.0
    
    # 11: Pe Peak (0-127 -> -48 to +48 semitones). Let's jump up an entire octave (+12 semitones) 
    # Center is 64, so 64 + 16 = 80
    params[0, 11] = 80.0
    
    # --- FILTER SETTINGS ---
    # 2: Filter Freq (0-127 -> 20Hz to 20000Hz). Base cutoff very low (dial at 10)
    params[0, 2] = 10.0
    
    # 3: Filter Res (0-127 -> 0.1 to 4.0). High resonance for a squelchy sound (dial at 100)
    params[0, 3] = 100.0
    
    # 4: Fe Amount (0-127 -> -1.0 to 1.0). Max positive sweep (dial at 127)
    params[0, 4] = 127.0
    
    # 5: Fe Attack (0-127 -> 0-20s). Fast filter opening (dial at 5)
    params[0, 5] = 5.0
    
    # 6: Fe Decay (0-127 -> 0-60s). Slower closing (dial at 20)
    params[0, 6] = 20.0
    
    # 7: Fe Sustain (0-127). Hold at 20% open
    params[0, 7] = 25.0
    
    # 8: Fe Release (0-127 -> 0-60s). Fast release (dial at 5)
    params[0, 8] = 5.0
    
    print("\nSending raw Ableton dials through the Differentiable Synth...")
    audio_tensor = proxy(params, note_off_time=note_off_time, num_samples=num_samples)
    
    # Output to WAV
    output_path = "test_proxy_final.wav"
    torchaudio.save(output_path, audio_tensor, sample_rate)
    
    print(f"\nSuccess! Plucked Synth Audio saved to: {os.path.abspath(output_path)}")
    print(f"Shape: {audio_tensor.shape}")
