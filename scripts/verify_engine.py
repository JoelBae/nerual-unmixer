import sys
import os
sys.path.append('src')
from data.pedalboard_engine import PedalboardEngine
import numpy as np

def test_engine():
    print("Starting PedalboardEngine Verification...")
    engine = PedalboardEngine()
    
    # 1. Test Plugin Loading
    print("\nTesting Plugin Loading and Parameters:")
    plugins_to_test = ["ott", "kh_distortion", "kh_eq3band", "kh_reverb", "vital"]
    for p_id in plugins_to_test:
        try:
            plugin = engine.load_plugin(p_id)
            print(f"  [PASS] Successfully loaded {p_id}")
            print(f"    Parameters: {[p.name for p in plugin.parameters.values()]}")
        except Exception as e:
            print(f"  [FAIL] Failed to load {p_id}: {e}")

    # 2. Test Audio Rendering
    print("\nTesting Audio Rendering (OTT):")
    try:
        # Create 0.1s of silence
        duration = 0.1
        sr = 44100
        input_audio = np.random.uniform(-0.1, 0.1, (2, int(sr * duration))).astype(np.float32)
        
        # Render with OTT
        params = {"depth": 1.0}
        output = engine.render("ott", params, duration_sec=duration, input_audio=input_audio)
        
        if output.shape == input_audio.shape:
            print(f"  [PASS] Rendered audio shape matches: {output.shape}")
            if not np.allclose(output, input_audio):
                print("  [PASS] Output audio is different from input (OTT processed it)")
            else:
                print("  [WARNING] Output is identical to input - OTT might not be processing?")
        else:
            print(f"  [FAIL] Rendered audio shape mismatch: {output.shape} vs {input_audio.shape}")
    except Exception as e:
        print(f"  [FAIL] Error during rendering: {e}")

    # 3. Test Jacobian Computation
    print("\nTesting Jacobian Computation (KH Filter):")
    try:
        base_params = {"cutoff": 0.5}
        # Pedalboard might use normalized values [0,1] or raw values depending on how it's wrapped.
        # kHs Filter uses 'Cutoff' as a parameter name.
        plugin = engine.load_plugin("kh_filter")
        # print(f"Plugin parameters: {[p.name for p in plugin.parameters.values()]}")
        
        # Most pedalboard parameters are accessible by name if they are simple VSTs.
        
        target_param = "cutoff" 
        # If it fails, we'll see in the output.
        
        jacs = engine.compute_jacobian(
            "kh_filter", 
            base_params, 
            [target_param], 
            duration_sec=0.05, 
            input_audio=input_audio[:, :int(sr * 0.05)]
        )
        
        if target_param in jacs:
            grad = jacs[target_param]
            print(f"  [PASS] Jacobian for '{target_param}' computed. Shape: {grad.shape}")
            if np.any(grad != 0):
                print(f"  [PASS] Jacobian has non-zero values (Gradient detected)")
            else:
                print(f"  [WARNING] Jacobian is all zeros - parameter might not be affecting output or wrong name used.")
        else:
            print(f"  [FAIL] Jacobian for '{target_param}' not found in results")
            
    except Exception as e:
        print(f"  [FAIL] Error during Jacobian computation: {e}")

if __name__ == "__main__":
    test_engine()
