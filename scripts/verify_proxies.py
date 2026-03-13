import torch
import numpy as np
import sys
import os
import argparse
import json
from typing import Dict, Any, List

# Ensure we can import from src
sys.path.append('src')

from data.pedalboard_engine import PedalboardEngine
from utils.alignment_validator import AlignmentValidator
from models.proxies.ott_ddsp import OTTProxy
from models.proxies.vital_ddsp import VitalProxy
from models.proxies.kilohearts_ddsp import KHDistortionDSP, KHEQ3BandDSP, KHReverbDSP

def verify_ott(validator, engine, args) -> Dict[str, Any]:
    print("\n--- Verifying OTT Proxy ---")
    proxy = OTTProxy()
    mapping = {
        "depth": "depth", "thresh_l": "thresh_l", "thresh_m": "thresh_m", "thresh_h": "thresh_h",
        "gain_l_db": "gain_l", "gain_m_db": "gain_m", "gain_h_db": "gain_h"
    }
    vst_params = {p: 0.5 for p in mapping.keys()}
    # Use low_rich to ensure the Low band compressor is triggered
    input_audio = validator.generate_input_signal("low_rich", duration_sec=args.duration)
    
    alpha_metrics, _ = validator.compute_alignment_metrics("ott", proxy, vst_params, mapping, input_audio, duration_sec=args.duration, delta=args.delta, representation=args.representation)
    epsilon = validator.compute_input_output_divergence("ott", proxy, vst_params, mapping, input_audio, duration_sec=args.duration, representation=args.representation)
    
    for p, alpha in alpha_metrics.items():
        if p in ["depth", "thresh_l", "thresh_m", "thresh_h"]:
            print(f" - {p:10}: alpha_c = {alpha:.4f}")
    print(f" - Global epsilon (Input-Output): {epsilon:.6f}")
    
    return {"alpha": alpha_metrics, "epsilon": epsilon}

def verify_kh_distortion(validator, engine, args) -> Dict[str, Any]:
    print("\n--- Verifying Kilohearts Distortion ---")
    dsp = KHDistortionDSP()
    mapping = {"drive": "drive", "bias": "bias", "mix": "mix"}
    vst_params = {"drive": 0.5, "bias": 0.5, "mix": 1.0, "type": 0.0}
    input_audio = validator.generate_input_signal("noise", duration_sec=args.duration)
    
    class DSPWrapper(torch.nn.Module):
        def __init__(self, dsp, type_idx):
            super().__init__()
            self.dsp = dsp
            self.type_idx = type_idx
        def forward(self, x, **params):
            return self.dsp(x, params["drive"], params["bias"], params["mix"], self.type_idx)

    wrapper = DSPWrapper(dsp, type_idx=0)
    alpha_metrics, _ = validator.compute_alignment_metrics("kh_distortion", wrapper, vst_params, mapping, input_audio, duration_sec=args.duration, delta=args.delta, representation=args.representation)
    epsilon = validator.compute_input_output_divergence("kh_distortion", wrapper, vst_params, mapping, input_audio, duration_sec=args.duration, representation=args.representation)
    
    for p, alpha in alpha_metrics.items():
        print(f" - {p:10}: alpha_c = {alpha:.4f}")
    print(f" - Global epsilon (Input-Output): {epsilon:.6f}")
    
    return {"alpha": alpha_metrics, "epsilon": epsilon}

def verify_kh_eq(validator, engine, args) -> Dict[str, Any]:
    print("\n--- Verifying Kilohearts 3-Band EQ ---")
    dsp = KHEQ3BandDSP()
    mapping = {"low": "low", "mid": "mid", "high": "high", "low_split": "low_split", "high_split": "high_split"}
    vst_params = {p: 0.5 for p in mapping.keys()}
    input_audio = validator.generate_input_signal("noise", duration_sec=args.duration)
    
    class EQWrapper(torch.nn.Module):
        def __init__(self, dsp):
            super().__init__()
            self.dsp = dsp
        def forward(self, x, **params):
            return self.dsp(x, params["low"], params["mid"], params["high"], params["low_split"], params["high_split"])

    wrapper = EQWrapper(dsp)
    alpha_metrics, _ = validator.compute_alignment_metrics("kh_eq3band", wrapper, vst_params, mapping, input_audio, duration_sec=args.duration, delta=args.delta, representation=args.representation)
    epsilon = validator.compute_input_output_divergence("kh_eq3band", wrapper, vst_params, mapping, input_audio, duration_sec=args.duration, representation=args.representation)
    
    for p, alpha in alpha_metrics.items():
        print(f" - {p:15}: alpha_c = {alpha:.4f}")
    print(f" - Global epsilon (Input-Output): {epsilon:.6f}")
    
    return {"alpha": alpha_metrics, "epsilon": epsilon}

def verify_vital(validator, engine, args) -> Dict[str, Any]:
    print("\n--- Verifying Vital Proxy ---")
    proxy = VitalProxy()
    mapping = {
        "oscillator_1_level": "oscillator_1_level", "oscillator_1_wave_frame": "oscillator_1_wave_frame",
        "filter_1_cutoff": "filter_1_cutoff", "filter_1_resonance": "filter_1_resonance",
        "envelope_1_attack": "envelope_1_attack", "envelope_1_decay": "envelope_1_decay",
        "envelope_1_sustain": "envelope_1_sustain", "envelope_1_release": "envelope_1_release"
    }
    vst_params = {p: 0.5 for p in mapping.keys()}
    vst_params["oscillator_1_level"] = 0.8
    vst_params["oscillator_1_wave_frame"] = 0.0 
    vst_params["filter_1_type"] = 0.0 # Lowpass
    
    print(" - Rendering base VST audio for sync...")
    input_audio_dummy = np.zeros((2, int(args.duration * 44100)), dtype=np.float32)
    base_vst_audio = engine.render("vital", vst_params, duration_sec=args.duration, input_audio=input_audio_dummy)
    base_vst_audio_torch = torch.from_numpy(base_vst_audio[0]).unsqueeze(0).float()
    print(" - Base VST audio rendered.")

    class VitalWrapper(torch.nn.Module):
        def __init__(self, proxy, sync_audio):
            super().__init__()
            self.proxy = proxy
            self.sync_audio = sync_audio
        def forward(self, x, **params):
            f0 = torch.tensor([[261.625565]]).to(x.device) 
            gate = torch.ones((x.shape[0], x.shape[1])).to(x.device)
            return self.proxy(f0=f0, gate=gate, input_audio=self.sync_audio, temp=0.01, **params)

    wrapper = VitalWrapper(proxy, base_vst_audio_torch)
    print(f" - Computing alignment metrics (rep={args.representation})...")
    alpha_metrics, _ = validator.compute_alignment_metrics("vital", wrapper, vst_params, mapping, input_audio_dummy, duration_sec=args.duration, delta=args.delta, representation=args.representation)
    print(" - Metrics computed.")
    epsilon = validator.compute_input_output_divergence("vital", wrapper, vst_params, mapping, input_audio_dummy, duration_sec=args.duration, representation=args.representation)
    
    for p, alpha in alpha_metrics.items():
        print(f" - {p:25}: alpha_c = {alpha:.4f}")
    print(f" - Global epsilon (Input-Output): {epsilon:.6f}")
    
    return {"alpha": alpha_metrics, "epsilon": epsilon}

def print_summary_table(all_results: Dict[str, Any]):
    print("\n" + "="*70)
    print(f"{'Proxy':<15} | {'Parameter':<25} | {'alpha_c':<10} | {'epsilon (global)':<10}")
    print("-"*70)
    for proxy_name, res in all_results.items():
        epsilon = res["epsilon"]
        for p_name, alpha in res["alpha"].items():
            print(f"{proxy_name:<15} | {p_name:<25} | {alpha:>10.4f} | {epsilon:>10.4f}")
            # Only print epsilon once per proxy for clarity in table
            epsilon = float('nan') 
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", type=str, default="all", help="ott, vital, kh_distortion, kh_eq, or all")
    parser.add_argument("--output", type=str, default="test_audio_outputs/verification_results.json", help="Path to save output JSON")
    parser.add_argument("--duration", type=float, default=0.1, help="Duration of test signal in seconds")
    parser.add_argument("--delta", type=float, default=1e-3, help="Finite difference delta")
    parser.add_argument("--signal", type=str, default=None, help="Force a signal type (noise, low_rich, bass_saw, etc)")
    parser.add_argument("--representation", "-rep", type=str, default="mel", choices=["mel", "time"], help="Representation for Jacobian (mel or time)")
    args = parser.parse_args()
    
    engine = PedalboardEngine()
    validator = AlignmentValidator(engine)
    
    all_results = {}
    
    # Override signal in verify functions if args.signal is provided
    def get_signal(default_type):
        return args.signal if args.signal else default_type

    if args.proxy in ["ott", "all"]:
        signals = args.signal.split(",") if args.signal else ["low_rich", "bass_saw", "mid_saw", "noise"]
        print(f"Running OTT verification for signals: {signals}")
        
        for sig in signals:
            sig = sig.strip()
            # Intelligent representation selection
            rep = "time" if "saw" in sig else args.representation
            print(f"\n--- OTT Verification [Signal: {sig}, Rep: {rep}] ---")
            
            proxy = OTTProxy()
            mapping = {
                "depth": "depth", "thresh_l": "thresh_l", "thresh_m": "thresh_m", "thresh_h": "thresh_h",
                "gain_l_db": "gain_l", "gain_m_db": "gain_m", "gain_h_db": "gain_h"
            }
            vst_params = {p: 0.5 for p in mapping.keys()}
            input_audio = validator.generate_input_signal(sig, duration_sec=args.duration)
            
            alpha_metrics, _ = validator.compute_alignment_metrics("ott", proxy, vst_params, mapping, input_audio, duration_sec=args.duration, delta=args.delta, representation=rep)
            epsilon = validator.compute_input_output_divergence("ott", proxy, vst_params, mapping, input_audio, duration_sec=args.duration, representation=rep)
            
            res_key = f"ott_{sig}"
            all_results[res_key] = {"alpha": alpha_metrics, "epsilon": epsilon}
            
            for p, alpha in alpha_metrics.items():
                if p in ["depth", "thresh_l", "thresh_m", "thresh_h"]:
                    print(f" - {p:10}: alpha_c = {alpha:.4f}")
            print(f" - Global epsilon (Input-Output): {epsilon:.6f}")
    
    if args.proxy in ["vital", "all"]:
        all_results["vital"] = verify_vital(validator, engine, args)
    
    if args.proxy in ["kh_distortion", "all"]:
        all_results["kh_distortion"] = verify_kh_distortion(validator, engine, args)
        
    if args.proxy in ["kh_eq", "all"]:
        all_results["kh_eq3band"] = verify_kh_eq(validator, engine, args)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {args.output}")
    
    # Print summary table
    print_summary_table(all_results)

if __name__ == "__main__":
    main()
