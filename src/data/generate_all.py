import argparse
import subprocess
import time

def run_generation(effect_name, num_samples, duration):
    print(f"\nExample: Generatng data for {effect_name} (Duration: {duration}s)...")
    cmd = [
        "python", "src/data/generator.py",
        "--effect", effect_name,
        "--num_samples", str(num_samples),
        "--duration", str(duration)
    ]
    subprocess.run(cmd, check=True)
    time.sleep(2) # Cooldown

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--duration", type=float, default=2.0)
    args = parser.parse_args()
    
    effects = [
        "operator",
        "saturator",
        "eq8",
        "ott",
        "phaser",
        "reverb"
    ]
    
    for effect in effects:
        run_generation(effect, args.num_samples, args.duration)
        
    print("\n--- All Datasets Generated ---")
