import torch
import numpy as np
import json
import sys
import os

# Ensure we can import from src
sys.path.append('src')

from data.pedalboard_engine import PedalboardEngine
from utils.alignment_validator import AlignmentValidator
from data.sampler import ParameterSampler
from models.proxies.ott_ddsp import OTTProxy

def generate_ott_clusters(num_clusters: int = 100):
    print(f"Generating {num_clusters} Jacobian clusters for OTT...")
    engine = PedalboardEngine()
    validator = AlignmentValidator(engine)
    sampler = ParameterSampler(seed=42)
    
    # Define OTT bounds (0-1 for all internal DDSP parameters)
    ott_bounds = {
        "depth": (0.0, 1.0),
        "thresh_l": (0.0, 1.0),
        "thresh_m": (0.0, 1.0),
        "thresh_h": (0.0, 1.0),
        "gain_l": (0.0, 1.0),
        "gain_m": (0.0, 1.0),
        "gain_h": (0.0, 1.0)
    }
    
    # VST Parameter Mapping for Jacobian extraction
    proxy_param_mapping = {
        "depth": "depth",
        "thresh_l": "thresh_l",
        "thresh_m": "thresh_m",
        "thresh_h": "thresh_h",
        "gain_l_db": "gain_l",
        "gain_m_db": "gain_m",
        "gain_h_db": "gain_h"
    }
    
    # 1. Load LHS Samples (or generate if missing)
    input_path = "datasets/jacobian_clusters/ott_lh_clusters.json"
    if not os.path.exists(input_path):
        print(f"No existing clusters found. Generating {num_clusters} samples...")
        clusters_params = sampler.generate_lhs_clusters(ott_bounds, num_clusters)
    else:
        print(f"Loading existing clusters from {input_path}...")
        with open(input_path, "r") as f:
            full_clusters = json.load(f)
            clusters_params = [c["proxy_params"] for c in full_clusters]
    
    # 2. Prepare Input Signal - Full Spectrum is best for capturing all band sensitivities
    input_audio = validator.generate_input_signal("full_spectrum", duration_sec=0.1)
    
    dataset = []
    proxy = OTTProxy()
    
    for i, params in enumerate(clusters_params):
        print(f"Processing Cluster {i+1}/{len(clusters_params)}...")
        
        vst_params = {
            "depth": params["depth"],
            "thresh_l": params["thresh_l"],
            "thresh_m": params["thresh_m"],
            "thresh_h": params["thresh_h"],
            "gain_l_db": params["gain_l"],
            "gain_m_db": params["gain_m"],
            "gain_h_db": params["gain_h"]
        }
        
        # Compute VST Jacobian
        j_vst = validator.get_jacobian("ott", proxy, vst_params, proxy_param_mapping, input_audio, representation="mel", mode="vst")
        
        # Compute Proxy Jacobian
        j_proxy = validator.get_jacobian("ott", proxy, vst_params, proxy_param_mapping, input_audio, representation="mel", mode="proxy")
        
        # Compute Alpha (Directional Alignment)
        alphas = {k: float(validator.compute_cosine_similarity(j_vst[k], j_proxy[k])) for k in j_vst.keys()}
        
        # Store as lists for JSON (flattened)
        j_vst_serializable = {k: v.flatten().tolist() for k, v in j_vst.items()}
        j_proxy_serializable = {k: v.flatten().tolist() for k, v in j_proxy.items()}
        
        cluster_entry = {
            "id": i,
            "params": params,
            "alpha": alphas,
            "jacobian_vst": j_vst_serializable,
            "jacobian_proxy": j_proxy_serializable,
            "status": "complete"
        }
        dataset.append(cluster_entry)
        
    os.makedirs("datasets/jacobian_clusters", exist_ok=True)
    output_path = "datasets/jacobian_clusters/ott_manifold_dataset.json"
    with open(output_path, "w") as f:
        json.dump(dataset, f)
        
    print(f"Saved {len(dataset)} clusters with complete Jacobians to {output_path}")

if __name__ == "__main__":
    generate_ott_clusters(num_clusters=20)
