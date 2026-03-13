import torch
import numpy as np
import json
import os

class ManifoldMapper:
    """
    Solves for the parameter exchange rate K(theta) using the Manifold Mapping dataset.
    k_c = <J_vst, J_proxy> / ||J_proxy||^2
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = self._load_dataset()
        
    def _load_dataset(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        with open(self.dataset_path, "r") as f:
            return json.load(f)

    def compute_exchange_rates(self):
        """
        Computes k_c for every parameter in every cluster.
        Returns a list of k_c values mapped to parameter names.
        """
        results = []
        
        for entry in self.data:
            k_rates = {}
            params = entry["params"]
            j_vst = entry["jacobian_vst"]
            j_proxy = entry["jacobian_proxy"]
            
            for p_name in j_vst.keys():
                v = np.array(j_vst[p_name])
                p = np.array(j_proxy[p_name])
                
                # Equation 7: Projection of VST onto Proxy
                # k_c = (V . P) / (P . P)
                dot_product = np.dot(v, p)
                p_norm_sq = np.dot(p, p) + 1e-8
                
                k_c = dot_product / p_norm_sq
                k_rates[p_name] = float(k_c)
                
            results.append({
                "id": entry["id"],
                "params": params,
                "k_rates": k_rates,
                "alphas": entry.get("alpha", {})
            })
            
        return results

    def save_calibration_map(self, output_path: str):
        rates = self.compute_exchange_rates()
        with open(output_path, "w") as f:
            json.dump(rates, f, indent=2)
        print(f"Saved calibration map with {len(rates)} clusters to {output_path}")

if __name__ == "__main__":
    mapper = ManifoldMapper("datasets/jacobian_clusters/ott_manifold_dataset.json")
    mapper.save_calibration_map("datasets/jacobian_clusters/ott_k_map.json")
