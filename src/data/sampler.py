import numpy as np
from scipy.stats import qmc
from typing import Dict, List, Tuple, Optional

class ParameterSampler:
    """
    Handles Latin Hypercube Sampling (LHS) for parameter calibration clusters.
    """
    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed

    def generate_lhs_clusters(self, 
                              param_bounds: Dict[str, Tuple[float, float]], 
                              num_clusters: int) -> List[Dict[str, float]]:
        """
        Generates N clusters using Latin Hypercube Sampling.
        
        Args:
            param_bounds: Dictionary mapping parameter names to (min, max) tuples.
            num_clusters: Number of clusters to generate.
            
        Returns:
            List of parameter dictionaries.
        """
        names = list(param_bounds.keys())
        bounds = np.array([param_bounds[name] for name in names])
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        
        # Initialize LHS sampler
        sampler = qmc.LatinHypercube(d=len(names), seed=self.seed)
        sample = sampler.random(n=num_clusters)
        
        # Scale to bounds
        scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)
        
        # Convert to list of dicts
        clusters = []
        for i in range(num_clusters):
            clusters.append({names[j]: float(scaled_sample[i, j]) for j in range(len(names))})
            
        return clusters

if __name__ == "__main__":
    # Example usage for OTT
    ott_bounds = {
        "depth": (0.0, 1.0),
        "thresh_l": (0.0, 1.0),
        "thresh_m": (0.0, 1.0),
        "thresh_h": (0.0, 1.0),
        "gain_l": (0.0, 1.0),
        "gain_m": (0.0, 1.0),
        "gain_h": (0.0, 1.0)
    }
    
    sampler = ParameterSampler()
    clusters = sampler.generate_lhs_clusters(ott_bounds, num_clusters=10)
    
    print(f"Generated {len(clusters)} clusters for OTT:")
    for i, c in enumerate(clusters[:3]):
        print(f" Cluster {i}: {c}")
