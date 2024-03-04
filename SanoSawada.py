import numpy as np
from dataclasses import dataclass, field
from typing import Any
import matplotlib.pyplot as plt


@dataclass
class SanoSawada:
    ts: np.ndarray
    num_neighbor: int
    dt: float
    step_jac: int
    epsilon: float
    td_idx: np.ndarray
    sampling_neighbors: str = "sequential"
    model: Any = None

    def __post_init__(self):
        self.l_end = np.abs(np.min(self.td_idx))
        self.jac_len = (self.ts.shape[0]-1-self.step_jac-self.l_end)//self.step_jac + 1
        self.r_end = (self.jac_len-1)*self.step_jac + self.l_end

        acceptable_sampling_methods = ["sequential", "random"]
        if self.sampling_neighbors not in acceptable_sampling_methods:
            raise ValueError(f"{self.sampling_neighbors} is not an acceptable value")

    def set_radius_of_ball_for_neighboring_points_search(self):
        self.radius = self.epsilon*(np.max(self.ts)-np.min(self.ts))

    def delay_embedding(self):
        # Phase space reconstruction (simplified example)
        pass

    def compute_jacobian(self):
        # Compute the Jacobian matrix from the time series (simplified example)
        pass

    def estimate_lyapunov_spectrum(self):
        # Estimate the Lyapunov spectrum (simplified example)
        pass
