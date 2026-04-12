from abc import ABC, abstractmethod

import numpy as np

class HamiltonianSystem(ABC):
	
	@abstractmethod
	def ic(self, **kwargs) -> np.ndarray:
		pass
	
	@abstractmethod
	def __call__(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
		pass

class DuffingSystem(HamiltonianSystem):
	
	def ic(self, **kwargs) -> np.ndarray:
		omega = kwargs["omega"]
		period = 2 * np.pi / omega
		p_0 = np.random.uniform(-0.5, 0.5)
		p_tau_0 = 0.0
		q_0 = np.random.uniform(-0.5, 0.5)
		tau_0 = np.random.uniform(0, period)
		return np.array([p_0, p_tau_0, q_0, tau_0])
	
	def __call__(self, t: float, x: np.ndarray, **kwargs) -> np.ndarray:
		F = kwargs["F"]
		omega = kwargs["omega"]
		p, p_tau, q, tau = x
		p_dot = q - q**3 + F * np.cos(omega * tau)
		p_tau_dot = -F * omega * q * np.sin(omega * tau)
		q_dot = p
		tau_dot = 1.0
		return np.array([p_dot, p_tau_dot, q_dot, tau_dot])

duffing = DuffingSystem()