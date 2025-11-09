import numpy as np

def duffing(t: float, x: np.ndarray, F: float, omega: float) -> np.ndarray:
	p, p_tau, q, tau = x

	p_dot = q-q**3+F*np.cos(omega*tau)
	p_tau_dot = -F*omega*q*np.sin(omega*tau)
	q_dot = p
	tau_dot = 1.0
	
	return np.array([p_dot, p_tau_dot, q_dot, tau_dot])