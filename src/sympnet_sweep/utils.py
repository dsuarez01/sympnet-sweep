import functools

import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch.utils.data import Dataset

from sympnet_sweep.config import PROJECT_ROOT, SUPPORTED_SYSTEMS, OPTIMIZER_MAP
from sympnet_sweep import systems

def get_solver(system: str):
	return getattr(systems, system)

def get_ic_sampler(system: str):
	return getattr(systems, system).ic

def get_optimizer(optimizer: str):
	return getattr(torch.optim, OPTIMIZER_MAP[optimizer])

def solve(system: str, n_data: int, h: float, **system_kwargs) -> np.ndarray:
	assert system in SUPPORTED_SYSTEMS, "Unsupported system"
	solver = functools.partial(get_solver(system), **system_kwargs)
	ic_sampler = get_ic_sampler(system)
	dataset = []

	for i in range(n_data):
		x0 = ic_sampler(**system_kwargs)

		result = solve_ivp(
			fun=solver,
			t_span=[0.,h],
			y0=x0,
			method="DOP853", # 8th order RK
			rtol=3e-14, # scipy complains at ~ 2.2e-14
			atol=1e-15 * np.maximum(1.0, np.abs(x0)),
			dense_output=False
		)

		if not result.success:
			print(f"Integration failed at sample {i}")
			print(f"At dataset: n_data={n_data}, h={h}, {system_kwargs}")
			print(f"Msg.: {result.message}")
			continue

		truth = result.y[:, -1]
		dataset.append(np.array([x0, truth]))

	return np.array(dataset)

def load_dataset(system: str, n_data: int, h: float, **system_kwargs) -> np.ndarray:
	assert system in SUPPORTED_SYSTEMS, "Unsupported system"
	kwarg_str = "_".join(f"{k}{v}" for k, v in system_kwargs.items())
	filepath = PROJECT_ROOT / f"data/{system}/n{n_data}_h{h}/{kwarg_str}.npy"
	return np.load(filepath)

def save_dataset(system: str, n_data: int, h: float, **system_kwargs) -> None:
	assert system in SUPPORTED_SYSTEMS, "Unsupported system"
	kwarg_str = "_".join(f"{k}{v}" for k, v in system_kwargs.items())
	filepath = PROJECT_ROOT / f"data/{system}/n{n_data}_h{h}/{kwarg_str}.npy"
	if filepath.exists():
		return
	filepath.parent.mkdir(parents=True, exist_ok=True)
	np.save(filepath, solve(system, n_data, h, **system_kwargs))

class SympNetDS(Dataset):
	def __init__(self, data: np.ndarray) -> None:
		self.data: torch.Tensor = torch.from_numpy(data)
	
	def __len__(self) -> int:
		return self.data.shape[0]
	
	def __getitem__(self, idx) -> torch.Tensor:
		return self.data[idx]