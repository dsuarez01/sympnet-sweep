from typing import Literal
import pathlib
from dataclasses import dataclass, field
import uuid

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.integrate import solve_ivp

from sympnet_sweep.systems import duffing

SUPPORTED_SYSTEMS = ["duffing"]
SYSTEM_TO_METHOD= {"duffing": duffing}

def solve(system: str, n_data: int, h: float, F: float, omega: float) -> np.ndarray:
	assert system in SUPPORTED_SYSTEMS, "Unsupported system"
	period = 2*np.pi / omega
	dataset = []

	for i in range(n_data):
		p_0 = np.random.uniform(-2.0, 2.0)
		p_tau_0 = 0.0
		q_0 = np.random.uniform(-2.0, 2.0)
		tau_0 = np.random.uniform(0, period)
		x_aug_i = np.array([p_0, p_tau_0, q_0, tau_0])

		result = solve_ivp(
			fun=SYSTEM_TO_METHOD[system],
			t_span=[0.,h],
			y0=x_aug_i,
			method="DOP853", # 8th order RK
			args=(F,omega),
			rtol=3e-14, # scipy complains at ~ 2.2e-14
			atol=1e-15 * np.maximum(1.0, np.abs(x_aug_i)),
			dense_output=False
		)

		if not result.success:
			print(f"Integration failed at sample {i}")
			print(f"At dataset: n_data={n_data}, h={h}, F={F}, omega={omega}")
			print(f"Msg.: {result.message}")
			continue

		truth = result.y[:, -1]
		dataset.append(np.array([x_aug_i, truth]))

	return np.array(dataset)

def load_dataset(system: str, n_data: int, h: float, F: float, omega: float) -> np.ndarray:
	assert system in SUPPORTED_SYSTEMS, "Unsupported system"
	filepath = pathlib.Path(f"./data/{system}/n{n_data}_h{h}_F{F}_omega{omega}.npy")
	return np.load(filepath)

def save_dataset(system: str, n_data: int, h: float, F: float, omega: float) -> None:
	assert system in SUPPORTED_SYSTEMS, "Unsupported system"
	filepath = pathlib.Path(f"./data/{system}/n{n_data}_h{h}_F{F}_omega{omega}.npy")
	filepath.parent.mkdir(parents=True, exist_ok=True)
	np.save(
		filepath,
		solve(system, n_data, h, F, omega),
	)

class SympNetDS(Dataset):
	def __init__(self, data: np.ndarray) -> None:
		self.data: torch.Tensor = torch.from_numpy(data)
	
	def __len__(self) -> int:
		return self.data.shape[0]
	
	def __getitem__(self, idx) -> torch.Tensor:
		return self.data[idx]

@dataclass
class TrialConfig:
	# for wandb logging / gen
	system: str
	ts: str
	run_id: str

	# data/gen (non-opt)
	epochs: int
	lr: float
	h: float
	n_data: int
	F: float
	omega: float
	
	# gen. model config (non-opt)
	dim: int
	layers: int
	symmetric: bool
	method: str
	
	# train config
	val_size: float
	random_state: int
	batch_size: int
	activation: str | None
	volume_step: bool

	# checkpt
	checkpt: dict

	# method-specific model config (w/ defaults)
	width: int | None = None
	min_degree: int | None = None
	max_degree: int | None = None
	sublayers: int | None = None

	@property
	def model_name(self) -> str:
		name = f"symp{self.method}_l{self.layers}_h{self.h}_n{self.n_data}_sym{self.symmetric}"
		if self.width is not None:
			name += f"_w{self.width}"
		if self.min_degree is not None:
			name += f"_mind{self.min_degree}"
		if self.max_degree is not None:
			name += f"_maxd{self.max_degree}"
		if self.sublayers is not None:
			name += f"_subl{self.sublayers}"
		return name
	
	@property
	def checkpt_path(self) -> pathlib.Path:
		path = pathlib.Path(f"./checkpts/{self.system}/{self.ts}/{self.model_name}.pt")
		path.parent.mkdir(parents=True, exist_ok=True)
		return path

	def __post_init__(self):
		assert self.system in SUPPORTED_SYSTEMS