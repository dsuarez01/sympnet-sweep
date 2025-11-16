import itertools
import argparse
import uuid
from datetime import datetime

import dotenv
import ray
import wandb

from sympnet_sweep.trial import run_trial
from sympnet_sweep.utils import TrialConfig

SUPPORTED_SYSTEMS = ["duffing"]

def main(args: argparse.Namespace) -> None:
	"""
	(See Section 4.1, pg. 20 of [2]; Section 4.4, pg. 23 of [2])

	trial params:      applies to?             choose one of these vals for trial

	(data/general)
	- epochs         (all)                   : [50_000]
	- lr             (all)                   : [0.002 (Adam)]
	- h              (all)                   : [0.01, 0.1, 1.0]
	- n_data         (all)                   : [100, 200, 400]
	- F              (all)                   : [0.3]
	- omega          (all)                   : [1.0]

	(model config)
	- dim            (all)                   : [2]
	- layers         (all)                   : [8, 12, 16, 24, 32, 48, 64]
	- width          (G, GR, H, R)           : [4, 8, 16, 32, 64]
	- symmetric      (all)                   : [True, False]
	- method         (all)                   : ["R", "P", "G", "GR", "LA", "H"]
	- min_deg        (P only)                : [2]
	- max_deg        (P only)                : [2, 3, 4, 8, 12, 16, 24]
	- sublayers      (LA only)               : [3, 6, 9, 12]
	- activation     (N/A, see strupnet)     : [None]
	- volume_step    (N/A, see strupnet)     : [False]

	note:

	dim=2 since there are 2 degrees of freedom in the augmented Hamiltonian

	For P-SympNets, min_deg=2 by default. Refer to pg. 14 of [2]: 'In practice, we will let the [basis Hamiltonian] sum
	run from 2 to d to avoid linear terms in the Hamiltonian, an additional physical assumption
	that would otherwise correspond to a constant term in the ODE.'

	activation is only applicable to certain methods, hardcoded in repo if so

	volume_step unrelated to SympNets

	(reverse used internally if symmetric enabled)
	"""

	# ray + wandb handling
	print("run_sweep.py: init. ray cluster in main...")
	# init ray here
	ray.init(
		address="auto",
		runtime_env={
			"OMP_NUM_THREADS" : 1,
			"MKL_NUM_THREADS" : 1,
			"OPENBLAS_NUM_THREADS" : 1,
			"VECLIB_MAXIMUM_THREADS" : 1,
			"NUMEXPR_NUM_THREADS" : 1,
		},
	)

	print("run_sweep.py: ray cluster resources are", ray.cluster_resources())

	dotenv.load_dotenv()
	run_id = str(uuid.uuid4())

	if args.enable_wandb:
		wandb.login()
		wandb.setup()

	configs: list[ray.ObjectRef[TrialConfig]] = []
	base_kwargs = {
		"system": args.system,
		"ts": args.timestamp if args.timestamp else datetime.now().strftime('%m%d-%H%M'),
		"run_id": run_id,
		"checkpt": {},
		"epochs": 50000,
		"lr": 0.002,
		"F": 0.3,
		"omega": 1.0,
		"dim": 2,
		"weight_decay": 0.0,
		"val_size": 0.2,
		"random_state": 42,
		"batch_size": 100,
		"activation": None,
		"volume_step": False,
	}

	def add_config(h, n_data, layers, sym, method, **method_kwargs):
		config_dict = {**base_kwargs, "h": h, "n_data": n_data, 
			"layers": layers, "symmetric": sym, "method": method,
			**method_kwargs}
		trial_ref = ray.put(TrialConfig(**config_dict))
		configs.append(trial_ref)

	for (n_data, h), layers, sym, method in itertools.product(
		[(100, 0.01), (200, 0.1), (400, 1.0)],
		[8, 12, 16, 24, 32, 48, 64],
		[True, False],
		["R", "P", "G", "GR", "LA", "H"]
	):
		if method in ["G", "GR", "H", "R"]:
			for width in [4, 8, 16, 32, 64]:
				add_config(h, n_data, layers, sym, method, width=width)
		
		elif method == "P":
			for max_degree, weight_decay in itertools.product(
				[2, 3, 4, 8, 12, 16, 24],
				[0.0, 1e-4, 1e-3, 1e-2, 1e-1]
			):
				add_config(h, n_data, layers, sym, method, min_degree=2, max_degree=max_degree, weight_decay=weight_decay)
		
		elif method == "LA":
			for sublayers in [3, 6, 9, 12]:
				add_config(h, n_data, layers, sym, method, sublayers=sublayers)

	print(f"Total configs: {len(configs)}")

	futures = [run_trial.remote(c, args) for c in configs]
	results = []
	remaining = futures[:]

	while remaining:
		ready, remaining = ray.wait(remaining, num_returns=1)
		results.extend(ready)
	
	results = ray.get(results)
	print(f"Run finished: Ray completed {len(results)}/{len(configs)} trials")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--entity", required=False, default = "", help="Wandb username (for logging)")
	parser.add_argument("-s", "--system", required=True, choices=["duffing"], help="System name (resumes from this arg if passed and logging enabled)")
	parser.add_argument("-t", "--timestamp", required=False, default = "", help="Wandb timestamp (resumes from this arg if passed and logging enabled)")
	parser.add_argument("--enable-wandb", action="store_true", help="Enables Wandb logging if provided")
	args = parser.parse_args()
	
	if args.enable_wandb:
		assert args.entity, "Entity name must be passed in if Wandb logging is enabled"

	print("run_sweep.py: submitting to main...")
	main(args)