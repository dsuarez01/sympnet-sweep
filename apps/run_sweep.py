import argparse
from datetime import datetime

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from sympnet_sweep.config import PARAMS, PROJECT_ROOT, SUPPORTED_SYSTEMS
from sympnet_sweep.trial import run_trial

def main(args: argparse.Namespace) -> None:
	""" Sweep runner for SympNet hyperparameter search using Ray Tune """

	print("run_sweep.py: init. ray cluster in main...")

	ray.init(
		address="auto",
		runtime_env={
			"worker_process_setup_hook": "sympnet_sweep.trial.setup_torch",
			"env_vars": {
				"RAY_CHDIR_TO_TRIAL_DIR": "0",
				"OMP_NUM_THREADS": "1",
				"MKL_NUM_THREADS": "1",
				"OPENBLAS_NUM_THREADS": "1",
				"VECLIB_MAXIMUM_THREADS": "1",
				"NUMEXPR_NUM_THREADS": "1",
			}
		},
	)

	print("run_sweep.py: ray cluster resources are", ray.cluster_resources())

	timestamp = args.timestamp if args.timestamp else datetime.now().strftime('%m%d-%Y-%H%M') 

	experiment_path = PROJECT_ROOT / "checkpts" / f"{args.system}_{timestamp}"

	if tune.Tuner.can_restore(str(experiment_path)):
		tuner = tune.Tuner.restore(
			path=str(experiment_path),
			trainable=run_trial,
		)
	else:
		params_system = PARAMS[args.system]["system"]
		params_train = PARAMS[args.system]["train"]
		params_search = PARAMS[args.system]["search"]

		param_space = {
			"system": args.system,
			"ts": timestamp,
			**params_system,
			**params_train,
			"h": tune.grid_search(params_search["h"]),
			"n_data": tune.grid_search(params_search["n_data"]),
			"layers": tune.grid_search(params_search["layers"]),
			"symmetric": tune.grid_search(params_search["symmetric"]),
			"method": tune.grid_search(params_search["method"]),
			"_width": tune.grid_search(params_search["width"]),
			"_max_degree": tune.grid_search(params_search["max_degree"]),
			"_weight_decay": tune.grid_search(params_search["weight_decay"]),
			"_sublayers": tune.grid_search(params_search["sublayers"]),
			"min_degree": params_search["min_degree"],
			"activation": params_search["activation"],
			"volume_step": params_search["volume_step"],
			"width": tune.sample_from(lambda spec: spec.config._width if spec.config.method in ["R", "G", "GR", "H"] else None), # type: ignore
			"max_degree": tune.sample_from(lambda spec: spec.config._max_degree if spec.config.method == "P" else None), # type: ignore
			"weight_decay": tune.sample_from(lambda spec: spec.config._weight_decay if spec.config.method == "P" else None), # type: ignore
			"sublayers": tune.sample_from(lambda spec: spec.config._sublayers if spec.config.method == "LA" else None), # type: ignore
		}

		tuner = tune.Tuner(
			run_trial,
			param_space=param_space,
			tune_config=tune.TuneConfig(
				scheduler=ASHAScheduler(
					metric="val_loss",
					mode="min",
					max_t=50000,
					grace_period=2000,
					reduction_factor=2,
				),
			),
			run_config=tune.RunConfig(
				name=f"{args.system}_{timestamp}",
				storage_path=str(PROJECT_ROOT / "checkpts"),
			),
		)

	results = tuner.fit()
	print(f"Run finished: {len(results)} trials completed")

	if len(results) > 0:
		(PROJECT_ROOT / "results").mkdir(exist_ok=True)
		results_df = results.get_dataframe()
		results_df = results_df.drop(columns=[c for c in results_df.columns if c.startswith("config/_")])
		results_df.to_csv(PROJECT_ROOT / f"results/{args.system}_{timestamp}.csv", index=False)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--system", required=True, choices=SUPPORTED_SYSTEMS, help="System name (Ray resumes based on this arg if possible)")
	parser.add_argument("-t", "--timestamp", required=False, default = "", help="timestamp (Ray resumes based on this arg if possible)")
	args = parser.parse_args()

	print(f"run_sweep.py: submitting to main for {args.system}...")
	main(args)