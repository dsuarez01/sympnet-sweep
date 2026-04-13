import argparse
from datetime import datetime
import itertools

import ray
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler

from sympnet_sweep.config import PARAMS, PROJECT_ROOT, SUPPORTED_SYSTEMS
from sympnet_sweep.trial import run_trial
from sympnet_sweep.utils import save_dataset

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

		configs = []
		for method, method_params in params_search["methods"].items():
			keys = list(method_params.keys())
			for combo in itertools.product(*method_params.values()):
				configs.append({"method": method, **dict(zip(keys, combo))})

		param_space = {
			"system": args.system,
			"ts": timestamp,
			**params_system,
			**params_train,
			"layers": tune.grid_search(params_search["layers"]),
			"symmetric": tune.grid_search(params_search["symmetric"]),
			"min_degree": params_search["min_degree"],
			"activation": params_search["activation"],
			"volume_step": params_search["volume_step"],
			"_h_n_data": tune.grid_search(list(zip(params_search["h"], params_search["n_data"]))),
			"h": tune.sample_from(lambda spec: spec["_h_n_data"][0]), # type: ignore
			"n_data": tune.sample_from(lambda spec: spec["_h_n_data"][1]), # type: ignore
			"_method_config": tune.grid_search(configs),
			"method": tune.sample_from(lambda spec: spec["_method_config"]["method"]), # type: ignore
			"width": tune.sample_from(lambda spec: spec["_method_config"].get("width")), # type: ignore
			"max_degree": tune.sample_from(lambda spec: spec["_method_config"].get("max_degree")), # type: ignore
			"weight_decay": tune.sample_from(lambda spec: spec["_method_config"].get("weight_decay")), # type: ignore
			"sublayers": tune.sample_from(lambda spec: spec["_method_config"].get("sublayers")), # type: ignore
		}

		for h, n_data in param_space["_h_n_data"]["grid_search"]:
			save_dataset(args.system, n_data, h, **params_system)

		tuner = tune.Tuner(
			run_trial,
			param_space=param_space,
			tune_config=tune.TuneConfig(
				scheduler=ASHAScheduler(
					metric="val_loss",
					mode="min",
					max_t=500,
					grace_period=20,
					reduction_factor=2,
				),
			),
			run_config=tune.RunConfig(
				name=f"{args.system}_{timestamp}",
				storage_path=str(PROJECT_ROOT / "checkpts"),
				checkpoint_config=tune.CheckpointConfig(
					num_to_keep=1,
					checkpoint_score_attribute="val_loss",
					checkpoint_score_order="min",
				),
				callbacks=[MLflowLoggerCallback(
					tracking_uri=f"sqlite:///{PROJECT_ROOT}/mlruns/mlflow.db",
					experiment_name=f"{args.system}_{timestamp}",
				)],
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