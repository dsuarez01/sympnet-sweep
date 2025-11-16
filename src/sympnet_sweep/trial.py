import argparse
import time

import ray
import wandb
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from strupnet import SympNet

from sympnet_sweep.utils import load_dataset, SympNetDS, TrialConfig

def setup_torch():
	import torch
	torch.set_default_dtype(torch.float64)
	torch.set_num_threads(1)
	torch.set_num_interop_threads(1)

@ray.remote(num_cpus=1, runtime_env={"worker_process_setup_hook":"sympnet_sweep.trial.setup_torch"})
def run_trial(trial_config: TrialConfig, args: argparse.Namespace) -> None:

	if trial_config.checkpt_path.exists():
		trial_config.checkpt = torch.load(trial_config.checkpt_path)
		trial_config.run_id = trial_config.checkpt["run_id"]
		start_epoch = trial_config.checkpt["epoch"]+1
		train_losses = trial_config.checkpt["train_losses"]
		val_losses = trial_config.checkpt["val_losses"]
		best_val_loss = trial_config.checkpt["best_val_loss"]
		t_elapsed = trial_config.checkpt["elapsed_time"]
		t_0 = time.time()
	else:
		start_epoch = 0
		train_losses = []
		val_losses = []
		best_val_loss = float("inf")
		t_elapsed = 0.0
		t_0 = time.time()

	if start_epoch >= trial_config.epochs: return # early termination to avoid extra inits.

	config = {k:v for k,v in vars(trial_config).items() if k != "checkpt"}
	
	print(f"Starting model run: {trial_config.model_name}")

	run = wandb.init(
		entity=args.entity,
		project="sympnet-sweep",
		group=trial_config.system+trial_config.ts,
		name=trial_config.model_name,
		id=trial_config.run_id,
		config=config,
		resume="allow",
		mode="offline" if args.enable_wandb else "disabled",
	)

	dataset = load_dataset(
		system=trial_config.system,
		n_data=trial_config.n_data,
		h=trial_config.h,
		F=trial_config.F,
		omega=trial_config.omega,
	)

	train_ds, val_ds = train_test_split(dataset, test_size=trial_config.val_size, random_state=trial_config.random_state)

	train_dl = DataLoader(SympNetDS(train_ds), batch_size=trial_config.batch_size, shuffle=True, num_workers=0)
	val_dl = DataLoader(SympNetDS(val_ds), batch_size=trial_config.batch_size, shuffle=False, num_workers=0)

	model_kwargs = {
		"dim": trial_config.dim,
		"layers": trial_config.layers,
		"width": trial_config.width,
		"symmetric": trial_config.symmetric,
		"method": trial_config.method,
		"min_degree": trial_config.min_degree,
		"max_degree": trial_config.max_degree,
		"sublayers": trial_config.sublayers,
		"activation": trial_config.activation,
		"volume_step": trial_config.volume_step,
	}
	
	model = SympNet(**model_kwargs)
	opt = torch.optim.Adam(model.parameters(), lr=trial_config.lr, weight_decay=trial_config.weight_decay)

	if trial_config.checkpt:
		model.load_state_dict(trial_config.checkpt["model"])
		opt.load_state_dict(trial_config.checkpt["opt"])

	for epoch in range(start_epoch, trial_config.epochs):
		model.train()
		train_loss = 0.0

		for batch in train_dl:
			x_i, truth = batch[:, 0], batch[:, 1]
			opt.zero_grad()
			pred = model(x_i, dt=trial_config.h, symmetric=trial_config.symmetric)
			loss = ((pred-truth)**2).sum(dim=-1).mean() # avg l2 norms over batch
			loss.backward()
			opt.step()
			train_loss += loss.item()

		train_loss /= len(train_dl)
		train_losses.append(train_loss)

		model.eval()
		val_loss = 0.0

		with torch.no_grad():
			for batch in val_dl:
				x_i, truth = batch[:, 0], batch[:, 1]
				pred = model(x_i, dt=trial_config.h, symmetric=trial_config.symmetric)
				loss = ((pred-truth)**2).sum(dim=-1).mean()
				val_loss += loss.item()

		val_loss /= len(val_dl)
		val_losses.append(val_loss)

		# checkpt only on best val loss
		if val_loss < best_val_loss:
			t_now = time.time()
			t_elapsed += t_now-t_0
			t_0 = t_now
			best_val_loss = val_loss
			checkpt = {
				"model": model.state_dict(),
				"opt": opt.state_dict(),
				"epoch": epoch,
				"train_losses": train_losses,
				"val_losses": val_losses,
				"best_val_loss": best_val_loss,
				"t_elapsed": t_elapsed,
				"run_id": trial_config.run_id,
			}

			torch.save(checkpt, trial_config.checkpt_path)

			print(f"New best at epoch {epoch+1}/{trial_config.epochs}: val_loss={val_loss:.6e}, time={t_elapsed:.1f}s")

		run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

	run.finish()