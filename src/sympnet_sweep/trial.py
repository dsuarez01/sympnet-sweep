import argparse
import os

# probably overkill, but worth a shot
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ray
import wandb
from sklearn.model_selection import train_test_split
from strupnet import SympNet
import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from torch.utils.data import DataLoader

from sympnet_sweep.utils import load_dataset, SympNetDS, TrialConfig

@ray.remote(num_cpus=1)
def run_trial(trial_config: TrialConfig, args: argparse.Namespace) -> None:

	if trial_config.checkpt_path.exists():
		trial_config.checkpt = torch.load(trial_config.checkpt_path)
		trial_config.run_id = trial_config.checkpt["run_id"]
		start_epoch = trial_config.checkpt["epoch"]+1
		train_losses = trial_config.checkpt["train_losses"]
		val_losses = trial_config.checkpt["val_losses"]
	else:
		start_epoch = 0
		train_losses = []
		val_losses = []

	if start_epoch >= trial_config.epochs: return # early termination to avoid extra inits.

	config = {k:v for k,v in vars(trial_config).items() if k != "checkpt"}
	
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
	opt = torch.optim.Adam(model.parameters(), lr=trial_config.lr)

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

		# checkpt once every 5000 epochs, or at the very end
		if (epoch % 5000 == 0) or (epoch == trial_config.epochs-1):
			checkpt = {
				"model": model.state_dict(),
				"opt": opt.state_dict(),
				"epoch": epoch,
				"train_losses": train_losses,
				"val_losses": val_losses,
				"run_id": trial_config.run_id,
			}

			torch.save(
				checkpt,
				trial_config.checkpt_path,
			)

			print(f"Epoch progress at checkpt: {epoch}/{trial_config.epochs}")

		run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

	run.finish()