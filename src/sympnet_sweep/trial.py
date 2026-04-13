import tempfile

from ray import tune
from sklearn.model_selection import train_test_split
from strupnet import SympNet
import torch
from torch.utils.data import DataLoader

from sympnet_sweep.config import PARAMS
from sympnet_sweep.utils import load_dataset, get_optimizer, SympNetDS

def setup_torch():
	import torch
	torch.set_default_dtype(torch.float64)
	torch.set_num_threads(1)
	torch.set_num_interop_threads(1)

def run_trial(config: dict) -> None:

	checkpt = None
	checkpoint = tune.get_checkpoint()
	if checkpoint:
		with checkpoint.as_directory() as checkpoint_dir:
			checkpt = torch.load(f"{checkpoint_dir}/checkpoint.pt", weights_only=True)
		start_epoch = checkpt["epoch"] + 1
		best_val_loss = checkpt["best_val_loss"]
	else:
		start_epoch = 0
		best_val_loss = float("inf")

	if start_epoch >= config["epochs"]: return # early termination to avoid extra inits.

	system_kwargs = PARAMS[config["system"]]["system"]
	dataset = load_dataset(
		system=config["system"],
		n_data=config["n_data"],
		h=config["h"],
		**system_kwargs,
	)

	train_ds, val_ds = train_test_split(dataset, test_size=config["val_size"], random_state=config["random_state"])

	train_dl = DataLoader(SympNetDS(train_ds), batch_size=config["batch_size"], shuffle=True, num_workers=0)
	val_dl = DataLoader(SympNetDS(val_ds), batch_size=config["batch_size"], shuffle=False, num_workers=0)

	model_kwargs = {
		"dim": config["dim"],
		"layers": config["layers"],
		"width": config["width"],
		"symmetric": config["symmetric"],
		"method": config["method"],
		"min_degree": config["min_degree"],
		"max_degree": config["max_degree"],
		"sublayers": config["sublayers"],
		"activation": config["activation"],
		"volume_step": config["volume_step"],
	}
	
	model = SympNet(**model_kwargs)
	opt = get_optimizer(config["optimizer"])(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"] or 0.0)

	if checkpoint:
		assert checkpt is not None
		model.load_state_dict(checkpt["model"])
		opt.load_state_dict(checkpt["opt"])

	for epoch in range(start_epoch, config["epochs"]):
		model.train()
		train_loss = 0.0

		for batch in train_dl:
			x_i, truth = batch[:, 0], batch[:, 1]
			opt.zero_grad()
			pred = model(x_i, dt=config["h"], symmetric=config["symmetric"])
			loss = ((pred-truth)**2).sum(dim=-1).mean() # avg l2 norms over batch
			loss.backward()
			opt.step()
			train_loss += loss.item()

		train_loss /= len(train_dl)

		model.eval()
		val_loss = 0.0

		with torch.no_grad():
			for batch in val_dl:
				x_i, truth = batch[:, 0], batch[:, 1]
				pred = model(x_i, dt=config["h"], symmetric=config["symmetric"])
				loss = ((pred-truth)**2).sum(dim=-1).mean()
				val_loss += loss.item()

		val_loss /= len(val_dl)

		# checkpt only on best val loss
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			with tempfile.TemporaryDirectory() as checkpoint_dir:
				torch.save(
					{
						"model": model.state_dict(),
						"opt": opt.state_dict(),
						"epoch": epoch,
						"best_val_loss": best_val_loss,
					},
					f"{checkpoint_dir}/checkpoint.pt"
				)
				tune.report(
					{"epoch": epoch, "val_loss": val_loss, "train_loss": train_loss},
					checkpoint=tune.Checkpoint.from_directory(checkpoint_dir),
				)
				print(f"New best at epoch {epoch+1}/{config['epochs']}: val_loss={val_loss:.6e}")
		elif epoch % 100 == 0:
			tune.report({"epoch": epoch, "val_loss": val_loss, "train_loss": train_loss})
