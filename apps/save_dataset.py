import argparse

from sympnet_sweep.config import PARAMS, SUPPORTED_SYSTEMS
from sympnet_sweep.utils import save_dataset

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--system", required=True, choices=SUPPORTED_SYSTEMS, help="System name")
	args = parser.parse_args()

	save_dataset(system=args.system, n_data=100, h=0.01, **PARAMS[args.system]["system"])
	save_dataset(system=args.system, n_data=200, h=0.1, **PARAMS[args.system]["system"])
	save_dataset(system=args.system, n_data=400, h=1.0, **PARAMS[args.system]["system"])