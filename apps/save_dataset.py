import argparse

from sympnet_sweep.utils import save_dataset

SUPPORTED_SYSTEMS = ["duffing"]

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--system", required=True, help="System name")
	args = parser.parse_args()
	assert args.system in SUPPORTED_SYSTEMS, "Unsupported system"

	save_dataset(system="duffing", n_data=100, h=0.01, F=0.3, omega=1.0)
	save_dataset(system="duffing", n_data=200, h=0.1, F=0.3, omega=1.0)
	save_dataset(system="duffing", n_data=400, h=1.0, F=0.3, omega=1.0)