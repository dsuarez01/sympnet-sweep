import pathlib
import yaml

def find_project_root() -> pathlib.Path:
	path = pathlib.Path(__file__).resolve().parent
	while path != path.parent:
		if (path / "pyproject.toml").exists():
			return path
		path = path.parent
	raise RuntimeError("Could not find project root")

PROJECT_ROOT = find_project_root()

with open(PROJECT_ROOT / "configs/sweep.yaml") as f:
	PARAMS = yaml.safe_load(f)

SUPPORTED_SYSTEMS = list(PARAMS.keys())

OPTIMIZER_MAP = {
	"adam": "Adam",
	"sgd": "SGD",
}