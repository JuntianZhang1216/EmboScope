"""Locate the checkout and its fixed, user-populated data directory."""
from pathlib import Path


def project_root():
    checkout = Path(__file__).resolve().parents[2]
    current = Path.cwd().resolve()
    for root in (checkout, current, *current.parents):
        if ((root / 'evaluate.sh').is_file()
                and (root / 'pyproject.toml').is_file()
                and (root / 'eval/embodied_eval').is_dir()):
            return root
    raise ValueError('Run this command from the EmboScope checkout, or invoke its evaluate.sh by path.')


def dataset_path(root):
    root = Path(root)
    dataset = root / 'data/benchmark_data.jsonl'
    if not dataset.is_file():
        raise ValueError(f'Missing data/benchmark_data.jsonl; download the question JSONL into {root / "data"}.')
    if not (root / 'data/images').is_dir():
        raise ValueError(f'Missing data/images/; extract the complete media archive into {root / "data"}.')
    return dataset
