## Beyond Detection

Utilities and data generation code for the Beyond Detection project. The associated
project paper is a work in progress and attached separately.

## Repository layout

- `config/`: configuration files
- `src/`: Python package source
- `scripts/`: shell scripts for common runs
- `data/`: local datasets (gitignored) - but are available in HF
- `outputs/`: generated outputs and reports (gitignored, except `outputs/reports/`)

## Setup

1. Create and activate a Python 3.9+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

For editable installs:

```bash
pip install -e .
```

## Usage

Example run scripts live in `scripts/`. Adjust paths and config as needed:

```bash
bash scripts/run_sft.sh
```

## Notes

- This project is under active development; interfaces may change.
