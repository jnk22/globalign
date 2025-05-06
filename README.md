# globalign

[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13-blue)](./pyproject.toml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

A library for fast FFT-computed global mutual information-based rigid alignment
using the GPU.

Related to the article (if you use this code, please cite it):

> Johan Öfverstedt, Joakim Lindblad, and Nataša Sladoje. Fast computation of
> mutual information in the frequency domain with applications to global
> multimodal image alignment. _Pattern Recognition Letters_, Vol. 159, pp.
> 196-203, 2022.
> [doi:10.1016/j.patrec.2022.05.022](https://doi.org/10.1016/j.patrec.2022.05.022)

**Preprint**: <https://arxiv.org/abs/2106.14699>

**Main author of the code**: Johan Öfverstedt

## Usage

To use the library, please see the included example script
[examples/example.py](examples/example.py).

To run the example, use the following commands:

```bash
# Create a new virtual environment and install all dependencies:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the globalign example:
python examples/example.py
```

## Learn2Reg 2024 — Reference solution for the [COMULISglobe SHG-BF](https://learn2reg.grand-challenge.org/learn2reg-2024/#task-3-comulisglobe-shg-bf) challenge

```bash
# Create a new virtual environment and install all dependencies:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r examples/Learn2Reg/requirements.txt

# Download the Dataset for 'TASK 3: COMULISglobe SHG-BF':
unzip COMULISSHGBF.zip

# Run globalign/CMIF registration using a rather coarse (fast) search:
python examples/Learn2Reg/COMULISSHGBF_2024.py
```

Validation displacement fields are saved to the directory `output`.
