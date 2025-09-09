# A Deep Learning Framework for Pulmonary Disease Classification Using Volume-Rendered CTs (2025)

[![CI](https://github.com/<you>/pulmonary-dl-ct-2025/actions/workflows/ci.yml/badge.svg)](https://github.com/<you>/pulmonary-dl-ct-2025/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.TBD.svg)](https://doi.org/10.5281/zenodo.TBD)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)](https://www.python.org/)

Reproducible artifact for the paper **A Deep Learning Framework for Pulmonary Disease Classification Using Volume-Rendered CTs**.

**Authors**: Noemi Maritza L. Romero, Ricco V. C. Soares, Mariana Recamonde-Mendoza, João L. D. Comba  
**Affiliation**: Instituto de Informática, UFRGS

## Quick Start

```bash
conda env create -f environment.yml && conda activate pulmonet
pip install -e .
python scripts/run_demo.py
pytest -q
pytest --nbval-lax notebooks/00_quickstart.ipynb
```

## Citing

```bibtex
@software{pulmonet-2025,
  title        = {A Deep Learning Framework for Pulmonary Disease Classification Using Volume-Rendered CTs},
  author       = {Noemi Maritza L. Romero, Ricco V. C. Soares, Mariana Recamonde-Mendoza, João L. D. Comba},
  year         = {2025},
  url          = {https://github.com/<you>/pulmonary-dl-ct-2025}
}
```

## License
Apache License 2.0 (see `LICENSE`).
