<p align="center">
  <img src="logo.png" alt="PyAutoNerves" width="400">
</p>

# PyAutoNerves

[![PyAutoScientist GitHub](https://img.shields.io/badge/%E2%9A%97%EF%B8%8F%20PyAutoScientist-GitHub-181717?style=flat-square)](https://github.com/PyAutoLabs/PyAutoScientist) [![PyAutoScientist ReadTheDocs](https://img.shields.io/badge/%F0%9F%93%96%20PyAutoScientist-ReadTheDocs-8CA1AF?style=flat-square)](https://pyautoscientist.readthedocs.io)

**PyAutoNerves is the Nerves of the PyAutoScientist** — the configuration,
serialization, and I/O foundation (package `autonerves`) connecting the
workspace's conventions to every library. It provides a layered configuration
system with workspace overrides, dict / JSON / CSV serialization of arbitrary
objects, and FITS I/O.

`PyAutoFit`, `PyAutoArray`, `PyAutoGalaxy`, and `PyAutoLens` all depend on
autonerves: it supplies their packaged default config, the object-serialization
used to persist models and results, and shared utilities (`test_mode`,
`jax_wrapper`). Centralising these here keeps a single, consistent config and
I/O layer beneath every library. As a released library it ships with every
nightly train — see the
[PyAutoHands release board](https://pyautolabs.github.io/PyAutoHands/) for the
current version.

## Install

```bash
pip install autonerves
```

## Examples

Layered config — read a directory of YAML into a queryable `Config`:

```python
from autonerves.conf import Config

config = Config("path/to/config")          # directory of YAML files
value = config["general"]["model"]["section"]["value"]
```

JSON serialization — round-trip arbitrary Python objects:

```python
from autonerves.dictable import output_to_json, from_json

data = {"sersic_index": 4.0, "centre": [0.0, 0.0]}
output_to_json(data, "model.json")
restored = from_json("model.json")         # == data
```

FITS I/O — write and read a NumPy array:

```python
import numpy as np
from autonerves.fitsable import output_to_fits, ndarray_via_fits_from

arr = np.arange(12.0).reshape(3, 4)
output_to_fits(values=arr, file_path="demo.fits", overwrite=True)
loaded = ndarray_via_fits_from(file_path="demo.fits", hdu=0)   # np.allclose(arr, loaded)
```

## Nerves board

**<https://pyautolabs.github.io/PyAutoNerves/>** — a read-only browser of every
config file and option across the organism: each library's `<package>/config/`
(PyAutoFit, PyAutoArray, PyAutoGalaxy, PyAutoLens, PyAutoCTI) and each
workspace's `config/` that overrides them.

- **Per repo** (`repos/<Repo>.html`): every YAML file with its top-level keys,
  its source line-numbered with the comments kept (the comments *are* the
  option docs) and a GitHub link; prior files render as a table
  (Class · param · type · mean/σ or bounds · width modifier · limits).
- **Search**: the index page's box searches every key, file and comment
  across all repos and jumps to the line.
- **Override map**: autonerves resolves a key workspace → last-imported
  library → … → PyAutoFit (`Config.push(keep_first=True)`, keys lowercased),
  so each workspace file is compared, by relative path, against the same file
  across its library stack in that order (lens → galaxy → array → fit; cti →
  array → fit). Keys whose value differs, keys no library defines (*orphans*)
  and keys that fall through to the libraries are listed per file.
  `build/*.yaml` is workspace tooling and is grouped apart.
- **Environment variables**: the `PYAUTO_*` switches autonerves reads.
- **Cockpit feed** (`state.json`): green when every file parses and no
  workspace key is orphaned; yellow with one item per unparseable file or
  orphan-carrying workspace file; grey when nothing was collected. Never red.

Rendered daily by [`.github/workflows/nerves_board.yml`](.github/workflows/nerves_board.yml)
from sparse checkouts of the config folders; the renderer is
[`scripts/board.py`](scripts/board.py) (not part of the `autonerves` package).
Nothing on the board edits config.

## Links

- Source & tests: [`autonerves/`](autonerves), [`test_autonerves/`](test_autonerves)
- Agent/contributor instructions: [`AGENTS.md`](AGENTS.md)
- The organism this repo is the Nerves of:
  [PyAutoBrain/ORGANISM.md](https://github.com/PyAutoLabs/PyAutoBrain/blob/main/ORGANISM.md),
  documented in full at <https://pyautoscientist.readthedocs.io>
- Ecosystem: [PyAutoLabs on GitHub](https://github.com/PyAutoLabs)
