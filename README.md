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
- **Possibly unused config keys**: each library's Python (and autonerves'
  own) is scanned for `conf.instance[...]` lookups, and every key of a
  library settings file is classed *used* (a lookup reads it, or reads a key
  under it — PyAutoLens reading a PyAutoFit key counts, since autonerves
  merges every layer), *section-read* (a lookup reads an ancestor section
  whole, or a non-literal subscript such as `["plots"][section][name]` ends
  the chain above it) or *unused* (nothing references it). Unused keys get a
  chip on the repo page, per-file counts and an index section grouped by
  library with GitHub links. Prior files are exempt (looked up by class name).
  **Limits:** the scan is static — it follows literal subscript chains
  (including multi-line ones and `.get("k")`), sections bound to a local and
  indexed later in the same function or a closure, and the `should_output`
  helper; a key read any other way (built key strings, `getattr`, a section
  passed to another function and indexed there) shows as section-read at
  best and may show as unused. Treat the list as candidates, not a verdict.
- **Environment variables**: the `PYAUTO_*` switches autonerves reads.
- **Cockpit feed** (`state.json`): green when every file parses and no
  workspace key is orphaned; yellow with one item per unparseable file or
  orphan-carrying workspace file; grey when nothing was collected. Never red.
  Possibly unused keys add one *info* item per library — never yellow, until
  the scan is trusted.

Rendered daily by [`.github/workflows/nerves_board.yml`](.github/workflows/nerves_board.yml)
from sparse checkouts of the config folders (plus each library's package
`.py` files, for the lookup scan); the renderer is
[`scripts/board.py`](scripts/board.py) (not part of the `autonerves` package).
Nothing on the board edits config.

## Links

- Source & tests: [`autonerves/`](autonerves), [`test_autonerves/`](test_autonerves)
- Agent/contributor instructions: [`AGENTS.md`](AGENTS.md)
- The organism this repo is the Nerves of:
  [PyAutoBrain/ORGANISM.md](https://github.com/PyAutoLabs/PyAutoBrain/blob/main/ORGANISM.md),
  documented in full at <https://pyautoscientist.readthedocs.io>
- Ecosystem: [PyAutoLabs on GitHub](https://github.com/PyAutoLabs)

## JAX compatibility

The package retains JAX/JAXlib `>=0.7,<0.12`, excluding `0.10.*` and `0.11.0`.
Those excluded releases contain CPU batched LAPACK scheduling that can deadlock
while materializing a vectorized likelihood (PyAutoHeart#274). The failure was
captured on 0.10.2; tagged source identifies the same path in the other excluded
releases. Open-source builds disable it from 0.11.1. This restriction does not
replace the separate historical FFT/Eigen workaround.

The compatibility diagnostics exercise **0.9.2** as an older endpoint and
**0.11.2** as the recommended validation baseline. They do not certify every
version permitted by the retained range or promise identical performance across
versions. A new minimum of 0.11.2 is deliberately avoided: it would unnecessarily
force newer NumPy/SciPy requirements on users with working older environments.

Updated family packages require `autonerves>2026.10.4.1` so their resolver cannot
fall back to a Nerves release without the exclusions. This is a release ordering
requirement: publish a Nerves wheel containing this policy before the corresponding
family wheels. No release version is chosen by this bound, and existing published
wheels are not retroactively repaired. An unpinned family package can still be
backtracked to an entirely older family release; select the repaired family
release explicitly when applying the fix to a constrained environment.

Keep JAX, JAXlib and any CUDA plugin on a mutually compatible version. Use pip's
resolver rather than `--no-deps`, and verify the resulting environment with
`python -m pip check`. Existing lockfiles need regeneration to adopt the policy.
