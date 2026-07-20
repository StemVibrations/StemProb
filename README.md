# StemProb

Probabilistic and sensitivity analysis tools built on top of [STEM](https://github.com/StemVibrations/STEM), for studying how uncertain soil, load and damping parameters affect railway-induced vibration predictions.

## Getting started

Start with the tutorial: [`docs/tutorial_sensitivity.rst`](docs/tutorial_sensitivity.rst). It walks through building a small 3D embankment model, then applies three methods to it:

- **Uncertainty quantification** -- Monte Carlo / Latin Hypercube sampling and Random Fields, to see the distribution of a predicted response.
- **Sensitivity analysis** -- the Morris method (and RBD-FAST as an alternative), to screen which parameters matter most.

Each chapter's code blocks are complete and can be pasted into a single script, or run directly from the matching file in `docs/` (e.g. `tutorial_sensitivity_morris.py`).

## Repository structure

- `docs/` -- the tutorial and the scripts it's built from.
- `legacy/` -- earlier exploratory work, kept for reference but not maintained.

## Installation

Install dependencies with `pip install -r requirements.txt` (includes STEM and its Kratos wheels). See the tutorial for STEM-specific setup notes.
