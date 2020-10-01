# Measuring non-trivial compositionality

This repo contains the source code accompanying the paper *Measuring non-trivial compositionality in emergent communication* currently under review at NeurIPS 2020 workshop Talking to Strangers: Zero-Shot Emergent Communication.

All metrics and protocols implement a common API defined in `metrics.base.Metric` and `protocols.Protocol` and were designed to be reusable.

### Running

To install all the dependencies, run `pip install -r requirements.txt`. We assume Python 3.6+.

To reproduce the results and plots from the main experiment (how each metric scores each protocol), run `python experiment_1.py`.

To reproduce the results and plots from the experiment in appendix C (how the composition function in TRE affects compositionality scores), run run `python experiment_2.py`.
