# Measuring non-trivial compositionality

This repo contains the source code accompanying the paper [*Measuring non-trivial compositionality in emergent communication*](https://arxiv.org/abs/2010.15058) presented at NeurIPS 2020 workshop [Talking to Strangers: Zero-Shot Emergent Communication](https://sites.google.com/view/emecom2020).

All metrics and protocols implement a common API defined in `metrics.base.Metric` and `protocols.Protocol` and were designed to be reusable.

### Running

To install all the dependencies, run `pip install -r requirements.txt`. We assume Python 3.6+.

To reproduce the results and plots from the main experiment (how each metric scores each protocol), run `python experiment_1.py`.

To reproduce the results and plots from the experiment in appendix C (how the composition function in TRE affects compositionality scores), run `python experiment_2.py`.

To run init tests for the metrics, run `pytest`.

### Citing

```bibtex
@misc{korbak2020measuring,
      title={Measuring non-trivial compositionality in emergent communication}, 
      author={Tomasz Korbak and Julian Zubek and Joanna Rączaszek-Leonardi},
      year={2020},
      eprint={2010.15058},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```
