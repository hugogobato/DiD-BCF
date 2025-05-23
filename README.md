# DiD-BCF

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

This is the Github Repository for the paper "Forests for Differences: Robust Causal Inference Beyond Parametric DiD".

## Paper Abstract

This paper introduces the Difference-in-Differences Bayesian Causal Forest (DiD-BCF), a novel non-parametric model addressing key challenges in DiD estimation, such as staggered adoption and heterogeneous treatment effects. DiD-BCF provides a unified framework for estimating Average (ATE), Group-Average (GATE), and Conditional Average Treatment Effects (CATE). A core innovation, its Parallel Trends Assumption (PTA)-based reparameterization, enhances estimation accuracy and stability in complex panel data settings. Extensive simulations demonstrate DiD-BCF's superior performance over established benchmarks, particularly under non-linearity, selection biases, and effect heterogeneity. Applied to U.S. minimum wage policy, the model uncovers significant conditional treatment effect heterogeneity related to county population, insights obscured by traditional methods. DiD-BCF offers a robust and versatile tool for more nuanced causal inference in modern DiD applications.


## Citation

```
@misc{https://doi.org/10.48550/arxiv.2505.09706,
  doi = {10.48550/ARXIV.2505.09706},
  url = {https://arxiv.org/abs/2505.09706},
  author = {Souto,  Hugo Gobato and Neto,  Francisco Louzada},
  keywords = {Methodology (stat.ME),  Machine Learning (cs.LG),  Machine Learning (stat.ML),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Forests for Differences: Robust Causal Inference Beyond Parametric DiD},
  publisher = {arXiv},
  year = {2025},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
