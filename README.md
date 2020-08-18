# Uncertainty Estimation in Medical Image Denoising with Bayesian Deep Image Prior

Max-Heinrich Laves, Malte Tölle, Tobias Ortmaier

Code for our paper accepted at the MICCAI workshop on Uncertainty for Safe Utilization of Machine Learning in Medical Imaging (UNSURE) 2020.

## Abstract

Uncertainty quantification in inverse medical imaging tasks with deep learning has received little attention.
However, deep models trained on large data sets tend to hallucinate and create artifacts in the reconstructed output that are not anatomically present.
We use a randomly initialized convolutional network as parameterization of the reconstructed image and perform gradient descent to match the observation, which is known as deep image prior.
In this case, the reconstruction does not suffer from hallucinations as no prior training is performed.
We further extend this to a Bayesian approach with Monte Carlo dropout to quantify both aleatoric and epistemic uncertainty.
The presented method is evaluated on the task of denoising different medical imaging modalities.
The experimental results show that our approach yields well-calibrated uncertainty.
That is, the predictive uncertainty correlates with the predictive error.
This allows for reliable uncertainty estimates and can tackle the problem of hallucinations and artifacts in inverse medical imaging tasks.

## BibTeX

```
\inproceedings{Laves2020MCDIP,
    authors = {Max-Heinrich Laves, Malte T{\"o}lle, Tobias Ortmaier},
    title = {Uncertainty Estimation in Medical Image Denoising with Bayesian Deep Image Prior},
    booktitle = {MICCAI UNSURE Workshop},
    year = {2020}}
```

## Contact

Max-Heinrich Laves  
[laves@imes.uni-hannover.de](mailto:laves@imes.uni-hannover.de)  
[@MaxLaves](https://twitter.com/MaxLaves)

Institute of Mechatronic Systems  
Leibniz Universität Hannover  
An der Universität 1, 30823 Garbsen, Germany
