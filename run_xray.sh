#!/bin/sh

python bayesian_denoising.py --img 1 --num_iter 50000 --gpu 1 --seed 1
python bayesian_denoising.py --img 1 --num_iter 50000 --gpu 1 --seed 2
python bayesian_denoising.py --img 1 --num_iter 50000 --gpu 1 --seed 3
