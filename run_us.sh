#!/bin/sh

python bayesian_denoising.py --img 2 --num_iter 50000 --gpu 0 --seed 1
python bayesian_denoising.py --img 2 --num_iter 50000 --gpu 0 --seed 2
python bayesian_denoising.py --img 2 --num_iter 50000 --gpu 0 --seed 3
