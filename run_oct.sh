#!/bin/sh

python bayesian_denoising.py --img 3 --num_iter 50000 --gpu 0 --seed 4
python bayesian_denoising.py --img 3 --num_iter 50000 --gpu 0 --seed 5
python bayesian_denoising.py --img 3 --num_iter 50000 --gpu 0 --seed 6