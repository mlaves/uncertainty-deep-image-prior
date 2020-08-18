# Max-Heinrich Laves
# Institute of Mechatronic Systems
# Leibniz Universität Hannover, Germany
# 2020

import matplotlib
matplotlib.use('Agg')
import fire
import matplotlib.pyplot as plt
import os
import numpy as np
from models import get_net
import torch
import torch.optim
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import get_noisy_image_gaussian
from utils.bayesian_utils import gaussian_nll
from utils.common_utils import init_normal, crop_image, get_image, pil_to_np, np_to_pil, plot_image_grid, get_noise, get_params, optimize, np_to_torch, torch_to_np
from utils.uce import uceloss
from utils.calibration_plots import plot_uncert
import time


def main(img: int=0, num_iter: int=40000, lr: float=3e-4, gpu: int=0, seed: int=42, save: bool=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    global i, out_avg, psnr_noisy_last, last_net, net_input, losses, psnrs, ssims, average_dropout_rate, no_layers, \
           img_mean, sample_count, recons, uncerts, uncerts_ale, loss_last, roll_back

    imsize = (256, 256)
    PLOT = True

    timestamp = int(time.time())
    save_path = '/media/fastdata/laves/unsure'
    os.mkdir(f'{save_path}/{timestamp}')

    # denoising
    if img == 0:
        fname = '../NORMAL-4951060-8.jpeg'
        imsize = (256, 256)
    elif img == 1:
        fname = '../BACTERIA-1351146-0006.png'
        imsize = (256, 256)
    elif img == 2:
        fname = '../081_HC.png'
        imsize = (256, 256)
    elif img == 3:
        fname = '../CNV-9997680-30.png'
        imsize = (256, 256)
    else:
        assert False

    if fname == '../NORMAL-4951060-8.jpeg':

        # Add Gaussian noise to simulate speckle
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        print(img_np.shape)
        p_sigma = 0.1
        img_noisy_pil, img_noisy_np = get_noisy_image_gaussian(img_np, p_sigma)

    elif fname == '../BACTERIA-1351146-0006.png':

        # Add Poisson noise to simulate low dose X-ray
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        print(img_np.shape)
        #p_lambda = 50.0
        #img_noisy_pil, img_noisy_np = get_noisy_image_poisson(img_np, p_lambda)
        # for lam > 20, poisson can be approximated with Gaussian
        p_sigma = 0.1
        img_noisy_pil, img_noisy_np = get_noisy_image_gaussian(img_np, p_sigma)

    elif fname == '../081_HC.png':

        # Add Gaussian noise to simulate speckle
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        print(img_np.shape)
        p_sigma = 0.1
        img_noisy_pil, img_noisy_np = get_noisy_image_gaussian(img_np, p_sigma)

    elif fname == '../CNV-9997680-30.png':

        # Add Gaussian noise to simulate speckle
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        print(img_np.shape)
        p_sigma = 0.1
        img_noisy_pil, img_noisy_np = get_noisy_image_gaussian(img_np, p_sigma)

    else:
        assert False

    if PLOT:
            q = plot_image_grid([img_np, img_noisy_np], 4, 6)
            out_pil = np_to_pil(q)
            out_pil.save(f'{save_path}/{timestamp}/input.png', 'PNG')

    INPUT = 'noise'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    reg_noise_std = 1./10.
    LR = lr
    roll_back = False  # To solve the oscillation of model training
    input_depth = 32

    show_every = 100
    exp_weight = 0.99

    mse = torch.nn.MSELoss()

    img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)


    LOSSES = {}
    RECONS = {}
    UNCERTS = {}
    UNCERTS_ALE = {}
    PSNRS = {}
    SSIMS = {}

    # # SGD

    OPTIMIZER = 'adamw'
    weight_decay = 0
    LOSS = 'mse'
    figsize = 4

    NET_TYPE = 'skip'

    skip_n33d = 128
    skip_n33u = 128
    skip_n11 = 4
    num_scales = 5
    upsample_mode = 'bilinear'

    dropout_mode_down = 'None'
    dropout_p_down = 0.0
    dropout_mode_up = 'None'
    dropout_p_up = dropout_p_down
    dropout_mode_skip = 'None'
    dropout_p_skip = dropout_p_down
    dropout_mode_output = 'None'
    dropout_p_output = dropout_p_down

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    out_avg = None
    last_net = None
    mc_iter = 1

    def closure_dip():

        global i, out_avg, psnr_noisy_last, last_net, net_input, losses, psnrs, ssims, average_dropout_rate, no_layers,\
               img_mean, sample_count, recons, uncerts, loss_last

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)
        out[:,:1] = out[:,:1].sigmoid()

        _loss = mse(out[:,:1], img_noisy_torch)
        _loss.backward()

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        losses.append(mse(out_avg[:,:1], img_noisy_torch).item())

        _out = out.detach().cpu().numpy()[0,:1]
        _out_avg = out_avg.detach().cpu().numpy()[0,:1]

        psnr_noisy = compare_psnr(img_noisy_np, _out)
        psnr_gt    = compare_psnr(img_np, _out)
        psnr_gt_sm = compare_psnr(img_np, _out_avg)

        ssim_noisy = compare_ssim(img_noisy_np[0], _out[0])
        ssim_gt    = compare_ssim(img_np[0], _out[0])
        ssim_gt_sm = compare_ssim(img_np[0], _out_avg[0])

        psnrs.append([psnr_noisy, psnr_gt, psnr_gt_sm])
        ssims.append([ssim_noisy, ssim_gt, ssim_gt_sm])

        if  PLOT and i % show_every == 0:
            print(f'Iteration: {i} Loss: {_loss.item():.4f} PSNR_noisy: {psnr_noisy:.4f} PSRN_gt: {psnr_gt:.4f} PSNR_gt_sm: {psnr_gt_sm:.4f}')

            out_np = _out

            psnr_noisy = compare_psnr(img_noisy_np, out_np)
            psnr_gt    = compare_psnr(img_np, out_np)

            if sample_count != 0:
                psnr_mean = compare_psnr(img_np, img_mean / sample_count)
            else:
                psnr_mean = 0

            print('###################')

            recons.append(out_np)

        i += 1

        return _loss

    if '../NORMAL-4951060-8.jpeg':
        net = get_net(input_depth, NET_TYPE, pad,
                      skip_n33d=skip_n33d,
                      skip_n33u=skip_n33u,
                      skip_n11=skip_n11,
                      num_scales=num_scales,
                      n_channels=1,
                      upsample_mode=upsample_mode,
                      dropout_mode_down=dropout_mode_down,
                      dropout_p_down=dropout_p_down,
                      dropout_mode_up=dropout_mode_up,
                      dropout_p_up=dropout_p_up,
                      dropout_mode_skip=dropout_mode_skip,
                      dropout_p_skip=dropout_p_skip,
                      dropout_mode_output=dropout_mode_output,
                      dropout_p_output=dropout_p_output).type(dtype)
    else:
        assert False

    net.apply(init_normal)

    losses = []
    recons = []
    uncerts = []
    uncerts_ale = []
    psnrs = []
    ssims = []

    img_mean = 0
    sample_count = 0
    i = 0
    psnr_noisy_last = 0
    loss_last = 1e16

    parameters = get_params(OPT_OVER, net, net_input)
    out_avg = None
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=weight_decay)
    optimize(optimizer, closure_dip, num_iter)

    LOSSES['dip'] = losses
    RECONS['dip'] = recons
    UNCERTS['dip'] = uncerts
    UNCERTS_ALE['dip'] = uncerts_ale
    PSNRS['dip'] = psnrs
    SSIMS['dip'] = ssims

    to_plot = [img_np] + [np.clip(img, 0, 1) for img in RECONS['dip']]
    q = plot_image_grid(to_plot, factor=13)

    out_pil = np_to_pil(q)
    out_pil.save(f'{save_path}/{timestamp}/dip_recons.png', 'PNG')

    ## SGLD

    weight_decay = 1e-4
    LOSS = 'mse'
    input_depth = 32
    param_noise_sigma = 2

    NET_TYPE = 'skip'

    skip_n33d = 128
    skip_n33u = 128
    skip_n11 = 4
    num_scales = 5
    upsample_mode = 'bilinear'

    dropout_mode_down = 'None'
    dropout_p_down = 0.0
    dropout_mode_up = 'None'
    dropout_p_up = dropout_p_down
    dropout_mode_skip = 'None'
    dropout_p_skip = dropout_p_down
    dropout_mode_output = 'None'
    dropout_p_output = dropout_p_down

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    mc_iter = 25

    def add_noise(model):
        for n in [x for x in model.parameters() if len(x.size()) == 4]:
            noise = torch.randn(n.size())*param_noise_sigma*LR
            noise = noise.type(dtype)
            n.data = n.data + noise

    def closure_sgld():

        global i, out_avg, psnr_noisy_last, last_net, net_input, losses, psnrs, ssims, average_dropout_rate, no_layers, img_mean, sample_count, recons, uncerts, loss_last

        add_noise(net)

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)
        out[:,:1] = out[:,:1].sigmoid()

        _loss = mse(out[:,:1], img_noisy_torch)
        _loss.backward()

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        losses.append(mse(out_avg[:,:1], img_noisy_torch).item())

        _out = out.detach().cpu().numpy()[0,:1]
        _out_avg = out_avg.detach().cpu().numpy()[0,:1]

        psnr_noisy = compare_psnr(img_noisy_np, _out)
        psnr_gt    = compare_psnr(img_np, _out)
        psnr_gt_sm = compare_psnr(img_np, _out_avg)

        ssim_noisy = compare_ssim(img_noisy_np[0], _out[0])
        ssim_gt    = compare_ssim(img_np[0], _out[0])
        ssim_gt_sm = compare_ssim(img_np[0], _out_avg[0])

        psnrs.append([psnr_noisy, psnr_gt, psnr_gt_sm])
        ssims.append([ssim_noisy, ssim_gt, ssim_gt_sm])

        if  PLOT and i % show_every == 0:
            print(f'Iteration: {i} Loss: {_loss.item():.4f} PSNR_noisy: {psnr_noisy:.4f} PSRN_gt: {psnr_gt:.4f} PSNR_gt_sm: {psnr_gt_sm:.4f}')

            out_np = _out
            recons.append(out_np)

            out_np_var = np.var(np.array(recons[-mc_iter:]), axis=0)[:1]

            print('mean epi', out_np_var.mean())
            print('###################')

            uncerts.append(out_np_var)

        i += 1

        return _loss

    if '../NORMAL-4951060-8.jpeg':
        net = get_net(input_depth, NET_TYPE, pad,
                      skip_n33d=skip_n33d,
                      skip_n33u=skip_n33u,
                      skip_n11=skip_n11,
                      num_scales=num_scales,
                      n_channels=1,
                      upsample_mode=upsample_mode,
                      dropout_mode_down=dropout_mode_down,
                      dropout_p_down=dropout_p_down,
                      dropout_mode_up=dropout_mode_up,
                      dropout_p_up=dropout_p_up,
                      dropout_mode_skip=dropout_mode_skip,
                      dropout_p_skip=dropout_p_skip,
                      dropout_mode_output=dropout_mode_output,
                      dropout_p_output=dropout_p_output).type(dtype)
    else:
        assert False

    net.apply(init_normal)

    losses = []
    recons = []
    uncerts = []
    uncerts_ale = []
    psnrs = []
    ssims = []

    img_mean = 0
    sample_count = 0
    i = 0
    psnr_noisy_last = 0
    loss_last = 1e10
    out_avg = None
    last_net = None

    parameters = get_params(OPT_OVER, net, net_input)
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=weight_decay)
    optimize(optimizer, closure_sgld, num_iter)

    LOSSES['sgld'] = losses
    RECONS['sgld'] = recons
    UNCERTS['sgld'] = uncerts
    UNCERTS_ALE['sgld'] = uncerts_ale
    PSNRS['sgld'] = psnrs
    SSIMS['sgld'] = ssims

    to_plot = [img_np] + [np.clip(img, 0, 1) for img in RECONS['sgld']]
    q = plot_image_grid(to_plot, factor=13)

    out_pil = np_to_pil(q)
    out_pil.save(f'{save_path}/{timestamp}/sgld_recons.png', 'PNG')

    errs = img_noisy_torch.cpu()-torch.tensor(RECONS['sgld'][-1])
    uncerts_epi = torch.tensor(UNCERTS['sgld'][-1]).unsqueeze(0)
    uncerts = uncerts_epi
    uce, err, uncert, freq = uceloss(errs**2, uncerts, n_bins=21)
    fig, ax = plot_uncert(err, uncert, freq, outlier_freq=0.001)
    ax.set_title(f'U = {uncerts.mean().sqrt().item():.4f}, UCE = {uce.item()*100:.3f}')
    plt.tight_layout()
    fig.savefig(f'{save_path}/{timestamp}/sgld_calib.png')

    ## SGLD + NLL

    LOSS = 'nll'

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    def closure_sgldnll():

        global i, out_avg, psnr_noisy_last, last_net, net_input, losses, psnrs, ssims, average_dropout_rate, no_layers,\
               img_mean, sample_count, recons, uncerts, uncerts_ale, loss_last

        add_noise(net)

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)
        out[:,:1] = out[:,:1].sigmoid()

        _loss = gaussian_nll(out[:,:1], out[:,1:], img_noisy_torch)
        _loss.backward()

        out[:,1:] = torch.exp(-out[:,1:])  # aleatoric uncertainty

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        with torch.no_grad():
            mse_loss = mse(out_avg[:, :1], img_noisy_torch).item()

        losses.append(mse_loss)

        _out = out.detach().cpu().numpy()[0,:1]
        _out_avg = out_avg.detach().cpu().numpy()[0,:1]

        psnr_noisy = compare_psnr(img_noisy_np, _out)
        psnr_gt    = compare_psnr(img_np, _out)
        psnr_gt_sm = compare_psnr(img_np, _out_avg)

        ssim_noisy = compare_ssim(img_noisy_np[0], _out[0])
        ssim_gt    = compare_ssim(img_np[0], _out[0])
        ssim_gt_sm = compare_ssim(img_np[0], _out_avg[0])

        psnrs.append([psnr_noisy, psnr_gt, psnr_gt_sm])
        ssims.append([ssim_noisy, ssim_gt, ssim_gt_sm])

        if  PLOT and i % show_every == 0:
            print(f'Iteration: {i} Loss: {_loss.item():.4f} PSNR_noisy: {psnr_noisy:.4f} PSRN_gt: {psnr_gt:.4f} PSNR_gt_sm: {psnr_gt_sm:.4f}')

            out_np = _out
            recons.append(out_np)
            out_np_ale = out.detach().cpu().numpy()[0,1:]
            out_np_var = np.var(np.array(recons[-mc_iter:]), axis=0)[:1]

            print('mean epi', out_np_var.mean())
            print('mean ale', out_np_ale.mean())
            print('###################')

            uncerts.append(out_np_var)
            uncerts_ale.append(out_np_ale)

        i += 1

        return _loss

    if '../NORMAL-4951060-8.jpeg':
        net = get_net(input_depth, NET_TYPE, pad,
                      skip_n33d=skip_n33d,
                      skip_n33u=skip_n33u,
                      skip_n11=skip_n11,
                      num_scales=num_scales,
                      n_channels=2,
                      upsample_mode=upsample_mode,
                      dropout_mode_down=dropout_mode_down,
                      dropout_p_down=dropout_p_down,
                      dropout_mode_up=dropout_mode_up,
                      dropout_p_up=dropout_p_up,
                      dropout_mode_skip=dropout_mode_skip,
                      dropout_p_skip=dropout_p_skip,
                      dropout_mode_output=dropout_mode_output,
                      dropout_p_output=dropout_p_output).type(dtype)
    else:
        assert False

    net.apply(init_normal)

    losses = []
    recons = []
    uncerts = []
    uncerts_ale = []
    psnrs = []
    ssims = []

    img_mean = 0
    sample_count = 0
    i = 0
    psnr_noisy_last = 0
    loss_last = 1e6
    out_avg = None
    last_net = None

    parameters = get_params(OPT_OVER, net, net_input)
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=weight_decay)
    optimize(optimizer, closure_sgldnll, num_iter)

    LOSSES['sgldnll'] = losses
    RECONS['sgldnll'] = recons
    UNCERTS['sgldnll'] = uncerts
    UNCERTS_ALE['sgldnll'] = uncerts_ale
    PSNRS['sgldnll'] = psnrs
    SSIMS['sgldnll'] = ssims

    to_plot = [img_np] + [np.clip(img, 0, 1) for img in RECONS['sgldnll']]
    q = plot_image_grid(to_plot, factor=13)

    out_pil = np_to_pil(q)
    out_pil.save(f'{save_path}/{timestamp}/sgldnll_recons.png', 'PNG')

    errs = img_noisy_torch.cpu()-torch.tensor(RECONS['sgldnll'][-1])
    uncerts_epi = torch.tensor(UNCERTS['sgldnll'][-1]).unsqueeze(0)
    uncerts_ale = torch.tensor(UNCERTS_ALE['sgldnll'][-1]).unsqueeze(0)
    uncerts = uncerts_epi + uncerts_ale
    uce, err, uncert, freq = uceloss(errs**2, uncerts, n_bins=21)
    fig, ax = plot_uncert(err, uncert, freq, outlier_freq=0.001)
    ax.set_title(f'U = {uncerts.mean().sqrt().item():.4f}, UCE = {uce.item()*100:.3f}')
    plt.tight_layout()
    fig.savefig(f'{save_path}/{timestamp}/sgldnll_calib.png')

    errs = torch.tensor(img_np).unsqueeze(0)-torch.tensor(RECONS['sgldnll'][-1])
    uncerts_epi = torch.tensor(UNCERTS['sgldnll'][-1]).unsqueeze(0)
    uncerts_ale = torch.tensor(UNCERTS_ALE['sgldnll'][-1]).unsqueeze(0)
    uncerts = uncerts_epi + uncerts_ale
    uce, err, uncert, freq = uceloss(errs**2, uncerts, n_bins=21)
    fig, ax = plot_uncert(err, uncert, freq, outlier_freq=0.001)
    ax.set_title(f'U = {uncerts.mean().sqrt().item():.4f}, UCE = {uce.item()*100:.3f}')
    plt.tight_layout()
    fig.savefig(f'{save_path}/{timestamp}/sgldnll_calib2.png')

    ## MCDIP

    OPTIMIZER = 'adamw'
    weight_decay = 1e-4
    LOSS = 'nll'
    input_depth = 32
    figsize = 4

    NET_TYPE = 'skip'

    skip_n33d = 128
    skip_n33u = 128
    skip_n11 = 4
    num_scales = 5
    upsample_mode = 'bilinear'

    dropout_mode_down = '2d'
    dropout_p_down = 0.3
    dropout_mode_up = '2d'
    dropout_p_up = dropout_p_down
    dropout_mode_skip = 'None'
    dropout_p_skip = dropout_p_down
    dropout_mode_output = 'None'
    dropout_p_output = dropout_p_down

    net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    mc_iter = 25

    def closure_mcdip():

        global i, out_avg, psnr_noisy_last, last_net, net_input, losses, psnrs, ssims, average_dropout_rate, no_layers,\
               img_mean, sample_count, recons, uncerts, uncerts_ale, loss_last

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)
        out[:,:1] = out[:,:1].sigmoid()

        _loss = gaussian_nll(out[:,:1], out[:,1:], img_noisy_torch)
        _loss.backward()

        out[:,1:] = torch.exp(-out[:,1:])  # aleatoric uncertainty

        # Smoothing
        if out_avg is None:
            out_avg = out.detach()
        else:
            out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

        losses.append(mse(out_avg[:,:1], img_noisy_torch).item())

        _out = out.detach().cpu().numpy()[0,:1]
        _out_avg = out_avg.detach().cpu().numpy()[0,:1]

        psnr_noisy = compare_psnr(img_noisy_np, _out)
        psnr_gt    = compare_psnr(img_np, _out)
        psnr_gt_sm = compare_psnr(img_np, _out_avg)

        ssim_noisy = compare_ssim(img_noisy_np[0], _out[0])
        ssim_gt    = compare_ssim(img_np[0], _out[0])
        ssim_gt_sm = compare_ssim(img_np[0], _out_avg[0])

        psnrs.append([psnr_noisy, psnr_gt, psnr_gt_sm])
        ssims.append([ssim_noisy, ssim_gt, ssim_gt_sm])

        if  PLOT and i % show_every == 0:
            print(f'Iteration: {i} Loss: {_loss.item():.4f} PSNR_noisy: {psnr_noisy:.4f} PSRN_gt: {psnr_gt:.4f} PSNR_gt_sm: {psnr_gt_sm:.4f}')

            img_list = []
            aleatoric_list = []

            with torch.no_grad():
                net_input = net_input_saved + (noise.normal_() * reg_noise_std)

                for _ in range(mc_iter):
                    img = net(net_input)
                    img[:,:1] = img[:,:1].sigmoid()
                    img[:,1:] = torch.exp(-img[:,1:])
                    img_list.append(torch_to_np(img[:1]))
                    aleatoric_list.append(torch_to_np(img[:,1:]))

            img_list_np = np.array(img_list)
            out_np = np.mean(img_list_np, axis=0)[:1]
            out_np_ale = np.mean(aleatoric_list, axis=0)[:1]
            out_np_var = np.var(img_list_np, axis=0)[:1]

            psnr_noisy = compare_psnr(img_noisy_np, out_np)
            psnr_gt    = compare_psnr(img_np, out_np)

            print('mean epi', out_np_var.mean())
            print('mean ale', out_np_ale.mean())
            print('###################')

            recons.append(out_np)
            uncerts.append(out_np_var)
            uncerts_ale.append(out_np_ale)

        i += 1

        return _loss

    if '../NORMAL-4951060-8.jpeg':
        net = get_net(input_depth, NET_TYPE, pad,
                      skip_n33d=skip_n33d,
                      skip_n33u=skip_n33u,
                      skip_n11=skip_n11,
                      num_scales=num_scales,
                      n_channels=2,
                      upsample_mode=upsample_mode,
                      dropout_mode_down=dropout_mode_down,
                      dropout_p_down=dropout_p_down,
                      dropout_mode_up=dropout_mode_up,
                      dropout_p_up=dropout_p_up,
                      dropout_mode_skip=dropout_mode_skip,
                      dropout_p_skip=dropout_p_skip,
                      dropout_mode_output=dropout_mode_output,
                      dropout_p_output=dropout_p_output).type(dtype)
    else:
        assert False

    net.apply(init_normal)

    losses = []
    recons = []
    uncerts = []
    uncerts_ale = []
    psnrs = []
    ssims = []

    img_mean = 0
    sample_count = 0
    i = 0
    psnr_noisy_last = 0
    loss_last = 1e16
    out_avg = None
    last_net = None

    parameters = get_params(OPT_OVER, net, net_input)
    optimizer = torch.optim.AdamW(parameters, lr=LR, weight_decay=weight_decay)
    optimize(optimizer, closure_mcdip, num_iter)

    LOSSES['mcdip'] = losses
    RECONS['mcdip'] = recons
    UNCERTS['mcdip'] = uncerts
    UNCERTS_ALE['mcdip'] = uncerts_ale
    PSNRS['mcdip'] = psnrs
    SSIMS['mcdip'] = ssims


    # In[75]:


    to_plot = [img_np] + [np.clip(img, 0, 1) for img in RECONS['mcdip']]
    q = plot_image_grid(to_plot, factor=13)

    out_pil = np_to_pil(q)
    out_pil.save(f'{save_path}/{timestamp}/mcdip_recons.png', 'PNG')


    # In[85]:


    errs = img_noisy_torch.cpu()-torch.tensor(RECONS['mcdip'][-1])
    uncerts_epi = torch.tensor(UNCERTS['mcdip'][-1]).unsqueeze(0)
    uncerts_ale = torch.tensor(UNCERTS_ALE['mcdip'][-1]).unsqueeze(0)
    uncerts = uncerts_epi + uncerts_ale
    uce, err, uncert, freq = uceloss(errs**2, uncerts, n_bins=21)
    fig, ax = plot_uncert(err, uncert, freq, outlier_freq=0.001)
    ax.set_title(f'U = {uncerts.mean().sqrt().item():.4f}, UCE = {uce.item()*100:.3f}')
    plt.tight_layout()
    fig.savefig(f'{save_path}/{timestamp}/mcdip_calib.png')


    # In[86]:


    errs = torch.tensor(img_np).unsqueeze(0)-torch.tensor(RECONS['mcdip'][-1])
    uncerts_epi = torch.tensor(UNCERTS['mcdip'][-1]).unsqueeze(0)
    uncerts_ale = torch.tensor(UNCERTS_ALE['mcdip'][-1]).unsqueeze(0)
    uncerts = uncerts_epi + uncerts_ale
    uce, err, uncert, freq = uceloss(errs**2, uncerts, n_bins=21)
    fig, ax = plot_uncert(err, uncert, freq, outlier_freq=0.001)
    ax.set_title(f'U = {uncerts.mean().sqrt().item():.4f}, UCE = {uce.item()*100:.3f}')
    plt.tight_layout()
    fig.savefig(f'{save_path}/{timestamp}/mcdip_calib2.png')


    fig, ax0 = plt.subplots(1, 1)

    for key, loss in LOSSES.items():
        ax0.plot(range(len(loss)), loss, label=key)
        ax0.set_title('MSE')
        ax0.set_xlabel('iteration')
        ax0.set_ylabel('mse loss')
        ax0.set_ylim(0,0.03)
        ax0.grid(True)
        ax0.legend()

    plt.tight_layout()
    plt.savefig(f'{save_path}/{timestamp}/losses.png')
    plt.show()


    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    labels = ["psnr_noisy", "psnr_gt", "psnr_gt_sm"]

    for key, psnr in PSNRS.items():
        psnr = np.array(psnr)
        for i in range(psnr.shape[1]):
            axs[i].plot(range(psnr.shape[0]), psnr[:,i], label=key)
            axs[i].set_title(labels[i])
            axs[i].set_xlabel('iteration')
            axs[i].set_ylabel('psnr')
            axs[i].legend()

    plt.savefig(f'{save_path}/{timestamp}/psnrs.png')
    plt.show()


    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    labels = ["ssim_noisy", "ssim_gt", "ssim_gt_sm"]

    for key, ssim in SSIMS.items():
        ssim = np.array(ssim)
        for i in range(ssim.shape[1]):
            axs[i].plot(range(ssim.shape[0]), ssim[:,i], label=key)
            axs[i].set_title(labels[i])
            axs[i].set_xlabel('iteration')
            axs[i].legend()
            axs[i].set_ylabel('ssim')

    plt.savefig(f'{save_path}/{timestamp}/ssims.png')
    plt.show()

    # save stuff for plotting
    if save:
        np.savez(f"{save_path}/{timestamp}/save.npz",
                 noisy_img=img_noisy_np, losses=LOSSES, recons=RECONS,
                 uncerts=UNCERTS, uncerts_ale=UNCERTS_ALE, psnrs=PSNRS, ssims=SSIMS)


if __name__ == '__main__':
    fire.Fire(main)
