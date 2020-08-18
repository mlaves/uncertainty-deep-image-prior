import torch
import numpy as np
import time
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from .common_utils import torch_to_np, np_to_torch, normalize

def calc_uncert(net, net_input_saved, noise, reg_noise_std):
    img_list = []
    img_list_np = []

    with torch.no_grad():
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        for i in range(25):
            img = net(net_input)
            img_list.append(img)
            img_list_np.append(torch_to_np(img))

    uncertainty_map = np.mean(np.std(np.array(img_list_np), axis=0), axis = 0)

    img_list = torch.cat(img_list, dim=0)
    img_mean = img_list[:,:1].mean(dim=0, keepdim=True)
    ale = img_list[:,1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:,0], dim=0, keepdim=True)
    uncert = (ale.exp()+epi)
    #return ale.exp().mean().item(), epi.mean().item(), uncert.mean().item()
    return uncertainty_map, ale.exp(), epi, uncert

# Loss
def gaussian_nll(mu, neg_logvar, target, reduction='mean'):
    neg_logvar = torch.clamp(neg_logvar, min=-20, max=20)  # prevent nan loss
    loss = torch.exp(neg_logvar) * torch.pow(target - mu, 2) - neg_logvar
    return loss.mean() if reduction == 'mean' else loss.sum()
