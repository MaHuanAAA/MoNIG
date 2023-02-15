import torch
import os
import numpy as np
from src.dataset import Multimodal_Datasets


def get_data(args, dataset, split='train'):
    alignment = 'a' if args.aligned else 'na'
    data_path = os.path.join(args.data_path, dataset) + f'_{split}_{alignment}.dt'
    if not os.path.exists(data_path):
        print(f"  - Creating new {split} data")
        data = Multimodal_Datasets(args.data_path, dataset, split, args.aligned)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {split} data")
        data = torch.load(data_path)
    return data


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model


def criterion_gaussian(u, y, sigma):
    loss = sum(0.5 * (torch.log(2 * np.pi * sigma) + ((y - u) ** 2) / sigma)) / len(u)
    return loss


def moe_nig(u1, la1, alpha1, beta1, u2, la2, alpha2, beta2):
    # Eq. 9
    u = (la1 * u1 + u2 * la2) / (la1 + la2)
    la = la1 + la2
    alpha = alpha1 + alpha2 + 0.5
    beta = beta1 + beta2 + 0.5 * (la1 * (u1 - u) ** 2 + la2 * (u2 - u) ** 2)
    return u, la, alpha, beta


def criterion_nig(u, la, alpha, beta, y, hyp_params):
    # our loss function
    om = 2 * beta * (1 + la)
    loss = sum(
        0.5 * torch.log(np.pi / la) - alpha * torch.log(om) + (alpha + 0.5) * torch.log(la * (u - y) ** 2 + om) + torch.lgamma(alpha) - torch.lgamma(alpha+0.5)) / len(u)
    lossr = hyp_params.risk * sum(torch.abs(u - y) * (2 * la + alpha)) / len(u)
    loss = loss + lossr
    return loss
