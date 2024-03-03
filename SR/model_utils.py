import torch
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    print(checkpoint.keys())
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_moco(model, weights):
    checkpoint = torch.load(weights)
    print(checkpoint.keys())
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name =  k # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    print(checkpoint.keys())
    try:
        model.load_state_dict(checkpoint["state_dict_r"])
    except:
        state_dict = checkpoint["state_dict_r"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        num = name.split('.')
        a = "recon_trunk_"
        b = num[1]
        c = num[2]
        d = num[3]
        name = a+b+"."+c+"."+d
        if len(num)>4:
            e = num[4]
            name = name+"."+e
        print(name)
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch
def load_optim_1(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer_naf'])

def load_optim_contrast(optimizer, weights):
    checkpoint = torch.load(weights)

    optimizer.load_state_dict(checkpoint['optimizer'])
def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr
