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

def load_checkpoint_c(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict_c"])
    except:
        state_dict = checkpoint["state_dict_c"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint_r(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict_r"])
    except:
        state_dict = checkpoint["state_dict_r"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k[0] =='r':
                name = k[7:]  # remove `module.`
                num = name.split('.')
                a = "recon_trunk_"
                b = num[1]
                c = num[2]
                d = num[3]
                name = a + b + "." + c + "." + d
                if len(num) > 4:
                    e = num[4]
                    name = name + "." + e
                new_state_dict[name] = v
            else:
                name = k
                new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint_d(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict_d"])
    except:
        state_dict = checkpoint["state_dict_d"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:] # remove `module.`
            name = 'module.' + k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim_c(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer_c'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr
def load_optim_i(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer_i'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr
def load_optim_r(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer_r'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr

def load_optim_d(optimizer, weights):
    checkpoint = torch.load(weights)
    optimizer.load_state_dict(checkpoint['optimizer_d'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr
