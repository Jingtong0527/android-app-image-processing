import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from SRN import SRN
from PIL import Image
import time
from model_summary import get_model_flops, get_model_activation
import argparse
import numpy as np

import math
import utils
from skimage.metrics import structural_similarity as sim

def main(args):
    model = SRN()
    start_epoch = 1
    path_chk_rest = 'model_epoch_430.pth'
    utils.load_checkpoint_r(model, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    model.eval()
    scripted_module = torch.jit.script(model)
    optimized_scripted_module = optimize_for_mobile(scripted_module)

    # Export full jit version model (not compatible with lite interpreter)
    scripted_module.save("model_an.pt")
    # Export lite interpreter version model (compatible with lite interpreter)
    scripted_module._save_for_lite_interpreter("model_an.ptl")
    # using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
    optimized_scripted_module._save_for_lite_interpreter("model_an_optim.ptl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)