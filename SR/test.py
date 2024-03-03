import torch
import os
from torchvision.transforms import transforms
from SRN import SRN
from PIL import Image
import time
from model_summary import get_model_flops, get_model_activation
import argparse
import numpy as np
img_path = 'D:/DIV2K_valid_LR_bicubic/X4'
targeet_path = 'D:/DIV2K_valid_HR'
img_list = sorted(os.listdir(img_path))
num_img = len(img_list)
import math
import utils
from skimage.metrics import structural_similarity as sim
def zhibiao(pred, gt):
    spred = pred.squeeze().clamp(0, 1).permute(1,2,0).cpu().numpy()
    sgt = gt.squeeze().clamp(0, 1).permute(1,2,0).cpu().numpy()
    spred = spred.astype(np.float32)
    sgt = sgt.astype(np.float32)
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    ssim_value = sim(spred, sgt, data_range=gt.max()-gt.min(),channel_axis=2, multichannel=True)
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse), rmse, ssim_value
def main(args):
    model_r = SRN()
    start_epoch = 1
    path_chk_rest = 'model_epoch_430.pth'
    utils.load_checkpoint_r(model_r, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1


    for epoch in range(start_epoch, 1000):

        print("laileao")
        model_r.eval()
        transform = transforms.ToTensor()
        PSNR = 0
        MSE = 0
        SSIM = 0
        for img in img_list:
            image = Image.open(img_path + '/' + img).convert('RGB')
            target = Image.open(targeet_path + '/' + img).convert('RGB')
            image = transform(image)
            target = transform(target)
            [A, B, C] = image.shape
            image = image.reshape([1, A, B, C])
            [A, B, C] = target.shape
            target = target.reshape([1, A, B, C])
            # 保存超分图：建立文件夹并输到save_path，把下面四行注释打开
            save_path = 'D:\SR\output'
            with torch.set_grad_enabled(False):
                pre = model_r(image)
            psnr_out,mse,ssim = zhibiao(pre, target)
            PSNR += psnr_out
            MSE += mse
            SSIM += ssim
            # pre = pre.squeeze(0).cpu().detach().numpy()
            # prel = np.transpose(pre, axes=[1, 2, 0]).astype('float32')
            # prel = np.uint8(np.round(np.clip(prel, 0, 1) * 255.))[:, :, ::-1]
            # cv2.imwrite(save_path_1, prel)

        print("PSNR =", PSNR / num_img)
        print("MSE =", MSE / num_img)
        print("SSIM =", SSIM / num_img)

        # 测试 激活量，卷积量，FLOPS，参数量
        # input_dim = (3, 256, 256)  # set the input dimension
        # activations, num_conv = get_model_activation(model_r, input_dim)
        # activations = activations / 10 ** 6
        # print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
        # print("{:>16s} : {:<d}".format("#Conv2d", num_conv))
        #
        # flops = get_model_flops(model_r, input_dim, False)
        # flops = flops / 10 ** 9
        # print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))
        #
        # num_parameters = sum(map(lambda x: x.numel(), model_r.parameters()))
        # num_parameters = num_parameters / 10 ** 6
        # print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))

        # 测试推理时间
        # input_tensor = torch.randn(1,3,256,256).cuda()
        # start_time = time.time()
        # with torch.no_grad():
        #     out = model_r(input_tensor)
        # end_time = time.time()
        # print(end_time - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)