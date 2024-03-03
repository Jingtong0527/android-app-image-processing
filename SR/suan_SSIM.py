import os
import torch
from PIL import Image
from torchvision.transforms import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.nn.parallel import DataParallel
img_path = '/media/sr617/29b9171f-dba6-4703-98a5-43495a4a4fe6/DINO-IR/output_naf_rain100'
targeet_path = '/media/sr617/29b9171f-dba6-4703-98a5-43495a4a4fe6/DINO-IR/DATA/test/test_rainy/target100'# seg_path = '/media/sr617/29b9171f-dba6-4703-98a5-43495a4a4fe6/MPRNet-main/Denoising/DATA/test/test_rainy/segmentation_0_1'
import numpy as np
import cv2


import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1, img2, full=True)
    return score

def compare_folders(folder1, folder2):
    file_list1 = os.listdir(folder1)
    file_list2 = os.listdir(folder2)
    if len(file_list1) != len(file_list2):
        print("The number of images in two folders is not equal.")
        return
    SSIM = 0
    for i in range(len(file_list1)):
        img_path1 = os.path.join(folder1, file_list1[i])
        img_path2 = os.path.join(folder2, file_list2[i])
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)
        if img1.shape != img2.shape:
            print("The shape of images in two folders is not equal.")
            return
        ssim_score = calculate_ssim(img1, img2)
        print(f"SSIM score between {img_path1} and {img_path2}: {ssim_score}")
        SSIM += ssim_score

    print(SSIM/100)
# 设置两个文件夹路径
folder1 = '/media/sr617/29b9171f-dba6-4703-98a5-43495a4a4fe6/DINO-IR/output_naf_rain100'
folder2 = '/media/sr617/29b9171f-dba6-4703-98a5-43495a4a4fe6/DINO-IR/DATA/test/test_rainy/target100'
compare_folders(folder1, folder2)





# def ssim(img1, img2):
#     C1 = (0.01 * 255)**2
#     C2 = (0.03 * 255)**2
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1**2
#     mu2_sq = mu2**2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
#
# def calculate_ssim(img1, img2, border=0):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     #img1 = img1.squeeze()
#     #img2 = img2.squeeze()
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     h, w = img1.shape[:2]
#     img1 = img1[border:h-border, border:w-border]
#     img2 = img2[border:h-border, border:w-border]
#
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')
# img_list = sorted(os.listdir(img_path))
# transform = transforms.ToTensor()
# PSNR = 0
# for img in img_list:
#     # print(img)
#     image = Image.open(img_path + '/' + img).convert('RGB')
#     target = Image.open(targeet_path + '/' + img).convert('RGB')
#     # seg = Image.open(seg_path + '/' + img).convert('RGB')
#     image = imread
#     # image = transform(image)
#     # target = transform(target)
#     # #
#     # # # seg = transform(seg)
#     # # image = image.cuda()
#     # # target = target.cuda()
#     # # # seg = seg.cuda()
#     # #
#     # # [A, B, C] = image.shape
#     # # image = image.reshape([1, A, B, C])
#     # # [A, B, C] = target.shape
#     # # target = target.reshape([1, A, B, C])
#     # #
#     # # # pre = pre.squeeze(0).cpu().detach().numpy()
#     # # # prel = np.transpose(pre, axes=[1, 2, 0]).astype('float32')
#     # # # prel = np.uint8(np.round(np.clip(prel, 0, 1) * 255.))[:, :, ::-1]
#     # #
#     # image = image.cpu().numpy()
#     # target = target.cpu().numpy()
#     # #
#     # image = image.squeeze(0).cpu().detach().numpy()
#     # target = target.squeeze(0).cpu().detach().numpy()
#
#     SSIM = calculate_ssim(image, target)
#     # psnr_out = psnr(pre, target)
#     print(SSIM)
#     PSNR +=SSIM
# print("PSNR =", PSNR / 100)