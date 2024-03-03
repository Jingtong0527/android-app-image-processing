import torch
from model_utils import load_checkpoint
from torch.utils.mobile_optimizer import optimize_for_mobile
from SRN import SRN
from EIMN import EIMN
import utils
import os
import cv2
import util
import numpy as np
def evaluate(model, val_data_dir='./data'):
    box_size = 368
    scale_search = [0.5, 1.0, 1.5, 2.0]
    param_stride = 8

    # Predict pictures
    list_dir = os.walk(val_data_dir)
    for root, dirs, files in list_dir:
        for f in files:
            test_image = os.path.join(root, f)
            print("test image path", test_image)
            img_ori = cv2.imread(test_image)  # B,G,R order

            multiplier = [scale * box_size / img_ori.shape[0] for scale in scale_search]

            for i, scale in enumerate(multiplier):
                h = int(img_ori.shape[0] * scale)
                w = int(img_ori.shape[1] * scale)
                pad_h = 0 if (h % param_stride == 0) else param_stride - (h % param_stride)
                pad_w = 0 if (w % param_stride == 0) else param_stride - (w % param_stride)
                new_h = h + pad_h
                new_w = w + pad_w

                img_test = cv2.resize(img_ori, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                img_test_pad, pad = util.pad_right_down_corner(img_test, param_stride, param_stride)
                img_test_pad = np.transpose(np.float32(img_test_pad[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5

                feed = Variable(torch.from_numpy(img_test_pad))
                output1, output2 = model(feed)
                print(output1.shape, output2.shape)



model = SRN()
# model = load_checkpoint(model,'D:\轻量级SR\model_epoch_430.pth')

#model = torch.load('D:\轻量级SR\model_epoch_430.pth')
# model = EIMN()

path_chk_rest = 'model_epoch_430.pth'
utils.load_checkpoint_r(model, path_chk_rest)
# print(checkpoint.keys())
# try:
#     model.load_state_dict(checkpoint["state_dict_r"])
# except:
#         state_dict = checkpoint["state_dict_r"]
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = 'module.' + k # remove `module.`
#             new_state_dict[name] = v
#         model.load_state_dict(new_state_dict)
model.float()
model.eval()

model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# Prepare the model for static quantization. This inserts observers in
# the model that will observe activation tensors during calibration.
model_fp32_prepared = torch.quantization.prepare(model)

# calibrate the prepared model to determine quantization parameters for activations
# in a real world setting, the calibration would be done with a representative dataset
evaluate(model_fp32_prepared)

# Convert the observed model to a quantized model. This does several things:
# quantizes the weights, computes and stores the scale and bias value to be
# used with each activation tensor, and replaces key operators with quantized
# implementations.
model_int8 = torch.quantization.convert(model_fp32_prepared)
print("model int8", model_int8)

# Export full jit version model (not compatible with lite interpreter)
model_int8.save("model_an_new.pt")
# Export lite interpreter version model (compatible with lite interpreter)
model_int8._save_for_lite_interpreter("model_an_new.ptl")
# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
model_int8._save_for_lite_interpreter("model_an_optim_new.ptl")


