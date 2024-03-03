import copy
import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from SRN import SRN
import utils
model = SRN()
# model = load_checkpoint(model,'D:\轻量级SR\model_epoch_430.pth')
path_chk_rest = 'model_epoch_430.pth'
utils.load_checkpoint_r(model, path_chk_rest)

model.eval()
# from torchvision.models import resnet50
# fp32_model = resnet50().eval()
model = copy.deepcopy(model)

qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}
# `prepare_fx` inserts observers in the model based on the configuration in `qconfig_dict`
model_prepared = prepare_fx(model, qconfig_dict)

calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(100)]
for i in range(len(calibration_data)):
   model_prepared(calibration_data[i])
model_quantized = convert_fx(copy.deepcopy(model_prepared))
# benchmark
x = torch.randn(1, 3, 224, 224)
timeit = model(x)
timeit_ = model_quantized(x)
