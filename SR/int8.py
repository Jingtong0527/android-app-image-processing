import torch
import torchvision.models as models
import torch.quantization
from SRN import SRN
import utils
# model = SRN()
# model = load_checkpoint(model,'D:\轻量级SR\model_epoch_430.pth')

#model = torch.load('D:\轻量级SR\model_epoch_430.pth')
model = SRN()

path_chk_rest = 'model_epoch_430.pth'
utils.load_checkpoint_r(model, path_chk_rest)
# 加载预训练模型
model_fp32 = SRN()
quantization_config = torch.quantization.get_default_qconfig('fbgemm')
model_fp32.qconfig = quantization_config

# 创建量化模型
quantized_model = torch.quantization.prepare(model_fp32)

# 训练/评估你的量化模型，或者直接应用于你的数据集

# 完成量化
quantized_model = torch.quantization.convert(quantized_model)

# 保存量化后的模型
torch.jit.save(torch.jit.script(quantized_model), 'quantized_model.pt')
quantized_model._save_for_lite_interpreter("model_an_new.ptl")
# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
quantized_model._save_for_lite_interpreter("model_an_optim_new.ptl")