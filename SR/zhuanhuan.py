import torch
from model_utils import load_checkpoint
from torch.utils.mobile_optimizer import optimize_for_mobile
from SRN import SRN
from EIMN import EIMN
import utils
# model = SRN()
# model = load_checkpoint(model,'D:\轻量级SR\model_epoch_430.pth')

#model = torch.load('D:\轻量级SR\model_epoch_430.pth')
model = EIMN()

path_chk_rest = 'model_epoch_2000.pth'
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

model.eval()

scripted_module = torch.jit.script(model)
optimized_scripted_module = optimize_for_mobile(scripted_module)

# Export full jit version model (not compatible with lite interpreter)
scripted_module.save("model_an_new.pt")
# Export lite interpreter version model (compatible with lite interpreter)
scripted_module._save_for_lite_interpreter("model_an_new.ptl")
# using optimized lite interpreter model makes inference about 60% faster than the non-optimized lite interpreter model, which is about 6% faster than the non-optimized full jit model
optimized_scripted_module._save_for_lite_interpreter("model_an_optim_new.ptl")


