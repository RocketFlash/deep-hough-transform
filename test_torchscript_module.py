import torch
torch.ops.load_library("src/model/_cdht_torchscript/build/lib.linux-x86_64-3.8/deep_hough.so")
import deep_hough as dh

print(torch.ops.dh_ops.forward)
print(dh.forward)