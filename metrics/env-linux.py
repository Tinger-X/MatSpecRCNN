import os
import sys
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("Python:", sys.version)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("nvidia-smi:")
os.system("nvidia-smi")

print("device_count:", torch.cuda.device_count())
print("device_name:", torch.cuda.get_device_name(0))
print("free -h:")
os.system("free -h")

print("pip:")
os.system("pip list | grep -E 'torch|torchvision|opencv|numpy'")

"""result:
PyTorch: 2.6.0+cu126
TorchVision: 0.21.0+cu126
Python: 3.10.11 | packaged by conda-forge | (main, May 10 2023, 18:58:44) [GCC 11.3.0]
CUDA version: 12.6
cuDNN version: 90501
nvidia-smi:
Sat Jul  5 07:14:51 2025       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:09.0 Off |                    0 |
| N/A   37C    P8    11W /  70W |      4MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
device_count: 1
device_name: Tesla T4
free -h:
              total        used        free      shared  buff/cache   available
Mem:           30Gi       3.7Gi        22Gi       2.0Mi       4.2Gi        26Gi
Swap:            0B          0B          0B
pip:
numpy                    2.2.3
opencv-python-headless   4.11.0.86
torch                    2.6.0+cu126
torchaudio               2.6.0+cu126
torchvision              0.21.0+cu126
"""
