import sys
import platform
import subprocess
import torch
import torchvision


def get_gpu_info():
    try:
        output = subprocess.check_output(
            "nvidia-smi",
            shell=True,
            text=True,
            stderr=subprocess.STDOUT
        )
        return output
    except Exception as e:
        return f"nvidia-smi not available: {str(e)}"


def get_memory_info():
    try:
        output = subprocess.check_output(
            "wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value",
            shell=True,
            text=True
        )
        lines = output.strip().split("\n")
        mem_info = {}
        for line in lines:
            if "=" in line:
                key, val = line.split("=")
                mem_info[key.strip()] = val.strip()

        total = int(mem_info.get("TotalVisibleMemorySize", 0))
        free = int(mem_info.get("FreePhysicalMemory", 0))
        used = total - free

        # Convert to GB (Windows returns values in KB)
        return (
            f"Total: {total / 1024 ** 2:.1f}GB\n"
            f"Used : {used / 1024 ** 2:.1f}GB\n"
            f"Free : {free / 1024 ** 2:.1f}GB"
        )
    except Exception as e:
        return f"Memory info unavailable: {str(e)}"


def get_package_info():
    try:
        output = subprocess.check_output(
            "pip list",
            shell=True,
            text=True
        )
        packages = []
        for line in output.split("\n"):
            if any(kw in line.lower() for kw in ["torch", "opencv", "numpy"]):
                packages.append(line)
        return "\n".join(packages)
    except Exception as e:
        return f"Package info unavailable: {str(e)}"


print(f"PyTorch: {torch.__version__}")
print(f"TorchVision: {torchvision.__version__}")
print(f"Python: {sys.version}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")

print("\nnvidia-smi:")
print(get_gpu_info())

if torch.cuda.is_available():
    print(f"\ndevice_count: {torch.cuda.device_count()}")
    print(f"device_name: {torch.cuda.get_device_name(0)}")
else:
    print("device_count: 0")
    print("device_name: No CUDA devices available")

print("\nMemory Info:")
print(get_memory_info())

print("\nInstalled Packages:")
print(get_package_info())

# 补充系统信息
print("\nSystem Information:")
print(f"OS: {platform.system()} {platform.release()}")
print(f"Processor: {platform.processor()}")
print(f"Architecture: {'64-bit' if sys.maxsize > 2 ** 32 else '32-bit'}")

"""result:
PyTorch: 2.5.1+cu124
TorchVision: 0.20.1+cu124
Python: 3.11.6 (tags/v3.11.6:8b6ee5b, Oct  2 2023, 14:57:12) [MSC v.1935 64 bit (AMD64)]
CUDA version: 12.4
cuDNN version: 90100

nvidia-smi:
Sat Jul  5 15:12:51 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 576.28                 Driver Version: 576.28         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3060      WDDM  |   00000000:01:00.0  On |                  N/A |
|  0%   39C    P8             18W /  170W |     933MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

device_count: 1
device_name: NVIDIA GeForce RTX 3060

Memory Info:
Total: 15.8GB
Used : 8.0GB
Free : 7.8GB

Installed Packages:
numpy                          1.26.4
opencv-contrib-python          4.10.0.84
opencv-python                  4.10.0.84
torch                          2.5.1+cu124
torchaudio                     2.5.1+cu124
torchvision                    0.20.1+cu124

System Information:
OS: Windows 10
Processor: Intel64 Family 6 Model 191 Stepping 2, GenuineIntel
Architecture: 64-bit
"""
