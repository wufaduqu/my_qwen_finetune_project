# 安裝必要的套件
!pip install transformers torch accelerate peft bitsandbytes

# 檢查 CUDA 是否可用 (對於模型訓練很重要)
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")