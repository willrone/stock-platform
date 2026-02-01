# ROCm 安装指南

本项目支持AMD GPU的ROCm和NVIDIA GPU的CUDA。以下是ROCm的安装指南。

## 系统要求

- AMD GPU (RX 6000系列、RX 7000系列、或专业卡)
- Ubuntu 20.04/22.04 或其他支持的Linux发行版
- Python 3.9+

## ROCm 安装步骤

### 1. 安装ROCm驱动

```bash
# 添加ROCm仓库
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# 更新包列表
sudo apt update

# 安装ROCm
sudo apt install rocm-dkms rocm-dev rocm-libs
```

### 2. 配置用户权限

```bash
# 将用户添加到render和video组
sudo usermod -a -G render,video $USER

# 重启系统或重新登录
```

### 3. 验证ROCm安装

```bash
# 检查ROCm信息
rocm-smi

# 检查可用的GPU
rocminfo
```

### 4. 安装支持ROCm的PyTorch

```bash
# 激活虚拟环境
source venv/bin/activate

# 安装ROCm版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
```

### 5. 验证PyTorch ROCm支持

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
```

## 性能优化建议

### 1. 环境变量设置

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # 根据你的GPU调整
export PYTORCH_ROCM_ARCH=gfx1030        # 根据你的GPU架构调整
```

### 2. GPU架构对应表

| GPU系列 | 架构代码 | GFX版本 |
|---------|----------|---------|
| RX 6600/6700 | gfx1032 | 10.3.2 |
| RX 6800/6900 | gfx1030 | 10.3.0 |
| RX 7600 | gfx1102 | 11.0.2 |
| RX 7700/7800/7900 | gfx1100 | 11.0.0 |

### 3. 内存管理

```python
# 在训练前设置内存分配策略
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(0.8)  # 使用80%的GPU内存
```

## 故障排除

### 常见问题

1. **ImportError: No module named 'torch'**
   - 确保在正确的虚拟环境中安装了PyTorch

2. **RuntimeError: No HIP GPUs are available**
   - 检查ROCm驱动是否正确安装
   - 确保用户在render和video组中

3. **GPU内存不足**
   - 减少batch_size
   - 使用梯度累积
   - 启用混合精度训练

### 性能监控

```bash
# 监控GPU使用情况
watch -n 1 rocm-smi

# 查看详细的GPU信息
rocminfo | grep -A 10 "Agent 1"
```

## 模型训练配置

在使用ROCm时，建议的训练配置：

```python
config = TrainingConfig(
    model_type=ModelType.LSTM,
    batch_size=16,  # ROCm可能需要较小的batch size
    epochs=100,
    learning_rate=0.001,
    sequence_length=60
)
```

## 注意事项

1. ROCm的性能可能与CUDA略有不同，建议根据实际情况调整超参数
2. 某些高级功能可能在ROCm上支持有限，如果遇到问题可以回退到CPU训练
3. 定期更新ROCm驱动以获得最佳性能和兼容性