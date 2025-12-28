"""
现代深度学习模型测试（独立测试，不依赖其他服务）
"""

import pytest
import torch
import numpy as np

# 直接导入模型，避免通过services模块导入
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app', 'services'))

from modern_models import TimesNet, PatchTST, Informer


class TestModernModels:
    """现代模型测试类"""
    
    @pytest.fixture
    def sample_input(self):
        """创建样本输入数据"""
        batch_size = 4
        seq_len = 60
        input_dim = 10
        return torch.randn(batch_size, seq_len, input_dim)
    
    def test_timesnet_forward(self, sample_input):
        """测试TimesNet前向传播"""
        batch_size, seq_len, input_dim = sample_input.shape
        
        model = TimesNet(
            input_dim=input_dim,
            seq_len=seq_len,
            num_classes=2,
            d_model=32,  # 减小模型大小用于测试
            d_ff=64,
            num_kernels=3,
            top_k=2
        )
        
        output = model(sample_input)
        
        # 检查输出形状
        assert output.shape == (batch_size, 2)
        
        # 检查输出是否为有限值
        assert torch.isfinite(output).all()
    
    def test_patchtst_forward(self, sample_input):
        """测试PatchTST前向传播"""
        batch_size, seq_len, input_dim = sample_input.shape
        
        model = PatchTST(
            input_dim=input_dim,
            seq_len=seq_len,
            num_classes=2,
            patch_len=8,
            stride=4,
            d_model=32,  # 减小模型大小用于测试
            nhead=4,
            num_layers=2
        )
        
        output = model(sample_input)
        
        # 检查输出形状
        assert output.shape == (batch_size, 2)
        
        # 检查输出是否为有限值
        assert torch.isfinite(output).all()
    
    def test_informer_forward(self, sample_input):
        """测试Informer前向传播"""
        batch_size, seq_len, input_dim = sample_input.shape
        
        model = Informer(
            input_dim=input_dim,
            seq_len=seq_len,
            num_classes=2,
            d_model=64,  # 减小模型大小用于测试
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            factor=2
        )
        
        output = model(sample_input)
        
        # 检查输出形状
        assert output.shape == (batch_size, 2)
        
        # 检查输出是否为有限值
        assert torch.isfinite(output).all()
    
    def test_timesnet_different_seq_lengths(self):
        """测试TimesNet处理不同序列长度"""
        input_dim = 5
        
        for seq_len in [30, 60, 120]:
            sample_input = torch.randn(2, seq_len, input_dim)
            
            model = TimesNet(
                input_dim=input_dim,
                seq_len=seq_len,
                num_classes=2,
                d_model=16,
                d_ff=32,
                num_kernels=2,
                top_k=2
            )
            
            output = model(sample_input)
            assert output.shape == (2, 2)
    
    def test_patchtst_patch_calculation(self):
        """测试PatchTST的patch计算"""
        seq_len = 60
        patch_len = 10
        stride = 5
        input_dim = 8
        
        model = PatchTST(
            input_dim=input_dim,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            num_classes=2,
            d_model=32,
            nhead=4,
            num_layers=1
        )
        
        # 验证patch数量计算
        expected_patch_num = (seq_len - patch_len) // stride + 1
        assert model.patch_num == expected_patch_num
        
        # 测试前向传播
        sample_input = torch.randn(3, seq_len, input_dim)
        output = model(sample_input)
        assert output.shape == (3, 2)
    
    def test_model_parameters_count(self, sample_input):
        """测试模型参数数量"""
        batch_size, seq_len, input_dim = sample_input.shape
        
        # TimesNet
        timesnet = TimesNet(input_dim, seq_len, num_classes=2, d_model=32)
        timesnet_params = sum(p.numel() for p in timesnet.parameters())
        assert timesnet_params > 0
        
        # PatchTST
        patchtst = PatchTST(input_dim, seq_len, num_classes=2, d_model=32)
        patchtst_params = sum(p.numel() for p in patchtst.parameters())
        assert patchtst_params > 0
        
        # Informer
        informer = Informer(input_dim, seq_len, num_classes=2, d_model=64)
        informer_params = sum(p.numel() for p in informer.parameters())
        assert informer_params > 0
        
        print(f"TimesNet参数数量: {timesnet_params}")
        print(f"PatchTST参数数量: {patchtst_params}")
        print(f"Informer参数数量: {informer_params}")
    
    def test_model_training_mode(self, sample_input):
        """测试模型训练/评估模式切换"""
        model = TimesNet(
            input_dim=sample_input.shape[-1],
            seq_len=sample_input.shape[1],
            num_classes=2,
            d_model=32
        )
        
        # 训练模式
        model.train()
        assert model.training
        
        output_train = model(sample_input)
        
        # 评估模式
        model.eval()
        assert not model.training
        
        with torch.no_grad():
            output_eval = model(sample_input)
        
        # 输出形状应该相同
        assert output_train.shape == output_eval.shape
    
    def test_gradient_flow(self, sample_input):
        """测试梯度流"""
        model = PatchTST(
            input_dim=sample_input.shape[-1],
            seq_len=sample_input.shape[1],
            num_classes=2,
            d_model=32,
            num_layers=1
        )
        
        # 前向传播
        output = model(sample_input)
        
        # 计算损失
        target = torch.randint(0, 2, (sample_input.shape[0],))
        loss = torch.nn.CrossEntropyLoss()(output, target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                assert torch.isfinite(param.grad).all()
        
        assert has_grad, "模型应该有梯度"
    
    @pytest.mark.parametrize("model_class", [TimesNet, PatchTST, Informer])
    def test_model_device_compatibility(self, model_class, sample_input):
        """测试模型设备兼容性"""
        seq_len, input_dim = sample_input.shape[1], sample_input.shape[2]
        
        # 创建模型
        if model_class == TimesNet:
            model = model_class(input_dim, seq_len, num_classes=2, d_model=32)
        elif model_class == PatchTST:
            model = model_class(input_dim, seq_len, num_classes=2, d_model=32)
        else:  # Informer
            model = model_class(input_dim, seq_len, num_classes=2, d_model=64)
        
        # CPU测试
        model_cpu = model.to('cpu')
        input_cpu = sample_input.to('cpu')
        output_cpu = model_cpu(input_cpu)
        
        assert output_cpu.device.type == 'cpu'
        assert output_cpu.shape == (sample_input.shape[0], 2)
        
        # 如果有CUDA，测试GPU
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            input_gpu = sample_input.to('cuda')
            output_gpu = model_gpu(input_gpu)
            
            assert output_gpu.device.type == 'cuda'
            assert output_gpu.shape == (sample_input.shape[0], 2)


class TestModelIntegration:
    """模型集成测试"""
    
    def test_all_models_same_interface(self):
        """测试所有模型具有相同的接口"""
        batch_size = 2
        seq_len = 40
        input_dim = 6
        sample_input = torch.randn(batch_size, seq_len, input_dim)
        
        models = [
            TimesNet(input_dim, seq_len, num_classes=2, d_model=32),
            PatchTST(input_dim, seq_len, num_classes=2, d_model=32),
            Informer(input_dim, seq_len, num_classes=2, d_model=64)
        ]
        
        for model in models:
            output = model(sample_input)
            assert output.shape == (batch_size, 2)
            assert torch.isfinite(output).all()
    
    def test_model_memory_usage(self):
        """测试模型内存使用"""
        seq_len = 100
        input_dim = 15
        batch_size = 8
        
        sample_input = torch.randn(batch_size, seq_len, input_dim)
        
        # 测试每个模型的内存使用
        models = {
            'TimesNet': TimesNet(input_dim, seq_len, num_classes=2, d_model=64),
            'PatchTST': PatchTST(input_dim, seq_len, num_classes=2, d_model=64),
            'Informer': Informer(input_dim, seq_len, num_classes=2, d_model=128)
        }
        
        for name, model in models.items():
            # 前向传播
            output = model(sample_input)
            
            # 计算模型大小（参数数量）
            param_count = sum(p.numel() for p in model.parameters())
            
            print(f"{name} - 参数数量: {param_count}, 输出形状: {output.shape}")
            
            # 基本检查
            assert param_count > 0
            assert output.shape == (batch_size, 2)