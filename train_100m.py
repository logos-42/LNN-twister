"""
Twistor-LNN 100M 训练脚本
=========================
支持:
- 混合精度训练 (AMP)
- 梯度检查点
- 分布式训练 (DeepSpeed)
- 学习率调度 (cosine decay)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
import os

from twistor_llm_100m import TwistorLNN_100M, TwistorLNNConfig


# ============================================================================
# 训练配置
# ============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_size: str = "small"  # small, medium, large
    
    # 数据配置
    vocab_size: int = 32000
    max_seq_len: int = 512
    
    # 训练配置
    batch_size: int = 32
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    total_steps: int = 100000
    
    # 优化器配置
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # 混合精度
    use_amp: bool = True
    
    # 梯度检查点
    use_gradient_checkpointing: bool = True
    
    # 日志
    log_interval: int = 10
    save_interval: int = 1000
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 简单数据集 (用于测试)
# ============================================================================

class SimpleTextDataset(Dataset):
    """简单文本数据集 (用于测试)"""
    
    def __init__(self, vocab_size: int, max_seq_len: int, n_samples: int = 10000):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_samples = n_samples
        
        # 生成随机数据 (仅用于测试)
        self.data = torch.randint(0, vocab_size, (n_samples, max_seq_len))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================================
# 训练器
# ============================================================================

class TwistorTrainer:
    """Twistor-LNN 训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # 创建模型
        if config.model_size == "small":
            model_config = TwistorLNNConfig.small()
        elif config.model_size == "medium":
            model_config = TwistorLNNConfig.medium()
        else:
            model_config = TwistorLNNConfig.large_100m()
        
        print(f"创建模型 (size={config.model_size})...")
        print(f"  vocab_size: {model_config.vocab_size}")
        print(f"  hidden_dim: {model_config.hidden_dim}")
        print(f"  n_layers: {model_config.n_layers}")
        print(f"  参数量：{model_config.get_param_count():,}")
        
        self.model = TwistorLNN_100M(model_config).to(config.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            eps=config.eps,
        )
        
        # 学习率调度器 (cosine decay with warmup)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps,
        )
        
        # 混合精度
        self.scaler = GradScaler('cuda') if config.use_amp else None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            batch = batch.to(self.config.device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]
            
            loss = self.train_step(input_ids, targets)
            
            total_loss += loss
            n_batches += 1
            
            # 更新进度条
            if n_batches % self.config.log_interval == 0:
                avg_loss = total_loss / n_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'})
            
            # 梯度累积
            if n_batches % self.config.gradient_accumulation_steps == 0:
                self.global_step += 1
                
                # 保存检查点
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        return total_loss / n_batches
    
    def train_step(self, input_ids: torch.Tensor, targets: torch.Tensor) -> float:
        """单个训练步骤"""
        self.optimizer.zero_grad()
        
        # 混合精度训练
        if self.config.use_amp:
            with autocast('cuda'):
                logits = self.model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, self.model.config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss / self.config.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            if self.global_step % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
        else:
            logits = self.model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, self.model.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            loss = loss / self.config.gradient_accumulation_steps
            
            loss.backward()
            
            # 梯度裁剪
            if self.global_step % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def train(self, dataloader: DataLoader, n_epochs: int = 10):
        """开始训练"""
        print(f"开始训练...")
        print(f"  设备：{self.config.device}")
        print(f"  总步数：{self.config.total_steps}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  学习率：{self.config.learning_rate}")
        print()
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            avg_loss = self.train_epoch(dataloader)
            
            print(f"Epoch {epoch + 1}/{n_epochs} - Average Loss: {avg_loss:.4f}")
            
            # 保存 epoch 检查点
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
        print(f"检查点已保存到：{path}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        print(f"检查点已加载：{path}")


# ============================================================================
# 学习率调度器
# ============================================================================

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine decay with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("Twistor-LNN 100M 训练")
    print("=" * 70)
    
    # 训练配置
    config = TrainingConfig(
        model_size="small",  # 先测试小模型
        batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=100,
        total_steps=1000,
        use_amp=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # 创建训练器
    trainer = TwistorTrainer(config)
    
    # 创建数据集
    print("创建数据集...")
    dataset = SimpleTextDataset(
        vocab_size=trainer.model.config.vocab_size,
        max_seq_len=64,
        n_samples=1000,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # 开始训练
    trainer.train(dataloader, n_epochs=3)
    
    print()
    print("=" * 70)
    print("训练完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
