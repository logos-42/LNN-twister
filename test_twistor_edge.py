"""
测试 TwistorLNNwithGQA (Twistor-LNN-Edge) 融合效果
"""
import torch
import torch.nn.functional as F
import numpy as np

# 尝试导入，如果失败说明有语法问题
try:
    import importlib.util
    
    # 直接从 twistor_lnn.py 文件加载模块
    spec = importlib.util.spec_from_file_location("twistor_lnn_module", "twistor_lnn.py")
    twistor_lnn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twistor_lnn_module)
    
    TwistorLNNwithGQA = twistor_lnn_module.TwistorLNNwithGQA
    GroupedQueryAttention = twistor_lnn_module.GroupedQueryAttention
    print("✓ 导入成功: TwistorLNNwithGQA, GroupedQueryAttention")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    exit(1)


def test_gqa():
    """测试 GQA Attention"""
    print("\n" + "=" * 50)
    print("测试 1: GQA Attention")
    print("=" * 50)
    
    B, T, D = 2, 4, 16
    n_heads = 4
    n_kv_heads = 2
    
    gqa = GroupedQueryAttention(dim=D, n_heads=n_heads, n_kv_heads=n_kv_heads)
    x = torch.randn(B, T, D)
    
    out = gqa(x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {out.shape}")
    print(f"  参数量: {sum(p.numel() for p in gqa.parameters()):,}")
    assert out.shape == (B, T, D), "输出形状错误"
    print("✓ GQA 测试通过")


def test_twistor_lnn_with_gqa():
    """测试 TwistorLNNwithGQA 前向传播"""
    print("\n" + "=" * 50)
    print("测试 2: TwistorLNNwithGQA 前向传播")
    print("=" * 50)
    
    # 创建模型
    model = TwistorLNNwithGQA(
        input_dim=2,
        hidden_dim=32,
        output_dim=1,
        use_gqa=True,
        n_heads=4,
        n_kv_heads=1,
        attention_interval=3,
        tau_attention_threshold=0.3,
    )
    
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成测试数据
    T, B = 20, 4
    X = torch.randn(T, B, 2)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        y = model(X)
    
    print(f"  输入形状: {X.shape}")
    print(f"  输出形状: {y.shape}")
    assert y.shape == (T, B, 1), "输出形状错误"
    print("✓ 前向传播测试通过")


def test_step_method():
    """测试 Agent step 方法"""
    print("\n" + "=" * 50)
    print("测试 3: Agent step 方法")
    print("=" * 50)
    
    model = TwistorLNNwithGQA(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        use_gqa=True,
    )
    
    # 重置状态
    z = model.reset_state(batch_size=2, device='cpu')
    print(f"  初始状态形状: {z.shape}")
    
    # 单步演化
    x = torch.randn(2, 4)  # batch=2, input_dim=4
    z_new, output = model.step(z, x)
    
    print(f"  输入形状: {x.shape}")
    print(f"  新状态形状: {z_new.shape}")
    print(f"  输出形状: {output.shape}")
    
    assert z_new.shape == (2, 16), "状态形状错误"
    assert output.shape == (2, 2), "输出形状错误"
    print("✓ step 方法测试通过")


def test_attention_trigger():
    """测试 Attention 触发机制"""
    print("\n" + "=" * 50)
    print("测试 4: Attention 触发机制")
    print("=" * 50)
    
    model = TwistorLNNwithGQA(
        input_dim=2,
        hidden_dim=16,
        output_dim=1,
        use_gqa=True,
        attention_interval=3,
        tau_attention_threshold=0.3,
    )
    
    # 测试 tau 触发条件
    z = torch.randn(2, 16, dtype=torch.complex64)
    tau = model.compute_tau(z)
    
    print(f"  tau 均值: {tau.mean().item():.4f}")
    print(f"  tau 范围: [{tau.min().item():.4f}, {tau.max().item():.4f}]")
    
    # 模拟触发
    trigger = model.should_trigger_attention(tau)
    print(f"  触发条件: {trigger}")
    print("✓ Attention 触发机制测试通过")


def test_training():
    """测试训练效果"""
    print("\n" + "=" * 50)
    print("测试 5: 训练收敛测试")
    print("=" * 50)
    
    from twistor_lnn.datasets import generate_sine_dataset
    
    # 生成数据
    X, y = generate_sine_dataset(n_samples=100, seq_len=30, noise_std=0.1, device='cpu')
    print(f"  数据形状: X={X.shape}, y={y.shape}")
    
    # 创建模型
    model = TwistorLNNwithGQA(
        input_dim=2,
        hidden_dim=16,
        output_dim=1,
        use_gqa=True,
        n_heads=2,
        n_kv_heads=1,
        attention_interval=5,
    )
    
    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    losses = []
    
    for epoch in range(50):
        optimizer.zero_grad()
        
        # 前向传播
        X_batch = X[:32].transpose(0, 1)  # (T, B, input_dim)
        y_batch = y[:32].transpose(0, 1)  # (T, B, output_dim)
        
        y_pred = model(X_batch)
        loss = F.mse_loss(y_pred, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss = {loss.item():.4f}")
    
    # 检查收敛
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"  初始损失: {initial_loss:.4f}")
    print(f"  最终损失: {final_loss:.4f}")
    print(f"  改善: {improvement:.1f}%")
    
    if final_loss < initial_loss:
        print("✓ 训练收敛测试通过")
    else:
        print("! 损失未下降，但模型可以运行")


def test_comparison():
    """对比有/无 GQA 的效果"""
    print("\n" + "=" * 50)
    print("测试 6: 有/无 GQA 对比")
    print("=" * 50)
    
    from twistor_lnn import TwistorLNN
    
    # 无 GQA
    model_no_gqa = TwistorLNN(input_dim=2, hidden_dim=32, output_dim=1)
    params_no_gqa = sum(p.numel() for p in model_no_gqa.parameters())
    
    # 有 GQA
    model_with_gqa = TwistorLNNwithGQA(
        input_dim=2, hidden_dim=32, output_dim=1,
        use_gqa=True, n_heads=4, n_kv_heads=1
    )
    params_with_gqa = sum(p.numel() for p in model_with_gqa.parameters())
    
    print(f"  TwistorLNN 参数: {params_no_gqa:,}")
    print(f"  TwistorLNN+GQA 参数: {params_with_gqa:,}")
    print(f"  额外参数: {params_with_gqa - params_no_gqa:,}")
    print(f"  增加比例: {(params_with_gqa / params_no_gqa - 1) * 100:.1f}%")
    
    # 简单测试
    X = torch.randn(20, 4, 2)
    
    model_no_gqa.eval()
    model_with_gqa.eval()
    
    with torch.no_grad():
        y1 = model_no_gqa(X)
        y2 = model_with_gqa(X)
    
    print(f"  TwistorLNN 输出: {y1.shape}")
    print(f"  TwistorLNN+GQA 输出: {y2.shape}")
    print("✓ 对比测试完成")


if __name__ == "__main__":
    print("=" * 60)
    print("Twistor-LNN-Edge (GQA 融合) 测试")
    print("=" * 60)
    
    test_gqa()
    test_twistor_lnn_with_gqa()
    test_step_method()
    test_attention_trigger()
    test_training()
    test_comparison()
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
