"""
WikiText数据集上的语言模型训练验证
===================================
测试流形约束生长系统在真实文本数据上的效果
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig

print("=" * 60)
print("WikiText-2 语言模型训练验证")
print("=" * 60)

# 加载数据集
print("\n加载 WikiText-2 数据集...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

# 构建词汇表
print("构建词汇表...")
all_texts = [item["text"] for item in ds["train"] if len(item["text"].strip()) >= 33]

char_set = set()
for text in all_texts[:5000]:
    char_set.update(text)
char_set = sorted(list(char_set))
char2idx = {c: i for i, c in enumerate(char_set)}
idx2char = {i: c for i, c in enumerate(char_set)}
vocab_size = len(char_set)
print(f"词汇表大小: {vocab_size} (字符级)")
print(f"训练文本: {len(all_texts)} 条 (长度>=33)")

# 准备训练数据
seq_len = 32


def encode_text(text, max_len=seq_len + 1):
    return [char2idx.get(c, 0) for c in text[:max_len]]


# 预编码所有文本
print("预编码文本...")
encoded_texts = []
for text in all_texts:
    enc = encode_text(text, max_len=len(text))
    if len(enc) >= seq_len + 1:
        encoded_texts.append(enc)
print(f"有效编码文本: {len(encoded_texts)}")


def get_batch(n_batches=1):
    for _ in range(n_batches):
        if not encoded_texts:
            break
        enc = encoded_texts[torch.randint(len(encoded_texts), (1,)).item()]
        start = torch.randint(0, max(1, len(enc) - seq_len), (1,)).item()
        chunk = enc[start : start + seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        yield x, y


# 创建模型
config = GrowthConfig(
    min_hidden_dim=0,
    max_hidden_dim=256,
    growth_interval=5,
    prune_interval=50,
    enable_developmental_schedule=True,
)

model = GrowableTwistorLNN(
    input_dim=vocab_size,
    hidden_dim=0,
    output_dim=vocab_size,
    growth_config=config,
    enable_growth=True,
    enable_mobius=True,
)

riemannian_opt = model.create_riemannian_optimizer(lr=0.005)
criterion = nn.CrossEntropyLoss()

print(f"\n初始状态:")
print(f"  input_dim: {model.input_dim}")
print(f"  hidden_dim: {model.hidden_dim}")
print(f"  output_dim: {model.output_dim}")
print(f"  流形半径: {model.manifold_geometry.manifold_radius.item():.2f}")

# 训练循环
print(f"\n{'=' * 80}")
print(
    f"{'Step':>5} | {'Phase':>12} | {'Dim':>4} | {'Active':>5} | {'Conn':>5} | {'Loss':>8} | {'PPL':>8} | {'MaxGrad':>7}"
)
print(f"{'=' * 80}")

loss_history = []
phase_history = []
current_phase_name = None
total_steps = 500
max_grad_global = 0

for step in range(total_steps):
    batch_data = list(get_batch(n_batches=1))
    if not batch_data:
        continue

    x, y = batch_data[0]

    # One-hot编码输入
    x_onehot = torch.zeros(seq_len, 1, vocab_size)
    for t in range(seq_len):
        if x[t].item() < vocab_size:
            x_onehot[t, 0, x[t].item()] = 1.0

    max_grad = 0

    # 先growth(可能产生新神经元)
    result = model.growth_step()
    phase = model._get_current_developmental_phase()

    # 有hidden_dim就训练
    if model.hidden_dim > 0:
        riemannian_opt.zero_grad()
        y_pred = model(x_onehot)

        if y_pred.shape[0] == y.shape[0]:
            loss = criterion(y_pred.squeeze(1), y)
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    g = p.grad.norm().item()
                    if g > max_grad:
                        max_grad = g
            if max_grad > max_grad_global:
                max_grad_global = max_grad

            riemannian_opt.step()
            loss_history.append(loss.item())
        else:
            loss = torch.tensor(0.0)
    else:
        loss = torch.tensor(0.0)

    if phase.name != current_phase_name:
        if current_phase_name is not None:
            phase_history.append(
                {
                    "phase": current_phase_name,
                    "end_dim": model.hidden_dim,
                    "end_active": len(
                        [
                            s
                            for s in model.neuron_states
                            if s.active and s.neuron_type == "hidden"
                        ]
                    ),
                    "end_conn": model.get_diagnostics()["connection_count"],
                    "avg_loss": sum(loss_history[-100:]) / min(100, len(loss_history))
                    if loss_history
                    else 0,
                }
            )
        current_phase_name = phase.name

    if step % 50 == 0 or step in [20, 100, 200, 300, 400]:
        diag = model.get_diagnostics()
        active = len(
            [s for s in model.neuron_states if s.active and s.neuron_type == "hidden"]
        )
        conn_per_neuron = model._get_avg_connections_per_neuron()
        avg_loss = (
            sum(loss_history[-20:]) / min(20, len(loss_history)) if loss_history else 0
        )
        ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float("inf")
        print(
            f"{step:5d} | {phase.name:>12} | {diag['hidden_dim']:4d} | {active:5d} | {diag['connection_count']:5d} | {avg_loss:8.4f} | {ppl:8.2f} | {max_grad:7.4f}"
        )

if current_phase_name is not None:
    phase_history.append(
        {
            "phase": current_phase_name,
            "end_dim": model.hidden_dim,
            "end_active": len(
                [
                    s
                    for s in model.neuron_states
                    if s.active and s.neuron_type == "hidden"
                ]
            ),
            "end_conn": model.get_diagnostics()["connection_count"],
            "avg_loss": sum(loss_history[-100:]) / min(100, len(loss_history))
            if loss_history
            else 0,
        }
    )

print(f"\n{'=' * 80}")
print("发育阶段总结")
print(f"{'=' * 80}")
for ph in phase_history:
    ppl = (
        torch.exp(torch.tensor(ph["avg_loss"])).item()
        if ph["avg_loss"] > 0
        else float("inf")
    )
    print(
        f"  {ph['phase']:12s}: dim={ph['end_dim']:4d}, active={ph['end_active']:5d}, conn={ph['end_conn']:5d}, loss={ph['avg_loss']:.4f}, PPL={ppl:.2f}"
    )

if loss_history:
    early_loss = sum(loss_history[:20]) / min(20, len(loss_history))
    late_loss = sum(loss_history[-20:]) / min(20, len(loss_history))
    print(f"\n📊 训练效果:")
    print(
        f"  早期平均loss: {early_loss:.4f} (PPL: {torch.exp(torch.tensor(early_loss)).item():.2f})"
    )
    print(
        f"  晚期平均loss: {late_loss:.4f} (PPL: {torch.exp(torch.tensor(late_loss)).item():.2f})"
    )
    improvement = (early_loss - late_loss) / early_loss * 100 if early_loss > 0 else 0
    print(f"  改善幅度: {improvement:.1f}%")
    print(f"  最大梯度范数: {max_grad_global:.4f}")

# 推理测试
print(f"\n{'=' * 80}")
print("推理测试 - 文本生成")
print(f"{'=' * 80}")

if model.hidden_dim > 0:
    model.eval()
    prompt = "The "
    generated = prompt
    encoded = [char2idx.get(c, 0) for c in prompt]

    with torch.no_grad():
        for _ in range(50):
            x = torch.zeros(len(encoded), 1, vocab_size)
            for t in range(len(encoded)):
                if encoded[t] < vocab_size:
                    x[t, 0, encoded[t]] = 1.0

            output = model(x)
            probs = torch.softmax(output[-1, 0], dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            next_char = idx2char.get(next_char_idx, "?")
            generated += next_char
            encoded.append(next_char_idx)

    print(f"  Prompt: '{prompt}'")
    print(f"  Generated: '{generated}'")

print(f"\n{'=' * 80}")
print("WikiText-2 验证完成!")
print(f"{'=' * 80}")
