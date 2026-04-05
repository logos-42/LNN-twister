"""
长期训练实验 - 流形约束权重(振幅+相位)
======================================
验证几何约束是否促进秩序涌现
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig
import numpy as np

print("=" * 60)
print("长期训练 - 流形约束权重验证")
print("=" * 60)

# 加载数据集
print("\n加载 WikiText-2...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
all_texts = [item["text"] for item in ds["train"] if len(item["text"].strip()) >= 33]

char_set = set()
for text in all_texts[:5000]:
    char_set.update(text)
char_set = sorted(list(char_set))
char2idx = {c: i for i, c in enumerate(char_set)}
idx2char = {i: c for i, c in enumerate(char_set)}
vocab_size = len(char_set)

all_texts_enc = []
for text in all_texts:
    enc = [char2idx.get(c, 0) for c in text]
    if len(enc) >= 33:
        all_texts_enc.append(enc)

print(f"词汇表: {vocab_size}, 训练文本: {len(all_texts_enc)}")

# 创建模型 - 流形约束权重
config = GrowthConfig(
    min_hidden_dim=0,
    max_hidden_dim=128,
    growth_interval=10,
    prune_interval=20,
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

riemannian_opt = model.create_riemannian_optimizer(lr=0.002)
criterion = nn.CrossEntropyLoss()

seq_len = 32


def get_batch():
    if not all_texts_enc:
        return None
    enc = all_texts_enc[torch.randint(len(all_texts_enc), (1,)).item()]
    start = torch.randint(0, max(1, len(enc) - seq_len), (1,)).item()
    chunk = enc[start : start + seq_len + 1]
    x = torch.tensor(chunk[:-1], dtype=torch.long)
    y = torch.tensor(chunk[1:], dtype=torch.long)
    return x, y


def onehot(x):
    h = torch.zeros(seq_len, 1, vocab_size)
    for t in range(seq_len):
        if x[t].item() < vocab_size:
            h[t, 0, x[t].item()] = 1.0
    return h


# 记录指标
snapshots = []


def take_snapshot(step, label=""):
    weights = []
    for gene in model.connection_genes:
        if gene.enabled:
            weights.append(abs(gene.weight))

    active_neurons = [
        s for s in model.neuron_states if s.active and s.neuron_type == "hidden"
    ]
    cons_scores = [s.consolidation_score for s in active_neurons]
    decay_scores = [s.decay_counter for s in active_neurons]

    # 复数权重统计
    if model.hidden_dim > 0:
        W = model.get_complex_weight()
        amp_mean = W.abs().mean().item()
        amp_std = W.abs().std().item()
        phase_mean = W.angle().mean().item()
        phase_std = W.angle().std().item()
        sparse_ratio = (W.abs() < 0.1).float().mean().item()
    else:
        amp_mean = amp_std = phase_mean = phase_std = sparse_ratio = 0

    snapshot = {
        "step": step,
        "label": label,
        "hidden_dim": model.hidden_dim,
        "active_count": len(active_neurons),
        "connection_count": len(weights),
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "phase_mean": phase_mean,
        "phase_std": phase_std,
        "weight_sparse_ratio": sparse_ratio,
        "cons_mean": np.mean(cons_scores) if cons_scores else 0,
        "cons_std": np.std(cons_scores) if cons_scores else 0,
        "consolidated_count": len(
            [s for s in active_neurons if s.consolidation_score > 0.5]
        ),
        "decayed_count": len([s for s in active_neurons if s.decay_counter > 0.5]),
    }
    snapshots.append(snapshot)
    return snapshot


# 训练
total_steps = 2000
loss_history = []

print(f"\n{'=' * 100}")
print(
    f"{'Step':>5} | {'Phase':>12} | {'Dim':>3} | {'Active':>5} | {'Conn':>6} | {'Loss':>7} | {'Amp':>6} | {'Phase_std':>9} | {'Sparse':>6} | {'Cons>0.5':>8}"
)
print(f"{'=' * 100}")

for step in range(total_steps):
    batch = get_batch()
    if batch is None:
        continue

    x, y = batch
    x_onehot = onehot(x)

    model.growth_step()
    phase = model._get_current_developmental_phase()

    if model.hidden_dim > 0:
        riemannian_opt.zero_grad()
        y_pred = model(x_onehot)
        if y_pred.shape[0] == y.shape[0]:
            loss = criterion(y_pred.squeeze(1), y)
            loss.backward()
            riemannian_opt.step()
            loss_history.append(loss.item())

    if step in [0, 50, 100, 200, 500, 800, 1000, 1500, 2000]:
        avg_loss = (
            sum(loss_history[-50:]) / min(50, len(loss_history)) if loss_history else 0
        )
        snap = take_snapshot(step, phase.name)
        print(
            f"{step:5d} | {phase.name:>12} | {snap['hidden_dim']:3d} | {snap['active_count']:5d} | {snap['connection_count']:6d} | {avg_loss:7.3f} | {snap['amp_mean']:6.3f} | {snap['phase_std']:9.3f} | {snap['weight_sparse_ratio']:6.3f} | {snap['consolidated_count']:8d}"
        )

take_snapshot(total_steps, "final")

print(f"\n{'=' * 100}")
print("秩序涌现分析 (流形约束权重)")
print(f"{'=' * 100}")

if len(snapshots) >= 2:
    s0 = snapshots[0]
    s1 = snapshots[-1]

    print(f"\n📊 振幅-相位演化:")
    print(
        f"  初始: amp_mean={s0['amp_mean']:.4f}, amp_std={s0['amp_std']:.4f}, phase_std={s0['phase_std']:.4f}"
    )
    print(
        f"  最终: amp_mean={s1['amp_mean']:.4f}, amp_std={s1['amp_std']:.4f}, phase_std={s1['phase_std']:.4f}"
    )
    print(
        f"  稀疏化: {s0['weight_sparse_ratio']:.3f} → {s1['weight_sparse_ratio']:.3f}"
    )

    print(f"\n🧠 神经元固化:")
    print(
        f"  初始: cons_mean={s0['cons_mean']:.3f}, consolidated={s0['consolidated_count']}"
    )
    print(
        f"  最终: cons_mean={s1['cons_mean']:.3f}, consolidated={s1['consolidated_count']}"
    )
    print(f"  衰减: {s1['decayed_count']}")

    print(f"\n📈 秩序指标:")
    order_score = (
        s1["weight_sparse_ratio"] * 0.3
        + (s1["cons_std"] / max(0.01, s1["cons_mean"])) * 0.4
        + (1 - s1["decayed_count"] / max(1, s1["active_count"] + s1["decayed_count"]))
        * 0.3
    )
    print(f"  秩序指数: {order_score:.3f} (0=混沌, 1=秩序)")

    if loss_history:
        early_loss = sum(loss_history[:20]) / min(20, len(loss_history))
        late_loss = sum(loss_history[-20:]) / min(20, len(loss_history))
        print(f"\n📊 训练效果:")
        print(
            f"  早期loss: {early_loss:.4f} (PPL: {torch.exp(torch.tensor(early_loss)).item():.2f})"
        )
        print(
            f"  晚期loss: {late_loss:.4f} (PPL: {torch.exp(torch.tensor(late_loss)).item():.2f})"
        )
        improvement = (
            (early_loss - late_loss) / early_loss * 100 if early_loss > 0 else 0
        )
        print(f"  改善: {improvement:.1f}%")

# 文本生成
print(f"\n{'=' * 100}")
print("文本生成测试")
print(f"{'=' * 100}")

if model.hidden_dim > 0:
    model.eval()
    prompts = ["The ", "In ", "A ", "It "]
    for prompt in prompts:
        generated = prompt
        encoded = [char2idx.get(c, 0) for c in prompt]
        with torch.no_grad():
            for _ in range(80):
                x = torch.zeros(len(encoded), 1, vocab_size)
                for t in range(len(encoded)):
                    if encoded[t] < vocab_size:
                        x[t, 0, encoded[t]] = 1.0
                output = model(x)
                probs = torch.softmax(output[-1, 0], dim=0)
                next_idx = torch.multinomial(probs, 1).item()
                next_char = idx2char.get(next_idx, "?")
                generated += next_char
                encoded.append(next_idx)
        print(f"  '{prompt}' → '{generated}'")

print(f"\n{'=' * 100}")
print("实验完成!")
print(f"{'=' * 100}")
