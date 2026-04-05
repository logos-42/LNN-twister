"""
长期训练实验 - 验证秩序涌现
============================
核心假设:
  固化 + 修剪 + 持续训练 → 从混沌到秩序的相变

观察指标:
  1. 连接权重分布 (是否从均匀→稀疏)
  2. 神经元激活模式 (是否从随机→结构化)
  3. 巩固分数演化 (是否出现分层)
  4. 生成文本质量 (是否出现语法/语义)
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig
import numpy as np

print("=" * 60)
print("长期训练实验 - 秩序涌现验证")
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

# 创建模型 - 限制连接增长
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
    """记录当前网络状态"""
    # 连接权重分布
    weights = []
    for gene in model.connection_genes:
        if gene.enabled:
            weights.append(abs(gene.weight))

    # 神经元状态
    active_neurons = [
        s for s in model.neuron_states if s.active and s.neuron_type == "hidden"
    ]
    cons_scores = [s.consolidation_score for s in active_neurons]
    decay_scores = [s.decay_counter for s in active_neurons]

    snapshot = {
        "step": step,
        "label": label,
        "hidden_dim": model.hidden_dim,
        "active_count": len(active_neurons),
        "connection_count": len(weights),
        "weight_mean": np.mean(weights) if weights else 0,
        "weight_std": np.std(weights) if weights else 0,
        "weight_sparse_ratio": np.mean([w < 0.1 for w in weights]) if weights else 0,
        "cons_mean": np.mean(cons_scores) if cons_scores else 0,
        "cons_std": np.std(cons_scores) if cons_scores else 0,
        "decay_mean": np.mean(decay_scores) if decay_scores else 0,
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

print(f"\n{'=' * 90}")
print(
    f"{'Step':>5} | {'Phase':>12} | {'Dim':>3} | {'Active':>5} | {'Conn':>6} | {'Loss':>7} | {'W_sparse':>8} | {'Cons>0.5':>8} | {'Decay>0.5':>9}"
)
print(f"{'=' * 90}")

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

    # 快照
    if step in [0, 50, 100, 200, 500, 800, 1000, 1500, 2000]:
        avg_loss = (
            sum(loss_history[-50:]) / min(50, len(loss_history)) if loss_history else 0
        )
        snap = take_snapshot(step, phase.name)
        print(
            f"{step:5d} | {phase.name:>12} | {snap['hidden_dim']:3d} | {snap['active_count']:5d} | {snap['connection_count']:6d} | {avg_loss:7.3f} | {snap['weight_sparse_ratio']:8.3f} | {snap['consolidated_count']:8d} | {snap['decayed_count']:9d}"
        )

# 最终快照
take_snapshot(total_steps, "final")

print(f"\n{'=' * 90}")
print("秩序涌现分析")
print(f"{'=' * 90}")

if len(snapshots) >= 2:
    s0 = snapshots[0]
    s1 = snapshots[-1]

    print(f"\n📊 连接权重演化:")
    print(
        f"  初始: mean={s0['weight_mean']:.4f}, std={s0['weight_std']:.4f}, sparse={s0['weight_sparse_ratio']:.3f}"
    )
    print(
        f"  最终: mean={s1['weight_mean']:.4f}, std={s1['weight_std']:.4f}, sparse={s1['weight_sparse_ratio']:.3f}"
    )
    sparse_change = s1["weight_sparse_ratio"] - s0["weight_sparse_ratio"]
    print(f"  稀疏化趋势: {'↑' if sparse_change > 0 else '↓'} ({sparse_change:+.3f})")

    print(f"\n🧠 神经元固化演化:")
    print(
        f"  初始: cons_mean={s0['cons_mean']:.3f}, consolidated={s0['consolidated_count']}"
    )
    print(
        f"  最终: cons_mean={s1['cons_mean']:.3f}, consolidated={s1['consolidated_count']}"
    )
    print(f"  衰减神经元: {s1['decayed_count']}")

    print(f"\n📈 秩序指标:")
    # 秩序 = 高稀疏性 + 高固化分层 + 低衰减
    order_score = (
        s1["weight_sparse_ratio"] * 0.3
        + (s1["cons_std"] / max(0.01, s1["cons_mean"])) * 0.4
        + (1 - s1["decayed_count"] / max(1, s1["active_count"] + s1["decayed_count"]))
        * 0.3
    )
    print(f"  秩序指数: {order_score:.3f} (0=混沌, 1=秩序)")

# 文本生成测试
print(f"\n{'=' * 90}")
print("文本生成测试")
print(f"{'=' * 90}")

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

print(f"\n{'=' * 90}")
print("实验完成!")
print(f"{'=' * 90}")
