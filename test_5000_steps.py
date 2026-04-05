"""
5000步长期训练 - 流形约束权重 + 振幅正则化
==========================================
验证:
1. 振幅正则化是否进一步促进稀疏化
2. 秩序指数是否继续提升
3. 文本生成质量是否改善
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig
import numpy as np
from collections import Counter

print("=" * 60)
print("5000步长期训练 - 流形约束 + 振幅正则化")
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

# 创建模型
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
loss_history = []


def take_snapshot(step, label=""):
    active_neurons = [
        s for s in model.neuron_states if s.active and s.neuron_type == "hidden"
    ]
    cons_scores = [s.consolidation_score for s in active_neurons]

    if model.hidden_dim > 0:
        W = model.get_complex_weight()
        amp = W.abs()
        phase = W.angle()
        amp_mean = amp.mean().item()
        amp_std = amp.std().item()
        phase_std = phase.std().item()
        sparse_ratio = (amp < 0.01).float().mean().item()
        top10_pct = amp.flatten().topk(max(1, amp.numel() // 10)).values.mean().item()
    else:
        amp_mean = amp_std = phase_std = sparse_ratio = top10_pct = 0

    snapshot = {
        "step": step,
        "label": label,
        "hidden_dim": model.hidden_dim,
        "active_count": len(active_neurons),
        "cons_mean": np.mean(cons_scores) if cons_scores else 0,
        "cons_std": np.std(cons_scores) if cons_scores else 0,
        "consolidated_count": len(
            [s for s in active_neurons if s.consolidation_score > 0.5]
        ),
        "decayed_count": len([s for s in active_neurons if s.decay_counter > 0.5]),
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "phase_std": phase_std,
        "weight_sparse_ratio": sparse_ratio,
        "top10_amp": top10_pct,
    }
    snapshots.append(snapshot)
    return snapshot


def compute_ngram_stats(generated_text, n=3):
    """计算 n-gram 多样性"""
    chars = list(generated_text)
    if len(chars) < n:
        return {"unique_ratio": 0, "entropy": 0}
    ngrams = ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    unique = len(set(ngrams))
    total = len(ngrams)
    counter = Counter(ngrams)
    probs = np.array(list(counter.values())) / total
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return {"unique_ratio": unique / max(1, total), "entropy": entropy}


# 训练
total_steps = 5000
reg_weight = 0.005  # 振幅正则化权重

print(f"\n{'=' * 110}")
print(
    f"{'Step':>5} | {'Phase':>12} | {'Dim':>3} | {'Active':>5} | {'Loss':>7} | {'PPL':>8} | {'Amp':>6} | {'Sparse':>6} | {'Top10%':>6} | {'Phaseσ':>7} | {'Cons>0.5':>8}"
)
print(f"{'=' * 110}")

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

            # 添加振幅正则化
            reg_loss = model.compute_amplitude_regularization(
                l1_weight=reg_weight, l2_weight=reg_weight * 0.1
            )
            total_loss = loss + reg_loss
            total_loss.backward()
            riemannian_opt.step()
            loss_history.append(loss.item())

    if step % 200 == 0 or step in [50, 100, 300, 700, 1500, 3000, 4500]:
        avg_loss = (
            sum(loss_history[-100:]) / min(100, len(loss_history))
            if loss_history
            else 0
        )
        snap = take_snapshot(step, phase.name)
        ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float("inf")
        print(
            f"{step:5d} | {phase.name:>12} | {snap['hidden_dim']:3d} | {snap['active_count']:5d} | {avg_loss:7.3f} | {ppl:8.2f} | {snap['amp_mean']:6.4f} | {snap['weight_sparse_ratio']:6.3f} | {snap['top10_amp']:6.4f} | {snap['phase_std']:7.3f} | {snap['consolidated_count']:8d}"
        )

take_snapshot(total_steps, "final")

print(f"\n{'=' * 110}")
print("秩序涌现分析 (5000步)")
print(f"{'=' * 110}")

if len(snapshots) >= 2:
    s0 = snapshots[0]
    s_mid = snapshots[len(snapshots) // 2]
    s1 = snapshots[-1]

    print(f"\n📊 振幅-相位演化:")
    print(f"  初始: amp={s0['amp_mean']:.4f}, sparse={s0['weight_sparse_ratio']:.3f}")
    print(
        f"  中期: amp={s_mid['amp_mean']:.4f}, sparse={s_mid['weight_sparse_ratio']:.3f}"
    )
    print(f"  最终: amp={s1['amp_mean']:.4f}, sparse={s1['weight_sparse_ratio']:.3f}")
    print(f"  Top10%振幅: {s1['top10_amp']:.4f}")

    print(f"\n🧠 神经元固化:")
    print(
        f"  最终: cons_mean={s1['cons_mean']:.3f}, consolidated={s1['consolidated_count']}/{s1['active_count']}"
    )
    print(f"  衰减: {s1['decayed_count']}")

    print(f"\n📈 秩序指数:")
    order_score = (
        s1["weight_sparse_ratio"] * 0.3
        + (s1["cons_std"] / max(0.01, s1["cons_mean"])) * 0.4
        + (1 - s1["decayed_count"] / max(1, s1["active_count"] + s1["decayed_count"]))
        * 0.3
    )
    print(f"  秩序指数: {order_score:.3f}")

    if loss_history:
        for label, start, end in [
            ("0-500", 0, 500),
            ("500-2000", 500, 2000),
            ("2000-4000", 2000, 4000),
            ("4000-5000", 4000, 5000),
        ]:
            segment = loss_history[start:end]
            if segment:
                avg = sum(segment) / len(segment)
                ppl = torch.exp(torch.tensor(avg)).item()
                print(f"  {label}: loss={avg:.4f}, PPL={ppl:.2f}")

# 文本生成
print(f"\n{'=' * 110}")
print("文本生成测试")
print(f"{'=' * 110}")

if model.hidden_dim > 0:
    model.eval()
    prompts = ["The ", "In ", "A ", "It ", "He ", "She "]
    for prompt in prompts:
        generated = prompt
        encoded = [char2idx.get(c, 0) for c in prompt]
        with torch.no_grad():
            for _ in range(100):
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

        stats = compute_ngram_stats(generated, n=3)
        print(f"  '{prompt}' → '{generated}'")
        print(
            f"    3-gram: unique_ratio={stats['unique_ratio']:.3f}, entropy={stats['entropy']:.3f}"
        )

print(f"\n{'=' * 110}")
print("实验完成!")
print(f"{'=' * 110}")
