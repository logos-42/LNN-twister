"""
10000步长期训练 - 降低正则化 + 真实n-gram评估
==============================================
验证:
1. 降低正则化权重(0.005→0.001)是否改善稀疏化
2. 使用验证集真实n-gram统计评估生成质量
3. 观察秩序指数是否继续提升
"""

import torch
import torch.nn as nn
from datasets import load_dataset
from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig
import numpy as np
from collections import Counter

print("=" * 60)
print("10000步训练 - 低正则化 + 真实n-gram评估")
print("=" * 60)

# 加载数据集
print("\n加载 WikiText-2...")
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

# 训练集
train_texts = [item["text"] for item in ds["train"] if len(item["text"].strip()) >= 33]
# 验证集 (用于真实n-gram统计)
val_texts = [
    item["text"] for item in ds["validation"] if len(item["text"].strip()) >= 33
]

char_set = set()
for text in train_texts[:5000] + val_texts[:1000]:
    char_set.update(text)
char_set = sorted(list(char_set))
char2idx = {c: i for i, c in enumerate(char_set)}
idx2char = {i: c for i, c in enumerate(char_set)}
vocab_size = len(char_set)

train_enc = []
for text in train_texts:
    enc = [char2idx.get(c, 0) for c in text]
    if len(enc) >= 33:
        train_enc.append(enc)

# 计算验证集真实n-gram统计
print(f"\n计算验证集真实n-gram统计...")
val_chars = []
for text in val_texts[:200]:  # 减少验证集大小
    val_chars.extend(list(text.lower()))


def get_ngram_stats(chars, n=3):
    ngrams = ["".join(chars[i : i + n]) for i in range(len(chars) - n + 1)]
    counter = Counter(ngrams)
    total = len(ngrams)
    probs = {k: v / total for k, v in counter.most_common(1000)}
    return probs, counter


val_2gram_probs, val_2gram_counter = get_ngram_stats(val_chars, n=2)
val_3gram_probs, val_3gram_counter = get_ngram_stats(val_chars, n=3)

print(f"词汇表: {vocab_size}")
print(f"训练文本: {len(train_enc)}")
print(f"验证集: {len(val_texts)} 条")
print(f"验证集2-gram种类: {len(val_2gram_counter)}")
print(f"验证集3-gram种类: {len(val_3gram_counter)}")

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
    if not train_enc:
        return None
    enc = train_enc[torch.randint(len(train_enc), (1,)).item()]
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


# 评估函数
def evaluate_generation(generated_text):
    """使用验证集真实n-gram评估生成质量"""
    gen_chars = list(generated_text.lower())

    # 2-gram 匹配
    gen_2grams = ["".join(gen_chars[i : i + 2]) for i in range(len(gen_chars) - 1)]
    gen_2gram_counter = Counter(gen_2grams)

    # 3-gram 匹配
    gen_3grams = ["".join(gen_chars[i : i + 3]) for i in range(len(gen_chars) - 2)]
    gen_3gram_counter = Counter(gen_3grams)

    # 匹配度: 生成n-gram在验证集中出现的概率
    match_2 = sum(val_2gram_probs.get(ng, 0) for ng in gen_2grams) / max(
        1, len(gen_2grams)
    )
    match_3 = sum(val_3gram_probs.get(ng, 0) for ng in gen_3grams) / max(
        1, len(gen_3grams)
    )

    # 覆盖率: 生成n-gram中有多少在验证集中出现过
    coverage_2 = sum(1 for ng in gen_2gram_counter if ng in val_2gram_counter) / max(
        1, len(gen_2gram_counter)
    )
    coverage_3 = sum(1 for ng in gen_3gram_counter if ng in val_3gram_counter) / max(
        1, len(gen_3gram_counter)
    )

    # 多样性
    unique_2 = len(gen_2gram_counter) / max(1, len(gen_2grams))
    unique_3 = len(gen_3gram_counter) / max(1, len(gen_3grams))

    return {
        "match_2gram": match_2,
        "match_3gram": match_3,
        "coverage_2gram": coverage_2,
        "coverage_3gram": coverage_3,
        "unique_2gram": unique_2,
        "unique_3gram": unique_3,
    }


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


# 训练
total_steps = 5000
reg_weight = 0.001  # 降低正则化权重

print(f"\n{'=' * 120}")
print(
    f"{'Step':>5} | {'Phase':>12} | {'Dim':>3} | {'Active':>5} | {'Loss':>7} | {'PPL':>8} | {'Amp':>6} | {'Sparse':>6} | {'Top10%':>6} | {'Phaseσ':>7} | {'Cons>0.5':>8}"
)
print(f"{'=' * 120}")

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
            reg_loss = model.compute_amplitude_regularization(
                l1_weight=reg_weight, l2_weight=reg_weight * 0.1
            )
            total_loss = loss + reg_loss
            total_loss.backward()
            riemannian_opt.step()
            loss_history.append(loss.item())

    if step % 500 == 0 or step in [50, 100, 300, 700, 1500, 3000, 5000, 7500, 9500]:
        avg_loss = (
            sum(loss_history[-200:]) / min(200, len(loss_history))
            if loss_history
            else 0
        )
        snap = take_snapshot(step, phase.name)
        ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss > 0 else float("inf")
        print(
            f"{step:5d} | {phase.name:>12} | {snap['hidden_dim']:3d} | {snap['active_count']:5d} | {avg_loss:7.3f} | {ppl:8.2f} | {snap['amp_mean']:6.4f} | {snap['weight_sparse_ratio']:6.3f} | {snap['top10_amp']:6.4f} | {snap['phase_std']:7.3f} | {snap['consolidated_count']:8d}"
        )

take_snapshot(total_steps, "final")

print(f"\n{'=' * 120}")
print("秩序涌现分析 (10000步)")
print(f"{'=' * 120}")

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
            ("0-1000", 0, 1000),
            ("1000-3000", 1000, 3000),
            ("3000-6000", 3000, 6000),
            ("6000-9000", 6000, 9000),
            ("9000-10000", 9000, 10000),
        ]:
            segment = loss_history[start : min(end, len(loss_history))]
            if segment:
                avg = sum(segment) / len(segment)
                ppl = torch.exp(torch.tensor(avg)).item()
                print(f"  {label}: loss={avg:.4f}, PPL={ppl:.2f}")

# 文本生成 + n-gram评估
print(f"\n{'=' * 120}")
print("文本生成测试 + 真实n-gram评估")
print(f"{'=' * 120}")

if model.hidden_dim > 0:
    model.eval()
    prompts = ["The ", "In ", "A ", "It ", "He ", "She ", "They ", "This "]
    all_stats = []

    for prompt in prompts:
        generated = prompt
        encoded = [char2idx.get(c, 0) for c in prompt]
        with torch.no_grad():
            for _ in range(120):
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

        stats = evaluate_generation(generated)
        all_stats.append(stats)
        print(f"\n  '{prompt}' → '{generated}'")
        print(
            f"    2-gram: match={stats['match_2gram']:.4f}, coverage={stats['coverage_2gram']:.3f}, unique={stats['unique_2gram']:.3f}"
        )
        print(
            f"    3-gram: match={stats['match_3gram']:.4f}, coverage={stats['coverage_3gram']:.3f}, unique={stats['unique_3gram']:.3f}"
        )

    # 平均指标
    if all_stats:
        avg_match_2 = np.mean([s["match_2gram"] for s in all_stats])
        avg_match_3 = np.mean([s["match_3gram"] for s in all_stats])
        avg_cov_2 = np.mean([s["coverage_2gram"] for s in all_stats])
        avg_cov_3 = np.mean([s["coverage_3gram"] for s in all_stats])
        print(f"\n  📊 平均n-gram指标:")
        print(f"    2-gram: match={avg_match_2:.4f}, coverage={avg_cov_2:.3f}")
        print(f"    3-gram: match={avg_match_3:.4f}, coverage={avg_cov_3:.3f}")

print(f"\n{'=' * 120}")
print("实验完成!")
print(f"{'=' * 120}")
