"""完整训练验证 - 流形约束生长系统"""

import torch
import torch.nn as nn
from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig

print("=" * 60)
print("完整训练验证 - 流形约束生长")
print("=" * 60)

config = GrowthConfig(
    min_hidden_dim=0,
    max_hidden_dim=512,
    growth_interval=5,
    prune_interval=10,
    enable_developmental_schedule=True,
)

model = GrowableTwistorLNN(
    input_dim=2,
    hidden_dim=0,
    output_dim=1,
    growth_config=config,
    enable_growth=True,
    enable_mobius=True,
)

riemannian_opt = model.create_riemannian_optimizer(lr=0.005)
criterion = nn.MSELoss()

print(f"\n初始状态:")
print(f"  hidden_dim: {model.hidden_dim}")
print(f"  流形半径: {model.manifold_geometry.manifold_radius.item():.2f}")
print(f"  扭转率: {model.manifold_geometry.twist_rate.item():.2f}")

phase_history = []
current_phase_name = None
loss_history = []

print(f"\n{'=' * 70}")
print(
    f"{'Step':>5} | {'Phase':>12} | {'Dim':>5} | {'Active':>5} | {'Conn':>6} | {'Conn/N':>6} | {'Loss':>8} | {'MaxGrad':>7}"
)
print(f"{'=' * 70}")

for step in range(1500):
    x = torch.randn(5, 4, 2)
    y_target = x[:, :, 0:1] * 0.5 + x[:, :, 1:2] * 0.3 + torch.randn(5, 4, 1) * 0.1

    riemannian_opt.zero_grad()

    if model.hidden_dim > 0 and model.hidden_dim < 256:
        y_pred = model(x)
        loss = criterion(y_pred, y_target)
        loss.backward()

        max_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.norm().item()
                if g > max_grad:
                    max_grad = g

        riemannian_opt.step()
        loss_history.append(loss.item())
    elif model.hidden_dim >= 256:
        # 大规模时用小 batch 继续训练
        x_small = torch.randn(3, 2, 2)
        y_small = (
            x_small[:, :, 0:1] * 0.5
            + x_small[:, :, 1:2] * 0.3
            + torch.randn(3, 2, 1) * 0.1
        )
        y_pred = model(x_small)
        loss = criterion(y_pred, y_small)
        loss.backward()

        max_grad = 0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.norm().item()
                if g > max_grad:
                    max_grad = g

        riemannian_opt.step()
        loss_history.append(loss.item())
    else:
        loss = torch.tensor(0.0)
        max_grad = 0

    result = model.growth_step()
    phase = model._get_current_developmental_phase()

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
                    "avg_loss": sum(loss_history[-200:]) / min(200, len(loss_history))
                    if loss_history
                    else 0,
                }
            )
        current_phase_name = phase.name

    if step % 200 == 0 or step in [100, 300, 600, 900]:
        diag = model.get_diagnostics()
        active = len(
            [s for s in model.neuron_states if s.active and s.neuron_type == "hidden"]
        )
        conn_per_neuron = model._get_avg_connections_per_neuron()
        avg_loss = (
            sum(loss_history[-50:]) / min(50, len(loss_history)) if loss_history else 0
        )
        print(
            f"{step:5d} | {phase.name:>12} | {diag['hidden_dim']:5d} | {active:5d} | {diag['connection_count']:6d} | {conn_per_neuron:6.1f} | {avg_loss:8.4f} | {max_grad:7.4f}"
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
            "avg_loss": sum(loss_history[-200:]) / min(200, len(loss_history))
            if loss_history
            else 0,
        }
    )

print(f"\n{'=' * 70}")
print("发育阶段总结")
print(f"{'=' * 70}")
for ph in phase_history:
    print(
        f"  {ph['phase']:12s}: dim={ph['end_dim']:5d}, active={ph['end_active']:5d}, conn={ph['end_conn']:6d}, avg_loss={ph['avg_loss']:.4f}"
    )

consolidated = [
    s
    for s in model.neuron_states
    if s.neuron_type == "hidden" and s.consolidation_score > 0.3
]
decayed = [
    s
    for s in model.neuron_states
    if s.neuron_type == "hidden" and s.decay_counter > 0.5
]
pruned = [s for s in model.neuron_states if s.neuron_type == "hidden" and not s.active]

print(f"\n💾 巩固神经元 (长期记忆): {len(consolidated)}")
print(f"📉 衰减神经元 (待遗忘): {len(decayed)}")
print(f"✂️ 已修剪神经元 (已遗忘): {len(pruned)}")

if loss_history:
    early_loss = sum(loss_history[:50]) / min(50, len(loss_history))
    late_loss = sum(loss_history[-50:]) / min(50, len(loss_history))
    print(f"\n📊 训练效果:")
    print(f"  早期平均loss (前50步): {early_loss:.4f}")
    print(f"  晚期平均loss (后50步): {late_loss:.4f}")
    print(f"  改善幅度: {(early_loss - late_loss) / early_loss * 100:.1f}%")

if consolidated:
    print(f"\n🏆 核心神经元 Top 3:")
    top3 = sorted(consolidated, key=lambda s: s.consolidation_score, reverse=True)[:3]
    for s in top3:
        print(
            f"  Neuron {s.index}: cons={s.consolidation_score:.3f}, decay={s.decay_counter:.3f}, life={s.life_span}"
        )

print(f"\n{'=' * 70}")
print("验证完成!")
print(f"{'=' * 70}")
