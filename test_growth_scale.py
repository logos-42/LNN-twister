from twistor_lnn.growable import GrowableTwistorLNN, GrowthConfig
import torch

config = GrowthConfig(
    min_hidden_dim=0,
    max_hidden_dim=2048,
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
)

for step in range(1500):
    # 前向传播提供激活数据 (只在神经元少时)
    if model.hidden_dim > 0 and model.hidden_dim < 128:
        x = torch.randn(3, 2, 2)
        with torch.no_grad():
            model(x)

    result = model.growth_step()
    phase = model._get_current_developmental_phase()

    if step % 100 == 0 or step in [100, 300, 600, 900]:
        diag = model.get_diagnostics()
        active = len(
            [s for s in model.neuron_states if s.active and s.neuron_type == "hidden"]
        )
        conn_per_neuron = model._get_avg_connections_per_neuron()
        name = phase.name
        print(
            f"Step {step:4d}: [{name:12s}] dim={diag['hidden_dim']:5d}, active={active:5d}, conn={diag['connection_count']:6d}, conn/neuron={conn_per_neuron:.1f}"
        )

print()
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
print(f"巩固神经元: {len(consolidated)}")
print(f"衰减神经元: {len(decayed)}")
print(f"已修剪: {len(pruned)}")
print(f"最终: dim={model.hidden_dim}")
