"""
Twistor-inspired Liquid Neural Network (Complex-valued LNN)
============================================================
Implements continuous-time dynamics: dz/dt = (-z + W*tanh(z) + Ux) / tau(z)

Key features:
- Complex-valued hidden state z (torch.complex)
- State-dependent time constant tau(z)
- Euler integration for time evolution
- Stability regularization via ||dz/dt||^2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class TwistorLNN(nn.Module):
    """
    Twistor-inspired Liquid Neural Network with complex-valued states.

    The dynamics follow: dz/dt = (-z + W*tanh(z) + Ux) / tau(z)
    where z is complex, and tau(z) is state-dependent.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Weight matrices - SEPARATE for real and imag parts (as required)
        # W_real: recurrent weight for real part
        self.W_real = nn.Linear(hidden_dim, hidden_dim)
        # W_imag: recurrent weight for imag part
        self.W_imag = nn.Linear(hidden_dim, hidden_dim)
        # U: input weight (input -> hidden) - shared
        self.U = nn.Linear(input_dim, hidden_dim)
        # W_tau: for computing state-dependent time constant
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        # Output projection (real part only)
        self.out = nn.Linear(hidden_dim, output_dim)

        # Initialize weights with small values for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization for stability."""
        nn.init.orthogonal_(self.W_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
        nn.init.zeros_(self.W_real.bias)
        nn.init.zeros_(self.W_imag.bias)
        nn.init.zeros_(self.U.bias)
        nn.init.zeros_(self.W_tau.bias)

    def compute_tau(self, z_real: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent time constant.

        tau(z) = sigmoid(W_tau(z.real)) + epsilon

        Args:
            z_real: Real part of complex state (B, hidden_dim)

        Returns:
            tau: Time constant (B, hidden_dim), always positive
        """
        return F.sigmoid(self.W_tau(z_real)) + 1e-6

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative dz/dt.

        Dynamics: dz/dt = (-z + W*tanh(z) + Ux) / tau(z)

        Real part: dz_real/dt = (-z_real + W_real*tanh(z_real) + U*x) / tau
        Imag part: dz_imag/dt = (-z_imag + W_imag*tanh(z_imag)) / tau

        Args:
            z: Complex hidden state (B, hidden_dim), dtype=complex
            x: Input (B, input_dim)

        Returns:
            dzdt: Time derivative (B, hidden_dim), dtype=complex
        """
        # Extract real and imaginary parts
        z_real = z.real
        z_imag = z.imag

        # Apply tanh to real and imag parts separately
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        # Compute numerator: -z + W*tanh(z) + Ux
        # Use SEPARATE weight matrices for real and imag (as required)
        W_tanh_real = self.W_real(tanh_real)  # (B, hidden_dim)
        W_tanh_imag = self.W_imag(tanh_imag)  # (B, hidden_dim)
        Ux = self.U(x)  # (B, hidden_dim)

        # Compute real and imag derivatives separately
        dz_real = -z_real + W_tanh_real + Ux
        dz_imag = -z_imag + W_tanh_imag

        # Compute state-dependent time constant (from real part only)
        tau = self.compute_tau(z_real)

        # Divide by tau and recombine as complex
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        return dzdt

    def forward(self, x: torch.Tensor, return_states: bool = False) -> torch.Tensor:
        """
        Forward pass with Euler integration over time.

        Args:
            x: Input sequence (T, B, input_dim)
            return_states: If True, return all hidden states

        Returns:
            y: Output sequence (T, B, output_dim)
            states: All hidden states (T, B, hidden_dim) if return_states=True
        """
        T, B, _ = x.shape

        # Initialize complex hidden state to zero
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []

        # Time loop: Euler integration
        for t in range(T):
            # Get input at time t
            x_t = x[t]  # (B, input_dim)

            # Compute time derivative
            dzdt = self.compute_dzdt(z, x_t)

            # Euler step: z(t+dt) = z(t) + dt * dz/dt
            # Using dt=0.1 for numerical stability
            dt = 0.1
            z = z + dt * dzdt

            # Output from real part only
            y_t = self.out(z.real)  # (B, output_dim)

            outputs.append(y_t)
            if return_states:
                states.append(z)

        # Stack outputs: (T, B, output_dim)
        y = torch.stack(outputs, dim=0)

        if return_states:
            states = torch.stack(states, dim=0)  # (T, B, hidden_dim)
            return y, states

        return y


def generate_sine_dataset(
    n_samples: int = 1000,
    seq_len: int = 50,
    input_dim: int = 2,
    noise_std: float = 0.1,
    device: str = "cpu",
):
    """
    Generate synthetic sine wave prediction dataset.

    Input: sine waves with random frequencies and phases
    Target: next values in the sequence

    Args:
        n_samples: Number of sequences
        seq_len: Length of each sequence
        input_dim: Input dimension (default 2: sin + cos)
        noise_std: Standard deviation of noise

    Returns:
        X: Input sequences (n_samples, seq_len, input_dim)
        y: Target sequences (n_samples, seq_len, 1)
    """
    X = []
    y = []

    for _ in range(n_samples):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)

        # Time steps
        t = np.linspace(0, 4 * np.pi, seq_len + 1)  # +1 for target

        # Generate sine wave
        signal = np.sin(freq * t + phase)

        # Add noise
        signal += np.random.randn(len(t)) * noise_std

        # Input: sin and cos components
        sin_component = signal[:-1]
        cos_component = (
            np.cos(freq * t[:-1] + phase) + np.random.randn(seq_len) * noise_std
        )

        x_seq = np.stack([sin_component, cos_component], axis=-1)  # (seq_len, 2)

        # Target: next value (prediction task)
        y_seq = signal[1:].reshape(-1, 1)  # (seq_len, 1)

        X.append(x_seq)
        y.append(y_seq)

    X = torch.FloatTensor(np.stack(X)).to(device)  # (n_samples, seq_len, input_dim)
    y = torch.FloatTensor(np.stack(y)).to(device)  # (n_samples, seq_len, 1)

    return X, y


def train_twistor_lnn(
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-2,
    hidden_dim: int = 16,
    stability_weight: float = 0.01,
    device: str = "cpu",
):
    """
    Train the Twistor LNN on sine wave prediction.

    Args:
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        hidden_dim: Hidden dimension
        stability_weight: Weight for ||dz/dt||^2 regularization
        device: Device to train on
    """
    print("=" * 60)
    print("Twistor-inspired Liquid Neural Network Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Stability weight: {stability_weight}")
    print()

    # Generate dataset
    print("Generating synthetic sine wave dataset...")
    X_train, y_train = generate_sine_dataset(n_samples=1000, seq_len=50, device=device)
    X_val, y_val = generate_sine_dataset(n_samples=200, seq_len=50, device=device)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Sequence length: {X_train.shape[1]}, Input dim: {X_train.shape[2]}")
    print()

    # Initialize model
    model = TwistorLNN(
        input_dim=X_train.shape[2], hidden_dim=hidden_dim, output_dim=1
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    # Training loop
    n_batches = len(X_train) // batch_size
    history = {"train_loss": [], "val_loss": [], "train_mse": [], "val_mse": []}

    print("Starting training...")
    print("-" * 60)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0

        # Shuffle data
        perm = torch.randperm(len(X_train), device=device)
        X_train = X_train[perm]
        y_train = y_train[perm]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # Get batch (T, B, input_dim) format
            x_batch = X_train[start_idx:end_idx].transpose(
                0, 1
            )  # (seq_len, batch, input_dim)
            y_batch = y_train[start_idx:end_idx].transpose(0, 1)  # (seq_len, batch, 1)

            optimizer.zero_grad()

            # Forward pass with states for stability loss
            y_pred, states = model(x_batch, return_states=True)

            # MSE loss
            mse_loss = F.mse_loss(y_pred, y_batch)

            # Stability regularization: ||dz/dt||^2
            # Compute dz/dt for all time steps
            dzdt_norm_sq = 0.0
            for t in range(len(states) - 1):
                dzdt = states[t + 1] - states[t]  # Approximate dz/dt
                dzdt_norm_sq += (dzdt.abs() ** 2).mean()
            stability_loss = dzdt_norm_sq / (len(states) - 1)

            # Total loss
            loss = mse_loss + stability_weight * stability_loss

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()

        # Average losses
        avg_train_loss = epoch_loss / n_batches
        avg_train_mse = epoch_mse / n_batches
        history["train_loss"].append(avg_train_loss)
        history["train_mse"].append(avg_train_mse)

        # Validation
        model.eval()
        with torch.no_grad():
            x_val = X_val.transpose(0, 1)
            y_val_t = y_val.transpose(0, 1)
            y_val_pred = model(x_val)
            val_mse = F.mse_loss(y_val_pred, y_val_t).item()
            history["val_loss"].append(val_mse)  # Use MSE as val loss
            history["val_mse"].append(val_mse)

        # Update learning rate
        scheduler.step(avg_train_loss)

        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}/{n_epochs}: "
                f"Train Loss = {avg_train_loss:.6f}, "
                f"Train MSE = {avg_train_mse:.6f}, "
                f"Val MSE = {val_mse:.6f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.6f}"
            )

    print("-" * 60)
    print(f"Training complete! Final Val MSE: {history['val_mse'][-1]:.6f}")
    print()

    # Plot results
    plot_training_results(history)
    plot_predictions(model, X_val, y_val, device)

    return model, history


def plot_training_results(history):
    """Plot training curves."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_mse"], label="Train MSE")
    plt.plot(history["val_mse"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Mean Squared Error")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved to 'training_curves.png'")
    plt.close()


def plot_predictions(model, X_test, y_test, device, n_samples: int = 5):
    """Plot sample predictions."""
    model.eval()

    # Get predictions
    with torch.no_grad():
        x_test = X_test[:n_samples].transpose(0, 1)
        y_pred = model(x_test).transpose(0, 1)  # (n_samples, seq_len, 1)
        y_true = y_test[:n_samples]

    plt.figure(figsize=(14, 8))

    for i in range(n_samples):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(
            y_true[i].cpu().numpy().flatten(),
            "o-",
            label="True",
            alpha=0.7,
            markersize=4,
        )
        plt.plot(
            y_pred[i].cpu().numpy().flatten(),
            "s-",
            label="Predicted",
            alpha=0.7,
            markersize=4,
        )
        plt.ylabel("Amplitude")
        plt.title(f"Sample {i + 1}")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    print("Sample predictions saved to 'predictions.png'")
    plt.close()


if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train model
    model, history = train_twistor_lnn(
        n_epochs=200,
        batch_size=32,
        lr=1e-2,
        hidden_dim=16,
        stability_weight=0.01,
        device=device,
    )

    # Save model
    torch.save(model.state_dict(), "twistor_lnn.pth")
    print("Model saved to 'twistor_lnn.pth'")

    print()
    print("=" * 60)
    print("Training Summary:")
    print(f"  Initial Train Loss: {history['train_loss'][0]:.6f}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Initial Val MSE: {history['val_mse'][0]:.6f}")
    print(f"  Final Val MSE: {history['val_mse'][-1]:.6f}")
    print(
        f"  Convergence: {'Yes' if history['train_loss'][-1] < history['train_loss'][0] * 0.5 else 'Partial'}"
    )
    print("=" * 60)
