import argparse
import math
import time

import torch

from Data import DataLoder
from model import GPT, GPTConfig

max_lr = 6e-4
min_lr = max_lr * 0.1  # 10 %
warmup_steps = 10
max_steps = 4000
save_interval = 500  # Save the model every 500 steps

# Dictionary to store the model configurations
model_configs = {
    "gpt2": {"n_layer": 12, "n_head": 12, "n_embd": 768},  # 124M params
    "gpt2-medium": {"n_layer": 24, "n_head": 16, "n_embd": 1024},  # 350M params
    "gpt2-large": {"n_layer": 36, "n_head": 20, "n_embd": 1280},  # 774M params
    "gpt2-xl": {"n_layer": 48, "n_head": 25, "n_embd": 1600},  # 1558M params
}


def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model_name=None):
    if model_name is None:
        model = model = GPT(GPTConfig(vocab_size=50304))
    else:
        config = model_configs.get(model_name, {})
        if not config:
            raise ValueError(f"Unknown model name: {model_name}")
        model = GPT(GPTConfig(vocab_size=50304, **config))

    data_loader = DataLoder(B=4, T=128)

    model.eval()
    model.to(device)
    model = torch.compile(model)
    torch.set_float32_matmul_precision("high")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8
    )
    # Loop over the maximum number of training steps
    for step in range(max_steps):
        # Record the start time of the current training step
        t0 = time.time()

        # Get the next batch of data from the DataLoader
        x, y = data_loader.next_batch()

        # Move the input and target sequences to the device (GPU if available, otherwise CPU)
        x = x.to(device)
        y = y.to(device)

        # Zero out the gradients of the optimizer
        optimizer.zero_grad()

        # Uncomment the following lines to enable automatic mixed precision training
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        #     logits, loss = model(x, y)

        # Forward propagate the input sequence through the model and calculate the loss
        logits, loss = model(x, y)

        # Backpropagate the loss through the model to calculate the gradients
        loss.backward()

        # Clip the gradients to a maximum norm of 1.0 to prevent exploding gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Get the learning rate for the current training step using the learning rate scheduler
        lr = get_lr(step)

        # Update the learning rate for each parameter group in the optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Update the model's parameters using the optimizer
        optimizer.step()

        # Wait for the GPU to finish any remaining work
        torch.cuda.synchronize()

        # Record the end time of the current training step
        t1 = time.time()

        # Calculate the time difference between the start and end of the current training step
        dt = (t1 - t0) * 1000

        # Calculate the number of tokens processed per second
        tokens_per_sec = (data_loader.B * data_loader.T) / (t1 - t0)

        # Print the current training step, loss, time difference, gradient norm, and tokens processed per second
        print(
            f"step {step}, loss: {loss.item()}, dt: {dt:.2f}ms, norm : {norm:.4f} ,tok/sec: {tokens_per_sec:.2f}"
        )
        # Save the model periodically
        if (step + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"model_step_{step + 1}.pt")
            print(f"Model saved at step {step + 1}")

        # Save the final model
        torch.save(model.state_dict(), "model_final.pt")
        print("Final model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GPT model.")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="Name of the pre-trained model to load. If None, a gpt2 small model will be initialized.",
    )

    args = parser.parse_args()
    train(args.model_name)
