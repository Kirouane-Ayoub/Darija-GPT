from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This dictionary contains the configuration parameters for different versions of the GPT-2 model
# The parameters include the number of layers, the number of attention heads, and the embedding dimension
# The numbers provided are for the 'gpt2-large' model
"""
{
    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
}
"""


# The GPTConfig class is a data class that holds the configuration parameters for the GPT-2 model
# The parameters are initialized with the values for the 'gpt2-large' model
@dataclass
class GPTConfig:
    block_size: int = 1024  # The maximum context length for the model
    vocab_size: int = 50257  # The size of the vocabulary
    n_layer: int = 36  # The number of layers in the model
    n_head: int = 20  # The number of attention heads in each layer
    n_embd: int = 1280  # The embedding dimension


# The MLP class defines a multi-layer perceptron (MLP) module for the GPT-2 model
class MLP(nn.Module):
    # The __init__ method initializes the MLP module with the given configuration
    def __init__(self, config):
        super().__init__()
        # The first linear layer (c_fc) takes the input embedding and maps it to a higher-dimensional space
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # The gelu activation function is applied to the output of the first linear layer
        self.gelu = nn.GELU(approximate="tanh")
        # The second linear layer (c_proj) maps the output of the gelu activation function back to the original embedding dimension
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # The NANOGPT_SCALE_INIT attribute is set to 1 to scale the output of the second linear layer
        self.c_proj.NANOGPT_SCALE_INIT = 1

    # The forward method defines the forward pass of the MLP module
    def forward(self, x):
        # The input is passed through the first linear layer
        x = self.c_fc(x)
        # The output of the first linear layer is passed through the gelu activation function
        x = self.gelu(x)
        # The output of the gelu activation function is passed through the second linear layer
        x = self.c_proj(x)
        # The output of the second linear layer is returned as the output of the MLP module
        return x


# The CausalSelfAttention class defines a causal self-attention module for the GPT-2 model
class CausalSelfAttention(nn.Module):
    # The __init__ method initializes the causal self-attention module with the given configuration
    def __init__(self, config):
        super().__init__()
        # Assert that the embedding dimension is divisible by the number of attention heads
        assert config.n_embd % config.n_head == 0
        # The c_attn linear layer computes the query, key, and value vectors for all attention heads in batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # The c_proj linear layer computes the output projection of the attention module
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # The NANOGPT_SCALE_INIT attribute is set to 1 to scale the output of the c_proj linear layer
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # The number of attention heads and the embedding dimension are stored as attributes of the module
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    # The forward method defines the forward pass of the causal self-attention module
    def forward(self, x):
        # The input is expected to have shape (B, T, C), where B is the batch size, T is the sequence length, and C is the embedding dimensionality
        B, T, C = x.size()
        # The c_attn linear layer is applied to the input to compute the query, key, and value vectors for all attention heads in batch
        qkv = self.c_attn(x)
        # The query, key, and value vectors are split along the channel dimension
        q, k, v = qkv.split(self.n_embd, dim=2)
        # The key and query vectors are reshaped to have shape (B, nh, T, hs), where nh is the number of attention heads and hs is the head size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # The value vectors are reshaped to have shape (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # The scaled dot-product attention is computed using the query, key, and value vectors
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Flash attention
        # The output of the attention module is reshaped to have shape (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # The output projection is computed using the c_proj linear layer
        y = self.c_proj(y)
        # The output of the causal self-attention module is returned
        return y


# The DecoderBlock class defines a decoder block for the GPT-2 model
class DecoderBlock(nn.Module):
    # The __init__ method initializes the decoder block with the given configuration
    def __init__(self, config):
        super().__init__()
        # The first layer normalization layer (ln_1) is applied to the input before the causal self-attention module
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # The causal self-attention module (attn) computes the attention weights for the input sequence
        self.attn = CausalSelfAttention(config)
        # The second layer normalization layer (ln_2) is applied to the input before the multi-layer perceptron module
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # The multi-layer perceptron module (mlp) computes the output of the decoder block
        self.mlp = MLP(config)

    # The forward method defines the forward pass of the decoder block
    def forward(self, x):
        # The input is passed through the causal self-attention module, with layer normalization applied to the input before the module
        x = x + self.attn(self.ln_1(x))
        # The output of the causal self-attention module is passed through the multi-layer perceptron module, with layer normalization applied to the input before the module
        x = x + self.mlp(self.ln_2(x))
        # The output of the decoder block is returned
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(
                    config.vocab_size, config.n_embd
                ),  # Token Embedding
                "wpe": nn.Embedding(
                    config.block_size, config.n_embd
                ),  # Positional Encoding
                "h": nn.ModuleList(
                    [DecoderBlock(config) for _ in range(config.n_layer)]
                ),  # The Decoder Block
                "ln_f": nn.LayerNorm(config.n_embd),  # Layer Normalization
            }
        )

        # output projection
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing
        self.transformer.wte.weight = self.lm_head.weight

        #     # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # the Forward
    def forward(self, idx, target=None):
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of size {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_embed = self.transformer.wpe(pos)  # positional embedding
        tok_embed = self.transformer.wte(idx)  # token embedding
        x = tok_embed + pos_embed

        # forward the blocks ofthe transformer

        for block in self.transformer.h:
            x = block(x)

        # forward the final LayerNorm and the classifier

        x = self.transformer.ln_f(x)
        loss = None
        logits = self.lm_head(x)
        # Calculate the loss
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss
