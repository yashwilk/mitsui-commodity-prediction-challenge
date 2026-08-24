"""
rope_example.py
---------------
RoPE (Rotary Position Embedding) — example implementation.
For reference only — not used in the current project.

RoPE is used in: LLaMA, GPT-NeoX, Mistral, Falcon.

Core idea:
  Instead of ADDING a positional vector to the embedding,
  RoPE ROTATES the query and key vectors in attention by
  an angle proportional to their position.

  This means position is encoded in the RELATIONSHIP between
  tokens, not in the tokens themselves — it captures relative
  positions naturally.

Why sinusoidal adds position:
  x_new = x + positional_vector(position)

Why RoPE rotates instead:
  x_new = rotate(x, angle(position))
  where angle depends on position and dimension index

The key property RoPE gives you:
  dot_product(q_i, k_j) depends only on (i - j)
  — the relative distance between positions i and j
  — not their absolute positions
  This is why it generalises better to long sequences.


"""

import math
import torch
import torch.nn as nn


# ============================================================
# ROPE CORE FUNCTIONS
# ============================================================

def compute_rope_frequencies(
    d_model   : int,
    max_len   : int,
    base      : float = 10000.0,
    device    : torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the sin and cos rotation matrices for RoPE.

    For each position p and each pair of dimensions (2i, 2i+1):
        theta_i = 1 / (base ^ (2i / d_model))
        angle   = p * theta_i

    Returns cos and sin matrices of shape (max_len, d_model // 2).

    Parameters
    ----------
    d_model : embedding dimension (must be even)
    max_len : maximum sequence length
    base    : controls frequency range (10000 is standard)
    """
    assert d_model % 2 == 0, "d_model must be even for RoPE"

    # dimension indices: 0, 2, 4, ..., d_model-2
    # shape: (d_model // 2,)
    dim_idx = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)

    # theta for each dimension pair
    # theta_i = 1 / (base ^ (2i / d_model))
    # shape: (d_model // 2,)
    theta = 1.0 / (base ** (dim_idx / d_model))

    # position indices: 0, 1, 2, ..., max_len-1
    # shape: (max_len,)
    positions = torch.arange(max_len, dtype=torch.float32, device=device)

    # outer product: angle[p, i] = p * theta_i
    # shape: (max_len, d_model // 2)
    angles = torch.outer(positions, theta)

    # precompute cos and sin
    cos = torch.cos(angles)   # (max_len, d_model // 2)
    sin = torch.sin(angles)   # (max_len, d_model // 2)

    return cos, sin


def apply_rope(
    x  : torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE rotation to a tensor.

    For each pair of dimensions (x_2i, x_2i+1) at position p:
        x_2i_new   =  x_2i   * cos(p * theta_i)
                    - x_2i+1 * sin(p * theta_i)
        x_2i+1_new =  x_2i   * sin(p * theta_i)
                    + x_2i+1 * cos(p * theta_i)

    This is a 2D rotation matrix applied to each pair of dims.

    Parameters
    ----------
    x   : (batch, seq_len, d_model)
    cos : (seq_len, d_model // 2)
    sin : (seq_len, d_model // 2)

    Returns
    -------
    x_rotated : (batch, seq_len, d_model)
    """
    batch, seq_len, d_model = x.shape

    # split into even and odd dimensions
    # x_even: dimensions 0, 2, 4, ...  shape: (batch, seq_len, d_model//2)
    # x_odd:  dimensions 1, 3, 5, ...  shape: (batch, seq_len, d_model//2)
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]

    # add batch dimension to cos/sin for broadcasting
    # cos: (1, seq_len, d_model//2)
    cos = cos[:seq_len].unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0)

    # apply 2D rotation to each pair
    x_even_rotated = x_even * cos - x_odd * sin
    x_odd_rotated  = x_even * sin + x_odd * cos

    # interleave back: [even_0, odd_0, even_1, odd_1, ...]
    x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
    x_rotated = x_rotated.flatten(-2)    # (batch, seq_len, d_model)

    return x_rotated


# ============================================================
# ROPE POSITIONAL EMBEDDING MODULE
# ============================================================

class RoPEEmbedding(nn.Module):
    """
    RoPE module that can replace PositionalEncoding
    or nn.Embedding in the TransformerEncoderNet.

    Unlike sinusoidal (adds) and learned (adds), RoPE
    rotates — it is applied inside the attention mechanism
    to queries and keys, not to the full embedding.

    For simplicity this implementation applies it to the
    full embedding (same position as sinusoidal/learned)
    rather than inside the attention heads.

    In production LLM implementations (LLaMA etc) RoPE is
    applied inside each attention head separately, which
    requires modifying nn.TransformerEncoderLayer.
    That level of modification is not shown here.
    """

    def __init__(
        self,
        d_model : int,
        max_len : int   = 512,
        base    : float = 10000.0,
        dropout : float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # precompute frequencies — not parameters, not updated
        cos, sin = compute_rope_frequencies(d_model, max_len, base)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, d_model) — rotated
        """
        x = apply_rope(x, self.cos, self.sin)
        return self.dropout(x)


# ============================================================
# HOW TO SWAP INTO transformer.py
# ============================================================

"""
To use RoPE instead of learned embeddings in transformer.py,
replace the positional embedding section in __init__():

    # REMOVE these lines:
    self.pos_embedding = nn.Embedding(n_features, d_model)
    self.pos_dropout   = nn.Dropout(p=dropout)

    # ADD this line:
    self.pos_encoding  = RoPEEmbedding(d_model, max_len=n_features+1, dropout=dropout)

And in forward(), replace:

    # REMOVE these lines:
    positions = torch.arange(self.n_features, device=x.device)
    pos_emb   = self.pos_embedding(positions)
    x = x + pos_emb.unsqueeze(0)
    x = self.pos_dropout(x)

    # ADD this line:
    x = self.pos_encoding(x)

Everything else stays identical.
"""


# ============================================================
# COMPARISON OF ALL THREE APPROACHES
# ============================================================

"""
SINUSOIDAL
  Formula  : PE(pos, 2i)   = sin(pos / 10000^(2i/d))
             PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
  Operation: ADD to embedding
  Parameters: 0
  Learns: nothing — fixed forever
  Best for: when you want zero positional parameters

LEARNED
  Formula  : lookup table of shape (n_positions, d_model)
  Operation: ADD to embedding
  Parameters: n_features × d_model (e.g. 34 × 64 = 2,176)
  Learns: what positional structure is useful
  Best for: tabular data, short sequences, medium datasets

ROPE
  Formula  : rotate (q, k) by angle proportional to position
  Operation: ROTATE queries and keys in attention
  Parameters: 0
  Learns: nothing — but captures relative positions
  Best for: long sequences (1000+ tokens), LLM-style models
  Relative position property: dot(q_i, k_j) = f(i-j)

RECOMMENDATION FOR THIS PROJECT:
  Use learned embeddings (current implementation).
  34 features is too short for RoPE to shine.
  ~1875 rows per target is too small for sinusoidal to
  hurt but also too small for learned to overfit.
"""


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    batch    = 4
    seq_len  = 34     # n_features in your project
    d_model  = 64

    x   = torch.randn(batch, seq_len, d_model)
    cos, sin = compute_rope_frequencies(d_model, max_len=seq_len)
    out = apply_rope(x, cos, sin)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Shape preserved: {x.shape == out.shape}")
    print(f"Values changed : {not torch.allclose(x, out)}")

    # verify relative position property
    # dot(q_i, k_j) should only depend on (i - j)
    q = torch.randn(1, seq_len, d_model)
    k = torch.randn(1, seq_len, d_model)

    q_rot = apply_rope(q, cos, sin)
    k_rot = apply_rope(k, cos, sin)

    # dot product between position 2 and position 5
    dot_2_5 = (q_rot[0, 2] * k_rot[0, 5]).sum().item()

    # dot product between position 0 and position 3
    # relative distance is also 3 — should be similar magnitude
    dot_0_3 = (q_rot[0, 0] * k_rot[0, 3]).sum().item()

    print(f"\nRelative position property check:")
    print(f"dot(q_2, k_5) = {dot_2_5:.4f}  (distance=3)")
    print(f"dot(q_0, k_3) = {dot_0_3:.4f}  (distance=3)")
    print("Both have same relative distance — magnitudes should be similar")