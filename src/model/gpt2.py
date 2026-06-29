"""GPT-2 implementation from scratch with modern improvements.

Modern features (all optional, backward compatible):
- RMSNorm (instead of LayerNorm)
- RoPE (Rotary Position Embeddings)
- SwiGLU (instead of GELU MLP)
- GQA (Grouped Query Attention)
- QK-Norm (Q/K normalization)
- KV-Cache (faster generation)
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# Suppress harmless flash attention warning on GTX 1080 Ti and older GPUs
# (PyTorch falls back to efficient_attention which is nearly as fast)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")


@dataclass
class GPT2Config:
    """GPT-2 model configuration with modern improvements."""

    # Core architecture
    vocab_size: int = 32000
    n_positions: int = 1024  # Max sequence length
    n_embd: int = 768  # Embedding dimension
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    n_inner: int = 3072  # FFN hidden dimension

    # Label smoothing (0.0 = hard labels, 0.1 = 10% spread to other tokens)
    label_smoothing: float = 0.0

    # Pad token ID (ignored in loss computation)
    pad_token_id: int = 0

    # Regularization
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    # Normalization
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"

    # Legacy flags (kept for compatibility)
    use_rope: bool = False  # Rotary Positional Embeddings
    use_mqa: bool = False  # Multi-Query Attention (use use_gqa + n_kv_heads=1 instead)

    # Modern improvement flags
    use_rmsnorm: bool = False  # Replace LayerNorm with RMSNorm
    use_swiglu: bool = False  # Replace GELU MLP with SwiGLU
    use_gqa: bool = False  # Enable Grouped Query Attention
    n_kv_heads: Optional[int] = None  # Number of KV heads for GQA (default: n_head)
    use_qk_norm: bool = False  # Apply RMSNorm to Q and K

    # Flash Attention (uses F.scaled_dot_product_attention)
    use_flash_attention: bool = False

    # RoPE configuration
    rope_theta: float = 10000.0  # Base frequency for RoPE
    rope_scaling: Optional[float] = None  # Scaling factor for extended context

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

        # Set default n_kv_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_head

        # Validate GQA configuration
        if self.use_gqa or self.use_mqa:
            assert self.n_head % self.n_kv_heads == 0, "n_head must be divisible by n_kv_heads"

        # MQA is a special case of GQA with n_kv_heads=1
        if self.use_mqa:
            self.use_gqa = True
            self.n_kv_heads = 1

        # Adjust n_inner for SwiGLU to maintain approximate parameter parity
        # SwiGLU has 3 projections vs 2 for standard MLP, so we reduce hidden dim
        if self.use_swiglu and self.n_inner == 4 * self.n_embd:
            # Standard: 2 * n_embd * n_inner params
            # SwiGLU: 3 * n_embd * n_inner params
            # For parity: n_inner_swiglu = 2/3 * n_inner_standard
            # Round to multiple of 256 for efficiency
            self.n_inner = ((2 * self.n_inner // 3 + 255) // 256) * 256


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm:
    - Does not center (subtract mean)
    - Does not have bias parameter
    - Uses RMS for normalization: x / sqrt(mean(x^2) + eps) * weight

    Reference: https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Cast to float32 for numerical stability, then cast back
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Applies rotation to query and key vectors based on position.

    Key properties:
    - Encodes relative position through rotation
    - No learned parameters
    - Naturally decays attention with distance

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        scaling_factor: Optional[float] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling_factor = scaling_factor

        # Precompute frequency bands: theta_i = theta^(-2i/dim)
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build initial cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for positions [0, seq_len)."""
        self.max_seq_len_cached = seq_len

        # Position indices
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # Apply scaling if provided (for extended context)
        if self.scaling_factor is not None:
            t = t / self.scaling_factor

        # Outer product: (seq_len,) x (dim/2,) -> (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)

        # Concatenate for full dimension: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to q and k.

        Args:
            q: Query tensor of shape (batch, n_head, seq_len, head_dim)
            k: Key tensor of shape (batch, n_kv_heads, seq_len, head_dim)
            position_ids: Optional position indices (batch, seq_len)

        Returns:
            Tuple of rotated (q, k) with same shapes
        """
        seq_len = q.size(2)

        # Extend cache if needed
        if seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len)

        # Get cos/sin for this sequence
        if position_ids is not None:
            # Custom position indices (for KV-cache)
            cos = self.cos_cached[position_ids]  # (batch, seq_len, dim)
            sin = self.sin_cached[position_ids]
            cos = cos.unsqueeze(1)  # (batch, 1, seq_len, dim)
            sin = sin.unsqueeze(1)
        else:
            # Standard sequential positions
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)

        return q_rotated, k_rotated

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotary embedding: x * cos + rotate_half(x) * sin

        Computed in float32 for precision (fp16 RoPE drifts on long sequences),
        then cast back to the input dtype.
        """
        orig_dtype = x.dtype
        x = x.float()
        cos = cos.float()
        sin = sin.float()
        out = (x * cos) + (self._rotate_half(x) * sin)
        return out.to(orig_dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with modern improvements.

    Supports:
    - Standard MHA (Multi-Head Attention)
    - MQA (Multi-Query Attention): n_kv_heads = 1
    - GQA (Grouped Query Attention): 1 < n_kv_heads < n_head
    - QK-Norm: RMSNorm applied to Q and K before attention
    - RoPE: Rotary position embeddings
    - KV-Cache: Efficient autoregressive generation
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads if config.use_gqa else config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = self.n_head // self.n_kv_heads  # Repetition factor for KV

        self.use_rope = config.use_rope
        self.use_qk_norm = config.use_qk_norm
        self.use_gqa = config.use_gqa
        self.use_flash_attention = config.use_flash_attention

        # Separate projections for Q, K, V (modern style, no bias)
        if config.use_gqa or config.use_rope or config.use_qk_norm:
            # Modern architecture: separate projections
            self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
            self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_dim, bias=False)
            self._use_separate_proj = True
        else:
            # Legacy: combined QKV projection
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
            self._use_separate_proj = False

        # Output projection (no bias in modern mode for consistency)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=not self._use_separate_proj)

        # QK-Norm (optional)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.layer_norm_epsilon)
            self.k_norm = RMSNorm(self.head_dim, eps=config.layer_norm_epsilon)

        # RoPE (optional)
        if self.use_rope:
            self.rotary_emb = RotaryPositionEmbedding(
                dim=self.head_dim,
                max_seq_len=config.n_positions,
                theta=config.rope_theta,
                scaling_factor=config.rope_scaling,
            )

        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.attn_pdrop = config.attn_pdrop

        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV-cache.

        Args:
            x: Input tensor (batch, seq_len, n_embd)
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE (batch, seq_len)
            past_key_value: Cached (K, V) from previous forward passes
            use_cache: Whether to return updated cache

        Returns:
            output: (batch, seq_len, n_embd)
            present_key_value: Updated (K, V) cache if use_cache=True
        """
        B, T, C = x.size()

        # Project Q, K, V
        if self._use_separate_proj:
            q = self.q_proj(x)  # (B, T, n_head * head_dim)
            k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
            v = self.v_proj(x)  # (B, T, n_kv_heads * head_dim)

            # Reshape for multi-head attention
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        else:
            # Legacy combined projection
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply QK-Norm if enabled
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE if enabled
        if self.use_rope:
            # Determine position IDs
            if position_ids is None:
                if past_key_value is not None:
                    past_len = past_key_value[0].size(2)
                    position_ids = torch.arange(
                        past_len, past_len + T, device=x.device
                    ).unsqueeze(0).expand(B, -1)
                else:
                    position_ids = torch.arange(T, device=x.device).unsqueeze(0)

            q, k = self.rotary_emb(q, k, position_ids)

        # KV-Cache: concatenate past keys/values
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Store for cache if needed
        present_key_value = (k, v) if use_cache else None

        # Repeat K, V for GQA
        if self.n_rep > 1:
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)

        S = k.size(2)  # Total sequence length (including cache)
        is_prefill = past_key_value is None  # T_q == T_k when no cache

        if self.use_flash_attention:
            # SDPA (efficient/flash kernel). Works with KV-cache too: SDPA's
            # is_causal flag assumes T_q == T_k, so we only use it on prefill.
            # During decode (T_q < T_k) we build an explicit causal mask.
            dropout_p = self.attn_pdrop if self.training else 0.0

            if attention_mask is None and is_prefill:
                # Fast path: built-in causal flag, no mask materialized.
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True,
                )
            else:
                # Build (T, S) causal mask aligned to the end of the sequence.
                causal = torch.ones(T, S, device=q.device, dtype=torch.bool).tril(diagonal=S - T)
                attn_mask = torch.zeros(T, S, device=q.device, dtype=q.dtype)
                attn_mask.masked_fill_(~causal, float("-inf"))
                if attention_mask is not None:
                    attn_mask = attn_mask + attention_mask.to(q.dtype)
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False,
                )
            self._last_attention_weights = None  # Not available with fused kernel
        else:
            # Manual attention (exposes attention weights for inspection)
            att = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_head, T, S)

            # Causal mask aligned to the end (handles both prefill and decode)
            causal_mask = torch.ones(T, S, device=x.device, dtype=torch.bool).tril(diagonal=S - T)
            att = att.masked_fill(~causal_mask, float("-inf"))

            if attention_mask is not None:
                att = att + attention_mask.to(att.dtype)

            att = F.softmax(att, dim=-1)
            self._last_attention_weights = att.detach()
            att = self.attn_dropout(att)
            y = att @ v  # (B, n_head, T, head_dim)

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with dropout
        y = self.resid_dropout(self.c_proj(y))

        return y, present_key_value

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match number of query heads (for GQA)."""
        B, n_kv_heads, S, head_dim = x.shape
        if self.n_rep == 1:
            return x
        # Expand and reshape: repeat each KV head n_rep times
        x = x[:, :, None, :, :].expand(B, n_kv_heads, self.n_rep, S, head_dim)
        return x.reshape(B, self.n_head, S, head_dim)


class MLP(nn.Module):
    """Position-wise feed-forward network with optional SwiGLU activation.

    Standard GELU MLP:
        x -> Linear(n_embd, n_inner) -> GELU -> Linear(n_inner, n_embd)

    SwiGLU MLP:
        gate, up = Linear(n_embd, 2*n_inner).chunk(2)
        output = SiLU(gate) * up
        output = Linear(n_inner, n_embd)(output)

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.use_swiglu = config.use_swiglu

        if self.use_swiglu:
            # SwiGLU: gate and up projection combined (no bias, modern style)
            self.gate_up_proj = nn.Linear(config.n_embd, 2 * config.n_inner, bias=False)
            self.down_proj = nn.Linear(config.n_inner, config.n_embd, bias=False)
            self.act = nn.SiLU()  # Swish activation
        else:
            # Standard GELU MLP
            self.c_fc = nn.Linear(config.n_embd, config.n_inner, bias=True)
            self.c_proj = nn.Linear(config.n_inner, config.n_embd, bias=True)

            if config.activation_function == "gelu":
                self.act = nn.GELU()
            elif config.activation_function == "relu":
                self.act = nn.ReLU()
            else:
                raise ValueError(f"Unsupported activation: {config.activation_function}")

        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.use_swiglu:
            # SwiGLU: split into gate and value, apply gated activation
            gate_up = self.gate_up_proj(x)  # (B, T, 2*n_inner)
            gate, up = gate_up.chunk(2, dim=-1)  # Each (B, T, n_inner)
            x = self.act(gate) * up  # Gated activation
            x = self.down_proj(x)
        else:
            # Standard GELU
            x = self.c_fc(x)
            x = self.act(x)
            x = self.c_proj(x)

        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with modern improvements.

    Architecture (pre-norm):
        x -> Norm -> Attention -> + -> Norm -> MLP -> +
             |__________________|      |____________|
    """

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.gradient_checkpointing = False

        # Normalization layers (RMSNorm or LayerNorm)
        if config.use_rmsnorm:
            self.ln_1 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
            self.ln_2 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
            self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV-cache support.

        Args:
            x: Input (batch, seq_len, n_embd)
            attention_mask: Optional attention mask
            position_ids: Position indices for RoPE
            past_key_value: Cached (K, V) for this layer
            use_cache: Whether to return updated cache

        Returns:
            output: (batch, seq_len, n_embd)
            present_key_value: Updated cache if use_cache=True
        """
        # Pre-norm attention
        if self.training and self.gradient_checkpointing and not use_cache:
            attn_output, present_key_value = torch.utils.checkpoint.checkpoint(
                self.attn,
                self.ln_1(x),
                attention_mask,
                position_ids,
                past_key_value,
                use_cache,
                use_reentrant=False,
            )
        else:
            attn_output, present_key_value = self.attn(
                self.ln_1(x),
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
        x = x + attn_output

        # Pre-norm MLP
        if self.training and self.gradient_checkpointing:
            x = x + torch.utils.checkpoint.checkpoint(
                self.mlp, self.ln_2(x), use_reentrant=False,
            )
        else:
            x = x + self.mlp(self.ln_2(x))

        return x, present_key_value


class GPT2(nn.Module):
    """GPT-2 Language Model with modern improvements."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        # Token embeddings
        wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embeddings (only if not using RoPE)
        wpe = None
        if not config.use_rope:
            wpe = nn.Embedding(config.n_positions, config.n_embd)

        # Final normalization
        if config.use_rmsnorm:
            ln_f = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        else:
            ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        # Build transformer
        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                drop=nn.Dropout(config.embd_pdrop),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_f=ln_f,
            )
        )

        # Add position embeddings if not using RoPE
        if wpe is not None:
            self.transformer["wpe"] = wpe

        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights first, then tie (so tying survives init)
        self.apply(self._init_weights)

        # GPT-2 scaled init: shrink the residual output projections by
        # 1/sqrt(2 * n_layer) so the residual stream variance stays bounded
        # with depth. Applied to attn output proj and MLP down proj.
        scale = (2 * config.n_layer) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight") or name.endswith("down_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 * scale)

        # Weight tying (after init to avoid double-init of the shared tensor)
        self.transformer.wte.weight = self.lm_head.weight

        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for all transformer blocks."""
        for block in self.transformer.h:
            block.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all transformer blocks."""
        for block in self.transformer.h:
            block.gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and "wpe" in self.transformer:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[List]]:
        """
        Forward pass with optional KV-cache.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Optional attention mask
            position_ids: Optional position indices for RoPE
            labels: Optional labels for loss computation
            past_key_values: List of (K, V) tuples per layer
            use_cache: Whether to return updated cache

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: Scalar if labels provided
            hidden_states: (batch, seq_len, n_embd)
            present_key_values: Updated cache if use_cache=True
        """
        device = input_ids.device
        B, T = input_ids.size()

        # Calculate total sequence length (including cache)
        past_length = 0
        if past_key_values is not None and past_key_values[0] is not None:
            past_length = past_key_values[0][0].size(2)

        total_length = past_length + T
        assert total_length <= self.config.n_positions, \
            f"Sequence length {total_length} exceeds max {self.config.n_positions}"

        # Token embeddings
        x = self.transformer.wte(input_ids)

        # Position embeddings (if not using RoPE)
        if "wpe" in self.transformer:
            if position_ids is None:
                position_ids = torch.arange(past_length, total_length, device=device)
                position_ids = position_ids.unsqueeze(0).expand(B, -1)
            pos_emb = self.transformer.wpe(position_ids)
            x = x + pos_emb

        x = self.transformer.drop(x)

        # Transformer blocks with optional caching
        present_key_values = [] if use_cache else None

        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None

            x, present_kv = block(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids if self.config.use_rope else None,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(present_kv)

        # Final normalization
        x = self.transformer.ln_f(x)
        hidden_states = x

        # Language modeling head
        logits = self.lm_head(x)

        # Loss computation
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                label_smoothing=self.config.label_smoothing,
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss, hidden_states, present_key_values

    @torch.no_grad()
    def forward_layer_logits(self, input_ids: torch.Tensor, layers):
        """Return logits computed from selected intermediate layers (for DoLa).

        For each requested layer index, the layer's hidden state is passed
        through the final norm + lm_head, giving "early-exit" logits. The final
        layer's logits are always included under key -1.

        Args:
            input_ids: (batch, seq_len)
            layers: iterable of layer indices (0-based) to read hidden states
                    after. Negative or out-of-range values are ignored.

        Returns:
            dict {layer_index: logits (batch, seq_len, vocab)}, plus key -1 for
            the final layer.
        """
        device = input_ids.device
        B, T = input_ids.size()
        wanted = set(int(l) for l in layers)

        x = self.transformer.wte(input_ids)
        if "wpe" in self.transformer:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
            x = x + self.transformer.wpe(position_ids)
        else:
            position_ids = torch.arange(T, device=device).unsqueeze(0)
        x = self.transformer.drop(x)

        out = {}
        n_layer = len(self.transformer.h)
        for i, block in enumerate(self.transformer.h):
            x, _ = block(
                x,
                attention_mask=None,
                position_ids=position_ids if self.config.use_rope else None,
                past_key_value=None,
                use_cache=False,
            )
            if i in wanted and i != n_layer - 1:
                # Project this layer's hidden state with the shared head.
                h = self.transformer.ln_f(x)
                out[i] = self.lm_head(h)

        # Final layer logits (always provided, key -1).
        h = self.transformer.ln_f(x)
        out[-1] = self.lm_head(h)
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively with optional KV-cache.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            do_sample: Whether to sample or greedy decode
            eos_token_id: Stop token ID
            use_cache: Whether to use KV-cache (faster generation)

        Returns:
            Generated token IDs (batch, total_length)
        """
        training = self.training
        self.eval()

        past_key_values = None

        for _ in range(max_length):
            # Determine input for this step
            if use_cache and past_key_values is not None:
                # Only process the last token when using cache
                input_ids_cond = input_ids[:, -1:]
            else:
                # Full sequence (crop if needed)
                input_ids_cond = (
                    input_ids if input_ids.size(1) <= self.config.n_positions
                    else input_ids[:, -self.config.n_positions:]
                )

            # Forward pass
            logits, _, _, past_key_values = self(
                input_ids_cond,
                past_key_values=past_key_values if use_cache else None,
                use_cache=use_cache,
            )

            # Get logits for last position
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty (HF semantics: divide positive logits,
            # multiply negative ones, so both move toward less likely).
            # Vectorized: gather logits of already-seen tokens, rescale, scatter.
            if repetition_penalty != 1.0:
                seen = logits.gather(1, input_ids)
                seen = torch.where(seen > 0, seen / repetition_penalty, seen * repetition_penalty)
                logits.scatter_(1, input_ids, seen)

            # Top-k filtering (clamp k to vocab size to avoid topk error)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                indices_to_remove = logits < torch.topk(logits, k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop on EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        self.train(training)
        return input_ids

    def save_pretrained(self, save_path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
            },
            save_path,
        )
        print(f"Model saved to {save_path}")

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cpu"):
        """Load model checkpoint."""
        # weights_only=True blocks arbitrary code execution from a malicious
        # checkpoint. The only non-tensor object we serialize is GPT2Config
        # (a dataclass), so we allowlist it explicitly.
        #
        # add_safe_globals + weights_only=True only exist on torch >= 2.4.
        # On older versions we fall back to a plain (trusted) load, since we
        # only ever load checkpoints we produced ourselves.
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([GPT2Config])
            checkpoint = torch.load(load_path, map_location=device, weights_only=True)
        else:
            checkpoint = torch.load(load_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = cls(config)

        # Backward compat: older checkpoints stored a per-layer causal mask
        # buffer "attn.bias" (shape n_positions x n_positions). The mask is now
        # built on the fly, so these keys are obsolete — drop them before load.
        state_dict = checkpoint["model_state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.endswith("attn.bias")}

        # strict=False also tolerates the missing buffers cleanly.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if unexpected:
            print(f"Warning: ignored unexpected keys: {unexpected}")
        # Only the intended obsolete buffers may be "missing"; flag anything else.
        real_missing = [k for k in missing if not k.endswith("attn.bias")]
        if real_missing:
            print(f"Warning: missing keys not initialized from checkpoint: {real_missing}")

        model = model.to(device)
        print(f"Model loaded from {load_path} on device: {device}")
        return model
