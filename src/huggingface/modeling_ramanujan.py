"""
HuggingFace Transformers 兼容层实现 (v1.6)

提供:
    - RamanujanConfig: 配置类
    - RamanujanPreTrainedModel: 基类
    - RamanujanGPT2: GPT-2 风格实现
    - RamanujanLlama: Llama 风格实现
    - RamanujanMistral: Mistral 风格实现

依赖:
    pip install transformers
"""

import logging
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

logger = logging.getLogger('acx_ramanujan')

# ─── 配置类 ────────────────────────────────────────────────────────

class RamanujanConfig:
    """
    拉马努金 Transformer 配置

    兼容 HuggingFace PretrainedConfig 接口
    """

    model_type = "ramanujan"

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_len: int = 2048,
        decoder_only: bool = True,
        # Ramanujan 特有参数
        ramanujan_depth: Optional[int] = None,
        transition_depth: Optional[int] = None,
        max_depth: int = 1000,
        alpha: float = 0.3,
        lambda_decay: float = 0.5,
        quantization: str = 'none',
        long_context_seq_len: int = 2048,
        use_flash_attention: bool = True,
        sliding_window_size: Optional[int] = None,
        # MoE 参数
        num_experts: int = 0,
        top_k: int = 2,
        expert_dropout: float = 0.0,
        load_balancing_weight: float = 0.01,
        # 兼容参数
        hidden_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = hidden_size or d_model
        self.nhead = num_attention_heads or nhead
        self.num_layers = num_hidden_layers or num_layers
        self.dim_feedforward = intermediate_size or dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.max_len = max_position_embeddings or max_len
        self.decoder_only = decoder_only
        self.ramanujan_depth = ramanujan_depth
        self.transition_depth = transition_depth
        self.max_depth = max_depth
        self.alpha = alpha
        self.lambda_decay = lambda_decay
        self.quantization = quantization
        self.long_context_seq_len = long_context_seq_len
        self.use_flash_attention = use_flash_attention
        self.sliding_window_size = sliding_window_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dropout = expert_dropout
        self.load_balancing_weight = load_balancing_weight

        # HuggingFace 兼容属性
        self.hidden_size = self.d_model
        self.num_attention_heads = self.nhead
        self.num_hidden_layers = self.num_layers
        self.intermediate_size = self.dim_feedforward
        self.max_position_embeddings = self.max_len

        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RamanujanConfig':
        return cls(**config_dict)


# ─── 基类 ──────────────────────────────────────────────────────────

class RamanujanPreTrainedModel(nn.Module):
    """
    拉马努金 Transformer 预训练模型基类

    兼容 HuggingFace PreTrainedModel 接口
    """

    config_class = RamanujanConfig
    base_model_prefix = "ramanujan"

    def __init__(self, config: Optional[RamanujanConfig] = None):
        super().__init__()
        self.config = config or RamanujanConfig()

    def get_input_embeddings(self):
        if hasattr(self, 'embeddings'):
            return self.embeddings.token_embedding
        return None

    def set_input_embeddings(self, value):
        if hasattr(self, 'embeddings'):
            self.embeddings.token_embedding = value

    def get_output_embeddings(self):
        if hasattr(self, 'lm_head'):
            return self.lm_head
        return None

    def get_num_params(self, only_trainable: bool = True) -> int:
        params = self.parameters() if not only_trainable else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {'input_ids': input_ids}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        从预训练路径加载模型

        Args:
            pretrained_model_name_or_path: 模型路径或 HuggingFace 模型 ID
        """
        import json
        from pathlib import Path

        path = Path(pretrained_model_name_or_path)

        if path.exists():
            # 本地路径
            config_path = path / 'config.json'
            if config_path.exists():
                with open(config_path) as f:
                    config_dict = json.load(f)
                config = RamanujanConfig.from_dict(config_dict)
            else:
                config = RamanujanConfig()

            model = cls(config)

            # 加载权重
            weight_files = list(path.glob('*.bin')) + list(path.glob('*.safetensors'))
            if weight_files:
                state_dict = torch.load(weight_files[0], map_location='cpu', weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"从 {weight_files[0]} 加载权重")
        else:
            logger.warning(f"路径不存在: {pretrained_model_name_or_path}，使用随机初始化")
            config = kwargs.get('config', RamanujanConfig())
            model = cls(config)

        return model

    def save_pretrained(self, save_directory: str):
        """保存模型到目录"""
        import json
        from pathlib import Path

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        config_path = save_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # 保存权重
        weight_path = save_dir / 'pytorch_model.bin'
        torch.save(self.state_dict(), weight_path)

        logger.info(f"模型已保存至 {save_dir}")


# ─── GPT-2 风格 ───────────────────────────────────────────────────

class RamanujanGPT2(RamanujanPreTrainedModel):
    """
    GPT-2 风格的拉马努金 Transformer

    兼容 HuggingFace GPT2Model 接口
    """

    def __init__(self, config: Optional[RamanujanConfig] = None):
        super().__init__(config)
        config = self.config

        from ..ramanujan_transformer import RamanujanTransformerDecoder
        from ..moe import RamanujanMoETransformer

        if config.num_experts > 0:
            self.model = RamanujanMoETransformer(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers,
                dim_feedforward=config.dim_feedforward,
                num_experts=config.num_experts,
                top_k=config.top_k,
                dropout=config.dropout,
                activation=config.activation,
                max_len=config.max_len,
                max_depth=config.max_depth,
                decoder_only=True,
                expert_dropout=config.expert_dropout,
                load_balancing_weight=config.load_balancing_weight,
                long_context_seq_len=config.long_context_seq_len,
            )
            self._is_moe = True
        else:
            self.model = RamanujanTransformerDecoder(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                nhead=config.nhead,
                num_layers=config.num_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                max_len=config.max_len,
                max_depth=config.max_depth,
                alpha=config.alpha,
                lambda_decay=config.lambda_decay,
                quantization=config.quantization,
                long_context_seq_len=config.long_context_seq_len,
                use_flash_attention=config.use_flash_attention,
                sliding_window_size=config.sliding_window_size,
            )
            self._is_moe = False

        self.embeddings = self.model.embeddings
        self.lm_head = self.model.lm_head

    def forward(self, input_ids: torch.Tensor, **kwargs):
        if self._is_moe:
            logits, aux = self.model(input_ids, return_aux_loss=True)
            return {'logits': logits, 'aux_loss': aux}
        else:
            logits = self.model(input_ids)
            return {'logits': logits}

    def generate(self, input_ids: torch.Tensor, **kwargs):
        return self.model.generate(input_ids, **kwargs)


# ─── Llama 风格 ───────────────────────────────────────────────────

class RamanujanLlama(RamanujanPreTrainedModel):
    """
    Llama 风格的拉马努金 Transformer

    默认使用 SiLU 激活、RMSNorm、无 bias
    """

    def __init__(self, config: Optional[RamanujanConfig] = None):
        super().__init__(config)
        config = self.config
        config.activation = 'silu'

        from ..ramanujan_transformer import RamanujanTransformerDecoder
        self.model = RamanujanTransformerDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            max_len=config.max_len,
            max_depth=config.max_depth,
            alpha=config.alpha,
            lambda_decay=config.lambda_decay,
            quantization=config.quantization,
            long_context_seq_len=config.long_context_seq_len,
            use_flash_attention=config.use_flash_attention,
            sliding_window_size=config.sliding_window_size,
        )

        self.embeddings = self.model.embeddings
        self.lm_head = self.model.lm_head

    def forward(self, input_ids: torch.Tensor, **kwargs):
        logits = self.model(input_ids)
        return {'logits': logits}

    def generate(self, input_ids: torch.Tensor, **kwargs):
        return self.model.generate(input_ids, **kwargs)


# ─── Mistral 风格 ─────────────────────────────────────────────────

class RamanujanMistral(RamanujanPreTrainedModel):
    """
    Mistral 风格的拉马努金 Transformer

    默认使用 SiLU、滑动窗口注意力
    """

    def __init__(self, config: Optional[RamanujanConfig] = None):
        super().__init__(config)
        config = self.config
        config.activation = 'silu'
        if config.sliding_window_size is None:
            config.sliding_window_size = 4096

        from ..ramanujan_transformer import RamanujanTransformerDecoder
        self.model = RamanujanTransformerDecoder(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            max_len=config.max_len,
            max_depth=config.max_depth,
            alpha=config.alpha,
            lambda_decay=config.lambda_decay,
            quantization=config.quantization,
            long_context_seq_len=config.long_context_seq_len,
            use_flash_attention=config.use_flash_attention,
            sliding_window_size=config.sliding_window_size,
        )

        self.embeddings = self.model.embeddings
        self.lm_head = self.model.lm_head

    def forward(self, input_ids: torch.Tensor, **kwargs):
        logits = self.model(input_ids)
        return {'logits': logits}

    def generate(self, input_ids: torch.Tensor, **kwargs):
        return self.model.generate(input_ids, **kwargs)
