#    Copyright 2024 LASAR Navigation Team
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file implements LASAR (LAnguage-guided Spatial AtlAs Reasoning)
# Fixed version of VEME with improved [MAP] token injection

import inspect
import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.loss import soft_cross_entropy

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaLlamaConfig(LlavaConfig):
    model_type = "llava_llama"
    
    def __init__(
        self,
        # Cognitive Map parameters
        num_world_primitives: int = 256,
        world_primitive_dim: int = 4096,
        
        # Attention parameters
        num_map_heads: int = 8,
        
        # Loss weights
        lambda_qa: float = 1.0,
        lambda_semantic: float = 0.1,
        lambda_context: float = 0.1,
        lambda_retro: float = 0.1,
        
        # Contrastive learning parameters
        contrastive_temperature: float = 0.07,
        num_hard_negatives: int = 8,
        
        # VQ parameters
        vq_gamma: float = 0.1,  # entropy regularization weight

        # Runtime toggles (default off: match NaVILA forward for navigation)
        use_lasar_injection: bool = False,
        tune_lasar_modules: bool = False,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_world_primitives = num_world_primitives
        self.world_primitive_dim = world_primitive_dim
        self.num_map_heads = num_map_heads
        self.lambda_qa = lambda_qa
        self.lambda_semantic = lambda_semantic
        self.lambda_context = lambda_context
        self.lambda_retro = lambda_retro
        self.contrastive_temperature = contrastive_temperature
        self.num_hard_negatives = num_hard_negatives
        self.vq_gamma = vq_gamma
        self.use_lasar_injection = use_lasar_injection
        self.tune_lasar_modules = tune_lasar_modules


LASAR_MODULE_KEY_PREFIXES = (
    "cognitive_map_encoder.",
    "belief_projection.",
    "belief_gate",
    "episode_projection.",
)
LASAR_MODULES_FILENAME = "lasar_modules.bin"


class AttentionPooling(nn.Module):
    """
    Single-head attention pooling to summarize episodic memory.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            batch_first=True
        )
        
    def forward(self, episodic_memory):
        """
        Args:
            episodic_memory: [B, T, D] - sequence of visual features
        Returns:
            pooled: [B, D] - summarized representation
        """
        batch_size = episodic_memory.size(0)
        query = self.query.expand(batch_size, -1, -1)  # [B, 1, D]
        
        pooled, _ = self.attention(query, episodic_memory, episodic_memory)
        return pooled.squeeze(1)  # [B, D]


class SemanticAtlas(nn.Module):
    """
    Learnable codebook of world primitives for the cognitive map.
    Implements Vector Quantization with entropy regularization.
    """
    def __init__(self, num_primitives, primitive_dim):
        super().__init__()
        self.num_primitives = num_primitives
        self.primitive_dim = primitive_dim
        
        # Learnable world primitives: E_world = {e_1, ..., e_N}
        self.primitives = nn.Parameter(
            torch.randn(num_primitives, primitive_dim)
        )
        # Use Xavier initialization for better numerical stability
        nn.init.xavier_uniform_(self.primitives, gain=0.5)
        
    def forward(self, features, compute_loss=False):
        """
        Args:
            features: [B, D] or [B, T, D]
            compute_loss: whether to compute VQ loss
        Returns:
            If compute_loss:
                selected_primitives, loss, usage_probs
            Else:
                selected_primitives
        """
        original_shape = features.shape
        if len(original_shape) == 3:
            B, T, D = original_shape
            features = features.reshape(B * T, D)
        else:
            features = features
            
        # Compute distances to all primitives
        # features: [B, D], primitives: [N, D]
        distances = torch.cdist(features, self.primitives)  # [B, N]
        
        # Find nearest primitive for each feature
        nearest_idx = distances.argmin(dim=-1)  # [B]
        selected_primitives = self.primitives[nearest_idx]  # [B, D]
        
        if compute_loss:
            # Vector Quantization loss: ||sg(F'_vis) - e_j||^2
            vq_loss = F.mse_loss(features.detach(), selected_primitives)
            
            # Entropy regularization to prevent collapse
            # Compute usage probability across the batch
            usage_counts = torch.bincount(
                nearest_idx, 
                minlength=self.num_primitives
            ).float()
            usage_probs = usage_counts / usage_counts.sum()
            
            # Entropy: -sum(p_k * log(p_k))
            # We want to maximize entropy, so minimize negative entropy
            epsilon = 1e-8
            entropy = -(usage_probs * torch.log(usage_probs + epsilon)).sum()
            
            # Restore original shape if needed
            if len(original_shape) == 3:
                selected_primitives = selected_primitives.reshape(B, T, D)
            
            return selected_primitives, vq_loss, usage_probs, entropy
        else:
            if len(original_shape) == 3:
                selected_primitives = selected_primitives.reshape(B, T, D)
            return selected_primitives


class CognitiveMapEncoder(nn.Module):
    """
    Encodes episodic memory into a contextual belief state using Semantic Atlas.
    m_t = CrossAttn(AttnPool(M_epi,t), E_world, E_world)
    """
    def __init__(self, hidden_dim, num_primitives, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention_pooling = AttentionPooling(hidden_dim)
        self.semantic_atlas = SemanticAtlas(num_primitives, hidden_dim)
        
        # Cross-attention to query the atlas
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize cross-attention weights for better numerical stability
        self._init_cross_attention_weights()
    
    def _init_cross_attention_weights(self):
        """Initialize cross-attention weights with smaller values for stability."""
        for name, param in self.cross_attention.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name and param is not None:
                nn.init.zeros_(param)
        
    def forward(self, episodic_memory, compute_atlas_loss=False):
        """
        Args:
            episodic_memory: [B, T, D] - M_epi,t
            compute_atlas_loss: whether to compute semantic atlas loss
        Returns:
            belief_state: [B, D] - m_t
            (optional) atlas_loss, usage_probs, entropy
        """
        # Check for NaN in input
        if torch.isnan(episodic_memory).any():
            print("⚠️  Warning: NaN detected in episodic_memory input")
            episodic_memory = torch.nan_to_num(episodic_memory, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Step 1: Attention pooling to summarize episodic context
        pooled_context = self.attention_pooling(episodic_memory)  # [B, D]
        
        # Numerical stability: clamp extreme values
        pooled_context = torch.clamp(pooled_context, min=-1e4, max=1e4)
        pooled_context = pooled_context.unsqueeze(1)  # [B, 1, D]
        
        # Step 2: Get atlas primitives (all of them for cross-attention)
        atlas_primitives = self.semantic_atlas.primitives.unsqueeze(0)  # [1, N, D]
        batch_size = episodic_memory.size(0)
        atlas_primitives = atlas_primitives.expand(batch_size, -1, -1)  # [B, N, D]
        
        # Normalize for stable attention computation
        pooled_context_norm = F.layer_norm(pooled_context, (self.hidden_dim,))
        atlas_primitives_norm = F.layer_norm(atlas_primitives, (self.hidden_dim,))
        
        # Step 3: Cross-attention with atlas
        belief_state, _ = self.cross_attention(
            query=pooled_context_norm,
            key=atlas_primitives_norm,
            value=atlas_primitives_norm
        )
        belief_state = belief_state.squeeze(1)  # [B, D]
        belief_state = self.layer_norm(belief_state)
        
        # Final stability check
        if torch.isnan(belief_state).any():
            print("⚠️  Warning: NaN detected in belief_state, replacing with zeros")
            belief_state = torch.nan_to_num(belief_state, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if compute_atlas_loss:
            # Compute VQ loss using the episodic memory features
            _, vq_loss, usage_probs, entropy = self.semantic_atlas(
                episodic_memory, compute_loss=True
            )
            return belief_state, vq_loss, usage_probs, entropy
        
        return belief_state


class LlavaLlamaModel(LlavaMetaModel, LlavaMetaForCausalLM, PreTrainedModel):
    """
    LASAR Model with dual-memory system for embodied navigation.
    Fixed version of VEME with improved [MAP] token injection.
    
    Architecture:
    1. Visual encoder (Siglip + VGGT fusion) - already in veme_encoder.py
    2. Episodic Memory: M_epi,t = (F'_vis,0, ..., F'_vis,t)
    3. Cognitive Map: m_t = CrossAttn(AttnPool(M_epi,t), E_world, E_world)
    4. Unified LLM backbone for action and QA
    
    Key Improvements over VEME:
    - [MAP] token is inserted BEFORE <eos>, not after
    - Uses dedicated special token to avoid conflicts
    - Uses learnable gate for belief fusion instead of forced normalization
    """
    config_class = LlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: LlavaLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)
        
        # ============================================
        # LASAR Components: Cognitive Map & Episodic Memory
        # ============================================
        # Get hidden dimension from LLM
        self.hidden_dim = self.llm.config.hidden_size
        
        # Cognitive Map Encoder
        self.cognitive_map_encoder = CognitiveMapEncoder(
            hidden_dim=self.hidden_dim,
            num_primitives=config.num_world_primitives,
            num_heads=config.num_map_heads
        )
        
        # ✅ FIX 1: Use dedicated special token instead of vocab_size - 1
        # Add [MAP] as a special token to avoid conflicts
        if '[MAP]' not in self.tokenizer.get_vocab():
            num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': ['[MAP]']})
            if num_added > 0:
                # Resize token embeddings to accommodate new token
                self.llm.resize_token_embeddings(len(self.tokenizer))
                print(f"✓ Added [MAP] token to vocabulary (new vocab size: {len(self.tokenizer)})")
        
        self.map_token_id = self.tokenizer.convert_tokens_to_ids('[MAP]')
        print(f"📍 Using token ID {self.map_token_id} as [MAP] token")
        
        # Projection layer to align belief state with LLM embedding dimension
        self.belief_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        # Initialize with small weights for numerical stability
        nn.init.xavier_uniform_(self.belief_projection.weight, gain=0.5)
        nn.init.zeros_(self.belief_projection.bias)
        
        # ✅ FIX 4: Learnable gate for belief fusion (instead of forced normalization)
        self.belief_gate = nn.Parameter(torch.zeros(1))
        print(f"✓ Initialized learnable belief gate (initial value: {self.belief_gate.item():.4f})")
        
        # For episodic contrastive learning
        self.episode_projection = nn.Linear(self.hidden_dim, 256)
        nn.init.xavier_uniform_(self.episode_projection.weight, gain=1.0)
        nn.init.zeros_(self.episode_projection.bias)
        
        # Store config for loss weights
        self.lambda_qa = config.lambda_qa
        self.lambda_semantic = config.lambda_semantic
        self.lambda_context = config.lambda_context
        self.lambda_retro = config.lambda_retro
        self.contrastive_temp = config.contrastive_temperature
        self.vq_gamma = config.vq_gamma

        checkpoint_dir = getattr(config, "resume_path", None) or getattr(config, "_name_or_path", None)
        if checkpoint_dir and os.path.isdir(str(checkpoint_dir)):
            self.load_lasar_modules(checkpoint_dir)

    def _is_lasar_param(self, name: str) -> bool:
        return any(name.startswith(prefix) or name == prefix.rstrip(".") for prefix in LASAR_MODULE_KEY_PREFIXES)

    def _lasar_state_dict(self) -> dict:
        return {k: v for k, v in self.state_dict().items() if self._is_lasar_param(k)}

    def save_lasar_modules(self, output_dir: str) -> None:
        lasar_state = self._lasar_state_dict()
        if not lasar_state:
            return
        lasar_dir = os.path.join(output_dir, "lasar")
        os.makedirs(lasar_dir, exist_ok=True)
        save_path = os.path.join(lasar_dir, LASAR_MODULES_FILENAME)
        torch.save(lasar_state, save_path)
        print(f"✓ Saved LASAR modules ({len(lasar_state)} tensors) to {save_path}")

    def load_lasar_modules(self, checkpoint_dir: str) -> bool:
        if not hasattr(self, "cognitive_map_encoder"):
            return False
        load_path = os.path.join(checkpoint_dir, "lasar", LASAR_MODULES_FILENAME)
        if not os.path.isfile(load_path):
            return False
        lasar_state = torch.load(load_path, map_location="cpu")
        missing, unexpected = self.load_state_dict(lasar_state, strict=False)
        if missing:
            print(f"⚠️  LASAR load missing keys ({len(missing)}), e.g. {missing[:3]}")
        if unexpected:
            print(f"⚠️  LASAR load unexpected keys ({len(unexpected)}), e.g. {unexpected[:3]}")
        print(f"✓ Loaded LASAR modules from {load_path}")
        return True

    def save_pretrained(self, output_dir, state_dict=None):
        super().save_pretrained(output_dir, state_dict=state_dict)
        self.save_lasar_modules(output_dir)

    def post_config(self):
        super().post_config()
        checkpoint_dir = getattr(self.config, "resume_path", None) or getattr(self.config, "_name_or_path", None)
        if checkpoint_dir and os.path.isdir(str(checkpoint_dir)):
            self.load_lasar_modules(checkpoint_dir)

    def freezed_module_patch(self):
        super().freezed_module_patch()
        if not getattr(self.config, "tune_lasar_modules", False):
            if hasattr(self, "cognitive_map_encoder"):
                self.cognitive_map_encoder.eval()
            if hasattr(self, "belief_projection"):
                self.belief_projection.eval()
            if hasattr(self, "episode_projection"):
                self.episode_projection.eval()

    @staticmethod
    def set_lasar_trainable(model: "LlavaLlamaModel", tune_lasar: bool) -> None:
        if not hasattr(model, "cognitive_map_encoder"):
            return
        for name, param in model.named_parameters():
            if model._is_lasar_param(name):
                param.requires_grad_(tune_lasar)

    def _forward_navila_baseline(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """NaVILA-compatible forward without [MAP] / cognitive-map injection."""
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        support_packing = "seqlens_in_batch" in inspect.signature(self.llm.forward).parameters

        if self.training and support_packing and not dpo_forward:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int() if attention_mask is not None else seqlens_in_batch
            new_input_ids = input_ids

        if support_packing:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                seqlens_in_batch=sorted_seqlens_in_batch,
            )
        else:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.training and self.config.time_token_ids:
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                new_labels,
                soft_tokens=self.config.time_token_ids,
                std=self.config.soft_ce_std,
            )

        loss_weight = calculate_loss_weight(new_labels)
        outputs.loss = outputs.loss * loss_weight

        if dpo_forward:
            return outputs.logits, new_labels

        return outputs

    def _resolve_lasar_inject_mask(
        self,
        batch_size: int,
        device: torch.device,
        has_query: Optional[torch.BoolTensor],
    ) -> torch.BoolTensor:
        if not getattr(self.config, "use_lasar_injection", False):
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        if has_query is None:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return has_query.to(device=device, dtype=torch.bool).view(-1)

    def _register_gradient_hooks(self):
        """Register hooks to monitor and clip gradients for LASAR components."""
        def gradient_clip_hook(grad, max_norm=10.0):
            """Clip gradients to prevent explosion."""
            if grad is None:
                return None
            # Check for NaN
            if torch.isnan(grad).any():
                print("⚠️  NaN detected in gradient, replacing with zeros")
                return torch.zeros_like(grad)
            # Clip gradient norm
            grad_norm = grad.norm()
            if grad_norm > max_norm:
                return grad * (max_norm / (grad_norm + 1e-6))
            return grad
        
        # Register hooks for LASAR-specific components
        if hasattr(self.cognitive_map_encoder.semantic_atlas, 'primitives'):
            self.cognitive_map_encoder.semantic_atlas.primitives.register_hook(
                lambda grad: gradient_clip_hook(grad, max_norm=5.0)
            )
        
        for param in self.belief_projection.parameters():
            param.register_hook(lambda grad: gradient_clip_hook(grad, max_norm=10.0))
        
        for param in self.episode_projection.parameters():
            param.register_hook(lambda grad: gradient_clip_hook(grad, max_norm=10.0))

    def load_state_dict(self, state_dict, strict=True):
        """
        Override load_state_dict to handle missing LASAR-specific components.
        This allows loading from LlavaLlama checkpoints where these components don't exist.
        """
        # LASAR-specific components that might be missing in base LlavaLlama checkpoints
        lasar_components = [
            'cognitive_map_encoder',
            'belief_projection',
            'belief_gate',
            'episode_projection',
        ]
        
        # Check if any LASAR components are missing
        missing_lasar = False
        for key in list(state_dict.keys()):
            if any(comp in key for comp in lasar_components):
                break
        else:
            # No LASAR components found in state_dict, likely loading from LlavaLlama
            missing_lasar = True
        
        if missing_lasar and strict:
            print("⚠️  Loading from LlavaLlama checkpoint: LASAR components will be randomly initialized")
            strict = False
        
        # Load with potentially relaxed strictness
        return super().load_state_dict(state_dict, strict=strict)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        """
        Smart loading: automatically handles LlavaLlama -> LASAR conversion.
        
        When loading from LlavaLlama checkpoint:
        1. Loads base components (LLM, Vision, Projector) from checkpoint
        2. Initializes LASAR components randomly
        3. Works with DeepSpeed by avoiding optimizer state mismatch
        """
        # Check if config exists and has LASAR parameters
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        elif isinstance(config, str):
            config = AutoConfig.from_pretrained(config)
        
        # Ensure LASAR parameters exist in config
        lasar_params = {
            'num_world_primitives': 256,
            'world_primitive_dim': 4096,
            'num_map_heads': 8,
            'lambda_qa': 1.0,
            'lambda_semantic': 0.1,
            'lambda_context': 0.1,
            'lambda_retro': 0.1,
            'contrastive_temperature': 0.07,
            'num_hard_negatives': 8,
            'vq_gamma': 0.1,
            'use_lasar_injection': False,
            'tune_lasar_modules': False,
        }
        
        for key, default_val in lasar_params.items():
            if not hasattr(config, key):
                setattr(config, key, default_val)
        
        # Use parent class from_pretrained with strict=False to handle missing keys
        kwargs['ignore_mismatched_sizes'] = True
        
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaLlamaModel, cls).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def compute_infonce_loss(self, anchor, positive, negatives, temperature=0.07):
        """
        InfoNCE contrastive loss.
        
        Args:
            anchor: [B, D]
            positive: [B, D]
            negatives: [B, N, D] or list of negatives
        Returns:
            loss: scalar
        """
        # Normalize
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        if isinstance(negatives, list):
            negatives = torch.stack(negatives, dim=1)  # [B, N, D]
        negatives = F.normalize(negatives, dim=-1)
        
        # Compute similarities
        pos_sim = (anchor * positive).sum(dim=-1) / temperature  # [B]
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / temperature  # [B, N]
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dpo_forward: bool = False,
        
        # LASAR-specific inputs
        has_query: Optional[torch.BoolTensor] = None,  # [B] - whether timestep has query
        contrastive_samples: Optional[dict] = None,  # For ST-CRL loss
        episode_samples: Optional[dict] = None,  # For episodic discriminability loss
        
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        # Standard multimodal input preparation
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )

        batch_size = inputs_embeds.size(0)
        inject_mask = self._resolve_lasar_inject_mask(batch_size, inputs_embeds.device, has_query)
        if not inject_mask.any():
            return self._forward_navila_baseline(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                seqlens_in_batch=seqlens_in_batch,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                dpo_forward=dpo_forward,
                **kwargs,
            )
        
        # ============================================
        # ✅ FIX 2: Insert [MAP] token BEFORE <eos>, not after
        # ============================================
        seq_len = inputs_embeds.size(1)
        
        # Check if [MAP] token already exists in the sequence
        map_token_exists = (input_ids == self.map_token_id).any(dim=1) if input_ids is not None else torch.zeros(batch_size, dtype=torch.bool, device=inputs_embeds.device)
        
        if not map_token_exists.all():
            # Need to inject [MAP] token for samples that don't have it
            # Create [MAP] token tensor
            map_token_tensor = torch.full((batch_size, 1), self.map_token_id, dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device)
            
            # Get [MAP] token embedding from LLM's embedding layer
            with torch.no_grad():
                map_token_embeds = self.llm.get_input_embeddings()(map_token_tensor)  # [B, 1, D]
            
            # Insert [MAP] token BEFORE <eos> for each sample
            new_inputs_embeds = []
            new_input_ids = []
            new_attention_mask = []
            new_labels = []
            new_position_ids = []
            
            eos_token_id = self.tokenizer.eos_token_id
            
            for b in range(batch_size):
                if not inject_mask[b]:
                    if attention_mask is not None:
                        valid_len = attention_mask[b].sum().item()
                        new_inputs_embeds.append(inputs_embeds[b][:valid_len])
                        if input_ids is not None:
                            new_input_ids.append(input_ids[b][:valid_len])
                        new_attention_mask.append(attention_mask[b][:valid_len])
                        if labels is not None:
                            new_labels.append(labels[b][:valid_len])
                        if position_ids is not None:
                            new_position_ids.append(position_ids[b][:valid_len])
                    else:
                        new_inputs_embeds.append(inputs_embeds[b])
                        if input_ids is not None:
                            new_input_ids.append(input_ids[b])
                        if labels is not None:
                            new_labels.append(labels[b])
                        if position_ids is not None:
                            new_position_ids.append(position_ids[b])
                    continue
                if map_token_exists[b]:
                    # Already has [MAP] token, keep as is
                    # Need to handle variable sequence lengths within batch
                    # Extract valid tokens (where attention_mask == 1)
                    if attention_mask is not None:
                        valid_len = attention_mask[b].sum().item()
                        new_inputs_embeds.append(inputs_embeds[b][:valid_len])
                        if input_ids is not None:
                            new_input_ids.append(input_ids[b][:valid_len])
                        new_attention_mask.append(attention_mask[b][:valid_len])
                        if labels is not None:
                            new_labels.append(labels[b][:valid_len])
                        if position_ids is not None:
                            new_position_ids.append(position_ids[b][:valid_len])
                    else:
                        new_inputs_embeds.append(inputs_embeds[b])
                        if input_ids is not None:
                            new_input_ids.append(input_ids[b])
                        if labels is not None:
                            new_labels.append(labels[b])
                        if position_ids is not None:
                            new_position_ids.append(position_ids[b])
                else:
                    # Find <eos> position (if exists)
                    eos_positions = torch.where(input_ids[b] == eos_token_id)[0] if input_ids is not None and eos_token_id is not None else []
                    
                    if len(eos_positions) > 0:
                        # Insert [MAP] BEFORE the first <eos>
                        insert_pos = eos_positions[0].item()
                        
                        # Split at insert position and insert [MAP]
                        new_inputs_embeds.append(torch.cat([
                            inputs_embeds[b][:insert_pos],
                            map_token_embeds[b],
                            inputs_embeds[b][insert_pos:]
                        ], dim=0))
                        
                        if input_ids is not None:
                            new_input_ids.append(torch.cat([
                                input_ids[b][:insert_pos],
                                map_token_tensor[b],
                                input_ids[b][insert_pos:]
                            ], dim=0))
                        
                        if attention_mask is not None:
                            new_attention_mask.append(torch.cat([
                                attention_mask[b][:insert_pos],
                                torch.ones(1, dtype=attention_mask.dtype, device=attention_mask.device),
                                attention_mask[b][insert_pos:]
                            ], dim=0))
                        
                        if labels is not None:
                            # [MAP] token should be ignored in loss computation
                            ignore_label = torch.full((1,), -100, dtype=labels.dtype, device=labels.device)
                            new_labels.append(torch.cat([
                                labels[b][:insert_pos],
                                ignore_label,
                                labels[b][insert_pos:]
                            ], dim=0))
                        
                        if position_ids is not None:
                            # Adjust position IDs: insert current position and increment subsequent ones
                            current_pos = position_ids[b][insert_pos-1] + 1 if insert_pos > 0 else 0
                            new_position_ids.append(torch.cat([
                                position_ids[b][:insert_pos],
                                torch.tensor([current_pos], dtype=position_ids.dtype, device=position_ids.device),
                                position_ids[b][insert_pos:] + 1  # Shift subsequent positions
                            ], dim=0))
                    else:
                        # No <eos> found, append at the end (fallback)
                        new_inputs_embeds.append(torch.cat([inputs_embeds[b], map_token_embeds[b]], dim=0))
                        
                        if input_ids is not None:
                            new_input_ids.append(torch.cat([input_ids[b], map_token_tensor[b]], dim=0))
                        
                        if attention_mask is not None:
                            new_attention_mask.append(torch.cat([attention_mask[b], torch.ones(1, dtype=attention_mask.dtype, device=attention_mask.device)], dim=0))
                        
                        if labels is not None:
                            ignore_label = torch.full((1,), -100, dtype=labels.dtype, device=labels.device)
                            new_labels.append(torch.cat([labels[b], ignore_label], dim=0))
                        
                        if position_ids is not None:
                            last_pos = position_ids[b][-1] if len(position_ids[b]) > 0 else 0
                            new_position_ids.append(torch.cat([position_ids[b], torch.tensor([last_pos + 1], dtype=position_ids.dtype, device=position_ids.device)], dim=0))
            
            # ✅ FIX: Pad sequences to same length before stacking
            # Find max length in the batch
            max_len = max(emb.size(0) for emb in new_inputs_embeds)
            
            # Pad inputs_embeds
            padded_inputs_embeds = []
            for emb in new_inputs_embeds:
                if emb.size(0) < max_len:
                    pad_len = max_len - emb.size(0)
                    padding = torch.zeros(pad_len, emb.size(1), dtype=emb.dtype, device=emb.device)
                    padded_inputs_embeds.append(torch.cat([emb, padding], dim=0))
                else:
                    padded_inputs_embeds.append(emb)
            inputs_embeds = torch.stack(padded_inputs_embeds, dim=0)
            
            # Pad input_ids
            if input_ids is not None:
                padded_input_ids = []
                for ids in new_input_ids:
                    if ids.size(0) < max_len:
                        pad_len = max_len - ids.size(0)
                        padding = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=ids.dtype, device=ids.device)
                        padded_input_ids.append(torch.cat([ids, padding], dim=0))
                    else:
                        padded_input_ids.append(ids)
                input_ids = torch.stack(padded_input_ids, dim=0)
            
            # Pad attention_mask
            if attention_mask is not None:
                padded_attention_mask = []
                for mask in new_attention_mask:
                    if mask.size(0) < max_len:
                        pad_len = max_len - mask.size(0)
                        padding = torch.zeros(pad_len, dtype=mask.dtype, device=mask.device)
                        padded_attention_mask.append(torch.cat([mask, padding], dim=0))
                    else:
                        padded_attention_mask.append(mask)
                attention_mask = torch.stack(padded_attention_mask, dim=0)
            
            # Pad labels
            if labels is not None:
                padded_labels = []
                for lbl in new_labels:
                    if lbl.size(0) < max_len:
                        pad_len = max_len - lbl.size(0)
                        padding = torch.full((pad_len,), -100, dtype=lbl.dtype, device=lbl.device)
                        padded_labels.append(torch.cat([lbl, padding], dim=0))
                    else:
                        padded_labels.append(lbl)
                labels = torch.stack(padded_labels, dim=0)
            
            # Pad position_ids
            if position_ids is not None:
                padded_position_ids = []
                for pos in new_position_ids:
                    if pos.size(0) < max_len:
                        pad_len = max_len - pos.size(0)
                        # Position IDs should continue from last valid position
                        last_valid_pos = pos[-1] if len(pos) > 0 else 0
                        padding = torch.arange(last_valid_pos + 1, last_valid_pos + 1 + pad_len, dtype=pos.dtype, device=pos.device)
                        padded_position_ids.append(torch.cat([pos, padding], dim=0))
                    else:
                        padded_position_ids.append(pos)
                position_ids = torch.stack(padded_position_ids, dim=0)
            
            # Log auto-injection (once per mode)
            if not hasattr(self, '_auto_inject_logged'):
                num_injected = (~map_token_exists).sum().item()
                mode = "TRAINING" if self.training else "EVAL"
                print(f"✓ [LASAR {mode}] Injected [MAP] token BEFORE <eos> for {num_injected}/{batch_size} samples")
                print(f"   All sequences now have [MAP] token at correct position!")
                self._auto_inject_logged = True
        
        # ============================================
        # LASAR: Compute Cognitive Map Belief State
        # ============================================
        batch_size = inputs_embeds.size(0)
        
        # Compute belief state m_t from episodic memory (inputs_embeds)
        if self.training and self.lambda_semantic > 0:
            belief_state, vq_loss, usage_probs, entropy = self.cognitive_map_encoder(
                inputs_embeds.detach(), compute_atlas_loss=True
            )
        else:
            belief_state = self.cognitive_map_encoder(inputs_embeds.detach(), compute_atlas_loss=False)
            vq_loss = torch.tensor(0.0, device=inputs_embeds.device)
            entropy = torch.tensor(0.0, device=inputs_embeds.device)
        
        # Project belief state to LLM embedding space
        belief_embeds = self.belief_projection(belief_state).unsqueeze(1)  # [B, 1, D]
        
        # ✅ FIX 4: Use learnable gate instead of forced normalization
        # Gate controls the strength of belief injection
        gate_value = torch.tanh(self.belief_gate)  # Range: [-1, 1]
        belief_embeds = gate_value * belief_embeds
        
        # Inject belief state m_t by replacing [MAP] token embeddings
        map_token_mask = (input_ids == self.map_token_id) if input_ids is not None else None
        
        if map_token_mask is not None and map_token_mask.any():
            # Replace [MAP] token embeddings with gated belief state
            for b in range(batch_size):
                if not inject_mask[b]:
                    continue
                map_positions = torch.where(map_token_mask[b])[0]
                if len(map_positions) > 0:
                    # Replace first [MAP] occurrence
                    # Use additive injection instead of full replacement for stability
                    original_embed = inputs_embeds[b, map_positions[0]]
                    inputs_embeds[b, map_positions[0]] = original_embed + belief_embeds[b, 0]
        
        # ✅ FIX 3: Ensure sequence lengths are consistent before repack
        # Data repacking for sequence parallelism
        support_packing = "seqlens_in_batch" in inspect.signature(self.llm.forward).parameters

        if self.training and support_packing and not dpo_forward:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            new_input_ids = None
            past_key_values = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            new_input_ids = input_ids
        
        # Forward through LLM
        if support_packing:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                seqlens_in_batch=sorted_seqlens_in_batch,
            )
        else:
            outputs = self.llm.forward(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=new_inputs_embeds,
                labels=new_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Apply soft cross entropy for time tokens (same as llava_llama.py)
        if self.training and self.config.time_token_ids:
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                new_labels,
                soft_tokens=self.config.time_token_ids,
                std=self.config.soft_ce_std,
            )
        
        # ============================================
        # LASAR: Auxiliary Losses (commented out for now)
        # ============================================
        # Can be enabled if needed for advanced training
        
        # Loss rescale for SP & DP loss match
        loss_weight = calculate_loss_weight(new_labels)
        outputs.loss = outputs.loss * loss_weight

        if dpo_forward:
            return outputs.logits, new_labels

        return outputs


AutoConfig.register("llava_llama", LlavaLlamaConfig)
AutoModel.register(LlavaLlamaConfig, LlavaLlamaModel)

