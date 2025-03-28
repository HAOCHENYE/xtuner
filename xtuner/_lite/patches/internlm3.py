# Copyright (c) OpenMMLab. All rights reserved.
import gc
import os
import time
import types
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from accelerate.utils import set_module_tensor_to_device
from packaging import version
from torch import Tensor
from torch import distributed as dist
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.distributed_c10d import ReduceOp
from torch.nn import functional as F
from torch.nn.utils.clip_grad import _no_grad
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)
from tqdm import tqdm
from xtuner._lite.accelerate import liger_kernel_is_available
from xtuner._lite.modelings.internlm3.modeling_internlm3 import InternLM3RotaryEmbedding
from xtuner._lite.parallel.sequence import split_for_sequence_parallel
from xtuner._lite.patches.base import (
    FSDPConfig,
    HFCheckpointLoader,
    clip_grad_norm_,
    lazy_init_fn,
)
from xtuner._lite.patches.internlm3 import CUDAPatchedInternLM3ForCausalLM
from xtuner._lite.patches.utils import pad_to_max_length, pad_to_multiple_of

from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from xpuyu.modelings.internlm3.modeling_internlm3_moe import (
    InternLM3MoEAttention,
    InternLM3MoEDecoderLayer,
    InternLM3MoEForCausalLM,
    InternLM3MoEModel,
    InternLM3MoERMSNorm,
    KwargsForCausalLM,
    _compute_aux_loss_func,
)

from ..modelings.internlm3.moe_train import (
    InternLM3MoEDecoderLayer as InternLM3MoETrainDecoderLayer,
)


def _apply_ep(module, device_mesh):
    for m in module.modules():
        if type(m).__name__ in ("ExpertEp", "GroupedLinear"):
            fused_w1w3 = nn.Parameter(distribute_tensor(m.fused_w1w3, device_mesh, [Shard(0)]))
            m.register_parameter("fused_w1w3", fused_w1w3)
            w2 = nn.Parameter(distribute_tensor(m.w2, device_mesh, [Shard(0)]))
            m.register_parameter("w2", w2)


def _replicate_other_params(module, device_mesh):
    if type(module).__name__ in ("ExpertEp", "GroupedLinear"):
        return
    for name, param in module.named_parameters(recurse=False):
        dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Replicate()]))
        module.register_parameter(name, dist_param)
    for child in module.children():
        _replicate_other_params(child, device_mesh)


@_no_grad
def clip_grad_norm_ep_(
    moe_params,
    non_moe_params,
    experts_fsdp_mesh,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    if isinstance(moe_params, torch.Tensor):
        moe_params = [moe_params]
    if isinstance(non_moe_params, torch.Tensor):
        non_moe_params = [non_moe_params]
    moe_grads = [p.grad for p in moe_params if p.grad is not None]
    non_moe_grads = [p.grad for p in non_moe_params if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(moe_grads) + len(non_moe_grads) == 0:
        return torch.tensor(0.0)
    first_device = moe_grads[0].device
    grouped_moe_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] = (
        _group_tensors_by_device_and_dtype([moe_grads])
    )
    grouped_non_moe_grads: Dict[Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]] = (
        _group_tensors_by_device_and_dtype([non_moe_grads])
    )
    moe_norms: List[Tensor] = []
    non_moe_norms: List[Tensor] = []

    for (device, _), ([device_grads], _) in grouped_moe_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            moe_norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            moe_norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    for (device, _), ([device_grads], _) in grouped_non_moe_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            non_moe_norms.extend(torch._foreach_norm(device_grads, norm_type))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            non_moe_norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_grads])

    local_sharded_moe_norm = torch.linalg.vector_norm(
        torch.stack([norm.to_local().to(first_device) for norm in moe_norms]), norm_type, dtype=torch.float32
    )
    local_sharded_non_moe_norm = torch.linalg.vector_norm(
        torch.stack([norm.to_local().to(first_device) for norm in non_moe_norms]), norm_type, dtype=torch.float32
    )

    if norm_type == 2:
        total_sharded_moe_norm = local_sharded_moe_norm**norm_type
        total_sharded_non_moe_norm = local_sharded_non_moe_norm**norm_type
        dist.all_reduce(total_sharded_moe_norm)
        dist.all_reduce(total_sharded_non_moe_norm, group=experts_fsdp_mesh.get_group(mesh_dim=0))
        total_norm = (total_sharded_moe_norm + total_sharded_non_moe_norm) ** 0.5
    else:
        raise NotImplementedError

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for (device, _), ([device_grads], _) in grouped_moe_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

    for (device, _), ([device_grads], _) in grouped_non_moe_grads.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_grads, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in device_grads:
                g.mul_(clip_coef_clamped_device)

    return total_norm


def fuse_moe_expert_weights(num_experts, state_dict):
    """Fuse individual MoE expert weights into concatenated weights for
    efficient computation.

    This function takes individual expert weights and fuses them into concatenated weights to enable
    efficient grouped matrix multiplication operations. The weights are moved to CUDA memory and
    concatenated along appropriate dimensions.

    Args:
        num_experts (int): Number of experts in the MoE layer
        state_dict (dict): State dict containing individual expert weights
            Expected keys for each expert e:
            - f"block_sparse_moe.experts.{e}.w1.weight"
            - f"block_sparse_moe.experts.{e}.w2.weight"
            - f"block_sparse_moe.experts.{e}.w3.weight"

    Returns:
        dict: State dict with fused weights replacing individual expert weights
    """
    # Load weights to CUDA memory efficiently using pin_memory and non_blocking transfer
    w1_list = [
        state_dict[f"block_sparse_moe.experts.{e}.w1.weight"].pin_memory().to("cuda", non_blocking=True)
        for e in range(num_experts)
    ]
    w3_list = [
        state_dict[f"block_sparse_moe.experts.{e}.w3.weight"].pin_memory().to("cuda", non_blocking=True)
        for e in range(num_experts)
    ]
    w2_list = [
        state_dict[f"block_sparse_moe.experts.{e}.w2.weight"].pin_memory().to("cuda", non_blocking=True)
        for e in range(num_experts)
    ]

    torch.cuda.synchronize()

    # Fuse w1 and w3 weights by concatenating along dim=1 for each expert, then concatenating experts along dim=0
    fused_w1w3 = torch.cat([torch.cat([w1_list[e].T, w3_list[e].T], dim=1) for e in range(num_experts)], dim=0)
    # Fuse w2 weights by concatenating transposed weights along dim=0
    fused_w2 = torch.cat([w2_list[e].T for e in range(num_experts)], dim=0)

    # Store fused weights in state dict
    state_dict["block_sparse_moe.experts.fused_w1w3"] = fused_w1w3
    state_dict["block_sparse_moe.experts.w2"] = fused_w2

    # Clean up individual expert weights to free memory
    for e in range(num_experts):
        del state_dict[f"block_sparse_moe.experts.{e}.w1.weight"]
        del state_dict[f"block_sparse_moe.experts.{e}.w3.weight"]
        del state_dict[f"block_sparse_moe.experts.{e}.w2.weight"]
    gc.collect()

    return state_dict


def split_moe_expert_weights(intermediate_size, num_experts, layer_state_dict):
    """Split fused MoE expert weights back into individual expert weights.

    This function takes the fused weights used for efficient computation and splits them back into
    separate weights for each expert. The fused weights consist of concatenated w1/w3 weights and w2 weights.



    Returns:
        dict: State dict with split weights for each expert
    """
    result_dict = {}

    # Get the fused weights
    fused_w1w3 = layer_state_dict["block_sparse_moe.experts.fused_w1w3"]
    fused_w2 = layer_state_dict["block_sparse_moe.experts.w2"]

    # Split fused_w1w3 into w1 and w3 portions
    fused_w1 = fused_w1w3[:, :intermediate_size]  # [num_experts*H, M]
    fused_w3 = fused_w1w3[:, intermediate_size:]  # [num_experts*H, M]

    # Chunk the weights for each expert
    w1_chunks = torch.chunk(fused_w1.T, num_experts, dim=1)  # List of [M, H]
    w2_chunks = torch.chunk(fused_w2, num_experts, dim=0)  # List of [num_experts*M, H] -> List of [M, H]
    w3_chunks = torch.chunk(fused_w3.T, num_experts, dim=1)  # List of [M, H]

    # Create the result dictionary using zip to iterate over chunks together
    # NOTE: Adding `tensor.contiguous()` can significantly accelerate the process by approximately 4 times.
    # Discontinuous memory distribution can cause delays in the CUDA-to-CPU data transfer process.
    for e, (w1, w2, w3) in enumerate(zip(w1_chunks, w2_chunks, w3_chunks)):
        result_dict[f"block_sparse_moe.experts.{e}.w1.weight"] = w1.contiguous()
        result_dict[f"block_sparse_moe.experts.{e}.w2.weight"] = w2.T.contiguous()
        result_dict[f"block_sparse_moe.experts.{e}.w3.weight"] = w3.contiguous()

    # Copy over non-expert parameters
    for key, value in layer_state_dict.items():
        if "fused_w1w3" not in key and "w2" not in key:
            result_dict[key] = value

    return result_dict


class HybridInternLM3MoEForCausalLM(CUDAPatchedInternLM3ForCausalLM):
    rotary_emb_cls = InternLM3RotaryEmbedding
    attn_cls = InternLM3MoEAttention
    layer_cls = InternLM3MoEDecoderLayer
    causal_cls = InternLM3MoEForCausalLM
    model_cls = InternLM3MoEModel
    rms_norm_cls = InternLM3MoERMSNorm

    @classmethod
    def dispatch_hf_code(cls, model) -> InternLM3MoEForCausalLM:
        for name, module in model.named_modules():
            if isinstance(module, cls.attn_cls):
                module.forward = types.MethodType(cls.patched_attn_forward, module)
            if isinstance(module, cls.causal_cls):
                module.forward = types.MethodType(cls.patched_casual_forward, module)
            if isinstance(module, cls.model_cls):
                module.forward = types.MethodType(cls.patched_model_forward, module)

            if isinstance(module, cls.rms_norm_cls):
                module.forward = types.MethodType(cls.patched_rms_norm_forward, module)
            if isinstance(module, nn.Linear):
                module.forward = types.MethodType(cls.patched_linear_forward, module)
            elif isinstance(module, nn.Embedding):
                module.forward = types.MethodType(cls.patched_emb_forward, module)
        return model

    def trainable_parameters(self):
        requried_grad_moe_params = []
        requried_grad_non_moe_params = []
        for name, param in self.patched_model.named_parameters():
            if not param.requires_grad:
                continue
            if ".experts." in name:
                requried_grad_moe_params.append(param)
            else:
                requried_grad_non_moe_params.append(param)

        return [
            {"params": requried_grad_moe_params},
            {"params": requried_grad_non_moe_params},
        ]

    def clip_grad_norm(self, max_norm):
        if self.fsdp_config.ep_size > 1:
            # TODO:  之前是在每个梯度累加内部，现在移到外面，应该是一样
            for param in self.patched_model.parameters():
                # non moe params
                if isinstance(param, DTensor) and param.device_mesh.ndim == 2 and param.placements[1] == Replicate():
                    dist.all_reduce(param.grad.to_local(), ReduceOp.AVG, group=self.ep_mesh.get_group(mesh_dim=0))

            trainable_parameters_ = self.trainable_parameters()
            requried_grad_moe_params = trainable_parameters_[0]["params"]
            requried_grad_non_moe_params = trainable_parameters_[1]["params"]

            for param in requried_grad_moe_params:
                param.grad.div_(self.ep_mesh.size())

            grad_norm = clip_grad_norm_ep_(
                requried_grad_moe_params, requried_grad_non_moe_params, self.experts_fsdp_mesh, max_norm
            )
        else:
            _requried_grad_params = [param for param in self.patched_model.parameters() if param.requires_grad]
            grad_norm = clip_grad_norm_(_requried_grad_params, self.experts_fsdp_mesh, max_norm)
            return grad_norm
        return grad_norm

    def fully_shard(self, fsdp_config: FSDPConfig) -> None:
        if fsdp_config.tp_size > 1:
            raise NotImplementedError
        ep_size = fsdp_config.ep_size
        sp_size = fsdp_config.sp_size

        world_size = dist.get_world_size()
        experts_fsdp_size = world_size // ep_size
        model_mesh = init_device_mesh(
            self.device_type, (experts_fsdp_size, ep_size), mesh_dim_names=("experts_fsdp", "ep")
        )
        self.world_mesh = init_device_mesh(self.device_type, (world_size,), mesh_dim_names=("world",))

        self.tp_mesh = None  # TODO: Support

        data_mesh = init_device_mesh(
            self.device_type,
            (world_size // sp_size, sp_size),
            mesh_dim_names=("dp", "sp"),
        )
        self.dp_mesh = data_mesh["dp"]
        self.sp_mesh = data_mesh["sp"]

        _data_mesh = init_device_mesh(
            self.device_type,
            (world_size // sp_size, sp_size),
            mesh_dim_names=("dp", "same_data"),
        )
        self._data_mesh = _data_mesh["same_data"]

        self.ep_mesh = model_mesh["ep"]
        self.experts_fsdp_mesh = model_mesh["experts_fsdp"]

        hf_checkpoint_loader = HFCheckpointLoader(self.patched_model.config._name_or_path)
        param_init_fn = partial(
            lazy_init_fn,
            module2name={mod: name for name, mod in self.patched_model.named_modules()},
            checkpoint_loader=hf_checkpoint_loader,
        )

        mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_config.param_dtype, reduce_dtype=fsdp_config.reduce_dtype)

        self.patched_model.model.rotary_emb = self.rotary_emb_cls(self.patched_model.config)

        num_recompute_layers = int(self.model_config.num_hidden_layers * fsdp_config.recompute_ratio)

        # TODO: 只是为了方便测试，后续要移除
        import torch.nn.init as init

        def init_weights(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

        for layer_idx, layer in enumerate(tqdm(self.patched_model.model.layers)):
            with torch.device("meta"):
                new_decoder_layer = InternLM3MoETrainDecoderLayer(self.patched_model.config, layer_idx)
            new_decoder_layer.to_empty(device=torch.cuda.current_device())

            # Gather state dict for `layer_idx` layer
            layer_state_dict = {}
            for key, value in hf_checkpoint_loader.weight_map.items():
                if f"layers.{layer_idx}." in key:
                    layer_state_dict[key.split(f"layers.{layer_idx}.")[-1]] = hf_checkpoint_loader.load(key)

            # convert split version to fused ones
            layer_state_dict = fuse_moe_expert_weights(self.patched_model.config.num_experts, layer_state_dict)
            new_decoder_layer.load_state_dict(layer_state_dict)
            # new_decoder_layer.apply(init_weights) # for debug

            self.dispatch_hf_code(new_decoder_layer)

            if fsdp_config.torch_compile:
                if sp_size > 1:
                    # all-to-all 不支持 compile
                    self_attn = torch.compile(new_decoder_layer.self_attn)
                else:
                    self_attn = torch.compile(new_decoder_layer.self_attn, fullgraph=True)
                new_decoder_layer.self_attn = self_attn

            if ep_size > 1:
                _apply_ep(new_decoder_layer, self.ep_mesh)
                _replicate_other_params(new_decoder_layer, self.ep_mesh)

            if layer_idx < num_recompute_layers - 1:
                new_decoder_layer = ptd_checkpoint_wrapper(
                    new_decoder_layer, preserve_rng_state=False, checkpoint_impl=CheckpointImpl.REENTRANT
                )

            self.patched_model.model.layers[layer_idx] = new_decoder_layer

            if layer_idx >= len(self.patched_model.model.layers) - 1:
                reshard_after_forward = False
            else:
                reshard_after_forward = fsdp_config.reshard_after_forward

            fully_shard(
                new_decoder_layer,
                mesh=self.experts_fsdp_mesh,
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_forward,
                offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
            )

        if version.parse(torch.__version__) >= version.parse("2.5.0"):
            for layer_cur, layer_next in zip(
                self.patched_model.model.layers[:-1],
                self.patched_model.model.layers[1:],
            ):
                layer_cur.set_modules_to_forward_prefetch([layer_next])

        self.patched_model.lm_head.apply(param_init_fn)
        self.patched_model.model.embed_tokens.apply(param_init_fn)
        self.patched_model.model.norm.apply(param_init_fn)

        if ep_size > 1:
            _replicate_other_params(self.patched_model.lm_head, self.ep_mesh)
            _replicate_other_params(self.patched_model.model.embed_tokens, self.ep_mesh)
            _replicate_other_params(self.patched_model.model.norm, self.ep_mesh)

        fully_shard(
            self.patched_model,
            mesh=self.experts_fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=CPUOffloadPolicy() if fsdp_config.cpu_offload else None,
        )

    @staticmethod
    def patched_linear_forward(self, input):
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias
        return F.linear(input, w, b)

    @staticmethod
    def patched_rms_norm_forward(self, input):
        if hasattr(self, "weight"):
            if isinstance(self.weight, DTensor):
                w = self.weight.to_local()
            else:
                w = self.weight
        else:
            if isinstance(self.norm.weight, DTensor):
                w = self.norm.weight.to_local()
            else:
                w = self.norm.weight
        return F.rms_norm(input, w.shape, w, self.variance_epsilon)

    @staticmethod
    def patched_emb_forward(self, input):
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
        else:
            w = self.weight
        return F.embedding(
            input,
            w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

    @staticmethod
    def patched_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        if self.config.micro_forward:
            # position_embeddings is tuple
            position_embeddings_cos = position_embeddings[0]
            position_embeddings_sin = position_embeddings[1]
            position_embeddings = tuple(
                (position_embeddings_cos[i : i + 1, ...], position_embeddings_sin[i : i + 1, ...])
                for i in range(position_embeddings_cos.shape[0])
            )

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.config.micro_forward and not isinstance(hidden_states, (list, tuple)):
                hidden_states = torch.split(hidden_states, 1, dim=0)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                # 为了兼容可重入式 checkpoint，需要确保需要计算梯度的输入输出必须是 tensor，而非 list[tensor] 嵌套格式
                if isinstance(hidden_states, (list, tuple)):
                    layer_outputs = decoder_layer(
                        *hidden_states,  # 如果是 list 则拆开分成多个 tensor 输入，否则不会计算梯度
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )  # 对输出也有要求，嵌套太深对象也无法计算梯度

                    # 重新拆分
                    if output_router_logits:
                        hidden_states = layer_outputs[: len(layer_outputs) // 2]
                        router_logits = layer_outputs[len(layer_outputs) // 2 :]
                        layer_outputs = [hidden_states, router_logits]
                    else:
                        layer_outputs = [layer_outputs]
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

            hidden_states = layer_outputs[0]
            if self.config.micro_forward:
                assert isinstance(hidden_states, (list, tuple))

            if output_attentions:
                pass

            if output_router_logits and layer_outputs[-1] is not None:
                if self.config.micro_forward:
                    assert isinstance(layer_outputs[-1], (list, tuple))
                    x = torch.cat(layer_outputs[-1], dim=0)
                    all_router_logits += (x,)
                else:
                    all_router_logits += (layer_outputs[-1],)

        if self.config.micro_forward:
            assert isinstance(hidden_states, (list, tuple))
            hidden_states = torch.cat(hidden_states, dim=0)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
        return output if return_dict else output.to_tuple()

    @staticmethod
    def patched_casual_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        balancing_loss_global_average: bool = True,
        label_shifted=False,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]

        if labels is None:
            loss = None

            logits = self.lm_head(hidden_states)
            if isinstance(logits, DTensor):
                logits = logits.to_local()
        else:
            if liger_kernel_is_available():
                # unable to return logits when using Liger Kernel
                logits = None

                if label_shifted:
                    shift_hidden_states = hidden_states
                    shift_labels = labels
                else:
                    shift_hidden_states = hidden_states[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_hidden_states.device)

                from liger_kernel.transformers.fused_linear_cross_entropy import (
                    LigerFusedLinearCrossEntropyLoss,
                )

                loss_fct = LigerFusedLinearCrossEntropyLoss()

                lm_head_weight = self.lm_head.weight
                if isinstance(lm_head_weight, DTensor):
                    # assert isinstance(shift_hidden_states, DTensor)
                    # shift_hidden_states = shift_hidden_states.to_local()
                    lm_head_weight = self.lm_head.weight.to_local()

                loss = loss_fct(lm_head_weight, shift_hidden_states, shift_labels, self.lm_head.bias)
            else:
                logits = self.lm_head(hidden_states)

                if label_shifted:
                    shift_logits = logits
                    shift_labels = labels
                else:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)

                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits, shift_labels)

        aux_loss = None
        if output_router_logits:
            if attention_mask is None:
                # compatible with micro-batch
                attention_mask = input_ids.view(1, -1) != 0  # pad token id is 0
            balancing_loss, z_loss = _compute_aux_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask=attention_mask,
                balancing_loss_global_average=balancing_loss_global_average,
            )
            aux_loss = (balancing_loss, z_loss)
            if labels is not None:
                loss += self.balancing_loss_coef * balancing_loss.to(
                    loss.device
                ) + self.router_z_loss_coef * z_loss.to(loss.device)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        label_shifted: bool = False,
        gather_logprobs: bool = False,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        block_table: Optional[torch.LongTensor] = None,
        prefilling: bool = False,
        sequence_parallel_mesh: Optional[DeviceMesh] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        if gather_logprobs:
            assert labels is not None and label_shifted

        _input_ids = input_ids
        _labels = labels
        _position_ids = position_ids
        _cu_seq_lens_q = cu_seq_lens_q
        _cu_seq_lens_k = cu_seq_lens_k
        _max_length_q = max_length_q
        _max_length_k = max_length_k

        # 开启 micro_forward 时，外面已经 padding 到 max_length，这里不需要再 padding
        if not self.patched_model.config.micro_forward and self.fsdp_config.torch_compile:
            _input_ids = pad_to_max_length(_input_ids, 0, self.fsdp_config.max_length, 1)
            _position_ids = pad_to_max_length(_position_ids, 0, self.fsdp_config.max_length, 1)
            if labels is not None:
                _labels = pad_to_max_length(_labels, -100, self.fsdp_config.max_length, 1)
        else:
            multiple_of = 1
            if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
                multiple_of = sequence_parallel_mesh.size()
                if self.tp_mesh:
                    multiple_of = sequence_parallel_mesh.size() * self.tp_mesh.size()
            else:
                if self.tp_mesh:
                    multiple_of = self.tp_mesh.size()

            _input_ids = pad_to_multiple_of(_input_ids, 0, multiple_of, 1)
            _position_ids = pad_to_multiple_of(_position_ids, 0, multiple_of, 1)
            if labels is not None:
                _labels = pad_to_multiple_of(_labels, -100, multiple_of, 1)

        num_padded_tokens = _input_ids.numel() - input_ids.numel()

        if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
            _input_ids = split_for_sequence_parallel(_input_ids, dim=1, sp_mesh=sequence_parallel_mesh)
            _position_ids = split_for_sequence_parallel(_position_ids, dim=1, sp_mesh=sequence_parallel_mesh)

            if labels is not None:
                _labels = split_for_sequence_parallel(_labels, dim=1, sp_mesh=sequence_parallel_mesh)

        if self.tp_mesh and self.tp_mesh.size() > 1:
            if labels is not None:
                _labels = split_for_sequence_parallel(_labels, dim=1, sp_mesh=self.tp_mesh)

        if self.training and num_padded_tokens > 0:
            assert torch.any(cu_seq_lens_k == cu_seq_lens_q)
            _cu_seq_lens_q = _cu_seq_lens_q.tolist()
            _cu_seq_lens_q.append(_cu_seq_lens_q[-1] + num_padded_tokens)

            _cu_seq_lens_q = torch.IntTensor(_cu_seq_lens_q).to(cu_seq_lens_q.device)
            _cu_seq_lens_k = _cu_seq_lens_q

            _max_length_q = max(_max_length_q, num_padded_tokens)
            _max_length_k = _max_length_q

        outputs = self.patched_model(
            _input_ids,
            attention_mask,
            _position_ids,
            past_key_values,
            inputs_embeds,
            _labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cache_position,
            num_logits_to_keep,
            label_shifted=label_shifted,
            cu_seq_lens_q=_cu_seq_lens_q,
            cu_seq_lens_k=_cu_seq_lens_k,
            max_length_q=_max_length_q,
            max_length_k=_max_length_k,
            block_table=block_table,
            prefilling=prefilling,
            sequence_parallel_mesh=self.sequence_parallel_mesh,
        )

        if outputs.loss is not None:
            outputs.loss = outputs.loss * (_labels >= 0).sum()
            if self.tp_mesh and self.tp_mesh.size() > 1:
                outputs.loss = dist.nn.all_reduce(outputs.loss, group=self.tp_mesh.get_group())
            if sequence_parallel_mesh and sequence_parallel_mesh.size() > 1:
                outputs.loss = dist.nn.all_reduce(outputs.loss, group=sequence_parallel_mesh.get_group())
            if (labels >= 0).sum() > 0:
                outputs.loss = outputs.loss / (labels >= 0).sum()

        return outputs

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        if dist.is_initialized() and dist.is_available():
            rank = dist.get_rank()
        else:
            rank = 0

        torch.cuda.empty_cache()

        if rank == 0:
            self.rank0_model.to(torch.bfloat16)

        num_experts = self.patched_model.config.num_experts

        for layer_idx, layer in enumerate(tqdm(self.patched_model.model.layers, desc="All-gather and split experts")):
            layer_state_dict = get_model_state_dict(
                layer, options=StateDictOptions(full_state_dict=True, cpu_offload=False)
            )
            if rank == 0:
                layer_state_dict = split_moe_expert_weights(
                    self.patched_model.config.intermediate_size, num_experts, layer_state_dict
                )

                for _name, _param in layer_state_dict.items():
                    layer_state_dict[_name] = _param.to("cpu", non_blocking=True)

                torch.cuda.synchronize()
                torch.cpu.synchronize()

                for name, param in layer_state_dict.items():
                    set_module_tensor_to_device(self.rank0_model.model.layers[layer_idx], name, "cpu", param)

        head_param = self.patched_model.lm_head.weight.full_tensor()
        embed_param = self.patched_model.model.embed_tokens.weight.full_tensor()
        norm_param = self.patched_model.model.norm.weight.full_tensor()

        if rank == 0:
            set_module_tensor_to_device(self.rank0_model, "lm_head.weight", "cpu", head_param)
            set_module_tensor_to_device(self.rank0_model, "model.embed_tokens.weight", "cpu", embed_param)
            set_module_tensor_to_device(self.rank0_model, "model.norm.weight", "cpu", norm_param)

            # TODO: move it to the xtuner/_lite/patches/llama.py
            executor = ThreadPoolExecutor(max_workers=1)
            handler: Future = executor.submit(
                self.rank0_model.save_pretrained,
                save_directory,
                is_main_process,
                state_dict,
                save_function,
                push_to_hub,
                max_shard_size,
                safe_serialization,
                variant,
                token,
                save_peft_format,
                **kwargs,
            )

            handler.add_done_callback(lambda handler: executor.shutdown(wait=False))
            return handler
        else:
            return None


if __name__ == "__main__":
    from xtuner._lite.parallel import setup_parallel
    from xtuner._lite.patches import AutoPatch, FSDPConfig

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from xpuyu.modelings import register_remote_code
    from xpuyu.patches import register_patches

    setup_parallel()
    register_patches()
    register_remote_code()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_pretrained(
            "/fs-computility/llm/shared/gaojianfei/internlm3_moe_empty", attn_implementation="flash_attention_2"
        )
    tokenizer = AutoTokenizer.from_pretrained("internlm/internlm3-8b-instruct")
    fsdp_config = FSDPConfig(reshard_after_forward=False, tp_size=1)
    patched_model = AutoPatch.from_causal_lm(model, fsdp_config)
    torch.distributed.breakpoint()
    prompts = [
        "<s><|im_start|>user\n陈金山同志在上海市委有什么突出贡献？<|im_end|>\n<|im_start|>assistant\n",
        "<s><|im_start|>user\n我的前任男友伤害了我，我想报复他，但我不知道应该采取什么措施来打击他。<|im_end|>\n<|im_start|>assistant\n",
    ] * 4

    input_ids = [tokenizer.encode(prompt, return_tensors="pt").cuda() for prompt in prompts]

    from xtuner._lite import get_logger

    logger = get_logger()
    start = time.time()
    patched_model.eval()
    import torch

    torch.manual_seed(42)
    response = patched_model.generate(
        input_ids,
        stop_token_ids=[128131],
        max_batch_size=len(prompts),
        max_new_tokens=1024,
        max_length=2048,
        cuda_graph=True,
    )
    for res in response:
        logger.info(tokenizer.decode(res))

    logger.info(time.time() - start)


