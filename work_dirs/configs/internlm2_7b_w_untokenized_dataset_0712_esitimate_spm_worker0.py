# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, ConstantLR, CosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import BatchSampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.intern_repo import build_packed_dataset, load_intern_repo_untokenized_dataset
from xtuner.dataset.intern_repo_streaming import ROLE_CFG_CHAT, build_internlm_streaming_dataset, ROLE_CFG_PRETRAIN
from xtuner.dataset.samplers import InternRepoSampler
from xtuner.engine import (
    DatasetInfoHook,
    EvaluateChatHook,
    SamplerLevelIterTimerHook,
    ThroughputHook,
    VarlenAttnArgsToMessageHubHook,
)
from xtuner.engine.hooks import HFCheckpointHook
from xtuner.engine.runner.loops import IterableEpochBasedTrainLoop, TrainLoop
from xtuner.model import SupervisedFinetune
from xtuner.utils import PROMPT_TEMPLATE
from sentencepiece import SentencePieceProcessor
from pathlib import Path

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
pretrained_model_name_or_path = (
    "/mnt/hwfile/mm_dev/sft/model_releases/S_Ampere_7B_v1_1_enchance_FT_v1_0_0_s1_rc47_1970_hf"
)
use_varlen_attn = True

# Data
dataset_folder = "/mnt/hwfile/mm_dev/yehaochen/datasets/internlm2.5_s2_0621_rc4/"  # noqa: E501
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 32768
pack_to_max_length = True

# parallel
sequence_parallel_size = 1

# Scheduler & Optimizer
# batch size per device, set to 1 if `use_varlen_attn` = True
# To clarify, enlarging the batch size essentially enlarges the `max_length`.
# For example, doubling the max length is tantamount to doubling the batch size
batch_size = 1
accumulative_counts = 1  # 1bs * 1acc * 64gpu = 64 batchsize
accumulative_counts *= sequence_parallel_size
dataloader_num_workers = 0
max_epochs = 1
optim_type = AdamW
lr = 2e-5
betas = (0.9, 0.95)
weight_decay = 0.01
max_norm = 1  # grad clip
warm_up_ratio = 0.025

# Save
save_steps = 500
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ""
evaluation_inputs = [
    "请给我介绍五个上海的景点",
    "Please tell me five scenic spots in Shanghai",
]

#######################################################################
#                      PART 2  Model & Tokenizer                      #
#######################################################################
# tokenizer = dict(
#     type=AutoTokenizer.from_pretrained,
#     pretrained_model_name_or_path=pretrained_model_name_or_path,
#     trust_remote_code=True,
#     padding_side="right",
#     add_bos_token=False,
#     add_eos_token=False,
# )
work_dir = Path('/mnt/hwfile/mm_dev/yehaochen/ckpts/7B/ampere/xtuner_regression/') / Path(__file__).stem
tokenizer = dict(type=SentencePieceProcessor, model_file='/mnt/hwfile/mm_dev/yehaochen/tokenizers/v13.model')


model = dict(
    type=SupervisedFinetune,
    use_varlen_attn=use_varlen_attn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=True,
    ),
)

######  #################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
train_dataset = dict(
    type=build_internlm_streaming_dataset,
    data_roots=dataset_folder,
    tokenizer=tokenizer,
    role_cfg=ROLE_CFG_PRETRAIN,
    shuffle=True,
)

train_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    collate_fn=dict(type=default_collate_fn, use_varlen_attn=use_varlen_attn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1 / 40,
        by_epoch=True,
        begin=0,
        end=warm_up_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=lr * 0.15,
        by_epoch=True,
        begin=warm_up_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=1)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer, is_intern_repo_dataset=True),
    # dict(
    #     type=EvaluateChatHook,
    #     tokenizer=tokenizer,
    #     every_n_iters=evaluation_freq,
    #     evaluation_inputs=evaluation_inputs,
    #     system=SYSTEM,
    #     prompt_template=prompt_template,
    # ),
    dict(type=ThroughputHook),
    dict(type=HFCheckpointHook, out_dir=str(work_dir.parent / work_dir.name)),
]

if use_varlen_attn:
    custom_hooks += [dict(type=VarlenAttnArgsToMessageHubHook)]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 100 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=1),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=True,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl", port=29501),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

log_processor = dict(
    by_epoch=True,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)
work_dir = str(work_dir)
