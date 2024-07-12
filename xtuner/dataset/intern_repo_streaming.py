import random
from copy import copy
from pathlib import Path
from typing import Dict, List, TypedDict
from sentencepiece import SentencePieceProcessor


import jsonlines
from mmengine import ConfigDict
from mmengine.dist import all_gather_object, get_rank, get_world_size
from mmengine.logging import print_log
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm
from transformers import AutoTokenizer
from typing_extensions import NotRequired

from xtuner.registry import BUILDER
from xtuner.utils.constants import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX

ROLE_CFG_PRETRAIN = dict(
    system=dict(
        begin=dict(
            with_name="[UNUSED_TOKEN_146]system name={name}\n",
            without_name="[UNUSED_TOKEN_146]system\n",
            name={
                "interpreter": "[UNUSED_TOKEN_142]",
                "plugin": "[UNUSED_TOKEN_141]",
            },
        ),
        end="[UNUSED_TOKEN_145]\n",
        loss=dict(
            meta=False,
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    user=dict(
        begin=dict(
            with_name="[UNUSED_TOKEN_146]user name={name}\n",
            without_name="[UNUSED_TOKEN_146]user\n",
        ),
        end="[UNUSED_TOKEN_145]\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    assistant=dict(
        begin=dict(
            with_name="[UNUSED_TOKEN_146]assistant name={name}\n",
            without_name="[UNUSED_TOKEN_146]assistant\n",
            name={
                "interpreter": "[UNUSED_TOKEN_142]",
                "plugin": "[UNUSED_TOKEN_141]",
            },
        ),
        end="[UNUSED_TOKEN_145]\n",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    environment=dict(
        begin=dict(
            with_name="[UNUSED_TOKEN_146]environment name={name}\n",
            without_name="[UNUSED_TOKEN_146]environment\n",
            name={
                "interpreter": "[UNUSED_TOKEN_142]",
                "plugin": "[UNUSED_TOKEN_141]",
            },
        ),
        end="[UNUSED_TOKEN_145]\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    tool=dict(
        begin=dict(
            with_name="[UNUSED_TOKEN_144]{name}\n",
            name={
                "interpreter": "[UNUSED_TOKEN_142]",
                "plugin": "[UNUSED_TOKEN_141]",
            },
        ),
        end="[UNUSED_TOKEN_143]\n",
        belong="assistant",
    ),
    thought=dict(
        begin=dict(without_name=""),
        end="",
        belong="assistant",
    ),
)

ROLE_CFG_CHAT = dict(
    system=dict(
        begin=dict(
            with_name="<|im_start|>system name={name}\n",
            without_name="<|im_start|>system\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            meta=False,
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    user=dict(
        begin=dict(
            with_name="<|im_start|>user name={name}\n",
            without_name="<|im_start|>user\n",
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    assistant=dict(
        begin=dict(
            with_name="<|im_start|>assistant name={name}\n",
            without_name="<|im_start|>assistant\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=True,
            current=True,
            prefix=False,
            end=True,
        ),
    ),
    environment=dict(
        begin=dict(
            with_name="<|im_start|>environment name={name}\n",
            without_name="<|im_start|>environment\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|im_end|>\n",
        loss=dict(
            icl=False,
            current=False,
            prefix=False,
        ),
    ),
    tool=dict(
        begin=dict(
            with_name="<|action_start|>{name}\n",
            name={
                "interpreter": "<|interpreter|>",
                "plugin": "<|plugin|>",
            },
        ),
        end="<|action_end|>\n",
        belong="assistant",
    ),
    thought=dict(
        begin=dict(without_name=""),
        end="",
        belong="assistant",
    ),
)


class InternlmInstruction(TypedDict):
    loss: str
    role: str
    content: str
    name: str


class InternlmStreamingData(TypedDict):
    source: str
    data: List[InternlmInstruction]
    input_ids: NotRequired[List[int]]


class InternlmTokenizedData(TypedDict):
    input_ids: int
    cumulative_len: List[int]
    position_ids: List[int]
    labels: List[int]


class JsonlDataset:
    def __init__(self, data_roots: Path | str | List[str] | List[Path]):
        if isinstance(data_roots, (Path, str)):
            data_roots = [data_roots]
        self.data_roots = [Path(i) if isinstance(i, str) else i for i in data_roots]
        self._data: List[InternlmStreamingData] = []
        for data_root in self.data_roots:
            jsonlfiles = list(data_root.glob('**/processed/**/*.jsonl'))
            for jsonl in jsonlfiles:
                with jsonlines.open(jsonl) as f:
                    for item in f:
                        self._data.append(
                            InternlmStreamingData(
                                source=str(f),
                                data=[
                                    InternlmInstruction(
                                        loss=i.get('loss', True),
                                        role=i['role'],
                                        content=i['content'],
                                        name=i.get('name'),
                                    )
                                    for i in item
                                ],
                            )
                        )

    def __getitem__(self, index: int | slice) -> InternlmStreamingData:
        return self._data[index]

    def __len__(self):
        return len(self._data)


class TokenizerWrapper(IterableDataset):

    def __init__(
        self,
        dataset: List[InternlmStreamingData],
        tokenizer: dict | ConfigDict,
        role_cfg: dict,
        max_length: int,
        shuffle=True,
        estimate_num=1000,
        pad: bool = False,
        seed=0,
    ):
        if not isinstance(role_cfg, dict):
            raise TypeError(f'role_cfg should be a dict, but got {type(role_cfg)}')
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError('max_length should be an int larger than 0')

        self.dataset = dataset
        self.role_cfg = role_cfg
        self.max_length = max_length

        self.shuffle = shuffle
        self.seed = seed
        self.pad = pad
        self.random_generator = random.Random(seed)
        # self._tokenizer = tokenizer
        self.tokenizer = BUILDER.build(tokenizer)

        if self.shuffle:
            self.random_generator.shuffle(self.dataset)
        self.estimate_num = estimate_num
        self.length = self._estimate_length()

    # @property
    # def tokenizer(self):
    #     if isinstance(self._tokenizer, dict):
    #     return self._tokenizer

    def _estimate_length(self):
        samples = self.random_generator.sample(self.dataset, k=self.estimate_num)
        processed_samples = [i['data'] for i in samples]
        # with ProcessPoolExecutor(max_workers=4) as executor:
        #     results = []
        #     for result in tqdm(executor.map(self.tokenize, processed_samples), total=len(processed_samples)):
        #         results.append(result)
        results = []
        for sample in tqdm(processed_samples):
            result = self.tokenize(sample)
            results.append(result)

        num_tokens = sum(len(i[0]) for i in results)
        max_num_tokens = max(all_gather_object(num_tokens))
        return int(len(self.dataset) / self.estimate_num * max_num_tokens / self.max_length) + 1

    def __len__(self):
        return self.length

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            process_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            process_id = 0
            num_workers = 1

        def pad_to_max_len(
            pad_num: int, token_ids: List[int], labels: List[int], position_ids: List[int], cumulative_len: List[int]
        ):
            token_ids += [DEFAULT_PAD_TOKEN_INDEX] * pad_num
            labels += [IGNORE_INDEX] * pad_num
            position_ids.extend(range(pad_num))
            cumulative_len.append(cumulative_len[-1] + pad_num)

        def cut_to_max_len(
            pad_num: int,
            new_tokens: List[int],
            new_lables: List[int],
            token_ids: List[int],
            labels: List[int],
            position_ids: List[int],
            cumulative_len: List[int],
        ):
            token_ids += new_tokens[:pad_num]
            labels += new_lables[:pad_num]
            position_ids.extend(range(pad_num))
            cumulative_len.append(cumulative_len[-1] + pad_num)

        token_ids = []
        cumulative_len = [0]
        position_ids = []
        labels = []

        idx = process_id
        for _ in range(len(self)):
            while True:
                new_tokens, new_labels = self.tokenize(self.dataset[idx]['data'])
                if len(token_ids) + len(new_tokens) > self.max_length:
                    pad_num = self.max_length - len(token_ids)
                    if pad_num > 0:
                        if self.pad:
                            pad_to_max_len(pad_num, token_ids, labels, position_ids, cumulative_len)
                        else:
                            cut_to_max_len(
                                pad_num, new_tokens, new_labels, token_ids, labels, position_ids, cumulative_len
                            )
                            new_tokens = new_tokens[pad_num:]
                            new_labels = new_labels[pad_num:]
                    yield InternlmTokenizedData(
                        input_ids=token_ids, cumulative_len=cumulative_len, position_ids=position_ids, labels=labels
                    )
                    token_ids = new_tokens
                    labels = new_labels
                    position_ids = list(range(len(new_tokens)))
                    cumulative_len = [0, len(new_tokens)]
                    break
                else:
                    token_ids += new_tokens
                    labels += new_labels
                    cumulative_len.append(cumulative_len[-1] + len(new_tokens))
                    position_ids.extend(range(len(new_tokens)))
                idx = (idx + num_workers) % len(self.dataset)

        # if token_ids:
        #     pad_num = self.max_length - len(token_ids)
        #     pad_to_max_len(pad_num, token_ids, labels, position_ids, cumulative_len)
        #     yield InternlmTokenizedData(
        #         input_ids=token_ids, cumulative_len=cumulative_len, position_ids=position_ids, labels=labels
        #     )

        if self.shuffle:
            self.random_generator.shuffle(self.dataset)

    def tokenize(self, processed_data):
        # For FTDP style data, `processed_data` will be a list of dict.
        # For DDM style data, `processed_data` will be dict, where the `dialogs` field
        # is almost equivalent to the FTDP style data.
        if isinstance(processed_data, dict) and 'dialogs' in processed_data:
            processed_data = processed_data['dialogs']

        def format_begin(role_cfg, message):
            name = message.get('name', None)
            if name is not None:
                begin = role_cfg['begin'].get('with_name', '')
                if name in role_cfg['begin'].get('name', {}):
                    begin = begin.format(name=role_cfg['begin']['name'][name])
                else:
                    begin = begin.format(name=name)
            else:
                begin = role_cfg['begin'].get('without_name', '')
            return begin

        def format_sub_role(messages: List[Dict], roles_cfg) -> List[Dict]:
            new_message = list()
            for message in messages:
                if message['role'] in ['assistant', 'user', 'system', 'environment']:
                    new_message.append(message)
                    continue
                role_cfg = roles_cfg[message['role']]
                begin = format_begin(role_cfg, message)
                new_content = begin + message['content'] + role_cfg['end']
                if role_cfg.get('fallback_role'):
                    new_message.append(dict(role=role_cfg['fallback_role'], content=new_content))
                elif role_cfg.get('belong'):
                    if new_message[-1]['role'] != role_cfg.get('belong'):
                        new_message.append(dict(role=role_cfg.get('belong'), content=new_content))
                    else:
                        new_message[-1]['content'] += new_content
                else:
                    new_message.append(dict(role=message['role'], content=new_content))

            return new_message

        token_ids = []
        labels = []
        _processed_data = format_sub_role(processed_data, self.role_cfg)

        for dialog_item in _processed_data:
            role = dialog_item['role']
            content = dialog_item['content']
            # TODO: is strip necessary? or use lstrip? 避免开始有\n\n的情况
            # content = content.lstrip()
            begin = format_begin(self.role_cfg[role], dialog_item)
            end = self.role_cfg[role]['end']
            begin_token = self.tokenizer.encode(begin)
            begin_label = copy(begin_token)
            if not self.role_cfg[role]['loss'].get('beigin', False):
                begin_label = [IGNORE_INDEX] * len(begin_token)
            end_token = self.tokenizer.encode(self.role_cfg[role]['end'])
            end_label = copy(end_token)
            if not self.role_cfg[role]['loss'].get('end', False):
                end_label = [IGNORE_INDEX] * len(end_token)

            content_token = self.tokenizer.encode(begin + content + end)
            content_token = content_token[len(begin_token) : -len(end_token)]
            content_label = copy(content_token)

            if dialog_item.get('loss', True):
                loss_cfg = self.role_cfg[role]['loss']
            else:
                loss_cfg = dict(icl=False, current=False, meta=False)
            loss_key = dialog_item.get('type')
            if loss_key is None:
                loss_key = 'current'
            if not loss_cfg[loss_key]:
                content_label = [IGNORE_INDEX] * len(content_token)

            if begin == '':
                tokens = content_token
                label = content_label
            else:
                tokens = begin_token + content_token
                label = begin_label + content_label
            if end != '':
                tokens = tokens + end_token
                label = label + end_label

            token_ids += tokens
            labels += label

        bos_token = (
            self.tokenizer.bos_id()
            if isinstance(self.tokenizer, SentencePieceProcessor)
            else self.tokenizer.bos_token_id
        )
        token_ids = [bos_token] + token_ids
        labels = [bos_token] + labels
        token_ids = token_ids[: self.max_length]
        labels = labels[: self.max_length]
        return token_ids, labels


def build_internlm_streaming_dataset(
    data_roots: Path | str | List[str] | List[Path],
    tokenizer: dict | ConfigDict | AutoTokenizer,
    role_cfg: dict,
    max_length: int = 32768,
    shuffle: bool = True,
    estimate_num=1000,
    pad: bool = False,
    seed: int = 0,
):
    rank = get_rank()
    world_size = get_world_size()
    print_log(f'Slice data to {world_size} pieces for data parallel training...', 'current')
    jsonl_dataset = JsonlDataset(data_roots=data_roots)
    dist_sharded_dataset = jsonl_dataset[rank::world_size]
    # import jsonlines
    # writer = jsonlines.open(f'/mnt/petrelfs/yehaochen/codebase/xtuner/work_dirs/streaming_test/origin/{rank}.jsonl', 'w')
    # saved_data = [i['data'] for i in dist_sharded_dataset]
    # writer.write_all(saved_data)
    tokenized_dataset = TokenizerWrapper(
        dataset=dist_sharded_dataset,
        tokenizer=tokenizer,
        role_cfg=role_cfg,
        max_length=max_length,
        shuffle=shuffle,
        seed=seed,
        estimate_num=estimate_num,
        pad=pad,
    )
    return tokenized_dataset
