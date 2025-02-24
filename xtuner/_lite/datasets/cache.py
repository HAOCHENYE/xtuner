# Copyright (c) OpenMMLab. All rights reserved.
# This module implemented here for preventing from the circular import error.
# Actually, it should be include in `xtuner._lite.datasets.utils` and
# `xtuner._lite.datasets.utils` should not import any module
# from `xtuner._lite.datasets` because it is a utility module.
#
import hashlib
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict

from transformers import PreTrainedTokenizer


def tokenizer_hash(tokenizer: PreTrainedTokenizer):
    with tempfile.TemporaryDirectory() as tokenizer_tempdir:
        tokenizer.save_pretrained(tokenizer_tempdir)
        tokenizer_files = sorted(Path(tokenizer_tempdir).iterdir())
        file_hash = hashlib.sha256()
        for file in tokenizer_files:
            with open(file, "rb") as f:
                file_hash.update(f.read())
        return file_hash.hexdigest()


def calculate_jsonl_sha256(path):
    with open(path, "rb") as f:
        file_hash = hashlib.sha256()
        file_hash.update(f.read())
    return file_hash.hexdigest()


class CacheObj(TypedDict, total=False):
    num_tokens: int


class CachableTokenizeFunction(ABC):
    @abstractmethod
    def __call__(self, item: Any) -> CacheObj:
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> str:
        raise NotImplementedError
