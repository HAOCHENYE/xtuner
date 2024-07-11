# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union, Sequence

from mmengine.runner import IterBasedTrainLoop, BaseLoop
from mmengine.dist import all_gather_object, get_rank
from torch.utils.data import DataLoader
import torch


class TrainLoop(IterBasedTrainLoop):

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        max_iters: Optional[int] = None,
        max_epochs: Union[int, float] = None,
        **kwargs,
    ) -> None:

        if max_iters is None and max_epochs is None:
            raise RuntimeError('Please specify the `max_iters` or '
                               '`max_epochs` in `train_cfg`.')
        elif max_iters is not None and max_epochs is not None:
            raise RuntimeError('Only one of `max_iters` or `max_epochs` can '
                               'exist in `train_cfg`.')
        else:
            if max_iters is not None:
                iters = int(max_iters)
                assert iters == max_iters, ('`max_iters` should be a integer '
                                            f'number, but get {max_iters}')
            elif max_epochs is not None:
                if isinstance(dataloader, dict):
                    diff_rank_seed = runner._randomness_cfg.get(
                        'diff_rank_seed', False)
                    dataloader = runner.build_dataloader(
                        dataloader,
                        seed=runner.seed,
                        diff_rank_seed=diff_rank_seed)
                iters = max_epochs * len(dataloader)
            else:
                raise NotImplementedError
        super().__init__(
            runner=runner, dataloader=dataloader, max_iters=iters, **kwargs)


class IterableEpochBasedTrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        max_epochs: int,
    ) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader.dataset)
        self._epoch = 0
        self._iter = 0
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        datagen = self._next_data()
        for batch_idx, data_batch in datagen:
            consumed_sample = len(data_batch['data']['cumulative_len'][0]) - 1
            self.runner.call_hook('before_train_iter', batch_idx=self._iter, data_batch=data_batch)
            # Enable gradient accumulation mode and avoid unnecessary gradient
            # synchronization during gradient accumulation process.
            # outputs should be a dict of loss.
            outputs = self.runner.model.train_step(data_batch, optim_wrapper=self.runner.optim_wrapper)

            self.runner.call_hook('after_train_iter', batch_idx=self._iter, data_batch=data_batch, outputs=outputs)
            self._iter += consumed_sample

        self.runner.call_hook('after_train')
        return self.runner.model

    def _next_data(self):
        # self.runner.call_hook('before_train_epoch')
        dataloader_iter = iter(self.dataloader)
        batch_idx = 0
        epoch_finished = False
        self.runner.call_hook('before_train_epoch')
        while not self._should_stop():
            if all(all_gather_object(epoch_finished)):
                self.runner.call_hook('after_train_epoch')
                self.runner.call_hook('before_train_epoch')
                self._epoch += 1

                epoch_finished = False
                continue
            try:
                data_batch = next(dataloader_iter)
                batch_idx += 1
                yield batch_idx, data_batch
            except StopIteration:
                dataloader_iter = iter(self.dataloader)
                epoch_finished = True
                batch_idx = 0
                data_batch = next(dataloader_iter)
                yield batch_idx, data_batch

    def _should_stop(self):
        epochs = all_gather_object(self._epoch)
        return all(epoch >= self._max_epochs for epoch in epochs)

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.model.train()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
