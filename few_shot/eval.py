from typing import Callable

import torch

from few_shot.callbacks import Callback
from few_shot.metrics import categorical_accuracy


class EvaluateFewShot(Callback):
    """Evaluate a network on  an n-shot, k-way classification tasks after every epoch.

    # Arguments
        eval_fn: Callable to perform few-shot classification. Examples include `proto_net_episode`,
            `matching_net_episode` and `meta_gradient_step` (MAML).
        num_tasks: int. Number of n-shot classification tasks to evaluate the model with.
        n_shot: int. Number of samples for each class in the n-shot classification tasks.
        k_way: int. Number of classes in the n-shot classification tasks.
        q_queries: int. Number query samples for each class in the n-shot classification tasks.
        task_loader: Instance of NShotWrapper class
        prepare_batch: function. The preprocessing function to apply to samples from the dataset.
        prefix: str. Prefix to identify dataset.
    """

    def __init__(self,
                 eval_fn: Callable,
                 num_tasks: int,
                 n_shot: int,
                 k_way: int,
                 q_queries: int,
                 taskloader: torch.utils.data.DataLoader,
                 prepare_batch: Callable,
                 prefix: str = 'val_',
                 **kwargs):
        super(EvaluateFewShot, self).__init__()
        self.eval_fn = eval_fn
        self.num_tasks = num_tasks
        self.n_shot = n_shot
        self.k_way = k_way
        self.q_queries = q_queries
        self.taskloader = taskloader
        self.prepare_batch = prepare_batch
        self.prefix = prefix
        self.kwargs = kwargs
        self.metric_name = f'{self.prefix}{self.n_shot}-shot_{self.k_way}-way_acc'

    def on_train_begin(self, logs=None):
        self.loss_fn = self.params['loss_fn']
        self.optimizer = self.params['optimizer']

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        seen = 0
        totals = {'loss': 0, self.metric_name: 0}
        for batch_index, batch in enumerate(self.taskloader):
            x, y = self.prepare_batch(batch)

            loss, y_pred = self.eval_fn(
                self.model,
                self.optimizer,
                self.loss_fn,
                x,
                y,
                n_shot=self.n_shot,
                k_way=self.k_way,
                q_queries=self.q_queries,
                train=False,
                **self.kwargs
            )

            seen += y_pred.shape[0]

            totals['loss'] += loss.item() * y_pred.shape[0]
            totals[self.metric_name] += categorical_accuracy(y, y_pred) * y_pred.shape[0]

        logs[self.prefix + 'loss'] = totals['loss'] / seen
        logs[self.metric_name] = totals[self.metric_name] / seen