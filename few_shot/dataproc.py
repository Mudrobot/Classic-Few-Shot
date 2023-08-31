from torch.utils.data import Sampler
from typing import List, Iterable, Callable, Tuple
import numpy as np
import torch

from few_shot.metrics import categorical_accuracy

class NShotTaskSampler(Sampler):
    def __init__(self,
                 dataset: torch.utils.data.Dataset,
                 episodes_per_epoch: int = None,
                 n: int = None,
                 k: int = None,
                 q: int = None,
                 num_tasks: int = 1,
                 fixed_tasks: List[Iterable[int]] = None):
        """PyTorch Sampler subclass that generates batches of n-shot, k-way, q-query tasks.

        Each n-shot task contains a "support set" of `k` sets of `n` samples and a "query set" of `k` sets
        of `q` samples. The support set and the query set are all grouped into one Tensor such that the first n * k
        samples are from the support set while the remaining q * k samples are from the query set.

        The support and query sets are sampled such that they are disjoint i.e. do not contain overlapping samples.

        # Arguments
            dataset: Instance of torch.utils.data.Dataset from which to draw samples
            episodes_per_epoch: Arbitrary number of batches of n-shot tasks to generate in one epoch
            n_shot: int. Number of samples for each class in the n-shot classification tasks.
            k_way: int. Number of classes in the n-shot classification tasks.
            q_queries: int. Number query samples for each class in the n-shot classification tasks.
            num_tasks: Number of n-shot tasks to group into a single batch
            fixed_tasks: If this argument is specified this Sampler will always generate tasks from
                the specified classes
        """
        super(NShotTaskSampler, self).__init__(dataset)
        self.episodes_per_epoch = episodes_per_epoch
        self.dataset = dataset
        if num_tasks < 1:
            raise ValueError('num_tasks must be > 1')

        self.num_tasks = num_tasks
        # k way n shot task
        self.k = k
        self.n = n
        self.q = q
        self.fixed_tasks = fixed_tasks

        self.i_task = 0

    def __len__(self):
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            batch = []

            for task in range(self.num_tasks): # num_task = 1 (default)
                if self.fixed_tasks is None:
                    # Get random classes
                    episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k, replace=False)
                else:
                    # Loop through classes in fixed_tasks
                    episode_classes = self.fixed_tasks[self.i_task % len(self.fixed_tasks)]
                    self.i_task += 1

                df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]

                support_k = {k:None for k in episode_classes} # 针对不同的类别（k）挑选出支持集
                for k in episode_classes:
                    # Select support example
                    support = df[df['class_id'] == k].sample(self.n)
                    support_k[k] = support

                    for i,s in support.iterrows():
                        batch.append(s['id'])

                for k in episode_classes:
                    query = df[(df['class_id'] == k) & (~df['id'].isin(support_k[k]['id']))].sample(self.q)
                    for i, q in query.iterrows():
                        batch.append(q['id'])

            yield np.stack(batch)

def prepare_meta_batch(n, k, q, meta_batch_size, num_input_channels):
    def prepare_meta_batch_(batch):
        x, y = batch
        # Reshape to `meta_batch_size` number of tasks. Each task contains
        # n*k support samples to train the fast model on and q*k query samples to
        # evaluate the fast model on and generate meta-gradients
        x = x.reshape(meta_batch_size, n * k + q * k, num_input_channels, x.shape[-2], x.shape[-1])
        # Move to device
        x = x.double().cuda()
        # Create label
        y = create_nshot_task_label(k, q).cuda().repeat(meta_batch_size)
        return x, y

    return prepare_meta_batch_
def prepare_nshot_task(n: int, k: int, q: int) -> Callable:
    """Typical n-shot task preprocessing.

    # Arguments
        n: Number of samples for each class in the n-shot classification task
        k: Number of classes in the n-shot classification task
        q: Number of query samples for each class in the n-shot classification task

    # Returns
        prepare_nshot_task_: A Callable that processes a few shot tasks with specified n, k and q
    """
    def prepare_nshot_task_(batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create 0-k label and move to GPU
        :param batch:
        :return:
        """
        x, y = batch
        x = x.double().cuda()
        # Create dummy 0-(num_classes - 1) label
        y = create_nshot_task_label(k,q).cuda()
        return x, y

    return prepare_nshot_task_

def create_nshot_task_label(k: int,q: int) -> torch.Tensor:
    '''
    Creates an n-shot task label.
    :param k: k-way
    :param q: q-query
    :return: Label vector for n-shot task of shape [q * k, ]
    '''

    y = torch.arange(0, k , 1 / q).long() # 这里用long就取整了
    return y
