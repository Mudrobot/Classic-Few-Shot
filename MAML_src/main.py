"""
Reproduce Model-agnostic Meta-learning results (supervised only) of Finn et al
"""
from torch.utils.data import DataLoader
from torch import nn
import argparse

from MAML_src.datasets import OmniglotDataset, MiniImageNet
from few_shot.dataproc import NShotTaskSampler, create_nshot_task_label, prepare_meta_batch
from few_shot.eval import EvaluateFewShot
from MAML_src.fit_function import meta_gradient_step
from MAML_src.models import FewShotClassifier
from few_shot.train import fit
from few_shot.callbacks import *
from MAML_src.utils import setup_dirs
from config import PATH

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="omniglot")  # miniImageNet)
parser.add_argument('--n', default=1, type=int)
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--q', default=1, type=int)  # Number of examples per class to calculate meta gradients with
parser.add_argument('--inner-train-steps', default=1, type=int)
parser.add_argument('--inner-val-steps', default=3, type=int)
parser.add_argument('--inner-lr', default=0.4, type=float)
parser.add_argument('--meta-lr', default=0.001, type=float)
parser.add_argument('--meta-batch-size', default=4, type=int)
parser.add_argument('--order', default=1, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch-len', default=100, type=int) # 有多少episode
parser.add_argument('--eval-batches', default=20, type=int)

args = parser.parse_args()





if __name__ == '__main__':
    if args.dataset == 'omniglot':
        dataset_class = OmniglotDataset
        fc_layer_size = 2304 # 64
        num_input_channels = 1
    elif args.dataset == 'miniImageNet':
        dataset_class = MiniImageNet
        fc_layer_size = 1600
        num_input_channels = 3
    else:
        raise (ValueError('Unsupported dataset'))

    param_str = f'{args.dataset}_order={args.order}_n={args.n}_k={args.k}_metabatch={args.meta_batch_size}_' \
                f'train_steps={args.inner_train_steps}_val_steps={args.inner_val_steps}'
    print(param_str)

    ###################
    # Create datasets #
    ###################
    background = dataset_class('background')
    background_taskloader = DataLoader(
        background,
        batch_sampler=NShotTaskSampler(background, args.epoch_len, n=args.n, k=args.k, q=args.q,
                                       num_tasks=args.meta_batch_size),
        num_workers=8
    )
    evaluation = dataset_class('evaluation')
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, args.eval_batches, n=args.n, k=args.k, q=args.q,
                                       num_tasks=args.meta_batch_size),
        num_workers=8
    )

    ############
    # Training #
    ############
    print(f'Training MAML on {args.dataset}...')
    meta_model = FewShotClassifier(num_input_channels, args.k, fc_layer_size).to(device, dtype=torch.double)
    meta_optimiser = torch.optim.Adam(meta_model.parameters(), lr=args.meta_lr)
    loss_fn = nn.CrossEntropyLoss().to(device)

    callbacks = [
        EvaluateFewShot(
            eval_fn=meta_gradient_step,
            num_tasks=args.eval_batches,
            n_shot=args.n,
            k_way=args.k,
            q_queries=args.q,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size, num_input_channels),
            # MAML kwargs
            inner_train_steps=args.inner_val_steps,
            inner_lr=args.inner_lr,
            device=device,
            order=args.order,
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/maml/{param_str}.pth',
            monitor=f'val_{args.n}-shot_{args.k}-way_acc'
        ),
        ReduceLROnPlateau(patience=10, factor=0.5, monitor=f'val_loss'),
        CSVLogger(PATH + f'/logs/maml/{param_str}.csv'),
    ]

    fit(
        meta_model,
        meta_optimiser,
        loss_fn,
        epochs=args.epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_meta_batch(args.n, args.k, args.q, args.meta_batch_size, num_input_channels),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=meta_gradient_step,
        fit_function_kwargs={'n_shot': args.n, 'k_way': args.k, 'q_queries': args.q,
                             'train': True,
                             'order': args.order, 'device': device, 'inner_train_steps': args.inner_train_steps,
                             'inner_lr': args.inner_lr},
    )
