"""
Reproduce Omniglot results of Snell et al Prototypical networks.
"""
import torch.backends.cudnn
from torch.optim import Adam
import argparse

from few_shot.eval import EvaluateFewShot
from ProtoNet_src.datasets import OmniglotDataset, MiniImageNet
from ProtoNet_src.models import get_few_shot_encoder
from few_shot.dataproc import NShotTaskSampler, prepare_nshot_task
from ProtoNet_src.fit_function import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from ProtoNet_src.utils import setup_dirs
from config import PATH

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="omniglot")  # miniImageNet
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=5, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
args = parser.parse_args()

evaluation_episodes = 1000
episodes_per_epoch = 100  # 一个epoch有100个batch


def lr_schedule(epoch, lr):  # lr损失函数
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


if __name__ == '__main__':
    if args.dataset == 'omniglot':
        n_epochs = 40
        dataset_class = OmniglotDataset
        num_input_channels = 1
        drop_lr_every = 20
    elif args.dataset == 'miniImageNet':
        n_epochs = 80
        dataset_class = MiniImageNet
        num_input_channels = 3
        drop_lr_every = 40
    else:
        raise (ValueError, 'Unsupported dataset')

    param_str = f'{args.dataset}_nt={args.n_train}_kt={args.k_train}_qt={args.q_train}_' \
                f'nv={args.n_test}_kv={args.k_test}_qv={args.q_test}'

    print(param_str)
    ###################
    # Create datasets #
    ###################
    background = dataset_class('background')
    background_taskloader = DataLoader(
        background,
        batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
        num_workers=4
    )
    evaluation = dataset_class('evaluation')
    evaluation_taskloader = DataLoader(
        evaluation,
        batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
        num_workers=4
    )

    #########
    # Model #
    #########
    model = get_few_shot_encoder(num_input_channels)
    model = model.to(device, dtype=torch.double)

    ############
    # Training #
    ############
    print(f'Training Prototypical network on {args.dataset}...')
    optimiser = Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.NLLLoss().cuda()

    callbacks = [
        EvaluateFewShot(
            eval_fn=proto_net_episode,
            num_tasks=evaluation_episodes,
            n_shot=args.n_test,
            k_way=args.k_test,
            q_queries=args.q_test,
            taskloader=evaluation_taskloader,
            prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
            distance=args.distance
        ),
        ModelCheckpoint(
            filepath=PATH + f'/models/proto_nets/{param_str}.pth',
            monitor=f'val_{args.n_test}-shot_{args.k_test}-way_acc'
        ),
        LearningRateScheduler(schedule=lr_schedule),
        CSVLogger(PATH + f'/logs/proto_nets/{param_str}.csv'),
    ]

    fit(
        model,
        optimiser,
        loss_fn,
        epochs=n_epochs,
        dataloader=background_taskloader,
        prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
        callbacks=callbacks,
        metrics=['categorical_accuracy'],
        fit_function=proto_net_episode,
        fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                             'distance': args.distance},
    )
