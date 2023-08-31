import torch
from collections import OrderedDict
from torch.optim import Optimizer
from torch.nn import Module
from typing import Dict, List, Callable, Union

from few_shot.dataproc import create_nshot_task_label


def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_


def meta_gradient_step(model: Module,
                       optimizer: Optimizer,
                       loss_fn: Callable,
                       x: torch.Tensor,
                       y: torch.Tensor,
                       n_shot: int,
                       k_way: int,
                       q_queries: int,
                       order: int,
                       inner_train_steps: int,
                       inner_lr: float,
                       train: bool,
                       device: Union[str, torch.device]):
    """
    Perform a gradient step on a meta-learner.

    :param model: Base model of the meta-learner being trained
    :param optimizer: Optimiser to calculate gradient step from loss
    :param loss_fn: Loss function to calculate between predictions and outputs
    :param x: Input samples for all few shot tasks
    :param y: Input labels of all few shot tasks
    :param n_shot: Number of examples per class in the support set of each task
    :param k_way: Number of classes in the few shot classification task of each task
    :param q_queries: Number of examples per class in the query set of each task. The query set is used to calculate
        meta-gradients after applying the update to
    :param order: Whether to use 1st order MAML (update meta-learner weights with gradients of the updated weights on the
        query set) or 2nd order MAML (use 2nd order updates by differentiating through the gradients of the updated
        weights on the query with respect to the original weights).
    :param inner_train_steps: Number of gradient steps to fit the fast weights during each inner update
    :param inner_lr: Learning rate used to update the fast weights on the inner update
    :param train: Whether to update the meta-learner weights at the end of the episode.
    :param device: Device on which to run computation
    """
    data_shape = x.shape[2:] # [batch,task_num,data_shape]
    create_graph = (True if order == 2 else False) and train

    task_gradients = []
    task_losses = []
    task_predictions = []
    for meta_batch in x:
        # By construction x is a 5D tensor of shape: (meta_batch_size, n*k + q*k, channels, width, height)
        # Hence when we iterate over the first  dimension we are iterating through the meta batches
        x_task_train = meta_batch[:n_shot * k_way]
        x_task_val = meta_batch[n_shot * k_way:]

        # Create a fast model using the current meta model weights
        fast_weights = OrderedDict(model.named_parameters())

        # Train the model for 'inner_train_steps' iterations
        for inner_batch in range(inner_train_steps): # 内部针对每个Task参数更新步数
            # Perform update of model weights
            y = create_nshot_task_label(k_way, n_shot).to(device)
            logits = model.functional_forward(x_task_train, fast_weights)
            loss = loss_fn(logits, y)
            gradients = torch.autograd.grad(loss, fast_weights.values(),create_graph=create_graph) # 对当前的模型参数求导，并存在gradients中

            # Update weights manually
            fast_weights = OrderedDict(
                (name, param - inner_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )
        ### Below part use the query data
        # Do a pass of the model on the validation data from the current task
        y = create_nshot_task_label(k_way, q_queries).to(device)
        logits = model.functional_forward(x_task_val, fast_weights) # 使用刚才support set更新出来的模型参数对query set进行计算
        loss = loss_fn(logits, y)
        loss.backward(retain_graph=True)

        # Get post-update accuracies
        y_pred = logits.softmax(dim=1)
        task_predictions.append(y_pred)

        # Accumulate losses and gradients
        task_losses.append(loss)
        gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph) # 计算当前参数（support更新过）在query set下的梯度并保存
        named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}
        task_gradients.append(named_grads)

    if order == 1:
        if train:
            sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                  for k in task_gradients[0].keys()} # 对之前每步query_set的得到的梯度求平均
            hooks = []
            for name, param in model.named_parameters():
                hooks.append(
                    param.register_hook(replace_grad(sum_task_gradients, name)) # 重新设置梯度更新的梯度关联？
                )

            model.train()
            optimizer.zero_grad()
            # Dummy pass in order to create `loss` variable
            # Replace dummy gradients with mean task gradients using hooks
            logits = model(torch.zeros((k_way,) + data_shape).to(device, dtype=torch.double))
            loss = loss_fn(logits, create_nshot_task_label(k_way, 1).to(device))
            loss.backward()
            optimizer.step()

            for h in hooks:
                h.remove()

        return torch.stack(task_losses).mean(), torch.cat(task_predictions)

    elif order == 2:
        model.train()
        optimizer.zero_grad()
        meta_batch_loss = torch.stack(task_losses).mean() # 直接对loss求平均后反向传播？

        if train:
            meta_batch_loss.backward()
            optimizer.step()

        return meta_batch_loss, torch.cat(task_predictions)
    else:
        raise ValueError('Order must be either 1 or 2.')
