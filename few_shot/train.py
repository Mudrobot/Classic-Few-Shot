"""
The `fit` function in this file implements a slightly modified version
of the Keras `model.fit()` API.
"""
import torch
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union

from few_shot.callbacks import CallbackList, AvgMetrics, ProgressBarLogger,Callback
from few_shot.metrics import NAMED_METRICS


def gradient_step(model: Module, optimizer: Optimizer, loss_fn: Callable, x: torch.Tensor, y: torch.Tensor, **kwargs):
    '''
    Takes a single gradient step
    :param model: Model to be fitted
    :param optimizer:  Optimizer to calculate gradient step from loss
    :param loss_fn: Loss function to calculate between predictions and outputs
    :param x: Input samples
    :param y: Input targets
    :return: loss, y_pred
    '''
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    loss.backward()
    optimizer.step()

    return loss, y_pred

def batch_metrics(model: Module, y_pred: torch.Tensor, y: torch.Tensor, metrics: List[Union[str, Callable]],batch_logs: dict):
    '''
    Calculates metrics for the current training batch
    :param model: Model being fit
    :param y_pred: predictions for a particular batch
    :param y: labels for a particular batch
    :param metrics: Dictionary of Callable functions or string type which indicates the evaluation metrics
    :param batch_logs: Dictionary of logs for the current batch
    :return:
    '''
    model.eval()
    for m in metrics:
        if isinstance(m, str):
            batch_logs[m] = NAMED_METRICS[m](y, y_pred)
        else:
            # Assume metric is a callable function
            batch_logs = m(y, y_pred)

    return batch_logs



def fit(model: Module, optimizer: Optimizer, loss_fn: Callable, epochs: int, dataloader: DataLoader,
        prepare_batch: Callable, metrics: List[Union[str, Callable]] = None, callbacks: List[Callback] = None,
        verbose: bool =True, fit_function: Callable = gradient_step, fit_function_kwargs: dict = {}):
    '''
    Function to abstract away training loop.

    The benefit of this function is that allows training scripts to be much more readable and allows for easy re-use of
    common training functionality provided they are written as a subclass of voicemap.Callback (following the
    Keras API).

    :param model: Model to be fitted.
    :param optimizer: Optimiser to calculate gradient step from loss
    :param loss_fn: Loss function to calculate between predictions and outputs
    :param epochs: Number of epochs of fitting to be performed
    :param dataloader: `torch.DataLoader` instance to fit the model to
    :param prepare_batch: Callable to perform any desired preprocessing
    :param metrics: Optional list of metrics to evaluate the model with
    :param callbacks: Additional functionality to incorporate into training such as logging metrics to csv, model
            checkpointing, learning rate scheduling etc... See voicemap.callbacks for more.
    :param verbose: All print output is muted if this argument is `False`
    :param fit_function: Function for calculating gradients. Leave as default for simple supervised training on labelled
            batches. For more complex training procedures (meta-learning etc...) you will need to write your own
            fit_function
    :param fit_function_kwargs: Keyword arguments to pass to `fit_function`
    '''
    # Determine Number of Samples
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size

    callbacks = CallbackList([AvgMetrics()]+(callbacks or [])+[ProgressBarLogger()])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches,
        'batch_size': batch_size,
        'verbose': verbose,
        'metrics': (metrics or []),
        'prepare_batch': prepare_batch,
        'loss_fn': loss_fn,
        'optimizer': optimizer
    })

    if verbose:
        print('Begin training...')

    callbacks.on_train_begin()

    for epoch in range(1, epochs+1):
        callbacks.on_epoch_begin(epoch)

        epoch_logs = {}
        for batch_index, batch in enumerate(dataloader):
            batch_logs = dict(batch=batch_index, size=(batch_size or 1))

            callbacks.on_batch_begin(batch_index,batch_logs)

            x, y = prepare_batch(batch)

            loss,y_pred = fit_function(model, optimizer, loss_fn, x, y, **fit_function_kwargs)
            batch_logs['loss'] = loss.item()

            # Loops through all metrics
            batch_logs = batch_metrics(model, y_pred, y, metrics, batch_logs)

            callbacks.on_batch_end(batch_index, batch_logs)

        # Run on epoch end
        callbacks.on_epoch_end(epoch, epoch_logs)

    # Run on train end
    if verbose:
        print('Finished.')

    callbacks.on_train_end()
