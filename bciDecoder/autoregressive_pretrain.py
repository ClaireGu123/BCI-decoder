"""
pretrain template for autoregressive representation learning (GPT-2 style pretrain)

Change the following:
    1. dataset 
        (use custom in_dataset_transform_x, out_dataset_transform_x to make desired input[features, targets] and outputs)
    2. model
        initialize with the desired model (e.g. alternative implementation of transformer_AE.NeuralDecoder)
    3. optimizer, criterion(loss)
"""

import numpy as np
import pandas as pd

import hydra
import torch
import torch.utils.data
import ignite.distributed as idist
from ignite.engine import Events, Engine
from ignite.metrics import  Loss
from ignite.handlers import PiecewiseLinear

from torch.cuda.amp import autocast, GradScaler
import torch.utils.model_zoo as model_zoo

import torch.onnx
from torchmetrics.text import CharErrorRate


from datasets.neural_speech_dataset import NeuralDataset

from datasets.sequence_data_transformer import (   
                                            seq_collate_wrapper,
                                           in_dataset_transform_ctc,
                                           out_dataset_transform_ctc,)

from datasets.utils.text_processor import PHONE_DEF_SIL


from models.transformer_AE import NeuralDecoder

np.random.seed(0)
torch.manual_seed(0)
device = idist.device()
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True

# TODO: configuration and project setup 
    # tutorial: https://pytorch-ignite.ai/tutorials/intermediate/01-cifar10-distributed/
    # TODO: setup sufficient logging
    # TODO: setting up model checkpoint and save model
    # TODO: add configuration for optimizer and LRScheduler

# TODO: experiment todo
    # TODO: setup model/data analysis steps, make continuous model inspection
    # TODO: use logging to analize and improve models

# TODO: modeling
    # TODO: add more data transformation as needed
    # TODO: figure out the training procedure: 
        #   random sample-batch across all sessions vs. random sample-batch within session

def cer(pred_ids, targets):
    # def make_str(ids):
    #     return ''.join([chr(p) for p in ids])
    def id2phon(ids):
        return ' '.join([PHONE_DEF_SIL[id_] for id_ in ids])

    pred = np.apply_along_axis(id2phon, 1,pred_ids.cpu().numpy())
    target = np.apply_along_axis(id2phon, 1,targets.cpu().numpy())
    # print(f"prediction sample: {pred[0]}")
    # print(f"target sample: {target[0]}")
    return CharErrorRate()(pred, target)
    

def output_transform_cer(output):
    return torch.argmax(output[0], dim=-1), output[1]

def output_transform_ctc(output):
    y_pred_log_probs = torch.nn.functional.log_softmax(output[0], dim=-1)
    y = output[1]
    input_lengths = torch.full((y_pred_log_probs.shape[0], ), y_pred_log_probs.shape[1], dtype=torch.long).to(device)
        
    target_lengths = torch.full((y.shape[0],), y.shape[1],dtype=torch.long).to(device)
    return y_pred_log_probs.permute(1,0,2), output[1], {'input_lengths':input_lengths, 
                                                        'target_lengths':target_lengths,
                                                        }


def load_train_val_sets(config):

    neural_ctc_testset = NeuralDataset(config,
                                partition=config.dataset.test_partition,
                                input_transform=in_dataset_transform_ctc,
                                transform=out_dataset_transform_ctc)
    neural_ctc_trainset = NeuralDataset(config,
                                    partition=config.dataset.train_partition,
                                    input_transform=in_dataset_transform_ctc, 
                                    transform=out_dataset_transform_ctc)
    
    if not config['distributed']:
        test_ctc_loader = torch.utils.data.DataLoader(neural_ctc_testset, 
                                                    batch_size=config['batch_size'], 
                                                    collate_fn=seq_collate_wrapper,
                                                    )
        train_ctc_loader = torch.utils.data.DataLoader(neural_ctc_trainset, 
                                                    batch_size=config['batch_size'], 
                                                    collate_fn=seq_collate_wrapper,
                                                    )
    else:
        test_ctc_loader = idist.auto_dataloader(
            neural_ctc_testset,
            batch_size = config.batch_size,
            num_workers=config.num_workers,
            collate_fn=seq_collate_wrapper
        )

        train_ctc_loader = idist.auto_dataloader(
            neural_ctc_trainset,
            batch_size = config.batch_size,
            num_workers=config.num_workers,
            collate_fn=seq_collate_wrapper
        )
    return  train_ctc_loader, test_ctc_loader


def initialize_model(config, ):
    neural_decoder = NeuralDecoder(config,)
    optimizer  = torch.optim.Adam(neural_decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    if config['distributed']:
        neural_decoder = idist.auto_model(neural_decoder)
        optimizer = idist.auto_optim(optimizer)
    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
    return neural_decoder, optimizer, criterion


def create_trainer(model, optimizer, criterion, distributed=False):
    def step(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = batch.input_features, batch.seqClassIDs
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if distributed:
            with_amp=True
            with autocast(enabled=with_amp):
                y_pred = model(x)
                y_pred_log_probs = torch.nn.functional.log_softmax(y_pred, dim=-1)
                input_lengths = torch.full((y_pred_log_probs.shape[0], ), y_pred_log_probs.shape[1], dtype=torch.long).to(device)
                
                target_lengths = torch.full((y.shape[0],), y.shape[1],dtype=torch.long).to(device)
                loss = criterion(y_pred_log_probs.permute(1,0,2),
                                                    y,
                                                    input_lengths,
                                                    target_lengths,
                                                    ) 
                scaler = GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)
                scaler.scale(loss).backward()  # If with_amp=False, this is equivalent to loss.backward()
                scaler.step(optimizer)  # If with_amp=False, this is equivalent to optimizer.step()
                scaler.update()  # If with_amp=False, this step does nothing
        else:
            y_pred = model(x)
            y_pred_log_probs = torch.nn.functional.log_softmax(y_pred, dim=-1)
            input_lengths = torch.full((y_pred_log_probs.shape[0], ), y_pred_log_probs.shape[1], dtype=torch.long).to(device)
            
            target_lengths = torch.full((y.shape[0],), y.shape[1],dtype=torch.long).to(device)
            loss = criterion(y_pred_log_probs.permute(1,0,2),
                                                y,
                                                input_lengths,
                                                target_lengths,
                                                ) 
            loss.backward()
            optimizer.step()

        return loss.item()
    
    trainer = Engine(step)
    return trainer

# TODO: configure this into training
# def get_lr_scheduler(config, optimizer):
#     milestones_values = [
#         (0, 0.0),
#         (config["num_iters_per_epoch"] * config["num_warmup_epochs"], config["learning_rate"]),
#         (config["num_iters_per_epoch"] * config["num_epochs"], 0.0),
#     ]
#     lr_scheduler = PiecewiseLinear(
#         optimizer, param_name="lr", milestones_values=milestones_values
#     )
#     return lr_scheduler

def create_evaluator(model):
    def eval_step(engine, batch):
        model.eval()
        x, y = batch.input_features, batch.seqClassIDs
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return model(x), y
    evaluator = Engine(eval_step)

    metric = Loss(torch.nn.functional.ctc_loss, output_transform=output_transform_ctc)
    metric.attach(evaluator, 'ctc_loss')

    metric = Loss(cer, output_transform=output_transform_cer)
    metric.attach(evaluator, 'cer')
    return evaluator

def setup_rank_zero():
    # TODO: setting up things on the master process `rank`=0
    # TODO: save model and checkpoints
    pass

def training(local_rank, config):

    rank = idist.get_rank()

    # TODO: setup seed
    # TODO: setup logger

    # if rank == 0:
    #     setup_rank_zero(cofig)

    neural_decoder, optimizer, criterion = initialize_model(config)
    dl_train, dl_val = load_train_val_sets(config)

    # TODO: setup learning rate scheduler

    trainer = create_trainer(neural_decoder.to(device), optimizer, criterion, config['distributed'])
    evaluator = create_evaluator(neural_decoder)


    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate_model(trainer):
        
        evaluator.run(dl_train)
        metrics = evaluator.state.metrics

        print("Train Results - Epoch: {} CTC Loss: {:.2f} cer: {:.2f}"
        .format(trainer.state.epoch, metrics['ctc_loss'],
                metrics['cer'],
                ))
        
        evaluator.run(dl_val)
        metrics = evaluator.state.metrics

        print("Test Results - Epoch: {} CTC Loss: {:.2f} : {:.2f}"
        .format(trainer.state.epoch, metrics['ctc_loss'],
                metrics['cer'],
                ))
    
    trainer.run(dl_train, max_epochs=config['max_epochs'])

@hydra.main(config_path='configs', config_name='config_auto_pt')
def main(config):
    if config['distributed']:
        with idist.Parallel(backend=config['backend'],
                            nproc_per_node=config['nproc_per_node']) as parallel:
            parallel.run(training, config)
    else:
        training(0, config)
    

if __name__ == '__main__':
    main()