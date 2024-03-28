"""
contrastive pretraining representation learning for neural decoder
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.utils.data

import ignite
import ignite.distributed as idist
from ignite.engine import Events, Engine
from ignite.utils import manual_seed, setup_logger

from torch.cuda.amp import autocast, GradScaler

import torch.onnx
from torchmetrics.text import CharErrorRate
from torchsummary import summary
from omegaconf import OmegaConf


from datasets.neural_speech_dataset import NeuralDataset

from datasets.sequence_data_transformer import (   
                                            seq_collate_wrapper,
                                           in_dataset_transform_ctc,
                                           out_dataset_transform_ctc,)

from datasets.utils.text_processor import PHONE_DEF_SIL


from models.CLIP_model import CLIP
from models.decoder_metrics import CLIPLoss

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


def get_save_handler(config):
    if config["with_clearml"]:
        from ignite.contrib.handlers.clearml_logger import ClearMLSaver

        return ClearMLSaver(dirname=config["output_path"])

    return config["output_path"]

def load_checkpoint(resume_from):
    checkpoint_fp = Path(resume_from)
    assert (
        checkpoint_fp.exists()
    ), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
    checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
    return checkpoint


def id2phon(ids):
    return ' '.join([PHONE_DEF_SIL[id_] for id_ in ids])

def cer(pred_ids, targets):
    # def make_str(ids):
    #     return ''.join([chr(p) for p in ids])
    pred = np.apply_along_axis(id2phon, 1,pred_ids.cpu().numpy())
    target = np.apply_along_axis(id2phon, 1,targets.cpu().numpy())
    print(f"prediction sample: {pred[0][:100]}")
    print(f"target sample: {target[0][:100]}")
    return CharErrorRate()(pred, target)

# def clip_loss(embed1, embed2):
#     cross_cor = torch.matmul(torch.permute(text_embeds,(1,0)), ecog_embeds) / text_embeds.shape[0]
#     c_diff = (cross_cor - torch.eye(text_embeds.shape[1]).to(device)).pow(2)
#     scaled = c_diff.mul_(self.config.model.reg)
#     retrun torch.diagonal(c_diff).sum() + scaled.sum() - torch.diagonal(scaled).sum()


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
    clip_model = CLIP(config)
    summary(clip_model, [(config.model.num_frames, config.model.in_channels, ) ,
                         (config.model.text_embedding_dim,),])
    optimizer  = torch.optim.Adam(clip_model.parameters(), lr=1e-3, weight_decay=1e-4)
    if config['distributed']:
        clip_model = idist.auto_model(clip_model)
        optimizer = idist.auto_optim(optimizer)
    return clip_model, optimizer


def create_trainer(clip_model, optimizer, distributed=False):
    def step(engine, batch):
        clip_model.train()
        optimizer.zero_grad()
        x, text_emb = batch.input_features,  batch.text_embedings
        x = x.to(device, non_blocking=True)
        text_emb = text_emb.to(device, non_blocking=True).squeeze(1)
        if distributed:
            with_amp=True
            with autocast(enabled=with_amp):
                loss = clip_model(x, text_emb)
                
                scaler = GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000, enabled=True)
                scaler.scale(loss).backward()  # If with_amp=False, this is equivalent to loss.backward()
                scaler.step(optimizer)  # If with_amp=False, this is equivalent to optimizer.step()
                scaler.update()  # If with_amp=False, this step does nothing
        else:
            loss = clip_model(x, text_emb)
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

def create_evaluator(clip_model):
    def eval_step(engine, batch):
        clip_model.eval()
        x, text_emb = batch.input_features,  batch.text_embedings
        x = x.to(device, non_blocking=True)
        text_emb = text_emb.to(device, non_blocking=True).squeeze(1)
        with torch.no_grad():
            loss = clip_model(x, text_emb)
        return loss
    evaluator = Engine(eval_step)

    # metric = Loss(lambda x,y: x, output_transform=lambda x: x)
    # metric.attach(evaluator, 'clip_loss')
    metric = CLIPLoss(ignored_class=-100)
    metric.attach(evaluator, 'clip_loss')

    return evaluator

def setup_rank_zero(logger, config):
    # TODO: setting up things on the master process `rank`=0
    # TODO: save model and checkpoints
    device = idist.device()

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = config['output_path']

    folder_name = (
        f"{config['model']['name']}-{idist.backend()}-{idist.get_world_size()}-{now}"
    )

    output_path = Path(output_path) / folder_name


    if not output_path.exists():
        output_path.mkdir(parents=True)

    config['output_path'] = output_path.as_posix()
    logger.info(f"Output path: {config['output_path']}")

    if config['with_clearml']:
        from clearml import Task

        task = Task.init(
            project_name="contrastive_pretrain",
            task_name=output_path.stem,
        )
        config["clearml_task_id"] = task.id
        logger.info(f"ClearML task: {task.id}")

        task.connect_configuration(OmegaConf.to_container(config, resolve=True))

        hyper_params = [
            "model",
            "batch_size",
            "max_epochs",
        ]
        task.connect({k: v for k, v in config.items()})
 
def log_basic_info(logger, config):
    logger.info(f"Train on CIFAR10")
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")
    if torch.cuda.is_available():
        # explicitly import cudnn as torch.backends.cudnn can not be pickled with hvd spawning procs
        from torch.backends import cudnn

        logger.info(
            f"- GPU Device: {torch.cuda.get_device_name(idist.get_local_rank())}"
        )
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")



def training(local_rank, config):

    rank = idist.get_rank()

    # TODO: setup seed
    manual_seed(config['seed'] + rank)
    # TODO: setup logger
    logger = setup_logger(name="CLIP_pretrain")
    log_basic_info(logger, config)

    if rank == 0:
        setup_rank_zero(logger, config)

    clip_model, optimizer = initialize_model(config)
    dl_train, dl_val = load_train_val_sets(config)

    # TODO: setup learning rate scheduler

    trainer = create_trainer(clip_model.to(device), optimizer, config['distributed'])
    evaluator = create_evaluator(clip_model)


    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def evaluate_model(trainer):
        
        evaluator.run(dl_train)
        metrics = evaluator.state.metrics

        logger.info("Train Results - Iteration: {} CLIP Loss: {:.2f} "
        .format(trainer.state.iteration, metrics['clip_loss'],
                ))
        
        evaluator.run(dl_val)
        metrics = evaluator.state.metrics

        logger.info("Test Results - Iteration: {} CLIP Loss: {:.2f} "
        .format(trainer.state.iteration, metrics['clip_loss'],
                ))
    
    trainer.run(dl_train, max_epochs=config['max_epochs'])

global config

@hydra.main(config_path='configs', config_name='config_clip_pt', version_base='1.1')
def main(config):
    if config['distributed']:
        with idist.Parallel(backend=config['backend'],
                            nproc_per_node=config['nproc_per_node']) as parallel:
            parallel.run(training, config)
    else:
        training(0, config)
    

if __name__ == '__main__':
    main()