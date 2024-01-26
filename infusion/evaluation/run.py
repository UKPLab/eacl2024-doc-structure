import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Type, List

import omegaconf
import pytorch_lightning as pl
import torch.utils.data
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from evaluation.common import BaseModel, BaseTask, Statistics, CustomDataset, Partition, BasePrediction, BaseInstance
from structformer.scaffold_tasks import S2ORCITGSubsetDataHandler, S2ORCITGSubsetInstance

logger = logging.getLogger(__name__)


class CustomCallback(Callback):
    predictions: List[BasePrediction]
    task: BaseTask
    stats: Statistics
    config: omegaconf.DictConfig

    def __init__(self, task: BaseTask, stats: Statistics, config: omegaconf.DictConfig):
        super(CustomCallback, self).__init__()
        self.predictions = []
        self.task = task
        self.stats = stats
        self.config = config

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                                outputs, batch, batch_idx, dataloader_idx) -> None:
        self.predictions += outputs

    def on_test_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                          outputs, batch, batch_idx, dataloader_idx) -> None:
        self.predictions += outputs

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logger.info(f"Evaluation on {Partition.DEV.value} data after {trainer.global_step} steps:")
        assert len(self.predictions) == len(self.task.dev_instances)

        result = self.task.evaluate(self.task.dev_instances, self.predictions, Partition.DEV)
        self.stats.results_by_step.append((trainer.global_step, result))

        lines = json.dumps(result.table_entries, indent=4).split("\n")
        for line in lines:
            logger.info(line)

        self.log_dict(result.table_entries)

        # save this model if it is the best model
        if self.stats.results_by_step == [] or result.score >= max(r[1].score for r in self.stats.results_by_step):
            logger.info("New best model!")
            self.stats.best_num_steps = trainer.global_step
            path = os.path.join(self.config["location"]["models"], f"{self.stats.to_hash()}.pt")
            trainer.save_checkpoint(path)
        self.predictions = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log('train_loss', outputs['loss'].item())
        if self.task.task_name == 'Scaffold Task For Pretraining':
            self.log('max_doc_length', batch['inputs']['input_ids'].size(1), reduce_fx=torch.max)

    def on_before_optimizer_step(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer: Optimizer, opt_idx: int
    ) -> None:
        self.log('optimizer_learning_rate', optimizer.param_groups[0]['lr'])

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.config["use_dev_as_test_data"]:
            logger.info(f"Final evaluation on {Partition.DEV.value} data:")
            instances = self.task.dev_instances
            partition = Partition.DEV
        else:
            logger.info(f"Final evaluation on {Partition.TEST.value} data:")
            instances = self.task.test_instances
            partition = Partition.TEST

        assert len(self.predictions) == len(instances)

        result = self.task.evaluate(instances, self.predictions, partition)
        self.stats.test_result = result

        lines = json.dumps(result.table_entries, indent=4).split("\n")
        for line in lines:
            logger.info(line)

        if (
            self.config['save_predictions']
            and not (self.task.task_name == 'Scaffold Task For Pretraining')
        ):
            output_path = os.path.join(
                self.config['location']['predictions'],
                f"{self.stats.to_hash()}.jsonl"
            )
            self.task.save_predictions(self.predictions, Path(output_path))

        self.predictions = []


class ScaffoldDataset(torch.utils.data.Dataset):
    all_instances = List[BaseInstance]
    config: omegaconf.DictConfig

    def __init__(
            self,
            s2orc_itg_subset_instances: List[S2ORCITGSubsetInstance],
            task_instances: List[BaseInstance],
            config: omegaconf.DictConfig
    ) -> None:
        super(ScaffoldDataset, self).__init__()
        self.config = config

        # shuffle manually so that prefixes are random
        random.shuffle(task_instances)
        random.shuffle(s2orc_itg_subset_instances)

        # determine the limiting factor (do we need to up-sample task instances or S2ORC instances?)
        ratio = config["scaffold_tasks"]["instances_ratio"]
        should_be_task_instances = (1 - ratio) / ratio * len(s2orc_itg_subset_instances)
        if len(task_instances) < should_be_task_instances:  # we need to up-sample task instances
            logger.info("Up-sample the end task instances.")
            n = math.ceil(should_be_task_instances / len(task_instances))
            task_instances = task_instances * n
        else:  # we need to up-sample S2ORC-ITG-Subset instances
            logger.info("Up-sample the S2ORC-ITG-Subset instances.")
            should_be_s2orc_instances = ratio / (1 - ratio) * len(task_instances)
            n = math.ceil(should_be_s2orc_instances / len(s2orc_itg_subset_instances))
            s2orc_itg_subset_instances = s2orc_itg_subset_instances * n

        logger.info(f"Now there are {len(task_instances)} task instances and "
                    f"{len(s2orc_itg_subset_instances)} S2ORC-ITG-Subset instances.")
        self.all_instances = task_instances + s2orc_itg_subset_instances
        random.shuffle(self.all_instances)

    def __getitem__(self, item) -> BaseInstance:
        return self.all_instances[item]

    def __len__(self) -> int:
        return len(self.all_instances)


def run(
        model_class: Type[BaseModel],
        task_class: Type[BaseTask],
        config: omegaconf.DictConfig
) -> None:
    logger.info(f"Evaluation of '{model_class.model_name}' on '{task_class.task_name}':")
    logger.info(f"Description: '{config['description']}'")
    assert model_class.task_name == task_class.task_name, f"Model and tasks do not match!"

    stats = Statistics(
        model_name=model_class.model_name,
        task_name=task_class.task_name,
        description=config["description"],
        config=config
    )

    start_time = time.time()

    ############################################################################
    # model initialization
    ############################################################################
    logger.info("Initialize the model.")
    tick = time.time()
    if config["load_model"]:
        logger.info("Load existing model.")
        path = os.path.join(config["location"]["models"], f"{config['hash_to_load']}.pt")
        model = model_class.load_from_checkpoint(
            path,
            strict=config['load_strict'],
            config=config,
            stats=stats
        )
    else:
        logger.info("Create new model.")
        model = model_class(config, stats)
    tack = time.time()
    logger.info(f"Initialized the model in {tack - tick:0.4f}s.")

    # Set batch size and gradient accumulation steps depending
    # on model and GPU
    batch_size, gradient_accumulation_steps = model.get_batch_size_and_accumulation_steps(
        config['model']['batch_size'],
        config['model']['accumulate_grad_batches']
    )
    if batch_size != config['model']['batch_size']:
        logger.info(f"Overriding set batch size {config['model']['batch_size']} to new batch size: {batch_size}")
        logger.info(f"Overriding set gradient accumulation steps {config['model']['accumulate_grad_batches']} to new value: {gradient_accumulation_steps}.")
        config['model']['batch_size'] = batch_size
        config['model']['accumulate_grad_batches'] = gradient_accumulation_steps

    # Set config values for bos and eos token if they are not set
    if (
        config['input_sequence']['use_bos_eos_token']
        and hasattr(model, 'tokenizer')
    ):
        if config['input_sequence']['bos_token'] is None:
            config['input_sequence']['bos_token'] = model.tokenizer.bos_token
        if config['input_sequence']['eos_token'] is None:
            config['input_sequence']['eos_token'] = model.tokenizer.eos_token

    ############################################################################
    # task initialization
    ############################################################################
    logger.info("Initialize the task.")
    tick = time.time()
    task = task_class(config, stats)
    tack = time.time()
    logger.info(f"Initialized the task in {tack - tick:0.4f}s.")

    ############################################################################
    # S2ORC-ITG-Subset initialization
    ############################################################################
    if config["do_train"] and config["scaffold_tasks"]["mode"] != "vanilla" and config["scaffold_tasks"]["on_s2orc_itg_subset"]:
        logger.info("Initialize the S2ORC-ITG-Subset.")
        tick = time.time()

        s2orc_itg_subset = S2ORCITGSubsetDataHandler(config, stats)

        tack = time.time()
        logger.info(f"Initialized the S2ORC-ITG-Subset in {tack - tick:0.4f}s.")

    ############################################################################
    # dataloader initialization
    ############################################################################
    logger.info("Initialize the dataloaders.")
    tick = time.time()

    # initialize the dataloaders
    if config["do_train"] and config["scaffold_tasks"]["mode"] != "vanilla" and config["scaffold_tasks"]["on_s2orc_itg_subset"]:
        # noinspection PyUnboundLocalVariable
        train_dataset = ScaffoldDataset(
            s2orc_itg_subset_instances=s2orc_itg_subset.instances,
            task_instances=task.train_instances,
            config=config
        )
    else:
        train_dataset = CustomDataset(task.train_instances)
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=model.training_collate_fn,
        batch_size=config["model"]["batch_size"],
        shuffle=True,
        num_workers=config["dataloader_num_workers"]
    )

    dev_dataset = CustomDataset(task.dev_instances)
    dev_dataloader = DataLoader(
        dev_dataset,
        collate_fn=model.validation_collate_fn,
        batch_size=config["model"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader_num_workers"]
    )

    if config["use_dev_as_test_data"]:
        logger.info("Using the dev instances as test instances.")
        test_dataset = CustomDataset(task.dev_instances)
    else:
        logger.info("USING THE ACTUAL TEST INSTANCES!!!")
        test_dataset = CustomDataset(task.test_instances)
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=model.validation_collate_fn,
        batch_size= config["model"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader_num_workers"]
    )

    tack = time.time()
    logger.info(f"Initialized the dataloaders in {tack - tick:0.4f}s.")

    ############################################################################
    # callback initialization
    ############################################################################
    logger.info("Initialize the callbacks.")
    tick = time.time()

    custom_callback = CustomCallback(task, stats, config)

    tack = time.time()
    logger.info(f"Initialized the callbacks in {tack - tick:0.4f}s.")

    ############################################################################
    # logger initialization
    ############################################################################
    logger.info("Initialize the logger.")
    tick = time.time()

    tensorboard_logger = TensorBoardLogger(
        save_dir=config["location"]["tensorboard"],
        name=f"{stats.task_name}",
        version=f'{stats.model_name} {stats.description} {stats.to_hash()} {config["slurm_job_id"]}'
    )

    tack = time.time()
    logger.info(f"Initialized the logger in {tack - tick:0.4f}s.")

    ############################################################################
    # trainer initialization
    ############################################################################
    logger.info("Initialize the trainer.")
    tick = time.time()

    trainer = pl.Trainer(
        deterministic=config["random"]["deterministic_trainer"],
        accelerator=config["accelerator"],
        accumulate_grad_batches=config["model"]["accumulate_grad_batches"],
        default_root_dir=config["location"]["pytorch_lightning"],
        fast_dev_run=config["fast_dev_run"],
        max_steps=config["model"]["max_steps"],
        min_steps=config["model"]["min_steps"],
        precision=config["precision"],
        check_val_every_n_epoch=None,  # we check after a certain number of steps
        val_check_interval=config["model"]["val_check_interval"],
        callbacks=[custom_callback],
        logger=tensorboard_logger,
        log_every_n_steps=config['log_every_n_steps'],
        devices=1,
        enable_checkpointing=False,  # we checkpoint by hand...
        num_sanity_val_steps=config['num_sanity_val_steps']
    )

    tack = time.time()
    logger.info(f"Initialized the trainer in {tack - tick:0.4f}s.")

    ############################################################################
    # fine-tuning
    ############################################################################
    if config["do_train"]:
        logger.info("Fine-tuning.")
        tick = time.time()

        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=dev_dataloader,
        )

        tack = time.time()
        logger.info(f"Fine-tuning done in {tack - tick:0.4f}s.")

    ############################################################################
    # evaluation
    ############################################################################
    logger.info("Evaluation.")
    tick = time.time()

    if config["do_train"]:
        # load the best model
        path = os.path.join(config["location"]["models"], f"{stats.to_hash()}.pt")
        model = model_class.load_from_checkpoint(path, config=config, stats=stats)

    trainer.test(
        model=model,
        dataloaders=test_dataloader
    )

    tack = time.time()
    logger.info(f"Evaluation done in {tack - tick:0.4f}s.")

    ############################################################################
    # end
    ############################################################################

    end_time = time.time()
    logger.info(f"All done in {end_time - start_time:0.4f}s.")

    stats.total_time = end_time - start_time

    logger.info("Save the statistics.")
    path = os.path.join(config["location"]["results"], f"{stats.task_name} {stats.model_name} {stats.description} {stats.to_hash()}.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(stats.to_json_dict(), file, indent=4)
