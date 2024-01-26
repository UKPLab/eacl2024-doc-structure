"""
Run the evaluation environment that fine-tunes and evaluates a model on an end task.
"""
import logging

import hydra
import omegaconf
import pytorch_lightning as pl

from evaluation.baselines import OracleForEvidenceInferenceModel, OracleForQASPERModel
from evaluation.run import run
from evaluation.tasks.evidence_inference import EvidenceInferenceTask
from evaluation.tasks.qasper import QASPERTask
from evaluation.tasks.scaffold_task_for_pretraining import ScaffoldTaskForPretraining
from structformer.led import LEDForQASPERModel, LEDForEvidenceInferenceModel, \
    LEDForScaffoldTaskForPretrainingModel
from structformer.longt5 import (
    LongT5ForQASPERModel,
    LongT5ForEvidenceInferenceModel,
    LongT5ForScaffoldTaskForPretrainingModel
)

logger = logging.getLogger()

TASK_CLASSES = [
    EvidenceInferenceTask,
    QASPERTask,
    ScaffoldTaskForPretraining,
]


MODEL_CLASSES = [
    OracleForEvidenceInferenceModel,
    OracleForQASPERModel,


    LEDForQASPERModel,
    LEDForEvidenceInferenceModel,
    LEDForScaffoldTaskForPretrainingModel,

    LongT5ForQASPERModel,
    LongT5ForEvidenceInferenceModel,
    LongT5ForScaffoldTaskForPretrainingModel,
]


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: omegaconf.DictConfig) -> None:

    if config['remote_debug']:
        import pydevd_pycharm
        pydevd_pycharm.settrace('10.167.11.14', port=3851, stdoutToServer=True, stderrToServer=True)
        config['dataloader_num_workers'] = 0

    logger.info("Run training.")

    logger.info("Fix random seeds.")
    pl.seed_everything(config["random"]["seed"])



    # select task
    task_class = None
    for t in TASK_CLASSES:
        if t.task_name == config["task"]["task_name"]:
            task_class = t
            break

    if task_class is None:
        logger.error("Did not find a suitable task!")
        assert False, "Did not find a suitable task!"

    # select model
    model_class = None
    for mw in MODEL_CLASSES:
        if mw.model_name == config["model"]["model_name"] and mw.task_name == config["task"]["task_name"]:
            model_class = mw
            break

    if model_class is None:
        logger.error("Did not find a suitable model!")
        assert False, "Did not find a suitable model!"

    run(
        model_class=model_class,
        task_class=task_class,
        config=config
    )

    logger.info(f"All done!")


if __name__ == "__main__":
    import sys

    import absl.flags as flags

    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    main()
