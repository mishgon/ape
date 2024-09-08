from omegaconf import DictConfig
import hydra

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))


@hydra.main(version_base=None, config_path='../configs', config_name='train')
def main(config: DictConfig):
    # 1. Model
    model: pl.LightningModule = hydra.utils.instantiate(config.model)

    # 2. Datamodule
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule.prepare_data()
    datamodule.setup('fit')

    # 3. Callbacks
    callbacks = [
        ModelCheckpoint(
            filename='best',
            monitor='ape/loss',
            save_last=True,
            mode='min',
        ),
        LearningRateMonitor()
    ]

    # 4. Trainer
    trainer: pl.Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks)

    # 5. Run training
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.ckpt_path
    )


if __name__ == '__main__':
    main()
