from datetime import datetime
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from smart.utils.config import load_config_act
from smart.datamodules import MultiDataModule
from smart.model import SMART
from smart.utils.log import Logging

torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    parser = ArgumentParser()
    Predictor_hash = {"smart": SMART, }
    parser.add_argument('--config', type=str, default='configs/train/train_scalable_custom.yaml')
    parser.add_argument('--pretrain_ckpt', type=str, default="")
    parser.add_argument('--ckpt_path', type=str, default="")
    parser.add_argument('--save_ckpt_path', type=str, default=f"ckpt/{datetime.now().strftime('%Y%m%d_%H%M')}")
    args = parser.parse_args()

    config = load_config_act(args.config)

    Predictor = Predictor_hash[config.Model.predictor]
    strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True)
    Data_config = config.Dataset
    datamodule = MultiDataModule(**vars(Data_config))

    if args.pretrain_ckpt == "":
        model = Predictor(config.Model)
    else:
        logger = Logging().log(level='DEBUG')
        model = Predictor(config.Model)
        model.load_params_from_file(filename=args.pretrain_ckpt,
                                    logger=logger)
    trainer_config = config.Trainer
    model_checkpoint = ModelCheckpoint(dirpath=args.save_ckpt_path,
                                       filename="{epoch:02d}-{step}-{val_loss:.2f}",
                                       monitor='val_cls_acc',
                                       every_n_epochs=1,
                                       save_top_k=5,
                                       mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=trainer_config.accelerator, 
                         devices=trainer_config.devices,
                         max_epochs=trainer_config.max_epochs,
                         num_nodes=trainer_config.num_nodes,
                         precision=trainer_config.precision,
                         accumulate_grad_batches=trainer_config.accumulate_grad_batches,
                         num_sanity_val_steps=trainer_config.num_sanity_val_steps,
                         strategy=strategy,
                         callbacks=[model_checkpoint, lr_monitor],
                        #  limit_train_batches=0.0001,
                        #  limit_val_batches=0.0001,
                         gradient_clip_val=0.5)
    if args.ckpt_path == "":
        trainer.fit(model,
                    datamodule)
    else:
        trainer.fit(model,
                    datamodule,
                    ckpt_path=args.ckpt_path)
