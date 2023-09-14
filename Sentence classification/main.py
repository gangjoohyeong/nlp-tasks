import torch
import lightning.pytorch as pl
from dataset import YNATDataModule
from model import YNATModel

def run():
    klue_datamodule = YNATDataModule(model_name='klue/bert-base')
    klue_model = YNATModel(model_name='klue/bert-base')

    klue_trainer = pl.Trainer(
        max_epochs=1
    )

    klue_trainer.fit(klue_model, datamodule=klue_datamodule)
    klue_trainer.test(klue_model, datamodule=klue_datamodule)

if __name__ == '__main__':
    run()