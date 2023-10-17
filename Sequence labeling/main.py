
from transformers import AutoTokenizer 
import pytorch_lightning as pl

from args import parse_args
from dataset import load_data
from dataset import DPDataModule
from model import BERT_DP


def run(args):
    
    ## Load Data
    train_df, train_label, valid_df, test_df, test_label, labels, label_vocab = load_data('./data')


    ## Tokenizer
    # https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME)


    ## Data Module
    dm = DPDataModule(train_df = train_df, valid_df = valid_df, test_df = test_df, tokenizer = tokenizer, label_vocab = label_vocab, batch_size=args.batch_size, max_len=args.max_len)

    
    ## Model
    pl.seed_everything(args.random_initiallization)
    model = BERT_DP(label_vocab = label_vocab, tokenizer = tokenizer)
    
    # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss', dirpath=args.output_path, filename='DP_model')
    
    # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        devices=1
    )
    
    
    ## Train
    trainer.fit(model, datamodule=dm)
    
    # 현재 저장된 모델 파일 폴더를 찾을 수 있음
    model_path = checkpoint_callback.best_model_path
    
    
    ## Evaluation
    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=load_from_checkpoint#load-from-checkpoint
    model = BERT_DP.load_from_checkpoint(model_path)

    result = trainer.test(model, dataloaders=dm.test_dataloader())
    print(result)
    
    
if __name__ == "__main__":
    args = parse_args()
    run(args)