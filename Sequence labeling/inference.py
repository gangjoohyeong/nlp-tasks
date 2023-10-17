from transformers import AutoTokenizer 
import pytorch_lightning as pl

from args import parse_args
from dataset import load_data
from dataset import DPDataModule
from model import BERT_DP
import pandas as pd

def run(args):
    
    ## Load Data
    train_df, train_label, valid_df, test_df, test_label, labels, label_vocab = load_data('./data')
    
    ## Tokenizer
    # https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.MODEL_NAME)
    
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
    
    
    model.eval()

    predictions , true_labels = [], []
    inference_sents = [args.text]
    
    dm = DPDataModule(tokenizer=tokenizer, batch_size=len(inference_sents), max_len=200, inference_sents=inference_sents)
    outputs = trainer.predict(model, dataloaders=dm.inference_dataloader())

    for output in outputs:
        predictions.extend([list(p) for p in output])
        pred_labels = [list(label_vocab.keys())[p_i] for p in predictions for p_i in p]
        pred_labels = pred_labels[1:len(args.text)+1]
    
    result_data = {
                'char' : list(args.text),
                'pred_labels': pred_labels,
    }    
    
    print(pd.DataFrame(result_data))
    
if __name__ == "__main__":
    args = parse_args()
    run(args)