import torch
import lightning.pytorch as pl
from dataset import YNATDataModule
from model import YNATModel

# klue_tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
# kykim_tokenizer = AutoTokenizer.from_pretrained('kykim/bert-kor-base')
# snunlp_tokenizer = AutoTokenizer.from_pretrained('snunlp/KR-BERT-char16424')

# dataset = load_dataset('klue', 'ynat')

# train_test_dataset = dataset['train']
# valid_dataset = dataset['validation']

# train_test = train_test_dataset.train_test_split(train_size=0.9, shuffle=False)
# train_dataset, test_dataset = train_test['train'], train_test['test']

# klue = BertForSequenceClassification.from_pretrained('klue/bert-base', num_labels=7)
# kykim = BertForSequenceClassification.from_pretrained('kykim/bert-kor-base', num_labels=7)
# snunlp = BertForSequenceClassification.from_pretrained('snunlp/KR-BERT-char16424', num_labels=7)


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