import torch
from torch import nn
from transformers import AutoModel
import pytorch_lightning as pl
from args import parse_args
from evaluate import performance_eval

args = parse_args()

# https://pytorch.org/docs/stable/generated/torch.nn.Module.html
class TextReader(nn.Module):
    def __init__(self,
                 args = args):
        super(TextReader, self).__init__()

        # switchable 
        # 다른 모델도 forward 함수만 수정하면 적용 가능
        # https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModel
        self.text_reader = AutoModel.from_pretrained(args.MODEL_NAME)
        self.config = self.text_reader.config


    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.text_reader(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask
                                  )
        # last_hidden_state : [B, max_len, hidden_dim]
        return outputs.last_hidden_state 
    

# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
class BERT_DP(pl.LightningModule):

    def __init__(self,
                 label_vocab,
                 tokenizer,
                 args = args):
        super().__init__()
        self.save_hyperparameters()
        
        # padding 처리에 사용된 [PAD] 토큰은 loss 계산에서 제외
        self.ignore_label_idx = label_vocab['[PAD]']

        # text reader
        self.text_reader = TextReader()

        self.label_vocab = label_vocab
        self.tokenizer = tokenizer

        # to class (model_hidden_size -> output_label_vocab_size)
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=nn%20linear#torch.nn.Linear
        self.to_class = nn.Linear(self.text_reader.config.hidden_size, len(label_vocab))

        # loss
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html?highlight=nn%20crossentropyloss#torch.nn.CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_label_idx)

        self.outputs = []
        
    # sequence labeling의 cross entropy loss 계산
    def cal_loss(self, step_logits, step_labels, ignore_idx):
        # step_logits = [batch, max_len, hidden_dim]
        # step_labels = [batch, max_len]
        B, S, C = step_logits.shape
        
        # target label probability에 대한 계산을 위해, 벡터 사이즈 변경
        predicted = step_logits.view(-1, C)
        reference = step_labels.view(-1)

        # cross entropy 계산
        # probability에 대한 계산을 진행
        loss = self.criterion(predicted, reference.long())
        return loss

    def forward(self, input_ids, token_type_ids, attention_mask):
        # text reader() -> to_class
        last_hidden_state = self.text_reader(input_ids=input_ids.long(),
                                             token_type_ids=token_type_ids.long(),
                                             attention_mask=attention_mask.float()
                                             )
        # [B, max_len, hidden_dim] -> [B, max_len, num_labels]
        step_label_logits = self.to_class(last_hidden_state)
        return step_label_logits

    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#training
    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        pred_step_logits = self(input_ids, token_type_ids, attention_mask)

        # loss calculation
        loss = self.cal_loss(pred_step_logits, labels, ignore_idx=self.ignore_label_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#validation
    def validation_step(self, batch, batch_idx):
        # NOTE : "validation_step" is "RESERVED"
        input_ids, token_type_ids, attention_mask, labels = batch
        pred_step_logits = self(input_ids, token_type_ids, attention_mask)

        # loss calculation
        loss = self.cal_loss(pred_step_logits, labels, ignore_idx=self.ignore_label_idx)
        metrics = {'val_loss': loss}

        self.log('val_loss', loss, prog_bar=True, logger=True)
        return metrics

    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#testing
    def test_step(self, batch, batch_idx):
        # NOTE : "validation_step" is "RESERVED"
        input_ids, token_type_ids, attention_mask, labels = batch
        pred_step_logits = self(input_ids, token_type_ids, attention_mask)

        # loss calculation
        loss = self.cal_loss(pred_step_logits, labels, ignore_idx=self.ignore_label_idx)

        # 예측 라벨 출력
        # [B, max_len, hidden_dim] -> [B, max_len]
        best_sequences =  pred_step_logits.argmax(-1).detach().cpu()
        metrics = {'test_loss': loss,
                   'input_ids': input_ids.detach().cpu(),
                   'ne_output': labels.detach().cpu(),
                   'best_seq': best_sequences}
        self.outputs.append(metrics)
        self.log('test_loss', loss)
        return metrics

    def on_test_epoch_end(self):
        # outputs: 각 batch의 결과들을 모두 단순 concat한 형태
        # 평가를 위한 포맷팅이 필요
        input_ids = []
        ne_output = []
        best_seq = []
        label_vocab = self.label_vocab
        tokenizer = self.tokenizer
        
        for output in self.outputs:
            for node in output['input_ids'].tolist():
                input_ids.append(node)
            for node in output['ne_output'].tolist():
                ne_output.append(node)
            for node in output['best_seq']:
                best_seq.append(node)
        total_best_seq = best_seq
        total_ne_output = ne_output
        total_input_ids = input_ids

        performance_eval(label_vocab, tokenizer, total_input_ids, total_ne_output, total_best_seq)

    # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#prediction-loop
    def predict_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, labels = batch
        pred_step_logits = self(input_ids, token_type_ids, attention_mask)
        return pred_step_logits.argmax(-1).detach().cpu()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.learning_rate)
        return optimizer
