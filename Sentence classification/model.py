from transformers import BertForSequenceClassification, AdamW
import lightning.pytorch as pl
from torchmetrics import Accuracy

class YNATModel(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        
        self.model_name = model_name
        self.text_reader = BertForSequenceClassification.from_pretrained(model_name, num_labels=7)
        self.optimizer = AdamW(self.parameters(), lr=5e-5)
        self.accuracy = Accuracy(task='multiclass', num_classes=7)
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        outputs = self.text_reader(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        return outputs
    
    def configure_optimizers(self):
        
        return self.optimizer
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids, labels)
        self.log('train/loss', outputs.loss)
        
        return outputs.loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids, labels)
        self.log('valid/loss', outputs.loss)
        
        return outputs.loss
    
    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch
        outputs = self(input_ids, attention_mask, token_type_ids, labels)
        preds = outputs.logits.argmax(1)
        accuracy = self.accuracy(preds, labels)
        self.log('test/acc', accuracy, on_epoch=True)
            
        return accuracy