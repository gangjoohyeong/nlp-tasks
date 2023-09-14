from transformers import AutoTokenizer
from datasets import load_dataset
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset


class YNATDataset(Dataset):
    def __init__(self, input_data, labels):
        self.input_data = input_data
        self.labels = labels
    
    def __len__(self):
        return len(self.input_data['input_ids'])
    
    def __getitem__(self, idx):
        input_ids = self.input_data['input_ids'][idx]
        attention_mask = self.input_data['attention_mask'][idx]
        token_type_ids = self.input_data['token_type_ids'][idx]
        
        return input_ids, attention_mask, token_type_ids, self.labels[idx]
    
class YNATDataModule(pl.LightningDataModule):
    def __init__(self, model_name, max_len=128, truncation=True):
        super().__init__()
        self.max_len = max_len
        self.truncation = truncation
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        
        self.train_labels = None
        self.valid_labels = None
        self.test_labels = None
        
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        klue_df = load_dataset('klue', 'ynat')
        
        train_test_df = klue_df['train']
        valid_df = klue_df['validation']
        
        train_test = train_test_df.train_test_split(train_size=0.9, shuffle=False)
        train_df, test_df = train_test['train'], train_test['test']
        
        self.train_df = self.tokenizing(train_df)
        self.valid_df = self.tokenizing(valid_df)
        self.test_df = self.tokenizing(test_df)
        
        self.train_labels = train_df['label']
        self.valid_labels = valid_df['label']
        self.test_labels = test_df['label']
        
    def tokenizing(self, df):
        data = self.tokenizer(
            df['title'],
            padding='max_length',
            truncation=self.truncation,
            return_tensors='pt',
            max_length=self.max_len
        )
        
        return data
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_dataset = YNATDataset(self.train_df, self.train_labels)
            self.valid_dataset = YNATDataset(self.valid_df, self.valid_labels)
        else:
            self.test_dataset = YNATDataset(self.test_df, self.test_labels)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32, shuffle=False, num_workers=8)