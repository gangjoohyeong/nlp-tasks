import os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pytorch_lightning as pl



# KLUE DP 데이터셋에 맞게 character sequence, BIO label sequence를 생성
# 데이터 출처: https://github.com/KLUE-benchmark/KLUE/tree/main
def load_tsv(fn):
    sents = []
    labels = []
    with open(fn, 'r', encoding='utf-8') as f:
        sent = []
        label_sequence = []
        for line in f:
          # 주석인 경우 반영하지 않음
          if line[0:2]=='##':
              continue
          # 문장이 끝나면 데이터에 저장
          if line == '\n':
              sents.append({
                  'sentence': sent[:-1],
                  'labels': label_sequence[:-1]
              })
              sent = []
              label_sequence = []
          # 어절에 대해 음절 단위 분해, label sequence 생성
          else:
              index, word, lemma, pos, head, dp_tag = line.rstrip().split('\t')
              for i, c in enumerate(word):
                sent.append(c)
                # BIO 태그 부착
                if i==0:
                    label_sequence.append(f'B-{dp_tag}')
                else:
                    label_sequence.append(f'I-{dp_tag}')

              # 어절과 어절 사이엔 공백을 넣어줌
              sent.append(' ')
              label_sequence.append('O')

              labels.extend([f'B-{dp_tag}', f'I-{dp_tag}', 'O'])
    # 라벨 사전을 위한 데이터
    labels = set(labels)

    return pd.DataFrame.from_dict(sents), labels

def load_data(data_dir):
    train_df, train_label = load_tsv(os.path.join(data_dir, 'klue-dp-v1.1_train.tsv'))
    valid_df, _ = load_tsv(os.path.join(data_dir, 'klue-dp-v1.1_dev.tsv'))
    test_df, test_label = load_tsv(os.path.join(data_dir, 'klue-dp-v1.1_dev.tsv'))

    # 라벨 사전
    labels = train_label.union(test_label)
    labels = ['[PAD]']+list(labels) ## !!! 
    label_vocab = {label: id for id, label in enumerate(labels)}

    return train_df, train_label, valid_df, test_df, test_label, labels, label_vocab

def get_char_label_sequence(tokenizer, label_vocab, sents, max_len=512, truncate=True):
    data = []

    # 문장별 text, label sequence를 모델의 입력 가능한 숫자 데이터로 변형
    for idx, a_sent in tqdm(sents.iterrows()):
        chars = []
        labels = []

        # 주요한 내용을 저장
        for char, label in zip(a_sent['sentence'], a_sent['labels']):
            char_idx = tokenizer.convert_tokens_to_ids(char)
            label_idx = label_vocab[label]

            chars.append(char_idx)
            labels.append(label_idx)

        assert len(chars) == len(labels), "INVALID DATA LENGTH: N2N must get SAME INPUT and OUTPUT LENGTH"

        # max_len보다 길다면, truncate 시 뒷부분을 편집
        # 앞뒤로 [CLS] ~~ [SEP] 두 토큰이 위치해야 하므로, 감안하여 편집
        if truncate == True:
            chars = chars[:max_len-2]
            labels = labels[:max_len-2]

        # padding 처리
        # 모든 데이터의 text, label, attention_mask, token_type_ids에 대해 max_len 만큼 통일
        chars  = [tokenizer.cls_token_id]         + chars  + [tokenizer.sep_token_id]
        labels = [label_vocab['[PAD]']] + labels + [label_vocab['[PAD]']]

        attention_mask = [1] * len(chars)

        N = max_len - len(chars)
        chars = chars + [tokenizer.pad_token_id] * N
        token_type_ids = [0] * len(chars)
        attention_mask = attention_mask + [0] * N  # 1 for valid, 0 for pad

        labels = labels + [label_vocab['[PAD]']] * N

        # ((입력), 출력)
        # 입력과 출력 데이터를 각각 묶어서 저장
        data.append(((chars, token_type_ids, attention_mask), labels))
    return data

# 정답 label sequence 없이 데이터를 생성
def get_char_sequence_for_eval(tokenizer, sents:list, max_len=512, truncate=True):
    data = []
    for a_sent in tqdm(sents):
        # 인코딩된 text sequence 생성
        chars = []
        for char in a_sent:
            char_idx = tokenizer.convert_tokens_to_ids(char)
            chars.append(char_idx)
        # max_len보다 길면 편집
        if truncate == True:
            chars = chars[:max_len-2]

        # padding 처리
        chars = [tokenizer.cls_token_id] + chars + [tokenizer.sep_token_id]

        attention_mask = [1] * len(chars)

        N = max_len - len(chars)
        chars = chars + [tokenizer.pad_token_id] * N
        token_type_ids = [0] * len(chars)
        attention_mask = attention_mask + [0] * N

        # 가짜 라벨을 생성해서 함께 저장
        label = [0 for i in range(len(chars))]

        data.append(((chars, token_type_ids, attention_mask), label))
    return data



# Dataset 
# https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
class DPDataSet(Dataset):
    """DPDataSet dataset."""

    def __init__(self, 
                 raw_data,
                 tokenizer=None,
                 label_vocab=None,
                 max_len=512,
                 truncate=True,
                 is_inference=False,
                 ):
      
        if is_inference:
            self.data = get_char_sequence_for_eval(tokenizer, raw_data, max_len=max_len)
        else:
            self.data = get_char_label_sequence(tokenizer, label_vocab, raw_data, max_len=max_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 각 데이터를 하나씩 불러옴
        # batch를 만들 때 사용됨
        input, label = self.data[idx]

        chars, token_type_ids, attention_mask = input 

        # torch.Tensor와 호환 가능한 np.array 형태로 변환
        input_ids      = np.array(chars)
        token_type_ids = np.array(token_type_ids)
        attention_mask = np.array(attention_mask)

        label = np.array(label)

        # 각 데이터는 입출력의 필수요소를 모두 묶어서 들고 감
        item = [ input_ids, token_type_ids, attention_mask, label ]
        return item

# Data Module
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
class DPDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int = 32,
                 max_len: int = 512,
                 inference_sents: list = None,
                 train_df: pd.DataFrame = None,
                 valid_df: pd.DataFrame = None,
                 test_df: pd.DataFrame = None,
                 tokenizer = None,
                 label_vocab = None,
                 ):
        super().__init__()

        

        # batch size를 저장
        self.batch_size = batch_size

        # load label
        # 데이터에 대해 Dataset으로 만들어서 가져옴
                 
        if inference_sents is not None:
          self.inference_dataset = DPDataSet(raw_data = inference_sents, tokenizer = tokenizer, label_vocab = label_vocab, max_len=max_len, is_inference=True)
        else:
          self.train_dataset = DPDataSet(raw_data = train_df, tokenizer = tokenizer, label_vocab = label_vocab, max_len=max_len)
          self.valid_dataset = DPDataSet(raw_data = valid_df, tokenizer = tokenizer, label_vocab = label_vocab, max_len=max_len)
          self.test_dataset  = DPDataSet(raw_data = test_df, tokenizer = tokenizer, label_vocab = label_vocab, max_len=max_len)

    # 훈련 데이터
    # shuffle 옵션: 일정한 데이터가 한 batch에 뭉치지 않도록 섞어줌
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    # 검증 데이터
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    # 평가 데이터
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    # 예측 데이터
    def inference_dataloader(self):
        return DataLoader(self.inference_dataset, batch_size=self.batch_size)
