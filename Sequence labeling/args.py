import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--max_len", default=512, type=int, help="max sequence length")

    parser.add_argument("--MODEL_NAME", default="bert-base-multilingual-cased", type=str, help="MODEL NAME")
    
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--random_initiallization", default=1234, type=float, help="random initiallization")
    parser.add_argument("--max_epochs", default=1, type=int, help="max epochs")
    
    parser.add_argument("--output_path", default='./output/', type=str, help="output path")

    
    # inference
    parser.add_argument("--text", default='아이는 예정보다 일찍 태어나', type=str, help="inference text")
    
    
    args = parser.parse_args()

    return args