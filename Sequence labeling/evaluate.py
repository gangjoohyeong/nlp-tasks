import codecs

def performance_eval(label_vocab, tokenizer, input_ids, labels, total_best_seq):

    # 라벨 decoding을 위한 {int: label} 사전
    dict_id2label = {idx: label for idx, label in enumerate(label_vocab)}

    sent_list = []
    for bs_idx, bs in enumerate(input_ids):
        # 문장 별 분리
        text = input_ids[bs_idx]
        reference = labels[bs_idx]
        prediction = total_best_seq[bs_idx]
        a_sent = []
        for char_idx, char in enumerate(text):
            # 문장 속 char별 분리
            text_char = tokenizer.convert_ids_to_tokens(text[char_idx])
            
            # special token에 따른 처리
            # 시작토큰일 경우
            if text_char in ('[CLS]', '<s>'):
                continue
            # 모델이 모르는 토큰일 경우
            elif text_char in ('[UNK]', '<unk>'):
                text_char = ' '
            # 종료 토큰일 경우
            elif text_char in ('[SEP]', '</s>'):
                break
            # 라벨 디코딩
            refer_char = dict_id2label[int(reference[char_idx])]
            pred_char  = dict_id2label[int(prediction[char_idx])]
            
            # BIO tag를 뗀 라벨을 추출
            label = 'O' if refer_char == 'O' else refer_char[2:]
            a_sent.append((text_char, label, refer_char, pred_char))

        sent_list.append(a_sent)

    # Conlleval-2000 포맷으로 저장
    n2n_result_fn = './output/n2n_output.txt'
    with codecs.open(n2n_result_fn, 'w', encoding='utf-8') as f:
        for sent_idx, a_sent in enumerate(sent_list):
            for text_char, label, refer_char, pred_char in a_sent:
                print("{}\t{}\t{}\t{}\t".format(text_char, label, refer_char, pred_char), file=f)
            print("\n", file=f)
            
    print("\n Readable N2N result format is dumped at {}".format(n2n_result_fn))
    print("\n{}\n".format('=' * 50))