import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler,
                              SequentialSampler, TensorDataset)
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


#simcse_tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-large')#
#roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
#bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased/')
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

condition_token = ['<s1>', '<s2>', '<s3>', '<mask>'] # Maximum 3 speakers
special_tokens = {'additional_special_tokens': condition_token}
roberta_tokenizer.add_special_tokens(special_tokens)
#simcse_tokenizer.add_special_tokens(special_tokens)

def encode_right_truncated(text, tokenizer, max_length=511):
    tokenized = tokenizer.tokenize(text)
    truncated = tokenized[-max_length:]    
    ids = tokenizer.convert_tokens_to_ids(truncated)
    
    return [tokenizer.cls_token_id] + ids

def padding(ids_list, tokenizer, max_len=0):
    #if max_len ==0:
    for ids in ids_list:
        if len(ids) > max_len:
            max_len = len(ids)
        
    pad_ids = []
    for ids in ids_list:
        pad_len = max_len-len(ids)
        add_ids = [tokenizer.pad_token_id for _ in range(pad_len)]
        
        pad_ids.append(ids+add_ids)
    
    return torch.tensor(pad_ids), max_len




def make_batch_roberta(sessions):
    batch_input, batch_labels, batch_multi_labels, batch_multi_bool, batch_multi_labels_pseudo = [], [], [], [], []
    batch_speaker_tokens = []
    for session in sessions:
        data = session[0]
        label_list = session[1]
        label_dict = {index: label for index, label in enumerate(label_list)}
        pseudo_label = session[2]
        
        context_speaker, context, emotion, multi_emotion = data
        now_speaker = context_speaker[-1]
        speaker_utt_list = []
        
        conversation_len = len(context_speaker)
        
        inputString = ""
        for turn, (speaker, utt) in enumerate(zip(context_speaker, context)):
            inputString += '<s' + str(speaker+1) + '> ' # s1, s2, s3...
            inputString += utt + " "
            
            if turn<len(context_speaker)-1 and speaker == now_speaker:
                speaker_utt_list.append(encode_right_truncated(utt, roberta_tokenizer))
        
        concat_string = inputString.strip()
        batch_input.append(encode_right_truncated(concat_string, roberta_tokenizer))

        label_ind = label_list.index(emotion)
        batch_labels.append(label_ind)

        multi_ind = [0.0]*len(label_list)

        if len(multi_emotion) > 1:  # have multi-label 
            multi_ind[label_list.index(multi_emotion[0])] = 1.0
            multi_ind[label_list.index(multi_emotion[1])] = 1.0
            batch_multi_bool.append(True)
            multi__ = (label_list.index(multi_emotion[0]),label_list.index(multi_emotion[1]))
        else: 
            multi__ = (label_list.index(multi_emotion[0]),)
            if pseudo_label: #pseudo labeling
                multi_ind = pseudo_label
                batch_multi_bool.append(False)
                
            else:
                multi_ind[label_list.index(multi_emotion[0])] = 1.0
                batch_multi_bool.append(False)
              
        batch_multi_labels.append(multi_ind)
        batch_speaker_tokens.append(padding(speaker_utt_list, roberta_tokenizer))
    
    batch_input_tokens, max_len = padding(batch_input, roberta_tokenizer)
    batch_labels = torch.tensor(batch_labels)  
    batch_multi_labels = torch.tensor(batch_multi_labels, dtype=torch.float)
    batch_multi_bool = torch.tensor(batch_multi_bool)
    #print("batch_input_tokens, batch_labels, batch_multi_labels", batch_input_tokens.size(), batch_labels.size(), batch_multi_labels.size())
    return batch_input_tokens, batch_labels, batch_multi_labels, batch_multi_bool,label_dict

