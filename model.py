import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd
import pdb

from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoModel, AutoTokenizer

from sklearn.utils import Bunch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from weighted_supcon import MuliWeightedSupConLoss 
from utils import *
from transformers import RobertaTokenizer, RobertaModel

def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(dim=-1, keepdim=True)

def CELoss(pred_outs, labels):
    """
        pred_outs: [batch, clsNum]
        labels: [batch]
    """
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    loss_val = ce_loss(pred_outs, labels)
    return loss_val

class ERC_model(nn.Module):
    def __init__(self, model_type, clsNum, args):
        super(ERC_model, self).__init__()
        self.gpu = True
        self.args = args
        """Model Setting"""        
        model_path = model_type 
        
        
        self.model = RobertaModel.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            
        condition_token = ['<s1>', '<s2>', '<s3>'] # maximum 3 speakers
        special_tokens = {'additional_special_tokens': condition_token}
        self.tokenizer.add_special_tokens(special_tokens)
            
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.num_classes = clsNum
        self.hiddenDim = self.model.config.hidden_size
        self.pad_value = self.tokenizer.pad_token_id
        self.alpha = nn.Parameter(torch.tensor([self.args.alpha]))
        self.beta = nn.Parameter(torch.tensor([self.args.beta]))
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.supcon = SupconLoss(clsNum, self.args.temp, args)

        if self.args.loss in ['weighted', 'cew']:  
            self.multi_supcon = MuliWeightedSupConLoss(clsNum, self.args.temp, args)
        
            
        """score"""
        self.W = nn.Linear(self.hiddenDim, clsNum)
    def device(self):
        return self.model.device
    
    def hidden(self):
        return self.hiddenDim
        
    def forward(self, batch_input_tokens, batch_labels, batch_multi_labels, label_dict):
        """
            batch_input_tokens: (batch, len)
        """

        batch_size, max_len = batch_input_tokens.shape[0], batch_input_tokens.shape[-1]
        mask = 1 - (batch_input_tokens == (self.pad_value)).long()
        
        if self.args.pretrained != 'simcse':
            output = self.model(
                input_ids = batch_input_tokens, 
                attention_mask = mask,
                output_hidden_states = True,
                )

            batch_context_output = output.last_hidden_state[:,0,:]
        else:

            output = self.model(
                input_ids = batch_input_tokens, 
                attention_mask = mask,
                output_hidden_states = True,
                return_dict=True
            )['last_hidden_state']

            mask_pos = (batch_input_tokens == (self.mask_value)).long().max(1)[1]
            mask_outputs = output[torch.arange(mask_pos.shape[0]), mask_pos, :]
            # feature = torch.dropout(mask_outputs, 0.1, train=self.training)
            batch_context_output = mask_outputs


        context_logit = self.W(batch_context_output) # (batch, clsNum)


        # Calculate Entropy
        swapped_dict = {v: k for k, v in label_dict.items()}
        neu_value = swapped_dict['neutral']
        prob_dist = softmax(context_logit)
        entropy = -torch.sum(prob_dist * torch.log(prob_dist), dim=-1) #(batch_size,)
        max_entropy = torch.log(torch.tensor(self.num_classes, dtype=torch.float)).to(batch_input_tokens.device)
        entropy_weights = 1 - (entropy / max_entropy)  
        entropy_weights = (entropy_weights / entropy_weights.max()).to(batch_input_tokens.device).unsqueeze(1) # 0과 1 사이로 정규화
        
    
        for i in range(batch_size):
            if batch_labels[i] == neu_value:
                entropy_weights[i] = 1
        
        
        # Calculate Loss
        if self.args.test == False:    
            
            # Calculate Multi-label Emotion Contrastive learning loss
            multi_loss = self.loss_fn(context_logit,batch_multi_labels)
            weightedCL = self.multi_supcon(batch_context_output,batch_labels,entropy_weights, batch_multi_labels, label_dict)

            loss = multi_loss + self.alpha*weightedCL

        else:
            loss = None

        return context_logit, loss, batch_context_output, entropy_weights
