import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import pickle
import random
import time
import timeit
import warnings


def softmax(x):
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum(dim=-1, keepdim=True)


def cos_emotion3d(v1, v2):
    """
    Calculate the cosine similarity between two 3D vectors.
    """
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim

def cal_label_similar():
    # Calculate the similarity for all pairs of emotions

    # We Combine emotion wheel of Yang(2022)[1] and emotion representations in VAD space of Yang(2023)[2]
    # [1] Hybrid Curriculum Learning for Emotion Recognition in Conversation
    # [2] Cluster-Level Contrastive Learning for Emotion Recognition in Conversations

    
    emotions_2d = {
        'neutral': np.array([0.0,0.0]),
        'anger': np.array([-0.50,0.90]),
        'frustrated': np.array([-0.90, -0.20]),
        'happy': np.array([0.90, 0.10]),
        'excited': np.array([0.70, 0.80]),
        'sad': np.array([-0.70, -0.70]),
        'fear': np.array([-0.90, 0.30]),  
        'disgust': np.array([-0.80, 0.40]),
        'peaceful': np.array([0.90, -0.10]),
        'powerful': np.array([0.80, 0.60]),
        'surprise': np.array([0.10, 0.90])

    }

    similarity_matrix = {}

    for emotion1, vector1 in emotions_2d.items():
        for emotion2, vector2 in emotions_2d.items():
            if emotion1 != emotion2:
                pair = (emotion1, emotion2)
                if (emotion1 == 'neutral') or (emotion2 == 'neutral'):
                    similarity_matrix[pair] = 0.5
                else:
                    # Ensure that each pair is calculated only once
                    cos_similarity = cos_emotion3d(vector1, vector2)
                    normalized_similarity = (1 + cos_similarity) / 2  # -1~1 범위를 0~1로 조정
                    similarity_matrix[pair] = normalized_similarity

    
    return similarity_matrix

def build_batch_label_similarity_matrix(multi_label_tuple, label_similarity_matrix):
    """
    Build a similarity matrix for the labels within a batch based on the precomputed label similarity matrix.

    @param multi_label_tuple: containing the multi-labels with real label name ex) (fear, disgust) / (happy,)
    @param label_similarity_matrix: Precomputed matrix containing similarity scores between all pairs of labels

    @return: A batch label similarity matrix of shape (batch_size,)
    """
    #batch_size = multi_label_tuple.size(0)
    batch_label_similarity_matrix = []

    for idx, i in enumerate(multi_label_tuple):
        if len(i) <= 1:
            batch_label_similarity_matrix.append(0)
        elif len(i) > 1:
            batch_label_similarity_matrix.append(label_similarity_matrix.get(i))
    
    return torch.tensor(batch_label_similarity_matrix, dtype=torch.float)


class MuliWeightedSupConLoss(nn.Module):
    def __init__(self, num_classes, temp, args):
        super().__init__()
        self.temperature = temp
        self.num_classes = num_classes
        self.eps = 1e-8
        self.args = args
        self.label_similar_matrix = cal_label_similar()
        if self.args.multi_weighted:
            print("Multi weighted contrastive learning")
        if self.args.entropy:
            print("Entropy weighted")
        
    def score_func(self, x, y):
        return (1+F.cosine_similarity(x, y, dim=-1))/2 + self.eps
    
    def forward(self, reps, labels, entropy_weights, multi_labels, label_dict):
        '''
            @param reps: batch_size x ebd_dim
            @param labels: batch_size
            @param prob: batch_size x the number of classes

            @return loss: weighted contrastive learning loss

            This code is built upon the foundational work done by Yang et al. (https://github.com/caskcsg/SPCL/blob/master/spcl_loss.py). 
            We have adapted and extended their code to fit the specific needs of our work. 

            We express our gratitude to Yang et al. for their original contributions, which made this project possible. 
            For further details on the original work, please refer to the (https://github.com/caskcsg/SPCL/blob/master/spcl_loss.py).
        ''' 
        batch_size = reps.shape[0]

        swapped_dict = {v: k for k, v in label_dict.items()}
        neu_value = swapped_dict['neutral']
        
        mask1 = labels.unsqueeze(0).expand(labels.shape[0], labels.shape[0])
        mask2 = labels.unsqueeze(1).expand(labels.shape[0], labels.shape[0])
        
        mask = 1 - torch.eye(batch_size).to(reps.device)
        pos_mask = (mask1 == mask2).long()

        rep1 = reps.unsqueeze(0).expand(batch_size, batch_size, reps.shape[-1])
        rep2 = reps.unsqueeze(1).expand(batch_size, batch_size, reps.shape[-1])
        
        scores = self.score_func(rep1, rep2)
        scores *= 1 - torch.eye(batch_size).to(reps.device)
        scores /= self.temperature
        scores -= torch.max(scores).item()
        scores = torch.exp(scores)

        # multi-label supervised contrastive learning

        # Construct label similarity matrix based on the batch labels
        if self.args.multi_weighted:
            multi_label_tuple = []
            for row in multi_labels:
                temp = row.nonzero(as_tuple=True)[0].tolist()
                a = []
                for t in temp:
                    a.append(label_dict.get(t))
                multi_label_tuple.append(tuple(a))

            multi_label_similarity_matrix = build_batch_label_similarity_matrix(multi_label_tuple,self.label_similar_matrix ).to(reps.device)
            
            # Define A(i) and M(i) based on labels and multi_labels
            missing_label = multi_labels.clone()
            
            for idx, i in enumerate(labels):
                missing_label[idx][i] -= 1 # except for single label from multi-label set

            transformed_missing = torch.tensor([row.nonzero()[0].item() if row.nonzero().numel() > 0 else -1 for row in missing_label]).to(reps.device)
            missing = transformed_missing.unsqueeze(1).expand(transformed_missing.shape[0], transformed_missing.shape[0])
            missing_mask = (mask1 == missing).long()
            negative_mask = 1 - (pos_mask | missing_mask).long()

            negative_a_score = scores * negative_mask
            negative_m_score = scores * missing_mask
            
            if self.args.entropy:
                neg_scores = (negative_a_score + (1 - multi_label_similarity_matrix) * negative_m_score) * entropy_weights
                pos_scores = scores * (pos_mask * mask) * entropy_weights
            else:
                neg_scores = negative_a_score + (1 - multi_label_similarity_matrix) * negative_m_score
                pos_scores = scores * (pos_mask * mask) 
        elif self.args.entropy: 

            #Supervised-Contrastive learning
            pos_scores = scores * (pos_mask * mask) * entropy_weights
            neg_scores = scores * (1 - pos_mask) * entropy_weights

        else:
            pos_scores = scores * (pos_mask * mask)
            neg_scores = scores * (1 - pos_mask)

        # Calculate SupConLoss
        probs = pos_scores.sum(-1)/(pos_scores.sum(-1) + neg_scores.sum(-1))
        probs /= (pos_mask * mask).sum(-1) + self.eps
        loss = - torch.log(probs + self.eps)

        loss_mask = (loss > 0.3).long()
        loss = (loss * loss_mask).sum() / (loss_mask.sum().item() + self.eps)
        # loss = loss.mean()
            
        return loss
