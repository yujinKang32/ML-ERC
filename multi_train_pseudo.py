## -*- coding: utf-8 -*-
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from ERC_dataset_loader import MELD_loader, Emory_loader, IEMOCAP_loader, DD_loader
from model import ERC_model

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support, f1_score, roc_auc_score
from utils import encode_right_truncated, padding
from utils import make_batch_roberta
from transformers import RobertaTokenizer, RobertaModel

import numpy as np, pandas as pd 
import zipfile

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt 
import seaborn as sns

import time

from sklearn.utils import Bunch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import wandb

wandb.init(project="ML-ERC")

def T_SNE_vis(labels_reshape,reps_reshape, work_dir, epoch):
    
    '''    
    labels_reshape = labels_reshape.detach().cpu().numpy()
    reps_reshape = (reps_reshape).squeeze().detach().cpu().numpy()
    '''
    
    dialogue = Bunch(target = labels_reshape, data = reps_reshape)
    
    n_components = 2
    t_sne = TSNE(n_components = n_components).fit_transform(dialogue.data)
    #t = t_sne.fit_transform(dialogue.data)
    #print(t_sne)
    plt.figure(figsize=(10,10))
    plt.xlim(t_sne[:,0].min(), t_sne[:,0].max()+1)
    plt.ylim(t_sne[:,0].min(), t_sne[:,0].max()+1)
    colors = ['#476A2A', '#7851B8', '#BD3430', '#4A2D4E', '#875525','#A83683', '#4E655E', '#853541', '#3A3120', '#535D8E']
    for i in range(len(dialogue.data)):
        plt.text(t_sne[i,0], t_sne[i,1], str(int(dialogue.target[i])), color = colors[dialogue.target[i]])
    
    plt.savefig(work_dir+'/'+str(args.dataset)+'_tsne_'+str(epoch)+'.png')
    plt.close()

def hamming_loss(true_label, pred_label):

    true_array = np.stack(true_label)
    pred_array = np.stack(pred_label)

    # Calculate Hamming Distance for each sample
    hamming_distances = np.sum(true_array != pred_array, axis=1)
    num_labels = true_array.shape[1]
    normalized_hamming_distances = hamming_distances / num_labels

    # Calculate Hamming Loss
    hamming_loss = np.mean(normalized_hamming_distances)

    return hamming_loss 

def get_today():
    now = time.localtime()
    s = "%04d-%02d-%02d-%02d-%02d" %(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
    return s

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()   

def main():    
    """Dataset Loading"""
    batch_size = args.batch
    dataset = args.dataset
    dataclass = args.cls
    sample = args.sample
    model_type = args.pretrained
    loss_method = args.loss
    set_seed(args.seed)
    
    
    if dataset == 'iemocap':
        emo_len = 6
        neu_label = 4
    elif dataset == 'EMORY':
        emo_len = 7
        neu_label = 2
    else:
        emo_len = 7
        neu_label = 4
    
    
    dataType = 'multi'
    if dataset == 'MELD':
        if args.dyadic:
            dataType = 'dyadic'
        else:
            dataType = 'multi'
        data_path = 'dataset/MELD/'+dataType+'/'
        DATA_loader = MELD_loader
    elif dataset == 'EMORY':
        data_path = 'dataset/EMORY/'
        DATA_loader = Emory_loader
    elif dataset == 'iemocap':
        data_path = 'dataset/iemocap/'
        DATA_loader = IEMOCAP_loader
        
    if model_type == 'roberta-large':
        make_batch = make_batch_roberta
    
        
    train_path = data_path + dataset+'_train.txt'
    dev_path = data_path + dataset+'_dev.txt'
    test_path = data_path + dataset+'_test.txt'
    
    
    
    """logging and path"""
    save_path = os.path.join(dataset+'_models', args.loss)
    os.makedirs('test/diyi/temp', exist_ok=True)    
    print("###Save Path### ", save_path)

    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)     

    logger.info(args) 
    
 
    train_dataset = DATA_loader(train_path, dataset)
    clsNum = len(train_dataset.labelList)
    model = ERC_model(model_type, clsNum, args)
    model = model.cuda()
    wandb.watch(model)
    model.train() 
    
    """Training Setting"""        
    training_epochs = args.epoch
    save_term = int(training_epochs/5)
    max_grad_norm = args.norm
    lr = args.lr

    
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    """Input & Label Setting"""
    best_dev_fscore, best_test_fscore,best_dev_multi_fscore  = 0, 0,0
    best_dev_fscore_macro, best_dev_fscore_micro, best_test_fscore_macro, best_test_fscore_micro, best_test_multi_micro = 0, 0, 0, 0, 0
    best_dev_auc, best_test_auc = 0,0
    best_epoch = 0
    patience = 0

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, collate_fn=make_batch)
    train_sample_num = int(len(train_dataset)*sample)
    dev_dataset = DATA_loader(dev_path, dataset)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=make_batch)
        
    test_dataset = DATA_loader(test_path, dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=make_batch)
        

    for epoch in tqdm(range(training_epochs),desc="Epoch: ", position=0, ncols=150 ):
        
        total_loss = 0.0
        pseudo_label_list = []
        pseudo_label_list_len = 0
        for i_batch, data in (enumerate(tqdm(train_dataloader, desc="training: ", position=1, leave=False, ncols=150))):
            
            if i_batch > train_sample_num:
                print(i_batch, train_sample_num)
                break
            
            """Prediction"""
            
            batch_input_tokens, batch_labels, batch_multi_labels, batch_multi_bool,label_dict = data
            batch_input_tokens, batch_labels, batch_multi_labels = batch_input_tokens.cuda(), batch_labels.cuda(), batch_multi_labels.cuda()
                
            model.train()  
            pred_logits, loss,_, entropy = model(batch_input_tokens, batch_labels, batch_multi_labels, label_dict)
            
            
            if epoch > 20 :
                
                if args.pseudo:
                    normalize_logits = torch.tanh(pred_logits)

                    top2_values, top2_indices = normalize_logits.topk(2, dim=1)
                    thresholded_logits = torch.zeros_like(normalize_logits)

                    for i in range(thresholded_logits.size(0)):
                        if batch_labels[i] not in top2_indices[i]:
                            thresholded_logits[i,batch_labels[i]] = 1
                            if top2_indices[i][0] != neu_label:
                                thresholded_logits[i,top2_indices[i][0]] = 1
                        else:
                            for j in top2_indices[i]:
                                if j != neu_label: # Exclude the Neutral emotion
                                    thresholded_logits[i, j] = 1

                    
                    entropy_upper_bound = args.pseudo_entropy
                    
                    for i in range(len(batch_multi_bool)):
                        # Condition 1: Data not having multi labels
                        if not batch_multi_bool[i]:
                            # Condition 2: Data satisfying Entropy within specified range
                            if entropy[i] <= entropy_upper_bound:
                                # Condition 3: Data' single label is not 'neutral'
                                if batch_labels[i] != neu_label:
                                    pseudo_label_list.append(thresholded_logits[i].tolist())
                                    pseudo_label_list_len += 1
                                else:
                                    # Condition 3 not met, add an empty placeholder
                                    pseudo_label_list.append([])
                            else:
                                # Condition 2 not met, add an empty placeholder
                                pseudo_label_list.append([])
                        else:
                            # Condition 1 not met, add an empty placeholder
                            pseudo_label_list.append([])
                else:
                     pseudo_label_list = None
            
            else:
                pseudo_label_list = None
            loss.backward()
            total_loss += loss
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        #Reload the DataLoader with pseudo labels
        train_dataset = DATA_loader(train_path, dataset, pseudo_labels = pseudo_label_list)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, collate_fn=make_batch)
        train_sample_num = int(len(train_dataset)*sample)
            
        """Dev & Test evaluation"""
        model.eval()
        logger.info('Epoch: {}'.format(epoch))
        
        dev_prek, dev_pred_list, dev_label_list, dev_multi_label, dev_pred_multi = _CalACC(model, dev_dataloader,'dev')
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted', zero_division = 0)
        dev_multi_pre, dev_multi_rec, dev_multi_fbeta_macro, _ = precision_recall_fscore_support(dev_multi_label, dev_pred_multi , average='macro', zero_division = 0)
        dev_multi_pre, dev_multi_rec, dev_multi_fbeta_weighted, _ = precision_recall_fscore_support(dev_multi_label, dev_pred_multi , average='weighted', zero_division = 0)
        dev_auc = roc_auc_score(dev_multi_label, dev_pred_multi)
        #print("dev_multi_label, dev_pred_multi ", dev_multi_label, dev_pred_multi )

        test_prek, test_pred_list, test_label_list, test_multi_label, test_pred_multi = _CalACC(model, test_dataloader, 'test',epoch)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted') 
        test_multi_pre, test_multi_rec, test_multi_fbeta_macro, _ = precision_recall_fscore_support(test_multi_label, test_pred_multi, average='macro', zero_division = 0)
        test_multi_pre, test_multi_rec, test_multi_fbeta_weighted, _ = precision_recall_fscore_support(test_multi_label, test_pred_multi, average='weighted', zero_division = 0)
        test_auc = roc_auc_score(test_multi_label, test_pred_multi)

        dev_hamming_loss = hamming_loss(dev_multi_label, dev_pred_multi)
        test_hamming_loss = hamming_loss(test_multi_label, test_pred_multi)
        dev_multi_fbeta = dev_multi_fbeta_macro+dev_multi_fbeta_weighted
        
        """Best Score & Model Save"""
        if dev_fbeta >= best_dev_fscore:
        #if dev_multi_fbeta >= best_dev_multi_fscore:
            

            best_dev_fscore = dev_fbeta
            best_test_fscore = test_fbeta
            best_test_prek = test_prek

            best_dev_multi_fscore = dev_multi_fbeta
            best_test_multi_fscore_w = test_multi_fbeta_weighted
            best_test_multi_fscore_m = test_multi_fbeta_macro

            best_dev_auc = dev_auc
            best_test_auc = test_auc
            
            best_dev_hamming = dev_hamming_loss
            best_test_hamming = test_hamming_loss
            
            best_epoch = epoch
            _SaveModel(model, save_path)
            patience = 0
            logger.info("*** Best performance ***")
        else:
            patience += 1
                
        
        
        
        logger.info('Devleopment ## loss: {}, dev_pre: {}, recall: {}, fscore: {}'.format((total_loss/i_batch), dev_pre, dev_rec, dev_fbeta))
        logger.info('Devleopment for multi-label ## loss: {}, dev_pre: {}, recall: {}, macro_fscore: {}, weighted_fscore: {}, roc_auc_score: {}'.format((total_loss/i_batch), dev_multi_pre, dev_multi_rec, dev_multi_fbeta_macro,dev_multi_fbeta_weighted, dev_auc))
        
        logger.info('\nTest ## test_pre: {}, recall: {}, fscore: {}'.format(test_pre, test_rec, test_fbeta))
        logger.info('Test for multi-label ## test_pre: {}, recall: {}, macro-fscore: {}, weighted-fscore: {}, roc_auc_score: {}'.format(test_multi_pre, test_multi_rec, test_multi_fbeta_macro,test_multi_fbeta_weighted, test_auc))
        
        wandb.log({"loss": (total_loss/i_batch), "dev_single_f1": dev_fbeta, "test_single_f1": test_fbeta, "multi_dev_f1":dev_multi_fbeta, 
        "multi_test_pre":test_multi_pre,"multi_test_rec":test_multi_rec,"multi_test_macro_f1":test_multi_fbeta_macro,
        "multi_test_weighted_f1":test_multi_fbeta_weighted, "test_AUC":test_auc, "dev_hamming_loss": dev_hamming_loss, 
        "test_hamming_loss": test_hamming_loss, "pseudo_num":pseudo_label_list_len})

        logger.info('')
            
        if patience > 15:
            print("Early stop!")
            break  
        
    
    logger.info('Final Fscore ## test-fscore: {}, test_epoch: {}'.format(best_test_fscore, best_epoch))    
    logger.info('Final Fscore for multi-label ## test-macro-fscore: {}, test-weighted-fscore: {}, test-auc: {}'.format(best_test_multi_fscore_m, best_test_multi_fscore_w, best_test_auc))        
    wandb.log({"final_f1": best_test_fscore, "final_multi_weighted_f1": best_test_multi_fscore_w, "final_multi_macro_f1": best_test_multi_fscore_m, "final_AUC":best_test_auc, "final_hamming":test_hamming_loss})


def _CalACC(model, dataloader, mode, epoch =0, directory=''):
    model.eval()
    correct = 0
    label_list = []
    pred_list = []
    multi_label_list = []
    multi_pred_list = []
    tsne_embedding = []
    tsne_label = []
    p1num, p2num, p3num = 0, 0, 0    
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader, desc="Eval: ", position=1, leave=False, ncols=150)):            
            """Prediction"""
            batch_input_tokens, batch_labels, batch_multi_labels, batch_multi_bool,label_dict = data
            batch_input_tokens, batch_labels, batch_multi_labels = batch_input_tokens.cuda(), batch_labels.cuda(), batch_multi_labels.cuda()

            pred_logits, loss,_, entropy = model(batch_input_tokens, batch_labels, batch_multi_labels, label_dict)
            
            normalize_logits = torch.tanh(pred_logits)
            logit_mean = normalize_logits.mean(dim=1)
            over_labels = normalize_logits > logit_mean.unsqueeze(1)
            pred_multi_labels = over_labels.int()

            
            batch_multi_labels = batch_multi_labels.cpu().numpy()
            pred_multi_labels = pred_multi_labels.cpu().numpy()
            
            multi_label_list.append(batch_multi_labels[0])
            multi_pred_list.append(pred_multi_labels[0])

            """Calculation"""    
            pred_logits_sort = pred_logits.sort(descending=True)
            indices = pred_logits_sort.indices.tolist()[0]
            
            pred_label = indices[0] # pred_logits.argmax(1).item()
            true_label = batch_labels.item()
            
            pred_list.append(pred_label)
            label_list.append(true_label)
            if pred_label == true_label:
                correct += 1
                
            """Calculation precision"""
            if true_label in indices[:1]:
                p1num += 1
            if true_label in indices[:2]:
                p2num += 1/2
            if true_label in indices[:3]:
                p3num += 1/3
            
        p1 = round(p1num/len(dataloader)*100, 2)
        p2 = round(p2num/len(dataloader)*100, 2)
        p3 = round(p3num/len(dataloader)*100, 2)
        
        
    return [p1, p2, p3], pred_list, label_list, multi_label_list, multi_pred_list

def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))

if __name__ == '__main__':
    torch.cuda.empty_cache()

    '''
    we utilze structure made by Lee and Lee (https://github.com/rungjoo/CoMPM/tree/master). 
    We have adapted and extended their code to fit the specific needs of our work. 

    We express our gratitude to Lee and Lee for their original contributions, which made this project possible. 
    For further details on the original work, please refer to the (https://github.com/rungjoo/CoMPM/tree/master).
    '''
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--batch", type=int, help = "batch_size", default = 16)
    
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 30) # 12 for iemocap
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    parser.add_argument( "--sample", type=float, help = "sampling trainign dataset", default = 1.0) # 
    parser.add_argument( "--dataset", help = 'MELD or EMORY or iemocap or dailydialog', default = 'MELD')
    parser.add_argument( "--pretrained", help = 'roberta-large or simcse', default = 'roberta-large')
    parser.add_argument('-dya', '--dyadic', action='store_true', help='dyadic conversation')
    parser.add_argument( "--cls", help = 'emotion or sentiment', default = 'emotion')
    parser.add_argument( "--temp", type=float, help = 'temperature in contrastive learning loss', default = 0.05)    
    parser.add_argument( "--loss", help = 'method calculating loss.  ',default = 'multi')
    parser.add_argument( "--tsne", help = 'want to visualization embedding space via T-SNE  ',default = False)
    parser.add_argument( "--alpha", help = 'percentage between BCE and Weighted CL in loss', type=float,default = 0.1) 
    parser.add_argument( "--entropy", help = 'use entropy weight in WSCL', type=bool,default = True) 
    parser.add_argument( "--multi_weighted", help = 'use label relation in WSCL', type=bool,default = True) 
    parser.add_argument( "--seed", type=int, help = "set seed", default = 2333)
    parser.add_argument( "--pseudo", help = 'use pseudo labeling', type=bool,default = False) 
    parser.add_argument( "--pseudo_entropy", help = 'threshold for pseudo labeling', type=float,default = 0.7) 
    parser.add_argument( "--beta", help = 'percentage of single and multi in loss', type=float,default = 0.5) 
    parser.add_argument( "--test", help = 'test mode', type=bool,default = False) 
    
    
    args = parser.parse_args()
    wandb.config.update(args) # adds all of the arguments as config variables
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    
    main()
