from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence
import random


class IEMOCAP_loader(Dataset):
    def __init__(self, txt_file, dataclass, pseudo_labels=None):
        self.dialogs = []
        self.pseudo_labels = pseudo_labels
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        context_emotion = []
        self.speakerNum = []

        emodict = {'ang': "anger", 'exc': "excited", 'fru': "frustrated", 'hap': "happy", 'neu': "neutral", 'sad': "sad"}
        # use: 'hap', 'sad', 'neu', 'ang', 'exc', 'fru'
        # discard: disgust, fear, other, surprise, xxx        

        self.emoSet = set()
        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                context_emotion = []
                continue
            
            speaker = data.strip().split('\t')[0]
            utt = ' '.join(data.strip().split('\t')[1:-1])
            emo = data.strip().split('\t')[-1]
            
            context.append(utt)
            
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)

            # make multi-label automatically by emotion-shift phenomenon
            if (len(context_emotion) > 1) and (emodict[emo] != "neutral") :
                try:
                    #find previous utterance of same speaker of current utterance
                    recent_idx = max((index, value) for index, value in enumerate(context_speaker) if value == speakerCLS)[0]
                    
                    # make multi-label if emotion shift exist
                    if (emodict[emo] != context_emotion[recent_idx])and (context_emotion[recent_idx] != "neutral") :
                        multi_emo = [emodict[emo], context_emotion[recent_idx]]
                    else:
                        multi_emo = [emodict[emo]]
                except ValueError: # first utterance of speaker
                    multi_emo = [emodict[emo]]
            else:
                multi_emo = [emodict[emo]]

            context_speaker.append(speakerCLS)
            context_emotion.append(emodict[emo])

            self.dialogs.append([context_speaker[:], context[:], emodict[emo], multi_emo])
            self.emoSet.add(emodict[emo])
        
        self.emoList = sorted(self.emoSet)
        self.labelList = self.emoList     
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        if self.pseudo_labels is not None:
            return self.dialogs[idx], self.labelList, self.pseudo_labels[idx]
        else:
            return self.dialogs[idx], self.labelList, None

    
class MELD_loader(Dataset):
    def __init__(self, txt_file, dataclass, pseudo_labels=None):
        self.dialogs = []
        self.pseudo_labels = pseudo_labels
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()
        
        temp_speakerList = []
        context = []
        context_speaker = []
        context_emotion = []
        self.speakerNum = []
        
        # 'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "happy", 'neutral': "neutral", 'sadness': "sad", 'surprise': 'surprise'}
        self.emoSet = set()

        for i, data in enumerate(dataset):
            if i < 2:
                continue
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                context_emotion = []
                continue

            speaker, utt, emo, senti = data.strip().split('\t')
            
            context.append(utt)
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)

            # make multi-label automatically by emotion-shift phenomenon
            if (len(context_emotion) > 1) and (emodict[emo] != "neutral") :
                try:
                    #find previous utterance of same speaker of current utterance
                    recent_idx = max((index, value) for index, value in enumerate(context_speaker) if value == speakerCLS)[0]
                    
                    # make multi-label if emotion shift exist
                    if (emodict[emo] != context_emotion[recent_idx]) and (context_emotion[recent_idx] != "neutral") :
                        multi_emo = [emodict[emo], context_emotion[recent_idx]]
                    else:
                        multi_emo = [emodict[emo]]
                except ValueError: # first utterance of speaker
                    multi_emo = [emodict[emo]]
            else:
                multi_emo = [emodict[emo]]
                
            context_speaker.append(speakerCLS)
            context_emotion.append(emodict[emo])
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo],multi_emo])
            
            self.emoSet.add(emodict[emo])
            
        self.emoList = sorted(self.emoSet)
        self.labelList = self.emoList     
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        if self.pseudo_labels is not None:
            return self.dialogs[idx], self.labelList, self.pseudo_labels[idx]
        else:
            return self.dialogs[idx], self.labelList, None
    
    
class Emory_loader(Dataset):
    def __init__(self, txt_file, dataclass, pseudo_labels=None):
        self.dialogs = []
        self.pseudo_labels = pseudo_labels
        f = open(txt_file, 'r')
        dataset = f.readlines()
        f.close()

        # 'Joyful', 'Mad', 'Neutral', 'Peaceful', 'Powerful', 'Sad', 'Scared'
        emodict = {'Joyful': "happy", 'Mad': "anger", 'Peaceful': "peaceful", 'Powerful': "powerful", 'Neutral': "neutral", 'Sad': "sad", 'Scared': 'fear'}
        
        temp_speakerList = []
        context = []
        context_speaker = []      
        context_emotion = []  
        self.speakerNum = []
        self.emoSet = set()


        for i, data in enumerate(dataset):
            if data == '\n' and len(self.dialogs) > 0:
                self.speakerNum.append(len(temp_speakerList))
                temp_speakerList = []
                context = []
                context_speaker = []
                context_emotion = []
                continue
            
            speaker, utt, emo = data.strip().split('\t')
            context.append(utt)
            
            
            if speaker not in temp_speakerList:
                temp_speakerList.append(speaker)
            speakerCLS = temp_speakerList.index(speaker)

            # make multi-label automatically by emotion-shift phenomenon
            if (len(context_emotion) > 1) and (emodict[emo] != "neutral") :
                try:
                    #find previous utterance of same speaker of current utterance
                    recent_idx = max((index, value) for index, value in enumerate(context_speaker) if value == speakerCLS)[0]
                    
                    # make multi-label if emotion shift exist
                    if (emodict[emo] != context_emotion[recent_idx]) and (context_emotion[recent_idx] != "neutral") :
                        multi_emo = [emodict[emo], context_emotion[recent_idx]]
                    else:
                        multi_emo = [emodict[emo]]
                except ValueError: # first utterance of speaker
                    multi_emo = [emodict[emo]]
            else:
                multi_emo = [emodict[emo]]
                
            context_speaker.append(speakerCLS)
            context_emotion.append(emodict[emo])
            
            self.dialogs.append([context_speaker[:], context[:], emodict[emo],multi_emo])
            self.emoSet.add(emodict[emo])
            
        self.emoList = sorted(self.emoSet)
        self.labelList = self.emoList 
        self.speakerNum.append(len(temp_speakerList))
        
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        if self.pseudo_labels is not None:
            return self.dialogs[idx], self.labelList, self.pseudo_labels[idx]
        else:
            return self.dialogs[idx], self.labelList, None
    
