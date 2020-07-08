import re
import sqlite3
import datetime, monthdelta
import pandas as pd
import os
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, BertForTokenClassification, BertConfig
import time
from tqdm import tqdm
import numpy as np
import math
import itertools
from collections import OrderedDict
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.aws_s3 import *


def translate_terms(df, to_translate=False):
    if to_translate:
        trans1_from = ['新型冠状病毒感染的肺炎', '新型冠状病毒感染肺炎', '新型冠状病毒肺炎']
        trans1_to = '非典型肺炎'

        trans2_from = '新冠肺炎'
        trans2_to = '非典肺炎'

        trans3_from = '新型冠状病'
        trans3_to = '非典型肺炎病毒'

        trans4_from = '新冠病毒'
        trans4_to = '非典病毒'
        
        df['sentence'] = df['sentence'].str.replace('|'.join(trans1_from), trans1_to)
        df['sentence'] = df['sentence'].str.replace(trans2_from, trans2_to)
        df['sentence'] = df['sentence'].str.replace(trans3_from, trans3_to)
        df['sentence'] = df['sentence'].str.replace(trans4_from, trans4_to)
    return df


def duplicate_df_by(df, var):
    def duplicate_df_to(df, to_size):
        df = df.sample(frac=1, random_state=1)
        return pd.concat( [df] * math.ceil( to_size/df.shape[0] ) ).head(max(to_size, df.shape[0]))

    max_row = df.groupby(var).count().reset_index().iloc[:,1].max()
    return df.groupby(var).apply(duplicate_df_to, to_size = max_row).reset_index(drop=True)


def gen_stage_2_label(df, start, by_date = True):
    df['days_since'] = ((df['date']-start)/np.timedelta64(1, 'D')).astype('int64')

    if by_date: 
        df['label'] = df['days_since']
    else:
        df['label'] = (df['days_since']).apply(np.floor).astype('int64')

    df = df.drop(columns=['days_since'])
    return df


def gen_train_test_data(model_settings, stage):
    """
    stage: specify which of the two stages are the data generated for
    """
    assert stage in [1, 2]

    if stage==1:
        ## import preprocessed data
        df_sars = pd.read_pickle(gen_data_path(model_settings, 'SARS_sentences.pickle'))
        df_sars_irrel = pd.read_pickle(gen_data_path(model_settings, 'SARS_irrel_sentences.pickle'))

        ## generate labels (binary on whether it's relevant)
        df_sars['label'] = 1
        df_sars_irrel['label'] = 0

        ## train-test split
        df_sars_9, df_sars_1 = train_test_split(
            df_sars, test_size=0.1, random_state=model_settings['rand_state'], stratify=df_sars['date']
            )
        df_sars_irrel_9, df_sars_irrel_1 = train_test_split(
            df_sars_irrel, test_size=0.1, random_state=model_settings['rand_state'], stratify=df_sars_irrel['date']
            )

        ## balance data by date (within each df) and by label (between two df's)
        df_sars_9_bal = duplicate_df_by(df_sars_9, "date")
        df_sars_irrel_9_bal = duplicate_df_by(df_sars_irrel_9, "date")
        df_sars_irrel_9_bal_sample = df_sars_irrel_9_bal.sample(
            n=len(df_sars_9_bal), replace=False, random_state=model_settings['rand_state']
            )

        ## produce train and test df's
        df_train = pd.concat([df_sars_9_bal, df_sars_irrel_9_bal_sample])
        df_test = pd.concat([df_sars_1, df_sars_irrel_1])

        df_train_unbal = pd.concat([df_sars_9, df_sars_irrel_9])

    elif stage==2:
        ## import preprocessed data
        df_sars = pd.read_pickle(gen_data_path(model_settings, 'SARS_sentences.pickle'))

        ## extract start date of SARS data
        SARS_START = min(df_sars['date'])

        ## generate date-level labels
        df_sars = gen_stage_2_label(df_sars, SARS_START)

        ## train-test split
        df_train_unbal, df_test = train_test_split(
            df_sars, test_size=0.1, random_state=model_settings['rand_state'], stratify=df_sars['label']
            )

        ## balance data by label (date)
        df_train = duplicate_df_by(df_train_unbal, "label")

    print("Generated train and test data.")

    return df_train, df_test, df_train_unbal


def gen_forecast_data(model_settings, stage, to_translate=False):
    """
    stage: specify which of the two stages are the data generated for
    to_translate: whether to translated covid terms to respective sars terms
    """
    assert stage in [1, 2]

    if stage==1:
        ## import preprocessed data
        df_covid = pd.read_pickle(gen_data_path(model_settings, 'COVID_sentences.pickle'))
        df_covid_irrel = pd.read_pickle(gen_data_path(model_settings, 'COVID_irrel_sentences.pickle'))

        ## generate labels (binary on whether it's relevant)
        df_covid['label'] = 1
        df_covid_irrel['label'] = 0

        ## translate covid terms to sars terms, if called for
        df_covid = translate_terms(df_covid, to_translate=to_translate)

        return df_covid, df_covid_irrel

    elif stage==2:
        ## import stage-1 results as data
        df_covid = pd.read_pickle(gen_model_path(model_settings, 'stage_1_results_forecast.pickle'))
        # df_covid_irrel = pd.read_pickle(gen_model_path(model_settings, 'stage_1_results_placebo.pickle'))

        ## only take data with stage-1 pred = 1
        df_covid = df_covid[df_covid['pred']==1]
        # df_covid_irrel = df_covid_irrel[df_covid_irrel['pred']==1]
        return df_covid, None 

    print("Generated forecast data.")

    


class gen_dataset(Dataset):
    def __init__(self, model_settings):

        self.stage = None
        
        self.max_length = model_settings['max_length']

        self.df = pd.DataFrame()

        self.len = 0

        self.contain_labels = True

        self.tokenizer = model_settings['tokenizer']

        self.text = model_settings['text']


    def populate_from_pickle(self, stage, path=None, custom_df=None, contain_labels=True):
        assert stage in [1, 2]
        self.stage = stage

        if custom_df is None:
            temp_df = pd.read_pickle(path)
        else:
            temp_df = custom_df

        assert all(elem in temp_df.columns for elem in [self.text])
        self.df[self.text] = temp_df[self.text]

        self.len = len(self.df)

        if contain_labels:
            self.label = temp_df['label']
        else:
            self.contain_labels = False

        print( "- # of obs: " + str(self.len) )


    def __getitem__(self, idx):
        sentence = self.df[self.text].iloc[idx,]

        if self.contain_labels:
            label = self.label.iloc[idx,]
            if self.stage==1:
                label_tensor = torch.from_numpy(np.array(label))
            elif self.stage==2:
                label_tensor = torch.from_numpy(np.array(label)).float()
        else:
            label_tensor = torch.empty(0)
        
        if self.max_length != None:
            sentence = sentence[:self.max_length]

        tokens_tensor, segments_tensor = self.get_tokenized_text(sentence, self.tokenizer)

        return tokens_tensor, segments_tensor, label_tensor

    def __len__(self):
        return self.len

    @staticmethod
    def get_tokenized_text(sentence, tokenizer):

        token = tokenizer.tokenize(sentence)

        word_pieces = ['[CLS]'] + token + ["[SEP]"]
        len_text = len(word_pieces)

        ids = tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)

        segments_tensor = torch.tensor([0] * len_text, dtype=torch.long)

        return tokens_tensor, segments_tensor


def load_data_for_pred(model_settings, stage, df):
    dataset = gen_dataset(model_settings)
    dataset.populate_from_pickle(stage = stage, custom_df = df)
    loader = DataLoader(dataset, batch_size=model_settings['predict_batch_size'], collate_fn=create_mini_batch, shuffle=False)
    return loader


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]

    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None

    # zero pad 
    tokens_tensors = pad_sequence(tokens_tensors,
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors,
                                    batch_first=True)

    masks_tensors = torch.zeros(tokens_tensors.shape,
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)

    return tokens_tensors, segments_tensors, masks_tensors, label_ids


def train_bert(model_settings, training_data, stage, prev_epoch=None, from_s3 = False):

    """
    stage: specify which of the two stages is the training for
    prev_epoch: whether to continue training from a given previous epoch
    from_s3: if continuing training, whether to get the model from s3
    """
    assert stage in [1, 2]

    if prev_epoch != None:
        epoch_start = prev_epoch + 1 
        model, optimizer, device = create_model(model_settings, stage, prev_epoch = prev_epoch, from_s3 = from_s3)
        print("Continue from previous epoch: " + str(prev_epoch))
    else:
        epoch_start = 1
        model, optimizer, device = create_model(model_settings, stage)
    
    training_dataset = gen_dataset(model_settings)
    training_dataset.populate_from_pickle(stage = stage, custom_df = training_data)   
    trainloader = DataLoader(training_dataset, batch_size=model_settings['batch_size'], collate_fn=create_mini_batch, shuffle=True)

    num_batches = trainloader.dataset.len//trainloader.batch_size    
    print("- # of batches: " + str(num_batches))

    model.train()

    if len(range(epoch_start, model_settings['epochs']+1))==0:
        print("All required epochs are trained.")

    for epoch in range(epoch_start, model_settings['epochs']+1):
        print('[epoch %d]' % (epoch))
        running_loss = 0.0

        with tqdm(total=num_batches) as pbar:
            for data in trainloader:

                tokens_tensors, segments_tensors, \
                masks_tensors, labels = [t.to(device) for t in data if (t is not None and isinstance(t, torch.Tensor ) ) ]

                optimizer.zero_grad()

                outputs = model(input_ids=tokens_tensors,
                                token_type_ids=segments_tensors,
                                attention_mask=masks_tensors,
                                labels=labels)

                loss = outputs[0]
                # backward
                loss.sum().backward()
                optimizer.step()
                running_loss += loss.sum().item()

                pbar.update(1)

        print('Finished epoch.')

        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()

        if not os.path.exists(gen_model_path(model_settings)):
            os.mkdir(gen_model_path(model_settings))

        torch.save(
            {
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss,
            }, 
            gen_model_path(model_settings, 'stage_'+str(stage)+'_model_epoch_'+str(epoch)+'.tar')
        )
        print('Saved model.')

        upload_model_s3(model_settings, stage, epoch)
        print('Uploaded model to s3.')

    return model


def create_model(model_settings, stage, prev_epoch = None, from_s3 = False):
    assert stage in [1, 2]

    if prev_epoch == None:
        model, optimizer, device = load_model(model_settings, stage)
        print("Started a new model")

    elif from_s3:
        download_model_s3(model_settings, stage, prev_epoch)
        print("Downloaded the requested model")
        model, optimizer, device = load_model(model_settings, stage, 'stage_'+str(stage)+'_model_epoch_'+str(prev_epoch)+'.tar')
        print("Loaded the requested model")

    else:
        model, optimizer, device = load_model(model_settings, stage, 'stage_'+str(stage)+'_model_epoch_'+str(prev_epoch)+'.tar')
        print("Loaded the requested model")

    return model, optimizer, device


def load_model(model_settings, stage, model_fname=None):
    assert stage in [1, 2]

    model = get_pretrained_model(model_settings, stage)
    optimizer = pci_optimizer(model)

    if model_fname != None:
        if torch.cuda.is_available():
            checkpoint = torch.load(gen_model_path(model_settings, model_fname))
        else:
            checkpoint = torch.load(gen_model_path(model_settings, model_fname), map_location = torch.device('cpu'))

        old_model_state_dict = checkpoint['model_state_dict']
        new_model_state_dict = OrderedDict()
        for k, v in old_model_state_dict.items():
            if k[:7]=="module.":
                name = k[7:] # remove string 'module.' of from head of k
            else:
                name = k
            new_model_state_dict[name]=v

        model.load_state_dict(new_model_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if model_settings['cuda_index'] == None:
        cuda = "cuda"
    else:
        cuda = "cuda:"+str(model_settings['cuda_index'])

    if torch.cuda.is_available():
        device = torch.device(cuda)
        if model_settings['multi_gpu'] and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

        if model_fname != None:
            for state in optimizer.state.values():  
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(cuda)
    else:
        device = torch.device("cpu")

    model.to(device)

    return model, optimizer, device


def get_pretrained_model(model_settings, stage):
    assert stage in [1, 2]
    if stage==1:
        config = BertConfig.from_pretrained(
            model_settings['pretrained_model_name'], 
            num_labels = model_settings['stage_1_num_labels'], 
            max_length = model_settings['max_length']
            )
    elif stage==2:
        config = BertConfig.from_pretrained(
            model_settings['pretrained_model_name'], 
            num_labels = model_settings['stage_2_num_labels'], 
            max_length = model_settings['max_length']
            )
    return BertForSequenceClassification.from_pretrained(
        model_settings['pretrained_model_name'], 
        config = config
        )


def pci_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-5)


def get_predictions(stage, model, device, dataloader, evaluate=True):
    """
    stage: specify which of the two stages are the predictions for
    evaluate: whether to produce metrics (acc for step 1 and mae for step 2) to evaluate the performance
    """
    assert stage in [1, 2]

    predictions = None
    correct = 0
    mae = 0
    total = 0

    model.eval()

    num_batches = dataloader.dataset.len//dataloader.batch_size
    print(str(num_batches) + " batches")

    with torch.no_grad():
        with tqdm(total=num_batches) as pbar:
            for data in dataloader:

                if next(model.parameters()).is_cuda:
                    data = [t.to(device) for t in data if (t is not None and isinstance(t, torch.Tensor ) ) ]

                tokens_tensors, segments_tensors, masks_tensors, label_ids = data[:4]

                outputs = model(input_ids=tokens_tensors,
                                token_type_ids=segments_tensors,
                                attention_mask=masks_tensors)

                logits = outputs[0]

                if stage==1:
                    _, pred = torch.max(logits.data, 1)
                elif stage==2:
                    pred = logits.data[:,0]

                labels = data[3]

                if stage==1 and evaluate:
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()

                elif stage==2 and evaluate:
                    total += labels.size(0)
                    mae += (torch.abs(pred - labels)).sum().item()

                if predictions is None:
                    predictions = pred
                else:
                    predictions = torch.cat((predictions, pred))

                pbar.update(1)

    if stage==1 and evaluate:
        acc = correct / total
        print('acc: %.3f' % acc)

        return predictions, acc

    elif stage==2 and evaluate:
        mae_out =  mae / total 
        print('mae: %.3f' % mae_out)

        return predictions, mae_out

    else:
        return predictions


def assess_training(model_settings, stage, df_train, df_test):
    """
    stage: specify which of the two stages to assess the training performance for
    """
    print("Training data:")
    trainloader = load_data_for_pred(model_settings, stage, df_train)
    print("Testing data:")
    testloader = load_data_for_pred(model_settings, stage, df_test)

    for epoch in [model_settings['epochs'], model_settings['epochs']-2, model_settings['epochs']-4]:
        print("==============")
        print("|| Epoch: " + str(epoch) + ' ||')
        print("==============")

        model, _, device = create_model(model_settings, stage, prev_epoch = epoch)
        ## if needed to download from s3
        # model, _, device = create_model(model_settings, stage, prev_epoch = epoch, from_s3 = True)

        print("Training performance:")
        get_predictions(stage, model, device, trainloader, evaluate=True)
        print("Testing performance:")
        get_predictions(stage, model, device, testloader, evaluate=True)


def calc_results(model_settings, stage, model, device, df, fname):
    """
    stage: specify which of the two stages to calculate the results for
    """
    assert stage in [1, 2]

    dataloader = load_data_for_pred(model_settings, stage, df)

    pred = get_predictions(stage, model, device, dataloader, evaluate=False)
    
    df['pred'] = pred.cpu().numpy()
    df_drop = df.drop(columns=['sentence'])

    if stage==1:
        pd.DataFrame(confusion_matrix(df['label'], df['pred'])).to_excel(gen_model_path(model_settings,
            'stage_'+str(stage)+'_matrix_'+fname+".xlsx"))
        df.to_pickle(gen_model_path(model_settings,
            'stage_'+str(stage)+'_results_'+fname+'.pickle'))

    elif stage==2:
        df_drop.to_excel(gen_model_path(model_settings,
            'stage_'+str(stage)+'_results_'+fname+'.xlsx'))
        df.to_pickle(gen_model_path(model_settings,
            'stage_'+str(stage)+'_results_'+fname+'.pickle'))
        
    return df