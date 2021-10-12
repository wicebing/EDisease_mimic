import os
import time
import unicodedata
import random
import string
import re
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import AutoConfig, AutoTokenizer, AutoModel, BertConfig, BertModel
import torch

from EDisease_utils import count_parameters
from EDisease_config import EDiseaseConfig 

class adjBERTmodel(nn.Module):
    def __init__(self, bert_ver, embedding_size, fixBERT=True):
        super(adjBERTmodel, self).__init__()
        
        self.Emodel = AutoModel.from_pretrained(bert_ver)
        self.config = self.Emodel.config


        print('baseBERT PARAMETERS: ' ,count_parameters(self.Emodel))        
        if fixBERT:
            for param in self.Emodel.parameters():
                param.requires_grad = False   
            print(' ** fix pretrained BERT WEIGHT ** ')
            print('baseBERT PARAMETERS: ' ,count_parameters(self.Emodel))
        
        self.emb_emb = nn.Sequential(nn.Linear(self.config.hidden_size,2*embedding_size),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(2*embedding_size,embedding_size),
                                     nn.LayerNorm(embedding_size),
                                     )

    def forward(self, inputs:dict, cls_token=None):
        outputs = self.Emodel(**inputs)
        last_hidden_states = outputs.last_hidden_state          
        heads = last_hidden_states[:,0,:]
        
        convert_head = self.emb_emb(heads)
        # convert_cls = self.emb_emb(self.CLS_emb)
        
        if cls_token is None:
            return convert_head, heads
        else:
            CLS_emb = self.Emodel.embeddings.word_embeddings(cls_token)
            convert_cls = self.emb_emb(CLS_emb)
            return convert_head, convert_cls

class float2spectrum(nn.Module):
    def __init__(self, embedding_size):
        super(float2spectrum, self).__init__()
        self.thida = torch.linspace(0,2*math.pi,int(embedding_size/2))
        
    def forward(self, tensor):
        k_thida = torch.einsum("nm,k->nmk", tensor, self.thida)
        emb_x = torch.cat((k_thida.cos(),k_thida.sin()), dim=-1)
        return emb_x        

class structure_emb(nn.Module):
    def __init__(self, config):
        super(structure_emb, self).__init__()
        self.float2emb = float2spectrum(config.hidden_size)
        
        self.Config = BertConfig()
        self.Config.hidden_size = config.hidden_size
        self.Config.num_hidden_layers = config.num_hidden_layers
        self.Config.intermediate_size = 256
        self.Config.num_attention_heads = config.num_attention_heads
        self.Config.max_position_embeddings = config.max_position_embeddings
        self.Config.type_vocab_size= config.type_vocab_size
        self.Config.vocab_size=config.vocab_size
        
        self.BERTmodel = BertModel(self.Config)

    def forward(self, inputs,attention_mask,position_ids,token_type_ids=None):
        inputs_embeds = self.float2emb(inputs)
        
        outputs = self.BERTmodel(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 token_type_ids=token_type_ids)
        
        return outputs[0][:,:1,:]



class pickle_Model(nn.Module):
    def __init__(self,config,tokanizer,device='cpu'):
        super(pickle_Model, self).__init__() 
        self.stc2emb = float2spectrum(config)
                 
        self.Config = BertConfig()
        self.Config.hidden_size = config.hidden_size
        self.Config.num_hidden_layers = config.num_hidden_layers
        self.Config.intermediate_size = 256
        self.Config.num_attention_heads = config.num_attention_heads
        self.Config.max_position_embeddings = config.max_position_embeddings
        self.Config.type_vocab_size= config.type_vocab_size
        self.Config.vocab_size=config.vocab_size
        
        self.BERTmodel = BertModel(self.Config)
        
        self.tokanizer = tokanizer
        self.device = device
        
    def forward(self,
                baseBERT,
                inputs,
                normalization=None, 
                noise_scale=0.001,
                mask_ratio=0.15, 
                mask_ratio_pi=0.5,
                token_type_ids=None, 
                expand_data=None,
                use_pi=False):
        s,c_emb_,h_emb_mean_ = inputs['structure'],inputs['ccemb'],inputs['hxemb']
        s,c_emb_,h_emb_mean_ = s.to(self.device),c_emb_.to(self.device),h_emb_mean_.to(self.device)
               
        normalization = torch.ones(s.shape).to(self.device)
        noise_ = normalization*noise_scale*torch.randn_like(s,device=self.device)
        s_noise = s+noise_   
#         s_noise = nn.Dropout(0.1*random.random())(s_noise)
            
        baseBERT.eval()
        s_emb = self.stc2emb(s_noise)
        s_emb_org = self.stc2emb(s)
        
        CLS_emb = baseBERT.bert_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.cls_token_id],device=self.device))
        SEP_emb = baseBERT.bert_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.sep_token_id],device=self.device))
        PAD_emb = baseBERT.bert_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.pad_token_id],device=self.device))

        normalization_c = torch.ones(c_emb_.shape).to(self.device)
        cc_noise_ = normalization_c*0.0*noise_scale*torch.randn_like(c_emb_,device=self.device)
        c_emb = c_emb_+cc_noise_  
#         c_emb = nn.Dropout(0.01*random.random())(c_emb)

        normalization_h = torch.ones(h_emb_mean_.shape).to(self.device)
        hx_noise_ = normalization_h*0.0*noise_scale*torch.randn_like(h_emb_mean_,device=self.device)
        h_emb_mean = h_emb_mean_+hx_noise_
#         h_emb_mean = nn.Dropout(0.01*random.random())(h_emb_mean)        

        c_emb_emb = self.emb_emb(c_emb)
        h_emb_emb = self.emb_emb(h_emb_mean)

        CLS_emb_emb = self.emb_emb(CLS_emb)
        SEP_emb_emb = self.emb_emb(SEP_emb)

        CLS_emb_emb = CLS_emb_emb.expand(c_emb_emb.shape)
        SEP_emb_emb = SEP_emb_emb.expand(c_emb_emb.shape)

        CLS_emb_emb.unsqueeze_(1)
        SEP_emb_emb.unsqueeze_(1)        
        
        if use_pi:
            p_emb_ = inputs['piemb']
            p_emb_ = p_emb_.to(self.device)
            
            normalization_p = torch.ones(p_emb_.shape).to(self.device)
            pi_noise_ = normalization_p*0.1*noise_scale*torch.randn_like(p_emb_,device=self.device)
            p_emb = p_emb_+pi_noise_  
            p_emb = nn.Dropout(0.1)(p_emb) 
            
            p_emb_emb = self.emb_emb(p_emb)
            
            expand_data_sz = 1
            
            input_emb = torch.cat([CLS_emb_emb,s_emb.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),p_emb_emb.unsqueeze(1)],dim=1)
            input_emb_org = torch.cat([CLS_emb_emb,s_emb_org.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),p_emb_emb.unsqueeze(1)],dim=1)
           
            nohx = inputs['stack_hx_n'] < 2
            attention_mask = torch.ones(input_emb.shape[:2],device=self.device)
            for i,e in enumerate(nohx):
                if e:
                    attention_mask[i,-1-expand_data_sz] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio:
                        attention_mask[i,-1-expand_data_sz] = 0
            pi_num = inputs['pi_num']
            nopi = pi_num < 1            
            for i,e in enumerate(nopi):            
                if e:
                    attention_mask[:,-1] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio_pi:
                        attention_mask[i,-1] = 0                    
       
        else:
            p_emb = None
            pi_num = None
            expand_data_sz = 0
            if expand_data is None:
                input_emb = torch.cat([CLS_emb_emb,s_emb.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1)],dim=1)
                input_emb_org = torch.cat([CLS_emb_emb,s_emb_org.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1)],dim=1)
            else:
                input_emb = torch.cat([CLS_emb_emb,s_emb.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),expand_data['emb']],dim=1)
                expand_data_sz = expand_data['emb'].shape[1]
                input_emb_org = torch.cat([CLS_emb_emb,s_emb_org.unsqueeze(1),c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),expand_data['emb']],dim=1)
                expand_data_sz = expand_data['emb'].shape[1]

            nohx = inputs['stack_hx_n'] < 2
            attention_mask = torch.ones(input_emb.shape[:2],device=self.device)
            for i,e in enumerate(nohx):
                if e:
                    attention_mask[i,-1-expand_data_sz] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio:
                        attention_mask[i,-1-expand_data_sz] = 0
            if expand_data is not None:
                attention_mask[:,-1*expand_data_sz:] = expand_data['mask']

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0   

        output = self.embeddings(input_emb,token_type_ids)
        output = self.encoder(output,extended_attention_mask)

        EDisease = output[:,:1]

        output = self.EDis(EDisease)
        return output,EDisease,(s,input_emb,input_emb_org),(CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,pi_num), inputs['stack_hx_n'],expand_data 


if __name__ == '__main__':
    pass

    tensor = torch.arange(0,6)
    tensor_b = (1/96)*tensor
    tensor_c = (1/192)*tensor
    inputs = torch.stack([tensor,tensor_b,tensor_c])
        
    config = EDiseaseConfig()
    structure_emb = structure_emb(config)
    attention_mask = torch.ones(inputs.shape[:2],dtype=torch.long)
    position_ids = torch.ones(inputs.shape[:2],dtype=torch.long)
    
    output = structure_emb(inputs=inputs,
                  attention_mask=attention_mask,
                  position_ids=position_ids)

    import AIED_dataloader_nhamcs
    model_name = "bert-base-multilingual-cased"
    
    adjBERT = adjBERTmodel(bert_ver=model_name,
                           embedding_size=96,
                           fixBERT=False)
    
    BERT_tokenizer = AutoTokenizer.from_pretrained(model_name)

    all_datas = AIED_dataloader_nhamcs.load_datas()

    data15_triage_train = all_datas['data15_triage_train']
    data01_person = all_datas['data01_person']
    data02_wh = all_datas['data02_wh']
    data25_diagnoses = all_datas['data25_diagnoses']
    dm_normalization_np = all_datas['dm_normalization_np']
    data15_triage_train = all_datas['data15_triage_train']

    EDEW_DS = AIED_dataloader_nhamcs.EDEW_Dataset(ds=data15_triage_train,
                           tokanizer = BERT_tokenizer,
                           data01_person = data01_person,
                           data02_wh = data02_wh,
                           data25_diagnoses= data25_diagnoses,
                           normalization = dm_normalization_np, 
                          )

    EDEW_DL = DataLoader(dataset = EDEW_DS,
                             shuffle = True,
                             num_workers=8,
                             batch_size=8,
                             collate_fn=AIED_dataloader_nhamcs.collate_fn)

    

    config = EDiseaseConfig()
            
    

    
    device = 'cuda'
    test_model = ewed_Model(config=config,
                            tokanizer=BERT_tokenizer,
                            device=device)
    
    test_model.to(device)
    
    
    
    for batch_idx, sample in enumerate(EDEW_DL):                  
        y = test_model(sample)
        print(batch_idx,)

        if batch_idx > 2:
            break



