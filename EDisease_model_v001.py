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
from EDisease_config import EDiseaseConfig, StructrualConfig


class adjBERTmodel(nn.Module):
    def __init__(self, bert_ver,T_config, fixBERT=True):
        super(adjBERTmodel, self).__init__()
        
        self.Emodel = AutoModel.from_pretrained(bert_ver)
        self.config = self.Emodel.config
        # self.emb_emb = emb_emb(T_config)

        print('baseBERT PARAMETERS: ' ,count_parameters(self.Emodel))        
        if fixBERT:
            for param in self.Emodel.parameters():
                param.requires_grad = False   
            print(' ** fix pretrained BERT WEIGHT ** ')
            print('baseBERT PARAMETERS: ' ,count_parameters(self.Emodel))
        
    def forward(self, input_ids,attention_mask, position_ids=None,token_type_ids=None,return_dict=True):    
        inputs = {'input_ids':input_ids,
                  'attention_mask':attention_mask}
        outputs = self.Emodel(**inputs,return_dict=return_dict)
        last_hidden_states = outputs.last_hidden_state 
        
        heads = last_hidden_states[:,0,:]
        

        outputs = {'heads':heads,
                   # 'em_heads':self.emb_emb(heads),
                   }
        
        return outputs


class float2spectrum(nn.Module):
    def __init__(self, embedding_size, spectrum_type=None):
        super(float2spectrum, self).__init__()
        self.embedding_size = embedding_size
        self.spectrum_type = spectrum_type
        
    def forward(self, tensor, time=False):
        '''
        limitation: the cycle will happen in the embedding_size
        such as embedding_size =96
        the emb_x will be the same on 1 -> 96, 2-> 97, ...
        due to the minus exist the minmax must be the half of embedding_size to avoid aliasing effect 2x sample freq
        '''
       
        device = tensor.device
        if self.spectrum_type is None:
            minmax = 0.5
            tensor = tensor.clamp(min=-1*minmax,max=minmax)
            thida = torch.linspace(0,math.pi,self.embedding_size,device=device).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            
            if time:
                emb_x = 0.1*k_thida.cos()
            else:
                emb_x = k_thida.sin()
                
        elif self.spectrum_type == 'cossin':
            # experimental 0 [cos x, sin x] auc 0.846
            thida = torch.linspace(0,math.pi,int(self.embedding_size/2),device=device)
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = torch.cat((k_thida.cos(),k_thida.sin()), dim=-1)
                        
        elif self.spectrum_type == 'sigmoid':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,10,self.embedding_size,device=device).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.sigmoid()
            
        elif self.spectrum_type == 'sinh':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,2*math.pi,self.embedding_size,device=device).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.sinh()

        elif self.spectrum_type == 'exp':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,self.embedding_size,device=device).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.exp()
            
        elif self.spectrum_type == 'parabolic':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,self.embedding_size,device=device).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida**2
            
        elif self.spectrum_type == 'linear':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,1,self.embedding_size,device=device).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida
            
        elif self.spectrum_type == 'constant':
            # experimental 2 [transformer position token]
            thida = torch.linspace(1,1,self.embedding_size,device=device).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida

        elif self.spectrum_type == 'sin_period':
            # experimental 2 [transformer position token]
            thida = torch.linspace(0,4*math.pi,self.embedding_size,device=device).float()
            k_thida = torch.einsum("nm,k->nmk", tensor, thida)
            emb_x = k_thida.sin()
        
        return emb_x        

class structure_emb(nn.Module):
    def __init__(self, config, spectrum_type=None):
        super(structure_emb, self).__init__()
        self.float2emb = float2spectrum(config.hidden_size,spectrum_type)
        
        self.Config = BertConfig()
        self.Config.hidden_size = config.hidden_size
        self.Config.num_hidden_layers = config.num_hidden_layers
        self.Config.intermediate_size = config.intermediate_size
        self.Config.num_attention_heads = config.num_attention_heads
        self.Config.max_position_embeddings = config.max_position_embeddings
        self.Config.type_vocab_size= config.type_vocab_size
        self.Config.vocab_size=config.vocab_size
        
        self.BERTmodel = BertModel(self.Config)

    def forward(self, inputs,attention_mask,position_ids,time_ids=None,token_type_ids=None):
        if time_ids is None:
            inputs_embeds = self.float2emb(0.05*inputs)
        else:
            inputs_embeds = self.float2emb(0.05*inputs) + self.float2emb(0.1*time_ids,time=True)            
            
        outputs = self.BERTmodel(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 token_type_ids=token_type_ids)
        
        return outputs[0][:,:1,:]

class structure_emb_mlp(nn.Module):
    def __init__(self, config):
        super(structure_emb_mlp, self).__init__()
        self.stc2emb_0 = nn.Sequential(nn.Linear(config.structure_size,4*config.hidden_size),
                                     nn.LayerNorm(4*config.hidden_size),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(4*config.hidden_size,4*config.hidden_size),
                                     nn.LayerNorm(4*config.hidden_size),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     )
        self.stc2emb_1 = nn.Sequential(nn.Linear(4*config.hidden_size,4*config.hidden_size),
                                     nn.LayerNorm(4*config.hidden_size),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(4*config.hidden_size,4*config.hidden_size),
                                     nn.LayerNorm(4*config.hidden_size),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     )
        self.stc2emb_2 = nn.Sequential(nn.Linear(4*config.hidden_size,2*config.hidden_size),
                                     nn.LayerNorm(2*config.hidden_size),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(2*config.hidden_size,config.hidden_size),
                                     nn.LayerNorm(config.hidden_size),
                                     )

    def forward(self, inputs,attention_mask=None,position_ids=None,token_type_ids=None):
        pooled_output = self.stc2emb_0(inputs)
        for i in range(4):
            hidden = self.stc2emb_1(pooled_output)
            pooled_output = pooled_output + hidden
        pooled_output = self.stc2emb_2(pooled_output)
        return pooled_output.unsqueeze(1)

class emb_emb(nn.Module):
    def __init__(self, config):
        super(emb_emb, self).__init__()
        self.emb_emb = nn.Sequential(nn.Linear(config.bert_hidden_size,2*config.hidden_size),
                                     nn.LayerNorm(2*config.hidden_size),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(2*config.hidden_size,config.hidden_size),
                                     nn.LayerNorm(config.hidden_size),
                                     )

    def forward(self, hidden_states):
        pooled_output = self.emb_emb(hidden_states)
        return pooled_output    

class EDisease_Model(nn.Module):
    def __init__(self,T_config,S_config):
        super(EDisease_Model, self).__init__()                     
        self.Config = BertConfig()
        self.Config.hidden_size = T_config.hidden_size
        self.Config.num_hidden_layers = T_config.num_hidden_layers
        self.Config.intermediate_size = T_config.intermediate_size
        self.Config.num_attention_heads = T_config.num_attention_heads
        self.Config.max_position_embeddings = T_config.max_position_embeddings
        self.Config.type_vocab_size= T_config.type_vocab_size
        self.Config.vocab_size=T_config.vocab_size
        
        self.EDisease_Transformer = BertModel(self.Config)
        
        self.EDs_embeddings = nn.Embedding(T_config.vocab_size, T_config.hidden_size)
        
        self.classifier = classifier(T_config)
    
    def forward(self,
                things,
                mask_ratio=0.15, 
                token_type_ids=None, 
                test=False):
        
        
        emb_ = []
        attention_mask_ = []
        position_id_ = []
        
        for k,v in things.items():
            emb_.append(v['emb'])
            attention_mask_.append(v['attention_mask'])
            position_id_.append(v['position_id'])
            
        bs = emb_[0].shape[0]
        
        device = emb_[0].device
                               
        em_CLS = self.EDs_embeddings(1*torch.ones((bs, 1), device=device, dtype=torch.long))
        em_SEP = self.EDs_embeddings(2*torch.ones((bs, 1), device=device, dtype=torch.long))
        em_PAD = self.EDs_embeddings(0*torch.ones((bs, 1), device=device, dtype=torch.long))
                
        input_emb = torch.cat([em_CLS,*emb_],dim=1)
        input_emb_org = torch.cat([em_CLS,*emb_],dim=1)
        
        attention_mask = torch.cat([torch.ones([bs,1],device=device),*attention_mask_],dim=1)
        position_ids = torch.cat([torch.zeros([bs,1],device=device),*position_id_],dim=1) 
        
        output = self.EDisease_Transformer(inputs_embeds = input_emb,
                                           attention_mask = attention_mask.long(),
                                           position_ids = position_ids.long(),
                                           return_dict=True)

        last_hidden_states = output.last_hidden_state 
        
        EDisease = last_hidden_states[:,0,:]
        
        predict = self.classifier(EDisease)
        
        outp ={'output':output,
               'EDisease':EDisease,
               'predict':predict,
               'input_emb_org':input_emb_org,
               'position_ids':position_ids,
               'attention_mask':attention_mask
               }

        return outp

class classifier(nn.Module):
    def __init__(self, config):
        super(classifier, self).__init__()
        self.avgpool = nn.Sequential(nn.Linear(config.hidden_size,4*config.hidden_size),
                                     nn.LayerNorm(4*config.hidden_size),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(4*config.hidden_size,2*config.hidden_size),
                                     nn.LayerNorm(2*config.hidden_size),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(2*config.hidden_size,config.classifier_num)
                                     )

    def forward(self, hidden_states):
        pooled_output = self.avgpool(hidden_states)
        return pooled_output

class GnLD(nn.Module):
    def __init__(self,T_config):
        super(GnLD, self).__init__()
        
        self.Config = BertConfig()
        self.Config.hidden_size = T_config.hidden_size
        self.Config.num_hidden_layers = T_config.num_hidden_layers
        self.Config.intermediate_size = T_config.intermediate_size
        self.Config.num_attention_heads = T_config.num_attention_heads
        self.Config.max_position_embeddings = T_config.max_position_embeddings
        self.Config.type_vocab_size= T_config.type_vocab_size
        self.Config.vocab_size=T_config.vocab_size
        
        self.EDisease_Transformer = BertModel(self.Config)
        self.classifier = classifier(T_config)
        self.EDs_embeddings = nn.Embedding(T_config.vocab_size, T_config.hidden_size)
        
    def forward(self,
                things,
                things_e,
                mask_ratio=0.15,
                token_type_ids=None,
                fake=False):
        EDisease = things_e['e']['emb'].squeeze(1)

        bs = EDisease.shape[0]
        device = EDisease.device

        emb_ = []
        attention_mask_ = []
        position_id_ = []
        
        for k,v in things.items():
            emb_.append(v['emb'])
            attention_mask_.append(v['attention_mask'])
            position_id_.append(v['position_id'])

        e_emb_ = []
        e_attention_mask_ = []
        e_position_id_ = []
        
        for k,v in things_e.items():
            if fake:
                e_emb_.append(v['embf'])
            else:
                e_emb_.append(v['emb'])
            e_attention_mask_.append(v['attention_mask'])
            e_position_id_.append(v['position_id'])            

        em_CLS = self.EDs_embeddings(1*torch.ones((bs, 1), device=device, dtype=torch.long))
        em_SEP = self.EDs_embeddings(2*torch.ones((bs, 1), device=device, dtype=torch.long))
        em_PAD = self.EDs_embeddings(0*torch.ones((bs, 1), device=device, dtype=torch.long))
        
        input_emb = torch.cat([em_CLS,*e_emb_,em_CLS,*emb_],dim=1)
        
        attention_mask = torch.cat([torch.ones([bs,1],device=device),
                                    *e_attention_mask_,
                                    torch.ones([bs,1],device=device),
                                    *attention_mask_
                                    ],dim=1)
        position_ids = torch.cat([torch.zeros([bs,1],device=device),
                                  *e_position_id_,
                                  torch.ones([bs,1],device=device),
                                  *position_id_
                                  ],dim=1) 

        input_shape = input_emb.size()[:-1]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            token_type_ids[:,2:] = 1
               
        output = self.EDisease_Transformer(inputs_embeds = input_emb,
                                           attention_mask = attention_mask.long(),
                                           position_ids = position_ids.long(),
                                           token_type_ids = token_type_ids.long(),
                                           return_dict=True)
        last_hidden_states = output.last_hidden_state 
        
        dim_H = last_hidden_states[:,0,:]
        
        output = self.classifier(dim_H)

        return output

class PriorD(nn.Module):
    def __init__(self,config):
        super(PriorD, self).__init__()
        self.config = config
        self.dense = nn.Sequential(nn.Linear(config.hidden_size,4*config.hidden_size),
                                   nn.LayerNorm(4*config.hidden_size),
                                   nn.GELU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(4*config.hidden_size,2*config.hidden_size),
                                   nn.LayerNorm(2*config.hidden_size),
                                   nn.GELU(),
                                   nn.Dropout(0.5),
                                   nn.LayerNorm(2*config.hidden_size),
                                   nn.Linear(2*config.hidden_size,1),
                                   nn.Sigmoid()
                                   )  
        
    def forward(self, EDisease):
        output = self.dense(EDisease)
        
        return output

class DIM(nn.Module):
    def __init__(self,T_config,alpha=1, beta=1, gamma=10):
        super(DIM, self).__init__()
        self.T_config = T_config
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.GnLD = GnLD(T_config)
        self.PriorD = PriorD(T_config)
        
    def shuffleE(self,EDiseaseFake,bs):
        t = random.randint(1,200)
        for _ in range(t):
            r = random.random()
            s = random.randint(1,min(95,bs-1))
            if r >0.5:
                EDiseaseFake = torch.cat([EDiseaseFake[s:],EDiseaseFake[:s]],dim=0)
            else:
                EDiseaseFake = torch.cat([EDiseaseFake[:,s:],EDiseaseFake[:,:s]],dim=1)
        return EDiseaseFake
    
    def target_real_fake(self, batch_size, soft, device):
        t = torch.ones(batch_size,1,device=device) 
        return soft*t, 1 - soft*t, t, 1-t
               
    def forward(self, 
                things,
                things_e,
                soft=0.7,
                mask_ratio=0.15,
                mode=None,
                ptloss=False,
                ep=0):
        
        EDisease = things_e['e']['emb'].squeeze(1)
        EDisease2 = things_e['e']['emb2'].squeeze(1)
        
        device = EDisease.device
        bs = EDisease.shape[0]
        
        things_e['e']['emb']
 
        fake_domain, true_domain, fake_em, true_em = self.target_real_fake(batch_size=bs, soft=soft,device=device)
        
        criterion_DANN = nn.MSELoss().to(device)
        criterion_em = nn.CrossEntropyLoss().to(device)
        #using Transformer to similar Global + Local diversity
        
        if self.alpha ==0:
            GLD_loss = torch.tensor(0)            
            GLD0_loss = torch.tensor(0)
            GLD1_loss = torch.tensor(0)

        else:
            GLD0 = self.GnLD(things=things,
                             things_e=things_e,
                             mask_ratio=mask_ratio)
            GLD1 = self.GnLD(things=things,
                             things_e=things_e,
                             mask_ratio=mask_ratio,
                             fake= True)

            Contrast0 = torch.cat([GLD0[:,:1],GLD1[:,:1]],dim=-1)
            Contrast1 = torch.cat([GLD0[:,1:],GLD1[:,1:]],dim=-1)

            GLD0_loss = criterion_em(Contrast0,true_em.view(-1).long())
            GLD1_loss = criterion_em(Contrast1,fake_em.view(-1).long())

            GLD_loss = self.alpha*(GLD0_loss+GLD1_loss)
        
        #SimCLR
        if (self.beta >0) & (EDisease2 is not None):
            Eall = torch.cat([EDisease.view(bs,-1),EDisease2.view(bs,-1)],dim=0)
            nEall = F.normalize(Eall,dim=1)
            simCLR = (1.+ep/150)*torch.mm(nEall,nEall.T)
            simCLR = simCLR - 1e3*torch.eye(simCLR.shape[0],device=device)

            simtrg= torch.arange(2*bs,dtype=torch.long,device=device)
            simtrg = torch.cat([simtrg[bs:],simtrg[:bs]])

            loss_simCLR = self.beta*criterion_em(simCLR,simtrg)
        else:
            loss_simCLR = torch.tensor(0.)
                           
        #using GAN method for train prior       
        fake_domain+=(1.1*(1-soft)*torch.rand_like(fake_domain,device=device))
        true_domain-=(1.1*(1-soft)*torch.rand_like(true_domain,device=device))        
             
        # Proir setting
        '''
        owing to y= x ln x, convex function, a+b+c=1; a,b,c>0, <=1; when a=b=c=1/3, get the min xln x
        set prior=[-1,1] uniform
        '''
        prior = torch.rand_like(EDisease.view(bs,-1),device=device)
        #prior = (prior - prior.mean())/(prior.std()+1e-6)   #fit to Domain of Layernorm()
        prior = 2*prior-1                                   #fit to Domain of Tanh()
        
        if self.gamma ==0:
            prior_loss = torch.tensor(0)
        else:
            if mode=='D':
                #only train the D , not G
                for param in self.PriorD.parameters():
                    param.requires_grad = True
                #d_EDisease = EDisease.view(bs,-1).detach()
                pred_domain_T = self.PriorD(EDisease.view(bs,-1).detach())
                loss_domain_T = criterion_DANN(pred_domain_T,true_domain)
                pred_domain_F = self.PriorD(prior) 
                loss_domain_F = criterion_DANN(pred_domain_F,fake_domain)          
            elif mode=='G':
                #only train the G , not D
                for param in self.PriorD.parameters():
                    param.requires_grad = False

                pred_domain_T = self.PriorD(EDisease.view(bs,-1))
                loss_domain_T = criterion_DANN(pred_domain_T,fake_domain)
                loss_domain_F = 0

            prior_loss = self.gamma*(loss_domain_T+loss_domain_F)
        
        if ptloss:
            with torch.no_grad():
                if EDisease2 is None:
                    print('GT:{:.4f}, GF:{:.4f}, Prior{:.4f}'.format(GLD0_loss.item(),
                                                                                  GLD1_loss.item(),
                                                                                  prior_loss.item()
                                                                                  ))
                else:
                    print('GT:{:.4f}, GF:{:.4f}, Sim:{:.4f}, Prior{:.4f}'.format(GLD0_loss.item(),
                                                                                 GLD1_loss.item(),
                                                                                 loss_simCLR.item(),
                                                                                 prior_loss.item()))          
 
                print(EDisease[0,:24])
                print(EDisease[1,:24])
                if self.alpha >0:                 
                    print('GLD0',GLD0[:2])#,true_em,true_domain)
                    print('GLD1',GLD1[:2])#,fake_em,fake_domain)
                    print('Cts0',Contrast0[:2])#,true_em,true_domain)
                    print('Cts1',Contrast1[:2])#,fake_em,fake_domain)
                if self.beta >0:
                    print('Sim',simCLR[bs-2:bs+4,:8])
                    print('Strg',simtrg[bs-2:bs+4])
                
        if EDisease2 is None:
            return GLD_loss+prior_loss
        else:
            return GLD_loss+prior_loss+loss_simCLR


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
    



