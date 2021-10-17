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
        self.emb_emb = emb_emb(T_config)

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
                   'em_heads':self.emb_emb(heads),
                   }
        
        return outputs


class float2spectrum(nn.Module):
    def __init__(self, embedding_size):
        super(float2spectrum, self).__init__()
        self.embedding_size = embedding_size
        
    def forward(self, tensor):
        device = tensor.device
        thida = torch.linspace(0,2*math.pi,int(self.embedding_size/2),device=device)
        k_thida = torch.einsum("nm,k->nmk", tensor, thida)
        emb_x = torch.cat((k_thida.cos(),k_thida.sin()), dim=-1)
        return emb_x        

class structure_emb(nn.Module):
    def __init__(self, config):
        super(structure_emb, self).__init__()
        self.float2emb = float2spectrum(config.hidden_size)
        
        self.Config = BertConfig()
        self.Config.hidden_size = config.hidden_size
        self.Config.num_hidden_layers = config.num_hidden_layers
        self.Config.intermediate_size = config.intermediate_size
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
    
    def forward(self,
                things,
                normalization=None, 
                noise_scale=0.001,
                mask_ratio=0.15, 
                mask_ratio_pi=0.5,
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
                       
        em_CLS = self.EDs_embeddings(torch.tensor([1],device=device))
        em_SEP = self.EDs_embeddings(torch.tensor([2],device=device))
        em_PAD = self.EDs_embeddings(torch.tensor([0],device=device))
        
        em_CLS = em_CLS.expand([bs,em_CLS.shape[-1]])
        em_SEP = em_SEP.expand([bs,em_SEP.shape[-1]])
        em_PAD = em_PAD.expand([bs,em_PAD.shape[-1]])
        
        input_emb = torch.cat([em_CLS.unsqueeze(1),*emb_],dim=1)
        input_emb_org = torch.cat([em_CLS.unsqueeze(1),*emb_],dim=1)
        
        attention_mask = torch.cat([torch.ones([bs,1],device=device),*attention_mask_],dim=1)
        position_ids = torch.cat([torch.zeros([bs,1],device=device),*position_id_],dim=1) 
        
        output = self.EDisease_Transformer(inputs_embeds = input_emb,
                                           attention_mask = attention_mask.long(),
                                           position_ids = position_ids.long(),
                                           return_dict=True)

        last_hidden_states = output.last_hidden_state 
        
        EDisease = last_hidden_states[:,0,:]
        
        outp ={'output':output,
               'EDisease':EDisease,
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
        
    def forward(self, 
                EDisease,
                M, 
                nohx,
                position_ids,
                attention_mask,
                token_type_ids=None, 
                mask_ratio=0.15):
        bs = EDisease.shape[0]
        eds = EDisease.unsqueeze(1)
        device = EDisease.device

        em_CLS = self.EDisease_Transformer.base_model.embeddings.word_embeddings(torch.tensor([1],device=device))
        em_SEP = self.EDisease_Transformer.base_model.embeddings.word_embeddings(torch.tensor([2],device=device))
        em_PAD = self.EDisease_Transformer.base_model.embeddings.word_embeddings(torch.tensor([0],device=device))
        
        em_CLS = em_CLS.expand(EDisease.shape)
        em_SEP = em_SEP.expand(EDisease.shape)
        em_PAD = em_PAD.expand(EDisease.shape)
        
        EM = torch.cat([M[:,:1],eds,em_SEP.unsqueeze(1),M[:,1:]],dim=1)
        
        new_position_ids = torch.cat([position_ids[:,:3],position_ids[:,1:]+10],dim=1)

        input_shape = EM.size()[:-1]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            token_type_ids[:,2:] = 1

        attention_mask = torch.ones(EM.shape[:2],device=device)

        for i,e in enumerate(nohx):
            if e<2:
                attention_mask[i,-1] = 0

            else:
                rd = random.random()
                if rd < mask_ratio:
                    attention_mask[i,-1] = 0              
     
        output = self.EDisease_Transformer(inputs_embeds = EM,
                                           attention_mask = attention_mask,
                                           position_ids = new_position_ids,
                                           token_type_ids = token_type_ids,
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
                EDisease,
                M,
                nohx,
                position_ids,
                attention_mask,
                token_type_ids=None,
                soft=0.7,
                mask_ratio=0.15,
                mode=None,
                ptloss=False,
                EDisease2=None,
                ep=0):
        device = EDisease.device
        bs = EDisease.shape[0]
        EDiseaseFake = torch.cat([EDisease[1:],EDisease[:1]],dim=0)
 
        fake_domain, true_domain, fake_em, true_em = self.target_real_fake(batch_size=bs, soft=soft)
        
        criterion_DANN = nn.MSELoss().to(device)
        criterion_em = nn.CrossEntropyLoss().to(device)
        #using Transformer to similar Global + Local diversity
        
        if self.alpha ==0:
            GLD_loss = torch.tensor(0)            
            GLD0_loss = torch.tensor(0)
            GLD1_loss = torch.tensor(0)

        else:
            GLD0 = self.GnLD(EDisease, 
                             M, 
                             nohx,
                             position_ids,
                             attention_mask,
                             token_type_ids=None,
                             mask_ratio=mask_ratio)
            GLD1 = self.GnLD(EDiseaseFake, 
                             M, 
                             nohx,
                             position_ids,
                             attention_mask,
                             token_type_ids=None,
                             mask_ratio=mask_ratio)

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



