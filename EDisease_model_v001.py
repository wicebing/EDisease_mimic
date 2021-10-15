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
    def __init__(self, bert_ver, fixBERT=True):
        super(adjBERTmodel, self).__init__()
        
        self.Emodel = AutoModel.from_pretrained(bert_ver)
        self.config = self.Emodel.config


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
        
        return heads


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
    def __init__(self,T_config,S_config,tokanizer,device='cpu'):
        super(EDisease_Model, self).__init__() 
        self.stc2emb = structure_emb(S_config)
        self.emb_emb = emb_emb(T_config)
                    
        self.Config = BertConfig()
        self.Config.hidden_size = T_config.hidden_size
        self.Config.num_hidden_layers = T_config.num_hidden_layers
        self.Config.intermediate_size = T_config.intermediate_size
        self.Config.num_attention_heads = T_config.num_attention_heads
        self.Config.max_position_embeddings = T_config.max_position_embeddings
        self.Config.type_vocab_size= T_config.type_vocab_size
        self.Config.vocab_size=T_config.vocab_size
        
        self.EDisease_Transformer = BertModel(self.Config)

        self.tokanizer = tokanizer
        self.device = device
        
        '''
        if fixpretrain:
            for param in self.baseBERT.parameters():
                param.requires_grad = False
        '''
    def forward(self,
                baseBERT,
                inputs,
                normalization=None, 
                noise_scale=0.001,
                mask_ratio=0.15, 
                mask_ratio_pi=0.5,
                token_type_ids=None, 
                expand_data=None,
                use_pi=False,
                test=False):
               
        s,c,cm,h,hm = inputs['structure'],inputs['cc'],inputs['mask_cc'],inputs['ehx'],inputs['mask_ehx']
        s,c,cm,h,hm = s.to(self.device),c.to(self.device),cm.to(self.device),h.to(self.device),hm.to(self.device)
        
        sp, sm = inputs['structure_position_ids'], inputs['structure_attention_mask']
        sp, sm = sp.to(self.device), sm.to(self.device)
               
        if normalization is None:
            s_noise = s
        else:
            #normalization = torch.tensor(normalization).expand(s.shape).to(self.device)
            normalization = torch.ones(s.shape).to(self.device)
            noise_ = normalization*noise_scale*torch.randn_like(s,device=self.device)
            s_noise = s+noise_
            
        baseBERT.eval()
        s_emb = self.stc2emb(inputs=s_noise,
                             attention_mask=sm,
                             position_ids=sp)
        s_emb_org = self.stc2emb(inputs=s,
                                 attention_mask=sm,
                                 position_ids=sp)
        c_emb = baseBERT(c.long(),cm.long())
        h_emb = baseBERT(h.long(),hm.long())
        
        CLS_emb = baseBERT.Emodel.base_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.cls_token_id],device=self.device))
        SEP_emb = baseBERT.Emodel.base_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.sep_token_id],device=self.device))
        PAD_emb = baseBERT.Emodel.base_model.embeddings.word_embeddings(torch.tensor([self.tokanizer.pad_token_id],device=self.device))
                
        cumsum_hx_n = torch.cumsum(inputs['stack_hx_n'],0)
        h_emb_mean_ = []
        for i,e in enumerate(cumsum_hx_n):            
            if inputs['stack_hx_n'][i]>1:
                h_mean = torch.mean(h_emb[1:cumsum_hx_n[i]],dim=0) if i < 1 else torch.mean(h_emb[1+cumsum_hx_n[i-1]:cumsum_hx_n[i]],dim=0)
                h_emb_mean_.append(h_mean)
            else:
                h_emb_mean_.append(PAD_emb.view(h_emb[0].shape))
                
        h_emb_mean = torch.stack(h_emb_mean_)
        h_emb_mean.to(self.device)
               
        c_emb_emb = self.emb_emb(c_emb)
        h_emb_emb = self.emb_emb(h_emb_mean)
               
        CLS_emb_emb = self.emb_emb(CLS_emb)
        SEP_emb_emb = self.emb_emb(SEP_emb)
        
        CLS_emb_emb = CLS_emb_emb.expand(c_emb_emb.shape)
        SEP_emb_emb = SEP_emb_emb.expand(c_emb_emb.shape)

        CLS_emb_emb.unsqueeze_(1)
        SEP_emb_emb.unsqueeze_(1)

        if use_pi:
            pi,pm,pil,yespi = inputs['pi'],inputs['mask_pi'],inputs['origin_pi_length'],inputs['yesPI']
            pi,pm,pil,yespi = pi.to(self.device),pm.to(self.device),pil.to(self.device),yespi.to(self.device)       
            p_emb = baseBERT(pi.long(),pm.long())
            
            p_emb_emb = self.emb_emb(p_emb)

            expand_data_sz = 1
            input_emb = torch.cat([CLS_emb_emb,s_emb,c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),p_emb_emb.unsqueeze(1)],dim=1)
            input_emb_org = torch.cat([CLS_emb_emb,s_emb_org,c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),p_emb_emb.unsqueeze(1)],dim=1)
            
            nohx = inputs['stack_hx_n'] < 2
            attention_mask = torch.ones(input_emb.shape[:2],device=self.device)
            for i,e in enumerate(nohx):
                if e:
                    attention_mask[i,-1-expand_data_sz] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio:
                        attention_mask[i,-1-expand_data_sz] = 0            
            
            nopi = inputs['yesPI'] < 1
            for i,e in enumerate(nopi):
                if e:
                    attention_mask[i,-1] = 0
                else:
                    rd = random.random()
                    if rd < mask_ratio_pi:
                        attention_mask[i,-1] = 0                               
        else:  
            p_emb = None
            yespi = None
            expand_data_sz = 0
            if expand_data is None:
                input_emb = torch.cat([CLS_emb_emb,s_emb,c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1)],dim=1)
                input_emb_org = torch.cat([CLS_emb_emb,s_emb_org,c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1)],dim=1)
            else:
                input_emb = torch.cat([CLS_emb_emb,s_emb,c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),expand_data['emb']],dim=1)
                expand_data_sz = expand_data['emb'].shape[1]
                input_emb_org = torch.cat([CLS_emb_emb,s_emb_org,c_emb_emb.unsqueeze(1),h_emb_emb.unsqueeze(1),expand_data['emb']],dim=1)
                expand_data_sz = expand_data['emb'].shape[1]

            nohx = inputs['stack_hx_n'] < 2
            attention_mask = torch.ones(input_emb.shape[:2],device=self.device)
            for i,e in enumerate(nohx):
                if e:
                    attention_mask[i,-1-expand_data_sz] = 0
                else:
                    if test:
                        pass
                    else:
                        rd = random.random()
                        if rd < mask_ratio:
                            attention_mask[i,-1-expand_data_sz] = 0
            if expand_data is not None:
                attention_mask[:,-1*expand_data_sz:] = expand_data['mask']
            position_ids = torch.arange(4,device=self.device).view(1,-1)
            position_ids = position_ids.expand(attention_mask.shape)

        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0   

        output = self.EDisease_Transformer(inputs_embeds = input_emb,
                                           attention_mask = attention_mask,
                                           position_ids = position_ids,
                                           token_type_ids = token_type_ids,
                                           return_dict=True)

        last_hidden_states = output.last_hidden_state 
        
        EDisease = last_hidden_states[:,0,:]

        return output,EDisease, (s,input_emb,input_emb_org), (CLS_emb_emb,SEP_emb_emb),(c_emb,h_emb_mean,p_emb,yespi), inputs['stack_hx_n'],expand_data

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
    def __init__(self,T_config,device):
        super(GnLD, self).__init__()
        self.device = device
        
        self.Config = BertConfig()
        self.Config.hidden_size = T_config.hidden_size
        self.Config.num_hidden_layers = T_config.num_hidden_layers
        self.Config.intermediate_size = T_config.intermediate_size
        self.Config.num_attention_heads = T_config.num_attention_heads
        self.Config.max_position_embeddings = T_config.max_position_embeddings
        self.Config.type_vocab_size= T_config.type_vocab_size
        self.Config.vocab_size=T_config.vocab_size
        
        self.EDisease_Transformer = BertModel(self.Config)
        self.classifier = classifier(self.Config)
        
    def forward(self, 
                EDisease,
                M, 
                SEP_emb_emb, 
                nohx,
                position_ids,
                attention_mask,
                token_type_ids=None, 
                mask_ratio=0.15):
        bs = EDisease.shape[0]
               
        EM = torch.cat([M[:,:1],EDisease,SEP_emb_emb,M[:,1:]],dim=1)
        
        new_position_ids = torch.cat([position_ids[:,:3],position_ids[:,1:]+10],dim=1)

        input_shape = EM.size()[:-1]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.device)
            token_type_ids[:,2:] = 1

        attention_mask = torch.ones(EM.shape[:2],device=self.device)

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
    def __init__(self,config,device):
        super(PriorD, self).__init__()
        self.config = config
        self.device = device
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
    def __init__(self,T_config,device='cpu',alpha=1, beta=1, gamma=10):
        super(DIM, self).__init__()
        self.T_config = T_config
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.GnLD = GnLD(T_config,device)
        # self.PriorD = PriorD(T_config,device)
        
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
    
    def target_real_fake(batch_size, device, soft):
        t = torch.ones(batch_size,1,device=device) 
        return soft*t, 1 - soft*t, t, 1-t
               
    def forward(self, 
                EDisease,
                M,
                SEP_emb_emb,
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
        
        bs = EDisease.shape[0]
        EDiseaseFake = torch.cat([EDisease[1:],EDisease[:1]],dim=0)
 
        fake_domain, true_domain, fake_em, true_em = self.target_real_fake(batch_size=bs, device=self.device, soft=soft)
        
        criterion_DANN = nn.MSELoss().to(self.device)
        criterion_em = nn.CrossEntropyLoss().to(self.device)
        #using Transformer to similar Global + Local diversity
        
        if self.alpha ==0:
            GLD_loss = torch.tensor(0)            
            GLD0_loss = torch.tensor(0)
            GLD1_loss = torch.tensor(0)

        else:
            GLD0 = self.GnLD(EDisease, 
                             M, 
                             SEP_emb_emb, 
                             nohx,
                             position_ids,
                             attention_mask,
                             token_type_ids=None,
                             mask_ratio=mask_ratio)
            GLD1 = self.GnLD(EDiseaseFake, 
                             M, 
                             SEP_emb_emb, 
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
        if self.beta ==0:
            loss_simCLR = torch.tensor(0)
        else:
            if EDisease2 is not None:
                Eall = torch.cat([EDisease.view(bs,-1),EDisease2.view(bs,-1)],dim=0)
                nEall = F.normalize(Eall,dim=1)
                simCLR = (1.+ep/150)*torch.mm(nEall,nEall.T)
                simCLR = simCLR - 1e3*torch.eye(simCLR.shape[0],device=self.device)

                simtrg= torch.arange(2*bs,dtype=torch.long,device=self.device)
                simtrg = torch.cat([simtrg[bs:],simtrg[:bs]])

                loss_simCLR = self.beta*criterion_em(simCLR,simtrg)
            else:
                loss_simCLR = torch.tensor(0)
                           
        #using GAN method for train prior       
        fake_domain+=(1.1*(1-soft)*torch.rand_like(fake_domain,device=self.device))
        true_domain-=(1.1*(1-soft)*torch.rand_like(true_domain,device=self.device))        
             
        # Proir setting
        '''
        owing to y= x ln x, convex function, a+b+c=1; a,b,c>0, <=1; when a=b=c=1/3, get the min xln x
        set prior=[-1,1] uniform
        '''
        prior = torch.rand_like(EDisease.view(bs,-1),device=self.device)
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
 
                print(EDisease[0,0,:24])
                print(EDisease[1,0,:24])
                if self.alpha >0:                 
                    print('GLD0',GLD0[:2])#,true_em,true_domain)
                    print('GLD1',GLD1[:2])#,fake_em,fake_domain)
                    print('Cts0',Contrast0[:2])#,true_em,true_domain)
                    print('Cts1',Contrast1[:2])#,fake_em,fake_domain)
                if self.beta >0:
                    print('Sim',simCLR[bs-1:bs+5,:8])
                    print('Strg',simtrg[bs-4:bs+8])
                
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



