# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 18:31:00 2022

@author: scottjhy
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.nn import init
import numpy as np
from torch.autograd import Variable

def masked_softmax(vec, mask, dim=-1):
    masked_vec = vec * mask.float()
    
    ## choose the max in masked_vec exculding those indices not masked
    with torch.no_grad():
        tmp_mask=1.0/mask.float()
        # print(tmp_mask)
        masked_max_vec=vec+(mask.float()-tmp_mask*torch.sign(tmp_mask))
        # print(masked_max_vec)
        max_vec = torch.max(masked_max_vec, dim=dim, keepdim=True)[0]
        # print(max_vec)
    # print(max_vec.size())
    
    exps = torch.exp((masked_vec-max_vec)*mask.float())
    masked_exps = (exps) * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/(masked_sums)

class policy_network(nn.Module):
    def __init__(self,input_num,output_num,n_hid,n_layers,dropout=0.0):
        ## input_num = state_dim + action_dim (800 + 360)
        ## output_num = action_dim (360)
        ## n_hid = 256
        ## n_layers = 3
        
        super(policy_network, self).__init__()
        self.input_num=input_num
        self.output_num=output_num
        self.n_hid=n_hid
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        
        ## MLP for s_t to x_t(GRU current information)
        self.linear1=torch.nn.Linear(input_num,n_hid)
        self.linear1_1=torch.nn.Linear(n_hid,n_hid)
        
        ## MLP for h_t to a_t(angle)
        self.linear2=torch.nn.Linear(n_hid,n_hid)
        self.linear_out=torch.nn.Linear(n_hid,output_num)
        
        ## from angle and hidden state to freq
        self.linear1_freq=torch.nn.Linear(n_hid+self.output_num,n_hid)
        self.linear1_1_freq=torch.nn.Linear(n_hid,n_hid)
        self.linear_out_freq=torch.nn.Linear(n_hid,2)
        
        
        ## what's this part?
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.in1_freq=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1_freq=torch.nn.InstanceNorm1d(n_hid)
        self.init_weights()
        
        ## note the starting position of used angle distribution
        self.start_act=self.input_num-self.output_num
        self.noise=1e-1

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.1)
        torch.nn.init.orthogonal_(self.linear1_freq.weight)
        torch.nn.init.constant_(self.linear1_freq.bias,0.1)
        torch.nn.init.orthogonal_(self.linear_out_freq.weight)
        torch.nn.init.constant_(self.linear_out_freq.bias,0.1)
        torch.nn.init.orthogonal_(self.linear1_1_freq.weight)
        torch.nn.init.constant_(self.linear1_1_freq.bias,0.1)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.1)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
    
    def init_hidden(self, batch_size):
        ## initial size is (3 * batch_size(1) * 256)
        return torch.zeros(self.n_layers, batch_size, self.n_hid).to(self.device)

    
    ## from h_{t-1} and s_t to h_t(hidden2) and a_t(angle)(out)
    def forward(self,state,hidden):
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        rnn_out,hidden2=self.gru(out,hidden)
        out=torch.relu(self.in2(self.linear2(rnn_out)))
        out=self.linear_out(out)
        # out1=torch.softmax(out,dim=2)
        return out,rnn_out,hidden2
    
    ## get doze distribution
    def forward2(self,out,one_hot):
        input=torch.cat([out,one_hot],dim=2)
        out1=torch.relu(self.in1_freq(self.linear1_freq(input)))
        out1=torch.relu(self.in1_1_freq(self.linear1_1_freq(out1)))
        out1=self.linear_out_freq(out1)
        batch_size=out1.size()[1]
        mean=out1[:,:,0].view(-1,batch_size)
        std=torch.exp(out1[:,:,1])#+self.noise
        std=std.view(-1,batch_size)
        return mean,std
    
    
    ## compute action
    def act(self,state,hidden,choose_act,deterministic=False):
        ## compute angle distribution and sample
        out1,rnn_out,_=self.forward(state,hidden)
        mask=torch.zeros(len(choose_act)).to(self.device)
        mask[abs(choose_act)<1e-8]=1.0
        out1=masked_softmax(out1,mask)
        angle=torch.multinomial(out1[0,0,:], 5)
        #print(angle)
        
        ## compute freq distribution
        angle_hot=(F.one_hot(angle,self.output_num)).sum(axis=0).type_as(rnn_out).detach()
        angle_hot=angle_hot.view(1,1,self.output_num)
        mean,std=self.forward2(rnn_out,angle_hot)
        normal1 = Normal(mean, std)
        
        if deterministic:
            angle=out1[0,0,:].sort(descending=True)[1][0:5]
            if torch.cuda.is_available():
                angle=angle.cpu()
                mean=mean.cpu()
            freq=mean
            return angle.numpy(),freq.numpy()[0,0]

        if torch.cuda.is_available():
            angle=angle.cpu()
            freq=normal1.sample().cpu()
        else:
            freq=normal1.sample()
        return angle.numpy(),freq.numpy()[0,0]
    


    def prob(self,state,hidden,act,freq):
        ## state = (collected new data(800), used angle distibution(360), used doze(1))
        
        ## from h_{t-1} and s_t to h_t(rnn_out) and a_t(angle)(out)
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        
        ## get used angles and mask the used position as 0
        one_hot=state[:,:,self.start_act:]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[abs(one_hot)<1e-8]=1
        # out1=mask*out1+1e-10
        # p=torch.sum(out1,dim=2).unsqueeze(2)
        # out1=out1/p
        # out1=torch.sum(act*out1,dim=2)
        
        
        out1=masked_softmax(out1,mask)
        out1=act*out1
        mask2=torch.zeros(one_hot.size()).to(self.device)
        mask2[out1==0]=1
        out1=out1+mask2
        out1=out1.prod(dim=-1)
        mean,std=self.forward2(rnn_out,act)
        freq=freq.view(-1,batch_size)
        normal1=Normal(mean,std)
        log_p1=(out1+1e-10).log()
        log_p2=normal1.log_prob(freq).view(-1,batch_size)
        p1=out1
        p2=log_p2.exp()
        
        ## p1 = p_{\pi_{\theta}}(angle | state), p2 is p_{\pi_{\theta}}(freq | state)
        return p1,p2,log_p1,log_p2,hidden2



class critic_network(nn.Module):
    def __init__(self,input_num,n_hid,n_layers,dropout=0.0):
        super(critic_network, self).__init__()
        self.input_num=input_num
        self.n_hid=n_hid
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.linear1=nn.Linear(input_num,n_hid)
        self.linear1_1=nn.Linear(n_hid,n_hid)
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
        self.linear2=nn.Linear(n_hid,n_hid)
        self.linear_out=nn.Linear(n_hid,1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.orthogonal_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear1_1.weight)
        torch.nn.init.constant_(self.linear1_1.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0.0)
        torch.nn.init.orthogonal_(self.linear_out.weight)
        torch.nn.init.constant_(self.linear_out.bias, 0.0)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                torch.nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                torch.nn.init.orthogonal_(param)
        
        

    def init_hidden(self, bsz):
        #weight = next(self.parameters())
        return torch.zeros(self.n_layers, bsz, self.n_hid).to(self.device)

    def forward(self,state,hidden):
        #print("state size{}".format(state.size()))
        batch_size=state.size()[1]
        out=torch.relu(self.in1(self.linear1(state)))
        out=torch.relu(self.in1_1(self.linear1_1(out)))
        out,hidden2=self.gru(out,hidden)
        out=torch.relu(self.in2(self.linear2(out)))
        out=self.linear_out(out).view(-1,batch_size)
        return out,hidden2
