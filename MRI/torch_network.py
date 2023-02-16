import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.nn import init
import numpy as np
from torch.autograd import Variable

def masked_softmax(vec,mask,dim=-1):
    masked_vec = vec * mask.float()
    ## choose the max in masked_vec exculding those indices not masked
    with torch.no_grad():
        tmp_mask=1.0/mask.float()
        masked_max_vec=vec+(mask.float()-tmp_mask*torch.sign(tmp_mask))
        max_vec = torch.max(masked_max_vec, dim=dim, keepdim=True)[0]
        if max_vec.cpu()[0].numpy()==torch.tensor([float("inf")]):
            print("wrong!")
            print("wrong!")
            print("wrong!")
            print("vec is",vec)
            print("mask is",mask)
    
    exps = torch.exp((masked_vec-max_vec)*mask.float())
    masked_exps = (exps) * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros=(masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps/(masked_sums)

class policy_network(nn.Module):
    def __init__(self,input_num,output_num,n_hid,n_layers,dropout=0.0):
        ## input_num = state_dim + action_dim (800 + 360)
        ## output_num = action_dim (360 --> number of pixels n1*n2)
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
        self.linear1=torch.nn.Linear(input_num+1,n_hid)
        self.linear1_1=torch.nn.Linear(n_hid,n_hid)
        
        ## MLP for h_t to a_t(angle --> fourier coefficient)
        self.linear2=torch.nn.Linear(n_hid,n_hid)
        self.linear_out=torch.nn.Linear(n_hid,output_num)
        
        ## what's this part?
        self.in1=torch.nn.InstanceNorm1d(n_hid)
        self.in1_1=torch.nn.InstanceNorm1d(n_hid)
        self.in2=torch.nn.InstanceNorm1d(n_hid)
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
    
    
    ## compute action
    def act(self,state,hidden,choose_act,deterministic=False):
        ## compute angle distribution and sample
        out1,rnn_out,_=self.forward(state,hidden)
        mask=torch.zeros(len(choose_act)).to(self.device)
        mask[abs(choose_act)<1e-8]=1.0
        out1=masked_softmax(out1,mask)
        out1+=1/360*mask
        out1=out1/torch.sum(out1,dim=2,keepdim=True)
        dist1=Categorical(out1)
        angle=dist1.sample()

        if deterministic:
            angle=out1[0,0,:].sort(descending=True)[1][0]
            if torch.cuda.is_available():
                angle=angle.cpu()
            return angle.numpy().reshape([1,])

        if torch.cuda.is_available():
            angle=angle.cpu()
        return angle.numpy()[0]
    


    def prob(self,state,hidden,act):
        ## state = (collected new data(800), used angle distibution(360), used doze(1))
        
        ## from h_{t-1} and s_t to h_t(rnn_out) and a_t(angle)(out)
        out1,rnn_out,hidden2=self.forward(state,hidden)
        batch_size=out1.size()[1]
        
        ## get used angles and mask the used position as 0
        one_hot=state[:,:,self.start_act:-1]
        mask=torch.zeros(one_hot.size()).to(self.device)
        mask[abs(one_hot)<1e-8]=1
        
        out1=masked_softmax(out1,mask)
        out1+=1/360*mask
        
        out1=out1/torch.sum(out1, dim=2,keepdim=True)
        out1=torch.sum(act*out1,dim=2)
        log_p1=(out1+1e-10).log()
        p1=out1
        
        ## p1 = p_{\pi_{\theta}}(angle | state)
        return p1,log_p1,hidden2



class critic_network(nn.Module):
    def __init__(self,input_num,n_hid,n_layers,dropout=0.0):
        super(critic_network, self).__init__()
        self.input_num=input_num
        self.n_hid=n_hid
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gru=nn.GRU(n_hid, n_hid, n_layers, dropout=dropout)
        self.linear1=nn.Linear(input_num+1,n_hid)
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