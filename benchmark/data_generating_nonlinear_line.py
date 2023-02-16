# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 16:58:07 2022

@author: scottjhy
"""

import timeit
import numpy as np
import random
import math
import pydicom as dicom
import os
import copy
import torch
import scipy

from sklearn import linear_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal
from torch.distributions.categorical import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.nn import init
import numpy as np
from torch.autograd import Variable

def read_img(img_path,size):
    """
    walk the dir to load all image
    """
    t=512//size
    img_list=[]
    print('image loading...')
    for _,_,files in os.walk(img_path):
        for f in files:
            if f.find('.IMA')>=0:
                tmp_img=dicom.dcmread(os.path.join(img_path,f))
                tmp_img=(tmp_img.pixel_array[0::t,0::t]-np.mean(tmp_img.pixel_array[0::t,0::t]))
                tmp_img=tmp_img/np.linalg.norm(tmp_img)
                img_list.append(tmp_img)
    img_data=np.array(img_list)
    print('done')
    return img_data


def error_eval(u_pre,u_true):
    e1=np.linalg.norm(u_pre-u_true,'fro')/np.linalg.norm(u_true,'fro')
    return e1

def psnr(u_pre,u_true):
    norm_pre=u_pre/np.max(u_pre)*255
    norm_true=u_true/np.max(u_true)*255
    mse=np.mean((norm_pre-norm_true)**2)
    if mse==0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr2(u_pre,u_true):
    mse=np.mean((u_pre-u_true)**2)
    if mse==0:
        return 100
    PIXEL_MAX = 4096
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def printlog(str1):
    fp=open('./test.out','a')
    fp.write(str1)
    fp.close()

def test(agent,k,r,f,deterministic=True):
    ran = 0
    train = 0
    uni = 0
    train_a = 0
    train_f = 0
    train_win = 0
    l1 = []
    l2 = []
    l3 = []
    l4 = []
    l5 = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    if agent.shape == "mnist":
        data_path = "C:/Users/scottjhy/Desktop/mnist_test.csv"
        data = np.loadtxt(data_path,delimiter=",")
        size = data.shape[0]
    for i in range(k):
        self=agent
        if self.shape == "circle":
            true_img=generate_circles(self.env.width, self.env.size)
        if self.shape == "oval":
            true_img=generate_tri_oval(self.env.width, self.env.size)
        if self.shape == "mnist":
            s = random.sample(range(0,size),1)
            d=data[s[0],1:]
            d=d/np.linalg.norm(d)*2
            d=d.reshape((28,28))
            true_img=d
        if self.shape == "dots":
            true_img=generate_dots(self.env.width, self.env.size)
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        angles=list(range(0,360))
        random.shuffle(angles)
        time_len =100
        #angles[0:20]=[18*i for i in range(20)]
        #angles[0:10]=self.env.angle_seq
        for i in range(time_len):
            #angle,doze=self.action(obs1,rest_doze,old_act,deterministic)
            angle=[angles[i]]
            next_obs,reward,done,_=self.env.step(angle,f)
            if done:
                break

        self = self.env
        state = self.state
        state_seq = self.state_proj_seq
        angle_seq = self.angle_seq
        A_seq = self.A_seq
        eta = reconstruct(self.state, self.state_proj_seq, self.angle_seq, self.A_seq,self.freq,self.freq_list, self.G, iter=10)
        size = np.int(np.sqrt(eta.size//3))
        eta0 = np.reshape(eta, (size,3*size))
        r1 = np.mean((eta0-true_img)**2)
        print("random error:",r1,"psnr:",psnr2(eta0,true_img))               
        l1.append(r1)
        p1.append(psnr2(eta0,true_img))
        ran+=r1/k
        
        self=agent
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        self.start_epoch(1)
        old_act=np.zeros((self.action_dim,))
        rest=r
        for i in range(self.long_time):
            if i!=0:
                obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            else:
                obs_norm=obs
            obs1=np.concatenate([obs_norm,np.zeros(self.state_dim-len(obs_norm)),old_act],axis=0)
            angle,freq,out=self.action(obs1,rest,old_act,deterministic)
            rest-=1
            next_obs,reward,done,_=self.env.step(angle,freq)
            self.next_hidden(obs1,rest+1)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=freq+1
            obs=next_obs
            if done:
                break
        self = self.env
        state = self.state
        state_seq = self.state_proj_seq
        angle_seq = self.angle_seq
        A_seq = self.A_seq
        eta = reconstruct(self.state, self.state_proj_seq, self.angle_seq, self.A_seq,self.freq,self.freq_list, self.G, iter=10)
        print(self.freq)
        print(self.angle_seq)
        size = np.int(np.sqrt(eta.size//3))
        eta = np.reshape(eta, (size,3*size))
        r2 = np.mean((eta-true_img)**2)
        print("trained error:",r2,"psnr:",psnr2(eta,true_img))
        l2.append(r2)
        p2.append(psnr2(eta,true_img))
        train +=r2/k
        
        '''if r1 > r2:
            train_win += 1'''
            
        
        self=agent
        angles=self.env.angle_seq
        freq=self.env.freq
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        time_len =100
        for i in range(time_len):
            #angle,doze=self.action(obs1,rest_doze,old_act,deterministic)
            angle=[angles[i]]
            next_obs,reward,done,_=self.env.step(angle,f)
            if done:
                break

        self = self.env
        state = self.state
        state_seq = self.state_proj_seq
        angle_seq = self.angle_seq
        A_seq = self.A_seq
        eta = reconstruct(self.state, self.state_proj_seq, self.angle_seq, self.A_seq,self.freq,self.freq_list, self.G, iter=20)
        size = np.int(np.sqrt(eta.size//3))
        eta0 = np.reshape(eta, (size,3*size))
        r4 = np.mean((eta0-true_img)**2)
        print("train_a error:",r4,"psnr:",psnr2(eta0,true_img))
        l4.append(r4)
        p4.append(psnr2(eta0,true_img))
        train_a+=r4/k    
        
        self=agent
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        time_len =100
        angles=[360*i/r for i in range(r)]
        for i in range(time_len):
            #angle,doze=self.action(obs1,rest_doze,old_act,deterministic)
            angle=[angles[i]]
            next_obs,reward,done,_=self.env.step(angle,int(freq[i]))
            if done:
                break

        self = self.env
        state = self.state
        state_seq = self.state_proj_seq
        angle_seq = self.angle_seq
        A_seq = self.A_seq
        eta = reconstruct(self.state, self.state_proj_seq, self.angle_seq, self.A_seq,self.freq,self.freq_list, self.G, iter=20)
        size = np.int(np.sqrt(eta.size//3))
        eta0 = np.reshape(eta, (size,3*size))
        r5 = np.mean((eta0-true_img)**2)
        print("train_f error:",r5,"psnr:",psnr2(eta0,true_img))
        l5.append(r5)
        p5.append(psnr2(eta0,true_img))
        train_f+=r5/k 
        
        
        self=agent
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        time_len =100
        angles=[360*i/r for i in range(r)]
        #angles[0:10]=self.env.angle_seq
        for i in range(time_len):
            #angle,doze=self.action(obs1,rest_doze,old_act,deterministic)
            angle=[angles[i]]
            next_obs,reward,done,_=self.env.step(angle,f)
            if done:
                break

        self = self.env
        state = self.state
        state_seq = self.state_proj_seq
        angle_seq = self.angle_seq
        A_seq = self.A_seq
        eta = reconstruct(self.state, self.state_proj_seq, self.angle_seq, self.A_seq,self.freq,self.freq_list, self.G, iter=5)
        size = np.int(np.sqrt(eta.size//3))
        eta0 = np.reshape(eta, (size,3*size))
        r3 = np.mean((eta0-true_img)**2)
        print("uniform error:",r3,"psnr:",psnr2(eta0,true_img))
        l3.append(r3)
        p3.append(psnr2(eta0,true_img))
        uni+=r3/k   
        
        if r2 == min(r1,r2,r3,r4,r5):
            train_win+=1

      
    print("final average trained error",train,"final train error variance",np.var(l2),"psnr:",np.average(p2),np.var(p2))
    print("final average random error",ran, "final random error variance",np.var(l1),"psnr:",np.average(p1),np.var(p1))
    print("final average uniform error",uni,"final train error variance",np.var(l3),"psnr:",np.average(p3),np.var(p3))
    print("final average trained_a error",train_a,"final train error variance",np.var(l4),"psnr:",np.average(p4),np.var(p4))
    print("final average trained_f error",train_f,"final train error variance",np.var(l5),"psnr:",np.average(p5),np.var(p5))
    
    print("train wins:",train_win,"train loses:",k-train_win)

    
def complex_mul(Ar, Ai, Br, Bi):
    Cr = Ar.dot(Br) - Ai.dot(Bi)
    Ci = Ar.dot(Bi) + Ai.dot(Br)
    return Cr, Ci

def approx_freq(freq, freq_list):
    l = np.abs(freq_list-freq)
    i = np.where(l==min(l))[0][0]
    return freq_list[i], i

class IS():
    def __init__(self,size,rest,num_freq=4,sample_size=1,have_noise=False,shape="oval",sensor_pos="circle"):
        self.size=size
        self.width=0.5
        #self.img_data = generate_eta_batch(self.width/2, sample_size ,size, 2, 10,0.01)
        #self.img_data=generate_image(size,sample_size)
        #self.img_data = read_img("D:/data/FD_1mm/full_1mm/L067/full_1mm",size)
        self.img_data=[]
        self.sensor_pos=sensor_pos
        if shape == "oval":
            for i in range(sample_size):
                self.img_data.append(generate_tri_oval(self.width, size))
        elif shape == "circle":
            for i in range(sample_size):
                self.img_data.append(generate_circles(self.width, size))
        elif shape == "mnist":
            size = 28
            data_path = "C:/Users/scottjhy/Desktop/mnist_test.csv"
            data = np.loadtxt(data_path,delimiter=",")
            size = data.shape[0]
            s = random.sample(range(0,size),sample_size)
            for i in s:
                d=data[i,1:]
                d=d/np.linalg.norm(d)*2
                d=d.reshape((28,28))
                self.img_data.append(d)
        elif shape == "dots":
            for i in range(sample_size):
                self.img_data.append(generate_dots(self.width, size))    
        self.img_data_size=len(self.img_data)
        self.have_noise=have_noise
        self.action_num=360
        self.max_photon=1.4*180/1e5
        self.rest=rest
        self.proj_data=0
        self.detector=0
        self.freq_list=np.array([size*j//num_freq for j in range(1,num_freq+1)])
        self.freq=[]
        self.reconstruct_alg=reconstruct
        self.G=[]
        x = np.linspace(-3*self.width,3*self.width,3*self.size)
        y = np.linspace(self.width,-self.width,self.size) 
        X = np.meshgrid(x,y)
        X0 = X[0].reshape((3*self.size**2,1))
        X1 = X[1].reshape((3*self.size**2,1))
        self.X = np.concatenate([X0,X1],axis=1)
        if self.sensor_pos=="line":
            self.r=0
            self.s=0
        for freq in self.freq_list:
            self.G.append(self.greenfunction(freq))

    def reset(self,rest=20,set_pic=None):
        self.detector=0
        self.state_proj_seq=[]
        self.angle_seq=[]
        self.A_seq=[]
        self.freq=[]
        self.true_img=self.img_data[random.randint(0,self.img_data_size-1)]

        if set_pic is not None:
            self.true_img=set_pic
        #self.true_img=self.img_data[10]
        init_act=random.randint(0,self.action_num)
        s1,s2=self.true_img.shape
        proj_data=np.zeros((2*self.detector**2,))
        img_size=self.true_img.shape
        self.state=np.zeros(img_size)
        #self.rest=1.+random.random()*0.2-0.1
        self.rest=rest
        self.start=True
        eta = np.reshape(self.true_img,(s1*s2,1))
        E = np.diag(eta.T[0])
        self.ReEE=[]
        self.ImEE=[]
        if self.sensor_pos=="line":
            self.r=0
            self.s=0
        '''for g in self.G:
            self.ReEE.append(np.multiply(np.multiply(np.diag(E)[:,None], np.real(g)),np.diag(E))+E)
            self.ImEE.append(np.multiply(np.multiply(np.diag(E)[:,None], np.imag(g)),np.diag(E)))'''
        for g in self.G:
            # A = EGE, B = EGEGE
            A = np.multiply(np.multiply(np.diag(E)[:,None], g),np.diag(E))
            B = np.multiply(A@g,np.diag(E))
            self.ReEE.append(E+np.real(A)+np.real(B))
            self.ImEE.append(np.imag(A)+np.imag(B))
        self.proj_data=proj_data.copy()
        return proj_data,0

    def step(self,action,freq):
        self.angle_seq = np.concatenate([self.angle_seq,action],axis=0)
        if self.sensor_pos=="line":
            sensor_loc = [1.5*np.sin((action[0]-180)*np.pi/360),0.51]
            tmp1 = np.sqrt(np.sum(np.power(abs(sensor_loc - self.X), 2), axis=1))
            kr = tmp1 * self.freq_list[freq]
            tmp = -1j/4*scipy.special.hankel1(0, kr)
            tmp = tmp.reshape([1,3*self.size**2])
            self.s=tmp
            if self.detector==0:
                self.r=tmp
            else:
                self.r=np.concatenate([self.r,tmp],axis=0)
        self.detector+=1
        self.freq = np.concatenate([self.freq, [freq]])
        proj_data=self.get_project_data(self.true_img,action,freq,self.have_noise)     
        self.state_proj_seq.append(proj_data)
        self.old_state=self.state
        self.state=self.reconstruct_alg(self.state,self.state_proj_seq,self.angle_seq,self.A_seq,self.freq,self.freq_list,self.G)
        reward=psnr2(self.state,self.true_img)-psnr2(self.old_state,self.true_img)
        
        if self.rest == 1:
            done = True
        else:
            done = False
        self.proj_data=proj_data.copy()
        self.rest-=1
        
        return proj_data,reward,done,None
    
    def greenfunction(self, freq):
        x = np.linspace(-self.width*3,self.width*3,self.size*3)
        y = np.linspace(self.width,-self.width,self.size) 
        X = np.meshgrid(x,y)
        X0 = X[0].reshape((3*self.size**2,1))
        X1 = X[1].reshape((3*self.size**2,1))
        X = np.concatenate([X0,X1],axis=1)
        s = X.shape[0]
        G = np.zeros((s,s),dtype=complex)
        for i in range(s):
            tmp1 = np.sqrt(np.sum(np.power(abs(X[[i], :] - X), 2), axis=1))
            kr = tmp1 * freq
            tmp = scipy.special.hankel1(0, kr)
            G[[i], :] = -1j / 4 * tmp.transpose()
            G[i, i] = 0
        return G
   

    def get_project_data(self,img,action,freq,noise=False):
        i=freq
        freq=self.freq_list[i]
        s1,s2=img.shape
        detector=self.detector
        eta = np.reshape(img,(s1*s2,1))
        x = np.linspace(-self.width*3,self.width*3,self.size*3)
        y = np.linspace(self.width,-self.width,self.size) 
        X = np.meshgrid(x,y)
        X0 = X[0].reshape((s1*s2,1))
        X1 = X[1].reshape((s1*s2,1))
        X = np.concatenate([X0,X1],axis=1)
        E = np.diag(eta.T[0])
        if self.sensor_pos == "circle":
            r = np.array([[1j*freq*np.cos(self.angle_seq[a]*np.pi/360),
                       1j*freq*np.sin(self.angle_seq[a]*np.pi/360)] for a in range(detector)])
            r2 = np.array([[1j*freq*np.cos(self.angle_seq[-1]*np.pi/360),
                       1j*freq*np.sin(self.angle_seq[-1]*np.pi/360)]])
            cc = X.dot(-r.T)
            cc = cc.T
            A1 = np.exp(cc)
            cc = X.dot(-r2.T)
            cc = cc.T
            A2 = np.exp(cc)
            
        elif self.sensor_pos == "line":
            A1 = self.r
            A2 = self.s
            '''r = np.array([[1j*freq*0.71*np.sin((self.angle_seq[a]-180)*np.pi/720),
                       1j*freq*0.51] for a in range(detector)])
            r2 = np.array([[1j*freq*0.71*np.sin((self.angle_seq[-1]-180)*np.pi/720),
                       1j*freq*0.51]])
            cc = X.dot(-r.T)
            cc = cc.T
            A1 = np.exp(cc)
            cc = X.dot(-r2.T)
            cc = cc.T
            A2 = np.exp(cc)'''
                                             
        Rr = np.real(A1)
        Ir = np.imag(A1)
        Rs = np.real(A2).T
        Is = np.imag(A2).T
        Rh = self.ReEE[i]
        Ih = self.ImEE[i]
        Rd = Rr@(Rh@Rs)-Ir@(Ih@Rs)-Ir@(Rh@Is)-Rr@(Ih@Is)
        Id = Rr@(Rh@Is)-Ir@(Ih@Is)+Ir@(Rh@Rs)+Rr@(Ih@Rs)
        '''print(np.linalg.norm(A1@np.diag(eta.T[0])@A2.T))
        print(np.linalg.norm(A1@(np.diag(eta.T[0])@self.G[i]@np.diag(eta.T[0]))@A2.T))
        print(np.linalg.norm(A1@(np.diag(eta.T[0])@self.G[i]@np.diag(eta.T[0])@self.G[i]@np.diag(eta.T[0]))@A2.T))'''
        self.A_seq.append((Rr,Ir,Rs,Is))
        data = np.concatenate([Rd,Id])
        return data.reshape((2*detector,)).copy()

    def show_psnr(self):
        return psnr2(self.state,self.true_img)

    def show_ssim(self):
        return ssim(self.state,self.true_img,data_range=4096)
    
'''state=self.state
state_seq=self.state_proj_seq
angle_seq=self.angle_seq
A_seq=self.A_seq
freq_seq=self.freq
freq_list=self.freq_list
G_list=self.G'''

# for 1 circle, alpha=0.0001, iter=3, lr=0.1   
# for 2, alpha=0.00001
def reconstruct(state, state_seq, angle_seq, A_seq, freq_seq, freq_list, G_list, alpha=0.00001, iter=2, stepsize=1e-6):  
    '''if len(angle_seq)<3:
        return state'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s1,s2=state.shape
    state0 = torch.tensor(state,dtype=torch.float).reshape((s1*s2)).to(device)
    O = opt_network(s1*s2)
    with torch.no_grad():
        O.linear.bias = nn.Parameter(state0.clone())
        O.linear.weight = nn.Parameter(O.linear.weight.to(device))
        O.linear.weight.requires_grad = False
    #optimizer = torch.optim.Adam(O.parameters(),1e-5,betas=(0.9,0.999),amsgrad=True)
    optimizer = torch.optim.LBFGS(O.parameters(),lr=1,line_search_fn= "strong_wolfe")
    O.train()
    ReG0_list = []
    ImG0_list = []
    for G0 in G_list:
        ReG0_list.append(torch.tensor(np.real(G0),dtype=torch.float).to(device))
        ImG0_list.append(torch.tensor(np.imag(G0),dtype=torch.float).to(device))
    for j in range(iter):
        ## this is the state (size**2)
        def closure():
            optimizer.zero_grad()
            s = O.forward(torch.tensor([0],dtype=torch.float).to(device))
            E = torch.diag(s)
            loss = 0
            for i in range(len(state_seq)):
                index = int(freq_seq[i])
                ReG0 = ReG0_list[index]
                ImG0 = ImG0_list[index]
                Rh = E + torch.multiply(torch.multiply(torch.diag(E)[:,None], ReG0),torch.diag(E))
                Ih = torch.multiply(torch.multiply(torch.diag(E)[:,None], ImG0),torch.diag(E))
                D = torch.tensor(state_seq[i],dtype=torch.float).to(device)
                Rd = D[0:(i+1)]
                Id = D[(i+1):]
                Rd = Rd.reshape((i+1,1))
                Id = Id.reshape((i+1,1))
                Rr, Ir, Rs, Is = A_seq[i]
                Rr = torch.tensor(Rr,dtype=torch.float).to(device)
                Ir = torch.tensor(Ir,dtype=torch.float).to(device)
                Rs = torch.tensor(Rs,dtype=torch.float).to(device)
                Is = torch.tensor(Is,dtype=torch.float).to(device)
                
                loss += torch.norm(Rd-(Rr@(Rh@Rs)-Ir@(Ih@Rs)-Ir@(Rh@Is)-Rr@(Ih@Is)))**2 + torch.norm(Id-(Rr@(Rh@Is)-Ir@(Ih@Is)+Ir@(Rh@Rs)+Rr@(Ih@Rs)))**2
            
            loss += alpha*torch.norm(s,p=1)        
            loss.backward()
            return loss
        
        '''s = O.forward(torch.tensor([0],dtype=torch.float).to(device))
        E = torch.diag(s)
        loss = 0
        for i in range(len(state_seq)):
                index = int(freq_seq[i])
                ReG0 = ReG0_list[index]
                ImG0 = ImG0_list[index]
                Rh = E + torch.multiply(torch.multiply(torch.diag(E)[:,None], ReG0),torch.diag(E))
                Ih = torch.multiply(torch.multiply(torch.diag(E)[:,None], ImG0),torch.diag(E))
                D = torch.tensor(state_seq[i],dtype=torch.float).to(device)
                Rd = D[0:(i+1)]
                Id = D[(i+1):]
                Rd = Rd.reshape((i+1,1))
                Id = Id.reshape((i+1,1))
                Rr, Ir, Rs, Is = A_seq[i]
                Rr = torch.tensor(Rr,dtype=torch.float).to(device)
                Ir = torch.tensor(Ir,dtype=torch.float).to(device)
                Rs = torch.tensor(Rs,dtype=torch.float).to(device)
                Is = torch.tensor(Is,dtype=torch.float).to(device)
                
                loss += torch.norm(Rd-(Rr@(Rh@Rs)-Ir@(Ih@Rs)-Ir@(Rh@Is)-Rr@(Ih@Is)))**2 + torch.norm(Id-(Rr@(Rh@Is)-Ir@(Ih@Is)+Ir@(Rh@Rs)+Rr@(Ih@Rs)))**2
  
        loss += alpha*torch.norm(s,p=1) 
        print(len(angle_seq),j,loss-alpha*torch.norm(s,p=1),alpha*torch.norm(s,p=1)  )'''
        optimizer.step(closure)
        
        
    state0 = O.linear.bias.clone().detach()
    if torch.cuda.is_available():
        state0=state0.cpu()
    img_rec=state0.numpy().reshape((s1,s2))   
    
    torch.cuda.empty_cache()   
    return img_rec.copy()

class opt_network(nn.Module):
    def __init__(self,n_hid,n_layers=1):
        super(opt_network, self).__init__()
        self.n_layers=n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear=nn.Linear(1,n_hid)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.constant_(self.linear.weight, 0.0)
        torch.nn.init.constant_(self.linear.bias, 0.0)

    def forward(self,state):
        out=self.linear(state)
        return out
    
def compare(agent1,agent2,k,r,deterministic=True):
    ran = 0
    train = 0
    train_win = 0
    l1 = []
    l2 = []
    for i in range(k):
        self=agent1
        true_img=generate_tri_oval(self.env.width, self.env.size)
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        self.start_epoch(1)
        old_act=np.zeros((self.action_dim,))
        for i in range(self.long_time):
            if i!=0:
                obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            else:
                obs_norm=obs
            obs1=np.concatenate([obs_norm,np.zeros(self.state_dim-len(obs_norm)),old_act],axis=0)
            angle=self.action(obs1,old_act,deterministic)
            next_obs,reward,done,_=self.env.step(angle)
            self.next_hidden(obs1)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=1
            obs=next_obs
            if done:
                break
        self = self.env
        state = self.state
        state_seq = self.state_proj_seq
        angle_seq = self.angle_seq
        A_seq = self.A_seq
        eta = reconstruct(state, state_seq, angle_seq, A_seq, self.G, iter=30)
        size = np.int(np.sqrt(eta.size))
        eta0 = np.reshape(eta, (size,size))
        r1 = np.mean((eta0-true_img)**2)
        print("agent1 error:",r1,"psnr:",self.show_psnr())
        l1.append(r1)
        ran+=r1/k
        
        self=agent2
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        self.start_epoch(1)
        old_act=np.zeros((self.action_dim,))
        for i in range(self.long_time):
            if i!=0:
                obs_norm=np.clip((obs-self.statestat.mean)/(self.statestat.std+1e-8),-40,40)
            else:
                obs_norm=obs
            obs1=np.concatenate([obs_norm,np.zeros(self.state_dim-len(obs_norm)),old_act],axis=0)
            angle,freq=self.action(obs1,rest,old_act,deterministic)
            freq1 = self.freq_bound/(1+np.exp(-freq))
            rest-=1
            next_obs,reward,done,_=self.env.step(angle,freq1)
            self.next_hidden(obs1,rest+1)
            if i==(self.long_time-1):
                done=True
            old_act[angle]+=freq
            obs=next_obs
            if done:
                break
        self = self.env
        state = self.state
        state_seq = self.state_proj_seq
        angle_seq = self.angle_seq
        A_seq = self.A_seq
        eta = reconstruct(self.state, self.state_proj_seq, self.angle_seq, self.A_seq,self.freq,self.freq_list, self.G, iter=30)
        size = np.int(np.sqrt(eta.size))
        eta = np.reshape(eta, (size,size))
        r2 = np.mean((eta-true_img)**2)
        print("agent2 error:",r2,"psnr:",self.show_psnr())
        l2.append(r2)
        train +=r2/k
        
        if r1 > r2:
            train_win += 1
    print("final average agent1 error",ran, "final agent1 error variance",np.var(l1))
    print("final average agent2 error",train,"final agent2 error variance",np.var(l2))
    print("agent1 wins:",k-train_win,"agent2 wins:",train_win)


def generate_circles(eta_domain, n):
    num=2
    c_x = [np.random.uniform(-eta_domain*3+0.25, -eta_domain*3+0.75, 1)[0], 
           np.random.uniform(-eta_domain*3+1, -eta_domain*3+2, 1)[0], 
           np.random.uniform(eta_domain*3-1, eta_domain*3-0.5, 1)[0]]
    c_y = [np.random.uniform(-eta_domain+0.1, -eta_domain+0.25,1)[0],
           np.random.uniform(-eta_domain+0.35, eta_domain-0.35,1)[0],
           np.random.uniform(eta_domain-0.25, eta_domain-0.1,1)[0]]
    '''c_x = np.random.uniform(-eta_domain, eta_domain, num)
    c_y = np.random.uniform(-eta_domain, eta_domain, num)'''
    x = np.linspace(-eta_domain*3, eta_domain*3, 3*n)
    y = np.linspace(-eta_domain, eta_domain, n)
    [X, Y] = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    X = np.mat(X)
    Y = np.mat(Y)
    eta = np.zeros([1, X.shape[1]])
    # print(eta.shape)
    for j in range(num):
        for i in range(X.shape[1]):
        # xx = X[0,i]
        # yy = Y[0,i]
        # print(xx,' ', yy)
           
            tmp = (np.power(X[0, i] - c_x[j], 2)  + np.power(Y[0, i] - c_y[j], 2) )/(j+1)*2
            # print(tmp)
            if tmp < 0.01:
                eta[0, i] = 1                
               
    # print(tmp.shape)
    eta = eta.transpose()
    eta=eta/np.linalg.norm(eta)*2/4
    return eta.reshape((n,3*n))
    

'''def generate_circles(eta_domain, n):
    num=3
    c_x = np.random.uniform(-eta_domain, eta_domain, num)
    c_y = np.random.uniform(-eta_domain, eta_domain, num)
    x = np.linspace(-eta_domain*3, eta_domain*3, 3*n)
    y = np.linspace(-eta_domain, eta_domain, n)
    [X, Y] = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    X = np.mat(X)
    Y = np.mat(Y)
    eta = np.zeros([1, X.shape[1]])
    # print(eta.shape)
    for j in range(num):
        for i in range(X.shape[1]):
        # xx = X[0,i]
        # yy = Y[0,i]
        # print(xx,' ', yy)
           
            tmp = (np.power(X[0, i] - c_x[j], 2)  + np.power(Y[0, i] - c_y[j], 2) )/(j+1)*2
            # print(tmp)
            if tmp < 0.01:
                eta[0, i] = 1                
               
    # print(tmp.shape)
    eta = eta.transpose()
    eta=eta/np.linalg.norm(eta)*2/4
    return eta.reshape((n,3*n))'''



def generate_tri_oval(eta_domain, n):
    num = 1
    c_x1 = np.random.uniform(-eta_domain*3+0.3, -0.1, num)
    c_y1 = np.random.uniform(-eta_domain+0.3, eta_domain-0.3, num)
    c_x2 = np.random.uniform(0.1,eta_domain-0.3, num)
    c_y2 = np.random.uniform(-eta_domain+0.3, eta_domain-0.3, num)
    x = np.linspace(-eta_domain*3, eta_domain*3, 3*n)
    y = np.linspace(-eta_domain, eta_domain, n)
    [X, Y] = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    X = np.mat(X)
    Y = np.mat(Y)
    n2 = X.shape[1]
    eta = np.zeros([1, n2])
    # print(eta.shape)
    t = random.uniform(-0.25,0.25)
    if t > 0:
        c_x1=-c_x1
        c_x2=-c_x2
    for i in range(X.shape[1]):
        # xx = X[0,i]
        # yy = Y[0,i]
        # print(xx,' ', yy)
           
            tmp = np.power(X[0, i] - c_x1, 2) / 0.002 + np.power(Y[0, i] - c_y1, 2) / (0.02)
            # print(tmp)
            if tmp < 1:
                eta[0, i] = 1                
               
            if (X[0, i] < 0.1+c_x2) and (Y[0, i] > 0.05+c_x2) and (-X[0, i] + Y[0, i] < 0.12):
                eta[0, i] = 1
             #print(tmp.shape)
    eta = eta.transpose()
    gpt = np.vstack((Y, X))
    eta=eta/np.linalg.norm(eta)**2
    return eta.reshape((n,3*n))#, gpt.transpose(), c_x, c_y


def plot_freq_prob(agent,ff=0):
    m = 0
    s = 0
    fig, axs = plt.subplots(5, 2)
    for pos in range(agent.rest):
        print(pos)
        a_list=[]
        for i in range(pos,len(agent.freqprob),agent.rest):
            s += agent.freqprob[i][0][0][ff]/4
            m += 1
            if m == 4:
                m = 0
                a_list.append(s)
                s = 0
        axs[pos//2,pos%2].plot(a_list)
        axs[pos//2,pos%2].set_title("Sensor"+str(pos))
        
def plot_freq_pos(agent,pos=0):
    m = 0
    s = 0
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("pos"+str(pos))
    for ff in range(4):
        print(ff)
        a_list=[]
        for i in range(pos,len(agent.freqprob),agent.rest):
            s += agent.freqprob[i][0][0][ff]/4
            m += 1
            if m == 4:
                m = 0
                a_list.append(s)
                s = 0
        axs[ff//2,ff%2].plot(a_list)
        axs[ff//2,ff%2].set_title("freq"+str(ff))
        
def generate_dots(eta_domain, n):
    num=3
    eta = np.zeros([1,n**2])
    # print(eta.shape)
    for j in range(num):
        c_x=np.random.randint(0,n**2)
        eta[0,c_x]=1
               
    # print(tmp.shape)
    eta=eta/np.linalg.norm(eta)**2
    return eta.reshape((n,n))

'''
plt.imshow(eta)
plt.axis('off')
'''