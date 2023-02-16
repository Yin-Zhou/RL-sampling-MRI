import timeit
import numpy as np
import random
import math
import pydicom as dicom
import os
import copy
import torch
import scipy
from skimage.metrics import structural_similarity as ssim
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
    for i in range(k):
        self=agent
        true_img=generate_tri_oval(self.env.width, self.env.size)
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        angles=list(range(0,360))
        random.shuffle(angles)
        time_len =100
        for i in range(time_len):
            angle=[angles[i]]
            next_obs,reward,done,_=self.env.step(angle,f)
            if done:
                break

        self = self.env
        state = self.state
        state_seq = self.state_proj_seq
        angle_seq = self.angle_seq
        A_seq = self.A_seq
        eta = reconstruct(self.state, self.state_proj_seq, self.angle_seq, self.A_seq,self.freq,self.freq_list, self.G,iter=20)
        size = np.int(np.sqrt(eta.size))
        eta0 = np.reshape(eta, (size,size))
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
        eta = reconstruct(self.state, self.state_proj_seq, self.angle_seq, self.A_seq,self.freq,self.freq_list, self.G, iter=20)
        print(self.freq)
        size = np.int(np.sqrt(eta.size))
        eta = np.reshape(eta, (size,size))
        r2 = np.mean((eta-true_img)**2)
        print("trained error:",r2,"psnr:",psnr2(eta,true_img))
        l2.append(r2)
        p2.append(psnr2(eta,true_img))
        train +=r2/k
            
        self=agent
        angles=self.env.angle_seq
        freq=self.env.freq
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        time_len =100
        for i in range(time_len):
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
        size = np.int(np.sqrt(eta.size))
        eta0 = np.reshape(eta, (size,size))
        r4 = np.mean((eta0-true_img)**2)
        print("train_a error:",r4,"psnr:",psnr2(eta0,true_img))
        l4.append(r4)
        p4.append(psnr2(eta0,true_img))
        train_a+=r4/k    
        
        self=agent
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        time_len =100
        angles=[36*i for i in range(10)]
        for i in range(time_len):
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
        size = np.int(np.sqrt(eta.size))
        eta0 = np.reshape(eta, (size,size))
        r5 = np.mean((eta0-true_img)**2)
        print("train_f error:",r5,"psnr:",psnr2(eta0,true_img))
        l5.append(r5)
        p5.append(psnr2(eta0,true_img))
        train_f+=r5/k  
        
        self=agent
        obs,_=self.env.reset(rest=r,set_pic=true_img)
        time_len =100
        angles=[36*i for i in range(10)]
        for i in range(time_len):
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
        size = np.int(np.sqrt(eta.size))
        eta0 = np.reshape(eta, (size,size))
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

class MRI():
    def __init__(self,size,rest,sample_size=1,have_noise=False,shape="oval"):
        self.size=size
        self.width=0.5
        self.img_data=[]

        if shape == "oval":
            for i in range(sample_size):
                self.img_data.append(generate_tri_oval(self.width, size))
        elif shape == "circle":
            for i in range(sample_size):
                self.img_data.append(generate_circles(self.width, size))
        elif shape == "mnist":
            # size = 28
            data_path = "/export/users/RL_MRI/code/MRI/mnist_test.csv"
            data = np.loadtxt(data_path,delimiter=",")
            size = data.shape[0]
            s = random.sample(range(0,size),sample_size)
            for i in s:
                d=data[i,1:]
                d=d/np.linalg.norm(d)*2
                d=d.reshape((28,28))
                self.img_data.append(d)

        self.img_data_size=len(self.img_data)
        self.have_noise=have_noise
        self.action_num=int((self.img_data[0].shape[0])*(self.img_data[0].shape[1]))
        self.rest=rest
        self.reconstruct_alg=reconstruct

    def reset(self,rest=20,set_pic=None):
        self.state_proj_seq=[]
        self.angle_seq=[]
        self.A_seq=[]
        self.freq=[]
        self.true_img=self.img_data[random.randint(0,self.img_data_size-1)]

        if set_pic is not None:
            self.true_img=set_pic

        init_act=random.randint(0,self.action_num)
        s1,s2=self.true_img.shape
        proj_data=np.zeros((s1*s2,))
        img_size=self.true_img.shape
        self.state=np.zeros(img_size)

        self.rest=rest
        self.start=True
        return proj_data,0

    def step(self,action): 
        self.angle_seq = np.concatenate([self.angle_seq,action],axis=0)
        self.old_state=self.state
        self.state=self.reconstruct_alg(self.state,self.angle_seq,self.img_data)
        reward=psnr2(self.state,self.true_img)-psnr2(self.old_state,self.true_img)
        if self.rest == 1:
            done = True
        else:
            done = False
        self.rest-=1
        
        proj_data = self.state.copy()

        return proj_data.reshape((-1,)),reward,done,None

    def show_psnr(self):
        return psnr2(self.state,self.true_img)

    def show_ssim(self):
        return ssim(self.state,self.true_img,data_range=4096)
 
    
def reconstruct(state,action,img_data,alpha=0.1,iter=3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s1,s2=state.shape
    state0 = torch.tensor(state,dtype=torch.float).reshape((s1*s2)).to(device)

    sample_size,s1,s2 = np.shape(img_data)
    img_data = np.reshape(img_data, (sample_size,s1*s2))
    img_data = torch.tensor(img_data,dtype=torch.float).to(device)

    action = torch.tensor(action,dtype=torch.int32).to(device)
    pattern = torch.zeros(s1*s2)
    pattern[action.long()] = 1
    pattern = torch.tensor(pattern,dtype=torch.float).to(device)

    O = opt_network(s1*s2)
    with torch.no_grad():
        O.linear.bias = nn.Parameter(state0.clone())
        O.linear.weight = nn.Parameter(O.linear.weight.to(device))
        O.linear.weight.requires_grad = False
    optimizer = torch.optim.LBFGS(O.parameters(),lr=1)
    O.train()

    for j in range(iter):
        ## this is the state (size**2)
        def closure():
            optimizer.zero_grad()
            loss = 0
            u = O.forward(torch.tensor([0],dtype=torch.float).to(device))
            r = torch.multiply(torch.diag(pattern), torch.fft.fftn(u,norm="ortho"))
            for i in range(sample_size):
                y = torch.fft.fftn(u,norm="ortho")
                loss += torch.norm(r - y)**2
            
            loss += alpha*torch.norm(u,p=1)   
            loss.backward()
            return loss
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
    num=3
    c_x = np.random.uniform(-eta_domain, eta_domain, num)
    c_y = np.random.uniform(-eta_domain, eta_domain, num)
    x = np.linspace(-eta_domain, eta_domain, n)
    y = np.linspace(-eta_domain, eta_domain, n)
    [X, Y] = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    X = np.mat(X)
    Y = np.mat(Y)
    eta = np.zeros([1, X.shape[1]])
    eta[0][0] = 1
    r = np.random.uniform(0.02,0.03,num)
    v = np.random.uniform(0.5,1,num)
    for j in range(num):
        for i in range(X.shape[1]):
           
            tmp = np.power(X[0, i] - c_x[j], 2)  + np.power(Y[0, i] - c_y[j], 2) 
            if tmp < r[j]**2:
                eta[0, i] = v[j]                

    eta = eta.transpose()
    eta=eta/np.linalg.norm(eta)*2
    return eta.reshape((n,n))


def generate_tri_oval(eta_domain, n):
    num = 3
    c_x = np.random.uniform(-eta_domain, eta_domain, num)
    c_y = np.random.uniform(-eta_domain, eta_domain, num)
    x = np.linspace(-eta_domain, eta_domain, n)
    y = np.linspace(-eta_domain, eta_domain, n)
    [X, Y] = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    X = np.mat(X)
    Y = np.mat(Y)
    n2 = X.shape[1]
    eta = np.zeros([1, n2])
    t = random.uniform(-0.4,0.4)
    for j in range(num): 
        t = random.uniform(-0.3,0.3)
        for i in range(X.shape[1]):

            tmp = np.power(X[0, i] - c_x[j], 2) / (0.002*(j+1)) + np.power(Y[0, i] - c_y[j], 2) / (0.02/(j+1))
            if tmp < 1:
                eta[0, i] = 1                
               
            if (X[0, i] < 0.1+t) and (Y[0, i] > 0.05+t) and (-X[0, i] + Y[0, i] < 0.06*j):
                eta[0, i] = 1

    eta = eta.transpose()
    gpt = np.vstack((Y, X))
    eta=eta/np.linalg.norm(eta)*2
    return eta.reshape((n,n))