import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import trange
from kmeans_pytorch import kmeans
device = 'cuda:0'#'cpu'#
from tensorboardX import SummaryWriter
import torch.nn.functional as F

# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

    

def error(a,b):
    return (0.5*torch.mean((a-b)**2))


class AdamOptimizer:
    def __init__(self, model_parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model_parameters = model_parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def minimize(self):
        if self.m is None:
            self.m = [torch.zeros_like(param) for param in self.model_parameters]
            self.v = [torch.zeros_like(param) for param in self.model_parameters]

        self.t += 1

        for i, param in enumerate(self.model_parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (param.grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            update = self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)

            param.data -= update

class GT2RBFnn(nn.Module):
    def __init__(self, input_size, mf_num, ouput_size, alpha_slice_num=101):
        super(GT2RBFnn, self).__init__()
        self.N = input_size
        self.M = mf_num
        self.P = ouput_size
        self.asn = alpha_slice_num
        self.type = type
        
        self.alpha = torch.tensor(np.linspace(0,1,alpha_slice_num)).reshape([1,1,1,-1]).to(device)
        self.mu = .5*torch.ones([1,input_size,mf_num,1]).to(torch.float64).to(device)
        self.sigma = torch.ones([1,input_size,mf_num,1]).to(torch.float64).to(device)
        self.delta_sig = .1*torch.ones([1,input_size,mf_num,1]).to(torch.float64).to(device)
        # self.weights = torch.ones([1,mf_num,1,ouput_size]).to(torch.float64).to(device)
        self.weights = torch.rand([mf_num,ouput_size]).to(torch.float64).to(device)
        self.bias = torch.rand([1, ouput_size]).to(torch.float64).to(device)*0

        self.tbwriter = SummaryWriter(log_dir='./log/GT2RBFnn/tblogs')

    def GT2_MF(self, x):
        sigma = self.sigma + self.delta_sig/2
        delta = self.delta_sig/2
        # Calculate the lower and upper membership grades based on the alpha level
        lower = torch.exp(-0.5 * ((x.unsqueeze(-1).unsqueeze(-1) - self.mu) / (sigma + delta*(1-self.alpha))) ** 2)  # GT2 Gaussian lower bound
        upper = torch.exp(-0.5 * ((x.unsqueeze(-1).unsqueeze(-1) - self.mu) / (sigma - delta*(1-self.alpha))) ** 2)  # GT2 Gaussian upper bound
        
        f = {'upper':upper.prod(1), 'lower':lower.prod(1)}
        return f
    def Center_E(self, X):
        cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=self.M, distance='euclidean', device=device)
        centers = torch.transpose(cluster_centers, 0, 1).to(torch.float64).to(device)
        self.mu = centers.reshape(self.mu.shape)
            
    def defuzz_GT2(self, f, type = 'Nie-Tan'):
        if type == 'Nie-Tan':
            y_alpha = (torch.swapaxes((f['upper']+f['lower']),1,2)@(self.weights.squeeze(0).squeeze(1)))/(f['upper'].sum(1)+f['lower'].sum(1)).unsqueeze(-1)
            y = (y_alpha*self.alpha.reshape([1,-1,1])).sum(1)/self.alpha.sum()
        return y
    def forward(self, X):
        f = self.GT2_MF(X)
        y = self.defuzz_GT2(f).T
        return y.T + self.bias
    def forward_Train(self, X):
        f = self.GT2_MF(X)
        y_alpha = (torch.swapaxes((f['upper']+f['lower']),1,2)@(self.weights.squeeze(0).squeeze(1)))/(f['upper'].sum(1)+f['lower'].sum(1)).unsqueeze(-1)
        y = (y_alpha*self.alpha.reshape([1,-1,1])).sum(1)/self.alpha.sum() + self.bias
        return (y,y_alpha,f)
    def error(self, a, b):
        return (0.5*torch.mean((a-b)**2))
    def grad(self, X, F):
        y, y_alpha, f = self.forward_Train(X)
        g_alpha = self.alpha/self.alpha.sum()

        g = {'sigma':None,'delta':None,'mu':None}#'weight':[],
        temp = (self.weights.unsqueeze(1)-y_alpha.unsqueeze(1).unsqueeze(1))/(f['upper'].sum(1)+f['lower'].sum(1)).unsqueeze(1).unsqueeze(1).unsqueeze(-1)
        l_alpha = (1-self.alpha).unsqueeze(-1)
        x_mu_2 = ((X.unsqueeze(-1).unsqueeze(-1) - self.mu).unsqueeze(-1))**2
        x_mu = ((X.unsqueeze(-1).unsqueeze(-1) - self.mu).unsqueeze(-1))
        d_temp = (y-F).reshape([-1,1,1,1,y.shape[1]])
        g['sigma'] = temp*x_mu_2*((f['upper'].unsqueeze(1).unsqueeze(-1)/(self.sigma.unsqueeze(-1)+(-1+l_alpha)*(-1)/2*self.delta_sig.unsqueeze(-1))**3)+(f['lower'].unsqueeze(1).unsqueeze(-1)/(self.sigma.unsqueeze(-1)+(1+l_alpha)/2*self.delta_sig.unsqueeze(-1))**3))
        g['sigma'] = g['sigma']*g_alpha.reshape([1,1,1,-1,1])*d_temp#[1000, in, M, alpha, out]
        g['sigma'] = g['sigma'].mean(4).mean(3).mean(0)
        self.sigma.grad = g['sigma'].unsqueeze(0).unsqueeze(-1)
        g['delta'] = temp*x_mu_2*((f['upper'].unsqueeze(1).unsqueeze(-1)*(-1+l_alpha)*(-1)/(self.sigma.unsqueeze(-1)+(-1+l_alpha)*(-1)/2*self.delta_sig.unsqueeze(-1))**3)+(f['lower'].unsqueeze(1).unsqueeze(-1)*(1+l_alpha)/(self.sigma.unsqueeze(-1)+(1+l_alpha)/2*self.delta_sig.unsqueeze(-1))**3))
        g['delta'] = g['delta']*g_alpha.reshape([1,1,1,-1,1])*d_temp
        g['delta'] = g['delta'].mean(4).mean(3).mean(0)
        self.delta_sig.grad = g['delta'].unsqueeze(0).unsqueeze(-1)

        # g['mu'] = temp*x_mu_2**.5*((f['upper'].unsqueeze(1).unsqueeze(-1)/(self.sigma.unsqueeze(-1)+(-1+l_alpha)*(-1)/2*self.delta_sig.unsqueeze(-1))**2)+(f['lower'].unsqueeze(1).unsqueeze(-1)/(self.sigma.unsqueeze(-1)+(1+l_alpha)/2*self.delta_sig.unsqueeze(-1))**2))
        g['mu'] = temp*x_mu*((f['upper'].unsqueeze(1).unsqueeze(-1)/(self.sigma.unsqueeze(-1)+(-1+l_alpha)*(-1)/2*self.delta_sig.unsqueeze(-1))**2)+(f['lower'].unsqueeze(1).unsqueeze(-1)/(self.sigma.unsqueeze(-1)+(1+l_alpha)/2*self.delta_sig.unsqueeze(-1))**2))
        g['mu'] = g['mu']*g_alpha.reshape([1,1,1,-1,1])*d_temp
        g['mu'] = g['mu'].mean(4).mean(3).mean(0)
        self.mu.grad = g['mu'].unsqueeze(0).unsqueeze(-1)

        temp1 = ((f['upper']+f['lower'])/(f['upper'].sum(1)+f['lower'].sum(1)).unsqueeze(1)).unsqueeze(-1)
        temp2 = (g_alpha.reshape([1,1,1,-1,1])*d_temp).squeeze(1)
        temp3 = temp1 * temp2
        self.weights.grad = temp3.mean(-2).mean(0)#.unsqueeze(0).unsqueeze(-2)

        self.bias.grad = d_temp.mean(-2).mean(0).squeeze(1)

        # g['sigma'] = torch.clip(g['sigma'], -1, 1)
        # g['mu'] = torch.clip(g['mu'], -1, 1)
        return y, f

    def Train(self, X_train, Y_train, Epochs = 90, batch_size=128, init_flag = True, lr = 0.02):#, iteration_num=1000
        iteration_num = int(X_train.shape[0]/batch_size)+1
        if init_flag:
            self.Center_E(X_train)
            model_parameters = [self.sigma, self.delta_sig, self.mu, self.weights] #, self.bias
            # self.optimizer = AdamOptimizer(model_parameters, learning_rate=0.02, beta1=0.9, beta2=0.999, epsilon=1e-8)
            self.optimizer = AdamOptimizer(model_parameters, learning_rate=lr, beta1=0.5, beta2=0.7, epsilon=1e-8)

        for Epoch in range(Epochs):
            # print(f'Epoch number {Epoch} from {Epochs}')
            if Epoch%3 == 0 and Epoch!=0:
                self.optimizer.learning_rate /= 10
            # tqdm_t = trange(iteration_num, desc='MSE:', leave=True)
            for i in range(iteration_num):#tqdm_t:
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                X = X_train[idx].to(device)
                Y = Y_train[idx].to(device)
                self.grad(X, Y)
                self.optimizer.minimize()

                y, _, f = self.forward_Train(X)
                # phi = ((f['upper']+f['lower']))/(f['upper'].sum(1)+f['lower'].sum(1)).unsqueeze(1)
                # phi = (phi*self.alpha.reshape([1,1,-1])).sum(2)/self.alpha.sum()
                # self.weights = torch.linalg.pinv(phi)@Y

                # loss = torch.clip(F.mse_loss(y, Y),0,1)#error(y, Y)
                loss = error(Y, y)
                self.tbwriter.add_scalar('loss', loss.item(), self.optimizer.t*8)
                # tqdm_t.set_description("Epoch: (%i) MSE: (%e)" % (Epoch,loss))
                # tqdm_t.refresh()
            # torch.save(self, './log/GT2RBFnn/models/GT2RBFnn_'+str(Epoch)+'.pt')

    
