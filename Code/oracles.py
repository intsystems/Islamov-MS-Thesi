import numpy as np
import random
import math
import sys
from scipy.stats import bernoulli

class LogReg:
    def __init__(self, A, b, reg_coef, n, m):
        '''
        This class is created to simulate Logistic Regression problem for classification
        
        A - data points
        b - targets
        reg_coef - L2 regularization coefficient
        n - number of nodes
        m - number of local data points
        '''
        self.A = A
        self.b = b
        self.lmb = reg_coef 
        self.n = n
        self.m = m
        self.d = A.shape[1]

    def function_value(self, x):
        '''
        x - current model weights
        
        return: P(x) = f(x) + lmb/2*||x||^2
        '''
        ans = 0
        N = self.n*self.m
        z = self.b * np.dot(self.A, x)
        tmp = np.minimum(z, 0)
        loss = np.log((np.exp(tmp) + np.exp(tmp - z)) / (np.exp(tmp)))
        ans = np.sum(loss)/N
        ans += self.lmb/2*np.linalg.norm(x)**2
        return ans
    
    def local_function_value(self, x, i):
        '''
        x - current model weights
        
        return: f_i(x) + lmb/2*||x||^2
        '''
        ans = 0
        left = i*self.m
        right = (i+1)*self.m
        z = self.b[left:right] * np.dot(self.A[left:right], x)
        tmp = np.minimum(z, 0)
        loss = np.log((np.exp(tmp) + np.exp(tmp - z)) / (np.exp(tmp)))
        ans = np.sum(loss)/self.m
        ans += self.lmb/2*np.linalg.norm(x)**2
        return ans
    
    def Hessian(self, x, i, j): 
        '''
        x - current model weights
        i - node number
        j - number of data point in local dataset
        
        return: Hessian of f_ij(x)
        '''
        l = i*self.m + j
        alpha = self.b[l]**2*np.exp(-self.b[l]*self.A[l].dot(x))/(1+np.exp(-self.b[l]*self.A[l].dot(x)))**2
        ans = alpha*self.A[l].reshape((self.d,1)).dot(self.A[l].reshape(1,self.d))
        return ans
    
    def alpha(self, x, i, j): 
        '''
        x - current model weights
        i - node number
        j - number of data point in local dataset
        
        return: alpha_ij(x)
        '''
        l = i*self.m + j
        alpha = self.b[l]**2*np.exp(-self.b[l]*self.A[l].dot(x))/(1+np.exp(-self.b[l]*self.A[l].dot(x)))**2
        return alpha
    
    def gradient(self, x, i, j): 
        '''
        x - current model weights
        i - node number
        j - number of data point in local dataset
        
        return: gradient of f_ij(x)
        '''
        l = self.m*i + j
        alpha = -self.b[l]*np.exp(-self.b[l]*self.A[l].dot(x))/(1+np.exp(-self.b[l]*self.A[l].dot(x)))
        ans = alpha*self.A[l]
        return ans
    
    def local_gradient(self, x, i):
        '''
        x - current model weights
        i - node number
        
        return: gradient of f_i(x)
        '''
        m = self.m
        left = i*m
        right = (i+1)*m
        z = self.b[left:right] * np.dot(self.A[left:right], x)
        tmp0 = np.minimum(z, 0)
        tmp1 = np.exp(-z) / ((1+ np.exp(-z)))
        tmp2 = - tmp1 * self.b[left:right]
        g = np.dot(self.A[left:right].T, tmp2) / m

        return g

    
    def local_Hessian(self, x, i):
        '''
        x - current model weights
        i - node number
        
        return: Hessian of f_i(x)
        '''
        m = self.m
        d = self.d
        H = np.zeros((d, d))
        for j in range(m):
            H += 1/m*self.Hessian(x, i, j)
        return H
    
    def full_Hessian(self, x):
        '''
        x - current model weights
        
        return: full Hessian of f(x)
        '''
        m = self.m
        n = self.n
        N = n*m
        d = self.d 
        H = np.zeros((d,d))
        for i in range(n):
            for j in range(m):
                H += 1/N*self.Hessian(x, i, j)
        return H

    def full_gradient(self, x):
        '''
        x - current model weights
        
        return: full gradient of f(x)
        '''
        N = self.n*self.m
        z = self.b * np.dot(self.A, x)
        tmp0 = np.minimum(z, 0)
        tmp1 = np.exp(-z) / ((1 + np.exp(-z)))
        tmp2 = - tmp1 * self.b
        g = np.dot(self.A.T, tmp2) / N
        return g


    def alphas(self, x, i):
        '''
        x - current model weights
        i - node number
        
        return: vector alpha_i(x), i.e. [alpha_i(x)]_j = alpha_ij(x)
        '''
        left = i*self.m
        right = (i+1)*self.m
        ans = np.zeros(self.m)
        for j in range(self.m):
            ans[j] = self.alpha(x, i, j)
        return ans
    
    def full_alphas(self, x):
        '''
        x - current model weights
        i - node number
        
        return: vector alpha(x), i.e. [full_alphas(x)]_{i*m+j} = alpha_ij(x)
        '''
        ans = np.zeros(self.n*self.m)
        for i in range(self.n):
            for j in range(self.m):
                ans[i*self.m+j] = self.alpha(x, i, j)
        return ans
    
        
    def get_reg_coef(self):
        '''
        return: regularization coefficient
        '''
        return self.lmb
    
    def get_number_of_weights(self):
        '''
        return: the dimension of weights space 
        '''
        return self.d
    
    def get_number_of_nodes(self):
        '''
        return: number of nodes n
        '''
        return self.n
    
    def get_number_of_local_data_points(self):
        '''
        return: number of data points m in local dataset
        '''
        return self.m
    
    def get_optimum(self):
        '''
        return: optimal solution of the problem
        '''
        return self.x_opt
    
    def set_optimum(self, x_opt):
        '''
        set the optimal solution of the problem
        '''
        self.x_opt = x_opt
#######################################################################################


class LinReg:
    def __init__(self, A, b, reg_coef, n, m, d):
        '''
        This class is created to simulate Linear Regression problem for classification
        
        
        A - data points
        b - targets
        reg_coef - L2 regularization coefficient
        n - number of nodes
        m - number of local data points
        d - dimension of the parameter space, i.e. the number of model parameters
        '''
        self.A = A
        self.b = b
        self.lmb = reg_coef 
        self.n = n
        self.m = m
        self.d = A.shape[1]

    def function_value(self, x):
        '''
        x - current model weights
        
        return: P(x) = f(x) + lmb/2*||x||^2
        '''
        N = self.n*self.m
        z = self.b * np.dot(self.A, x)
        ans = 1/(2*N)*np.linalg.norm(self.A.dot(x)-self.b)**2
        ans += self.lmb/2*np.linalg.norm(x)**2
        return ans
    
    def local_function_value(self, x, i):
        '''
        x - current model weights
        
        return: f_i(x) + lmb/2*||x||^2
        '''
        left = i*self.m
        right = (i+1)*self.m
        ans = 1/(2*N)*np.linalg.norm(self.A[left:right].dot(x)-self.b[left:right])**2
        ans += self.lmb/2*np.linalg.norm(x)**2
        return ans
    
    def Hessian(self, x, i, j): 
        '''
        x - current model weights
        i - node number
        j - number of data point in local dataset
        
        return: Hessian of f_ij(x)
        '''
        l = i*self.m + j
        ans = self.A[l].reshape((self.d,1)).dot(self.A[l].reshape(1,self.d))
        return ans
    
    def alpha(self, x, i, j): 
        '''
        x - current model weights
        i - node number
        j - number of data point in local dataset
        
        return: alpha_ij(x)
        '''
        return 1
    
    def gradient(self, x, i, j): 
        '''
        x - current model weights
        i - node number
        j - number of data point in local dataset
        
        return: gradient of f_ij(x)
        '''
        l = self.m*i + j
        alpha = self.A[l].dot(x)-self.b[l]
        ans = alpha*self.A[l]
        return ans
    
    def local_gradient(self, x, i):
        '''
        x - current model weights
        i - node number
        
        return: gradient of f_i(x)
        '''
        m = self.m
        left = i*m
        right = (i+1)*m
        alpha = self.A[left:right].dot(x)-self.b[left:right]
        g = np.dot(self.A[left, right].T, alpha) / m

        return g

    
    def local_Hessian(self, x, i):
        '''
        x - current model weights
        i - node number
        
        return: Hessian of f_i(x)
        '''
        m = self.m
        d = self.d
        H = np.zeros((d, d))
        for j in range(m):
            H += 1/m*self.Hessian(x, i, j)
        return H
    
    def full_Hessian(self, x):
        '''
        x - current model weights
        
        return: full Hessian of f(x)
        '''
        m = self.m
        n = self.n
        N = n*m
        d = self.d 
        H = np.zeros((d,d))
        for i in range(n):
            for j in range(m):
                H += 1/N*self.Hessian(x, i, j)
        return H

    def full_gradient(self, x):
        '''
        x - current model weights
        
        return: full gradient of f(x)
        '''
        N = self.m*self.n
        alpha = self.A.dot(x)-self.b
        g = np.dot(self.A.T, alpha) / N
        
        return g


    def alphas(self, x, i):
        '''
        x - current model weights
        i - node number
        
        return: vector alpha_i(x), i.e. [alpha_i(x)]_j = alpha_ij(x)
        '''
        return np.ones(self.m)
    
    def full_alphas(self, x):
        '''
        x - current model weights
        i - node number
        
        return: vector alpha(x), i.e. [full_alphas(x)]_{i*m+j} = alpha_ij(x)
        '''
        return np.ones(self.n*self.m)
    
        
    def get_reg_coef(self):
        '''
        return: regularization coefficient
        '''
        return self.lmb
    
    def get_number_of_weights(self):
        '''
        return: the dimension of weights space 
        '''
        return self.d
    
    def get_number_of_nodes(self):
        '''
        return: number of nodes n
        '''
        return self.n
    
    def get_number_of_local_data_points(self):
        '''
        return: number of data points m in local dataset
        '''
        return self.m
    
    def get_optimum(self):
        '''
        return: optimal solution of the problem
        '''
        return self.x_opt
    
    def set_optimum(self, x_opt):
        '''
        set the optimal solution of the problem
        '''
        self.x_opt = x_opt
        
        
        
##################################################################################
class SoftMax:
    def __init__(self, A, b, n, m, reg_coef, sigma):
        '''
        This class is created to simulate the problem for classification of the following form
        f_{ij}(x) = - log(b_{ij} - a_{ij}^T * x)
        
        A - data points
        b - targets
        reg_coef - L2 regularization coefficient
        n - number of nodes
        m - number of local data points
        '''
        self.B = A
        self.b = b
        self.lmb = reg_coef 
        self.n = n
        self.m = m
        self.d = A.shape[1]
        self.sigma=sigma

    def function_value(self, x):
        '''
        x - current model weights
        
        return: P(x) = f(x) + lmb/2*||x||^2
        '''
        ans = 0
        n = self.n
        m = self.m
        N = self.n*self.m
        for i in range(n):
            l = i*m
            r = (i+1)*m
            tmp = np.exp((self.A[l:r].dot(x) - self.b[l:r])/self.sigma)
            ans += self.sigma*np.log(np.sum(tmp))/n
        ans += self.lmb/2*np.linalg.norm(x)**2
        return ans
    
    def local_function_value(self, x, i):
        '''
        x - current model weights
        
        return: f_i(x) + lmb/2*||x||^2
        '''
        ans = 0
        m = self.m
        l = i*self.m
        r = (i+1)*self.m
        tmp = np.exp((self.A[l:r].dot(x) - self.b[l:r])/self.sigma)
        ans = self.sigma*np.log(np.sum(tmp))
        ans += self.lmb/2*np.linalg.norm(x)**2
        return ans
    
    
    def local_gradient(self, x, i):
        '''
        x - current model weights
        i - node number
        
        return: gradient of f_i(x)
        '''
        m = self.m
        l = i*m
        r = (i+1)*m
        sigma = self.sigma
        g = np.zeros(self.d)
        coef = 0
        for j in range(l,r):
            a = self.A[j]
            c = np.exp((a@x - self.b[j])/sigma)
            g += c*a
            coef += c
        return g/coef

    
    def local_Hessian(self, x, i):
        '''
        x - current model weights
        i - node number
        
        return: Hessian of f_i(x)
        '''
        m = self.m
        d = self.d
        sigma = self.sigma
        l = i*m
        r = (i+1)*m
        H1 = np.zeros((d, d))
        H2 = np.zeros((d, d))
        coef = 0
        for p in range(l,r):
            for q in range(l,r):
                a_p = self.A[p]
                a_q = self.A[q]
                if p == q:
                    c = np.exp((a_p@x - self.b[p])/sigma)
                    coef += c
                    H1 += c*a_p.reshape(self.d,1).dot(a_p.reshape(1,self.d))
                c = np.exp((a_p@x - self.b[p])/sigma)*np.exp((a_q@x - self.b[q])/sigma)
                H2 += c*a_p.reshape(self.d,1).dot(a_q.reshape(1,self.d))

        return 1/sigma*(H1/coef - H2/coef**2)
    
    def full_Hessian(self, x):
        '''
        x - current model weights
        
        return: full Hessian of f(x)
        '''
        n = self.n
        d = self.d 
        H = np.zeros((d,d))
        for i in range(n):
            H += 1/n*self.local_Hessian(x, i)
        return H

    def full_gradient(self, x):
        '''
        x - current model weights
        
        return: full gradient of f(x)
        '''
        n = self.n
        ans = np.zeros(self.d)
        for i in range(n):
            ans += 1/n*self.local_gradient(x,i)
        
        return ans

    def data_shift(self):
        '''
        perform data shift to set optimum at 0
        '''
        x = np.zeros(self.d)
        n = self.n
        m = self.m
        B = self.B
        for i in range(n):
            l = i*m
            r = (i+1)*m
            B[l:r] -= self.local_gradient(x,i)
        self.A = B
    
        
    def get_reg_coef(self):
        '''
        return: regularization coefficient
        '''
        return self.lmb
    
    def get_number_of_weights(self):
        '''
        return: the dimension of weights space 
        '''
        return self.d
    
    def get_number_of_nodes(self):
        '''
        return: number of nodes n
        '''
        return self.n
    
    def get_number_of_local_data_points(self):
        '''
        return: number of data points m in local dataset
        '''
        return self.m
    
    def get_optimum(self):
        '''
        return: optimal solution of the problem
        '''
        return self.x_opt
    
    def set_optimum(self, x_opt):
        '''
        set the optimal solution of the problem
        '''
        self.x_opt = x_opt