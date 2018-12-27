# -*- coding: utf-8 -*-
""" ELM network class for one-step-ahead forecasting time series """

import numpy as np
import pywt
#import statsmodels.api as sm
#import utils as ut
#from time import time

def reproducibility():
    """ Fix the seed for random number generator in order to reproduce the computational experiments"""
    np.random.seed(0)

def metricas(entrenado, real):
    """Forecasting performance metrics.
    Calculating the performance metrics MAE,MSE,RMSE,MAPE and THEIL
    
    return a tuple with the performance metrics    
    
    parameters: 
    ---------------------------------------------------------
    entrenado -- forecasting values from a model
    real      -- observed values    
    
    """
    mae = np.sum(np.abs(entrenado-real))/len(entrenado)
    mse = np.sum((entrenado-real)**2)/len(entrenado)
    rmse = np.sqrt(mse)
    mape = np.sum(np.abs((entrenado-real)/real))/len(entrenado)
    theil = (np.sum((real[1:]-entrenado[1:])**2)/np.sum((real[1:]-real[:-1])**2))
    return mae,mse,rmse,mape,theil

def procesar_wavelet(data,wavelet,mode):
    """ Processing the input vector using DWT from Pywaelets
    Calculate the DWT for each input vector
    
    return the DWT of each input vector.
    
    parameters:
    ---------------------------------------------------------
    data    -- input vector matrix
    wavelet -- Mother wavelet 
    mode    -- DWT mode
    """
    pat = pywt.wavedec(data,wavelet,mode)
    output = []
    for i in range(len(pat)):
        output += np.array(pat[i]).tolist()
    return np.array(output)
    
def logistica(x):
    """ Processing the logistic function  1/(1+exp(-x))
    
    return apply the logistic function to input x    
        
    parameters:
    ---------------------------------------------------------
    x  --  a float value
    """
    return (1.0 / (1.0 + np.exp(-1.0 * x)))

def factivacion(x,fa):
    """
    Processing the activation function according to x value (integer)
    
    return the activation function applied to x
    
    parameters:
    ---------------------------------------------------------
    x     -- a float value
    fa    -- a interger value (0,1,2,3,4,5)
    
    """
    if fa == 0:
        return 1.7159*np.tanh((float(2)/3)*x)
    elif fa==1:
        return logistica(x)-0.5
    elif fa==2:
        return x/(1+np.abs(x))
    elif fa==3:
        return logistica(x)
    elif fa==4:
        return np.log(1+np.exp(-1.0*x))    
    else:
        return x

class ELM_Network:
    """ ELM class"""
    def __init__(self,inp,hid,ac_function):
        """ Class contructor method 
        
        Building a ELM network instance
        Attributes:
        ---------------------------------------------------------
        inputs      -- number of input neurons       
        hiddens     -- number of hidden neurons
        fo          -- activation function
        num_params  -- number of parameters of the model
        w           -- input-hidden matrix (fixed)
        bias        -- bias term vector
        betas       -- hidden-output vector (to optimaze)
        ---------------------------------------------------------
        parameters:
        
        inp         -- number of input neurons
        hid         -- number of hidden neurons
        ac_function -- type of activation function
                
        """
        self.inputs     = inp        
        self.hidden     = hid
        self.fo         = ac_function        
        self.num_params = (inp+1) * hid + hid + 1
        sigma2 = (inp+hid)**(-0.5)
        self.w = np.random.uniform(-sigma2,sigma2,self.inputs*self.hidden).reshape((self.hidden,self.inputs))
        self.bias = np.random.uniform(-sigma2,sigma2,self.hidden)
        self.betas = np.random.uniform(-sigma2,sigma2,self.hidden)
        self.minimos = np.zeros(inp+1)
        self.maximos = np.zeros(inp+1)
        self.beta_0  = np.random.uniform(-sigma2,sigma2)
        
    def get_topology(self):
        """ Obtain the network topology
        return a tuple (num_inputs,num_hidden_ac_function) 
        """
        return self.inputs,self.hidden,self.fo
       
    def get_input_matrix(self):
        """ obtaining a copy of input-hidden matrix
            return a numpy array        
        """
        return np.array(self.w)
    
    def get_bias(self):
        """ obtaining a copy of bias coefficients
            return a numpy array        
        """
        return np.array(self.bias)
        
    def get_betas(self):
        """ obtaining a copy of beta coefficients 
            return a numpy array        
        """
        return np.array(self.betas),self.beta_0

    def get_min_max(self):
        """ obtaining a copy of minimum and maximum values 
            return two numpy arrays        
        """
        return np.copy(self.minimos),np.copy(self.maximos)

    def get_min(self):
        """obtaining the mamimum values of each input vector component
           return a numpy array                 
        """
        return np.copy(self.minimos)

    def get_max(self):
        """obtaining the maximum values of each input vector component
           return a numpy array                 
        """
        return np.copy(self.maximos)

    
    def set_min_max(self,v_min,v_max):
        """ Setting new values for minimum and maximum values
           
            parameters:
            ---------------------------------------------------------
            v_min  --  numpy array with minimum values
            v_max  --  numpy array with maximum values
            
        """
        self.minimos = np.copy(v_min)
        self.maximos = np.copy(v_max)
        
    def set_input_matrix(self,m):
        """ Setting new values for input-hidden matrix
        
            parameters:
            ---------------------------------------------------------
            m  --  numpy array with new values
        
        """
        self.w = np.array(m)
    
    def set_bias(self,b):
        """ Setting new value for bias term
        
            parameters:
            ---------------------------------------------------------
            b  --  the new value to replace
        """
        self.bias = np.array(b)
        
    def set_betas(self,b,b0):
        """ Setting new values for betas coefficients

            parameters:
            ---------------------------------------------------------
            b   --  numpy array with new beta values
            b0  --  a float value
        """
        self.betas = np.array(b)
        self.beta_0 = b0
    
    def copy(self):
        """ Copy method for ELM_Network class
            return an instace of ELM_Network class        
        """
        aux = ELM_Network(self.inputs,self.hidden,self.fo)
        aux.set_input_matrix(self.w)
        aux.set_bias(self.bias)
        aux.set_betas(self.betas,self.beta_0)
        aux.set_min_max(self.minimos,self.maximos)
        return aux
    
    def copy_network(self,net):
        """ Copy method from the other ELM_network object
        
            parameters:
            ---------------------------------------------------------
            net   -- an ELM_network instance
        """
        n,h,fo = net.get_topology()
        self.__init__(n,h,fo)
        self.set_input_matrix(net.get_input_matrix())
        self.set_bias(net.get_bias())
        b,b0 = net.get_betas()
        self.set_betas(b,b0)
        self.set_min_max(net.get_min(),net.get_max())
        
    def training_set(self,data,p):
        """ Bulding dataset from the data and p-lags
        
            parameters:
            ---------------------------------------------------------
            data  -- a time series
            p     -- number of lags
            ---------------------------------------------------------
            
            return two numpy arrays: input vectors and targets        
        """
        n = len(data)
        m = n-p
        inputs = np.zeros((m,p))
        for i in range(m):
            inputs[i] = data[i:i+p]
        return inputs,np.copy(data[p:])
        
    def training_set_wavelet(self,data,p,wavelet,mode):
        """ Bulding dataset from the data, p-lags and DWT parameters
        
            parameters:
            ---------------------------------------------------------
            data       -- a time series
            p          -- number of lags   
            wavelet    -- mother wavelet
            mode       -- DWT mode
            ---------------------------------------------------------
            
            return two numpy arrays: inputs and targets
        """
        n = len(data)
        m = n-p
        self.minimos = np.zeros(p+1,dtype=float)
        self.maximos  = np.zeros(p+1,dtype=float)
        normalizados = np.zeros((m,p))
        for i in range(m):
            normalizados[i] = procesar_wavelet(np.array(data[i:i+p]),wavelet,mode)
        for i in range(p):
            self.minimos[i] = np.min(normalizados[:,i])
            self.maximos[i] = np.max(normalizados [:,i]) 
            normalizados [:,i] = normalizados[:,i]*(self.maximos[i]-self.minimos[i])**-1 - self.minimos[i]*(self.maximos[i]-self.minimos[i])**-1 - 0.5
    
        targets = np.array(data[p:])
        self.minimos[p] = np.min(targets)
        self.maximos[p] = np.max(targets)
        targets = targets*(self.maximos[p]-self.minimos[p])**-1 - self.minimos[p]*(self.maximos[p]-self.minimos[p])**-1  - 0.5
        return normalizados,targets
        
        
    def normalizar(self,data,p):
        """ Normalize the input vectors to interval (-0.5,0.5)
            
            parameters:
            ---------------------------------------------------------
            data       -- a time series
            p          -- number of lags   
            ---------------------------------------------------------
    
            return two numpy arrays: inputs and targets 
        """
        n = len(data)
        m = n-p
        self.minimos = np.zeros(p+1,dtype=float)
        self.maximos  = np.zeros(p+1,dtype=float)
        normalizados = np.array([data[i:i+p] for i in range(m)])
    
        for i in range(p):
            self.minimos[i] = np.min(data[i:n-p+i])
            self.maximos[i] = np.max(data[i:n-p+i]) 
            normalizados [:,i] = normalizados[:,i]*(self.maximos[i]-self.minimos[i])**-1 - self.minimos[i]*(self.maximos[i]-self.minimos[i])**-1 - 0.5
    
        targets = np.array(data[p:])
        self.minimos[p] = np.min(targets)
        self.maximos[p] = np.max(targets)
        targets = targets*(self.maximos[p]-self.minimos[p])**-1 - self.minimos[p]*(self.maximos[p]-self.minimos[p])**-1  - 0.5
        return normalizados,targets

    def forward_pass(self,entrada):
        """ Apply a forward-pass throught the network

            parameters:
            ---------------------------------------------------------
            entrada       -- an input vector   
            
            ---------------------------------------------------------

            return the network output (float)
        
        """
        return np.dot(factivacion(self.w.dot(entrada)+self.bias,self.fo),self.betas)+self.beta_0
        
    def dnorm_output(self,out):
        """ denormalize the output of the network
        
            parameters:
            ---------------------------------------------------------
            output       -- a network output
            ---------------------------------------------------------
                            
            return the denormalize output (value in the original scale)
        
        """
        return (out+0.5)*(self.maximos[-1]-self.minimos[-1])+self.minimos[-1]
   
    def d_output(self,entrada,minimo,maximo):
        """ denormalize the output of the network"""
        out = self.forward_pass(entrada)
        return (out+0.5)*(maximo-minimo)+minimo
        
    def dnorm_forward_pass(self,entrada):
        """ denormalize the output of the network for a input vector"""
        out = self.forward_pass(entrada)
        return (out+0.5)*(self.maximos[-1]-self.minimos[-1])+self.minimos[-1]
        
    def dnorm_targets(self,targets):
        """ denormalize the targets values"""
        dtarget = np.copy(targets)
        dtarget =  (dtarget+0.5)*(self.maximos[-1]-self.minimos[-1])+self.minimos[-1]
        return dtarget
        
    def s_forward_pass(self,entrada):
        """ Processing the input vector until the hidden-layer
                        
            parameters:
            ---------------------------------------------------------
            entrada       -- an input vector
        
            ---------------------------------------------------------

            return a numpy array
                
        """
        return factivacion(self.w.dot(entrada)+self.bias,self.fo)

#    def cargar_matrix(self,row,col):
#        self.H = np.zeros(shape=(row,col))
        
    def to_file(self,filename):
        """ Save the network topology to file output

            parameters:
            ---------------------------------------------
            filename       -- the name of the output file
                
        
        """
        f  = open(filename,'w')
        f.write(str(self.inputs)+';'+str(self.hidden)+';'+str(self.fo)+'\n')
        result = np.array(self.w).flatten()
        result = ' '.join(list(map(str,result)))
        f.write(result+'\n')
        result2 = ';'.join(list(map(str,self.bias.flatten())))
        f.write(result2+'\n')
        result3 = ';'.join(list(map(str,self.betas.flatten())))
        f.write(result3+'\n')
        f.write(str(self.beta_0)+'\n')
        f.write(';'.join(list(map(str,self.minimos)))+'\n')
        f.write(';'.join(list(map(str,self.maximos)))+'\n') 
        f.close()
        
    def from_file(self,filename):
        """ Load the network topology from file
        
            parameters:
            ---------------------------------------------
            filename       -- file to load the ELM_Network instance       
                    
        """
        f  = open(filename,'r')
        linea = f.readline()
        self.inputs,self.hidden,self.fo = map(int,linea.split(';'))
        linea = f.readline()
        self.w= np.array(map(float,linea.split(' '))).reshape((self.hidden,self.inputs))
        linea = f.readline()
        self.bias = np.array(map(float,linea.split(';')))
        linea = f.readline()
        self.betas = np.array(map(float,linea.split(';')))
        self.beta_0 = float(f.readline())
        linea = f.readline()
        self.minimos = np.array(map(float,linea.split(';')))
        linea = f.readline()
        self.maximos = np.array(map(float,linea.split(';')))
        f.close()
        
       
    def training(self,entradas,targets):
        """ Training the network using Moore-Penrose Pseudo-inverse
        
            parameters:
            ---------------------------------------------
            entradas    -- numpy array with input vectors
            targets     -- numpy array with target values
          
        """
        H = np.zeros(shape=(len(entradas),self.hidden))
        for i in range(len(entradas)):
            H[i] = factivacion(self.w.dot(entradas[i])+self.bias,self.fo)
        self.betas = np.linalg.lstsq(H,targets.T)[0]
        
        output  = (np.dot(H,self.betas)+0.5)*(self.maximos[-1]-self.minimos[-1])+self.minimos[-1]
        dtarget = (targets+0.5)*(self.maximos[-1]-self.minimos[-1])+self.minimos[-1]
        return np.linalg.norm(output-dtarget)**2/len(entradas) 
