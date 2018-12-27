# -*- coding: utf-8 -*-
import ELM_Network as elm
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

""" Fix the random generator sequence for reproducibility"""
elm.reproducibility()

""" Load the data set from numpy array file"""

stock = 'INTC'
modes = ['periodization','symmetric','reflect','smooth','constant','zero','periodic']
sem = modes[0]
wavelet = 'sym2'

#""" Load the data set from numpy array file"""
stocks = ['IBM','CSCO','HPQ','INTC','MSFT','VZ']
stock  = stocks[-1]
path_base = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
path_data = path_base+'/data/'

tserie = np.load(path_data+stock+'.npy')

"""Networks topology setting """
elm_inputs = 8
elm_hidden = 4
elm_fo     = 5 

we_inputs = 8
we_hidden = 10
we_fo     = 5

net_ELM = elm.ELM_Network(elm_inputs,elm_hidden,elm_fo)
net_WE = elm.ELM_Network(we_inputs,we_hidden,we_fo)

betas = np.zeros(3)

""" Normalize the input and target """
patterns,targets       = net_ELM.normalizar(tserie,elm_inputs)
w_patterns,w_targets   = net_WE.training_set_wavelet(tserie,we_inputs,wavelet,sem)

min_values = net_ELM.get_min()
max_values =  net_ELM.get_max()


""" Portion of the data for training and forecasting """
trn_num        = int(0.7*len(patterns))
tst_num        = len(patterns) - trn_num
elm_trn_patterns     = patterns[0:trn_num]
elm_trn_targets      = targets[0:trn_num]
elm_tst_patterns     = patterns[trn_num:]

we_trn_patterns     = w_patterns[0:trn_num]
we_trn_targets      = w_targets[0:trn_num]
we_tst_patterns     = w_patterns[trn_num:]


""" Training the network using training set """
#trn_error = net.training(trn_patterns,trn_targets)
net_ELM.training(elm_trn_patterns,elm_trn_targets)
net_WE.training(we_trn_patterns,we_trn_targets)

t_matrix = np.ones((trn_num,3))
for h in range(trn_num):
    t_matrix[h][0] = net_WE.forward_pass(we_trn_patterns[h])
    t_matrix[h][1] = net_ELM.forward_pass(elm_trn_patterns[h])
    
H = np.matrix(t_matrix)
T = np.asmatrix(elm_trn_targets).T
betas = np.linalg.lstsq(H,T)[0]
trn_error = np.linalg.norm(np.dot(H,betas)-T)**2/trn_num
print "--------------------------------------------"
print "Training Error: ",trn_error
print "--------------------------------------------"


""" Save the network parameters to a file """
net_WE.to_file('ELM_network_parameters.csv')
net_ELM.to_file('WE_network_parameters.csv')
np.savetxt('betas.csv',betas,delimiter=";")


""" Forecasting with out-of-the-sample data """
min_values = net_ELM.get_min()
max_values = net_ELM.get_max()

forecasting = np.zeros(len(elm_tst_patterns))
for i in range(len(elm_tst_patterns)):
    elm_predict = net_ELM.forward_pass(elm_tst_patterns[i])
    we_predict =  net_WE.forward_pass(we_tst_patterns[i])
    output = elm_predict*betas[1] + we_predict*betas[0] + betas[2]
    forecasting[i] = (output+0.5)*(max_values[-1]-min_values[-1])+min_values[-1]

"""Forecasting performance assessment"""
print "--------------------------------------------"
metricas = elm.metricas(forecasting,tserie[-len(elm_tst_patterns):])
print "MAE: ",round(metricas[0],5)
print "MSE: ",round(metricas[1],5)
print "RMSE: ",round(metricas[2],5)
print "MAPE: ",round(metricas[3],5)
print "THEIL: ",round(metricas[4],5)
print "--------------------------------------------"