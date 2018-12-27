# -*- coding: utf-8 -*-
import ELM_Network as elm
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

#""" Fix the random generator sequence for reproducibility"""
elm.reproducibility()

#""" Load the data set from numpy array file"""
stocks = ['IBM','CSCO','HPQ','INTC','MSFT','VZ']
stock  = stocks[-1]
path_base = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
path_data = path_base+'/data/'

tserie = np.load(path_data+stock+'.npy')


"""Network topology setting """
inputs = 5
hidden = 4
fo     = 5 

""" Creating the ELM network"""
net = elm.ELM_Network(inputs,hidden,fo)


""" Normalize the input and target """
inputs,hidden,fo =net.get_topology()
patterns,targets   = net.normalizar(tserie,inputs)

""" Portion of the data for training and forecasting """
trn_num        = int(0.7*len(patterns))
tst_num        = len(patterns) - trn_num
trn_patterns    = patterns[0:trn_num]
trn_targets     = targets[0:trn_num]
tst_patetrn     = patterns[trn_num:]


""" Training the network using training set """
trn_error = net.training(trn_patterns,trn_targets)
print "--------------------------------------------"
print "Training Error: ",trn_error
print "--------------------------------------------"

""" Save the network parameters to a file """
net.to_file('network_parameters.csv')

""" Forecasting with out-of-the-sample data """
forecasting = np.zeros(len(tst_patetrn))
for i in range(len(tst_patetrn)):
    forecasting[i] = net.dnorm_forward_pass(tst_patetrn[i])

"""Forecasting performance assessment"""
metricas = elm.metricas(forecasting,tserie[-len(tst_patetrn):])
print "MAE: ",round(metricas[0],5)
print "MSE: ",round(metricas[1],5)
print "RMSE: ",round(metricas[2],5)
print "MAPE: ",round(metricas[3],5)
print "THEIL: ",round(metricas[4],5)
print "--------------------------------------------"