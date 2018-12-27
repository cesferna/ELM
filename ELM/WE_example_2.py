# -*- coding: utf-8 -*-
import ELM_Network as elm
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

""" Fix the random generator sequence for reproducibility"""
elm.reproducibility()

""" Load the data set from numpy array file"""
warnings.filterwarnings("ignore")

""" Load the data set from numpy array and CSV files"""
modes = ['periodization','symmetric','reflect','smooth','constant','zero','periodic']
sem = modes[0]

""" Select the mother wavelet according to available functions in pywavelets"""
wavelet = 'db1'

stocks = ['IBM','CSCO','HPQ','INTC','MSFT','VZ']
stock  = stocks[5]

path_base = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
path_data = path_base+'/data/'

tserie = np.load(path_data+stock+'.npy')


""" Network topology setting """
""" Note that inputs must be a power of two"""
inputs = 16
hidden = 4
fo     = 5 

""" Creating the ELM network"""
net = elm.ELM_Network(inputs,hidden,fo)

patterns,targets   = net.training_set_wavelet(tserie,inputs,wavelet,sem)


""" Portion of the data for training and forecasting """
trn_num        = int(0.7*len(patterns))
tst_num        = len(patterns) - trn_num
trn_patterns    = patterns[0:trn_num]
trn_targets     = targets[0:trn_num]
tst_patetrn     = patterns[trn_num:]

""" Training the network using training set """
trn_error = net.training(trn_patterns,trn_targets)
print "Training Error: ",trn_error

""" Save the network parameters to a file """
net.to_file('network_parameters_test.csv')

forecasting = np.zeros(len(tst_patetrn))
for i in range(len(tst_patetrn)):
    forecasting[i] = net.dnorm_forward_pass(tst_patetrn[i])


"""Forecasting performance assessment"""
metricas = elm.metricas(forecasting,tserie[-len(tst_patetrn):])
print "MAE: ",metricas[0]
print "MSE: ",metricas[1]
print "RMSE: ",metricas[2]
print "MAPE: ",metricas[3]
print "THEIL: ",metricas[4]
print "--------------------------------------------"