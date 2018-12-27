# -*- coding: utf-8 -*-
import ELM_Network as elm
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

""" Load the data set from numpy array file"""
stock = 'INTC'
modes = ['periodization','symmetric','reflect','smooth','constant','zero','periodic']
sem = modes[0]
wavelet = 'db1'

#""" Load the data set from numpy array file"""
stocks = ['IBM','CSCO','HPQ','INTC','MSFT','VZ']
stock  = stocks[5]
path_base = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
path_data = path_base+'/data/'

tserie = np.load(path_data+stock+'.npy')

path_file_WE = path_base+"/settings/ELM_WE/"+stock+"/"+stock+"_"+wavelet+"_network_setting.csv"
path_file_ELM = path_base+"/settings/ELM_WE/"+stock+"/"+stock+"_network_setting.csv"
path_betas = path_base+"/settings/ELM_WE/"+stock+"/"+"betas.csv"


""" Creating the ELM network"""
net_ELM = elm.ELM_Network(2,2,2)
net_WE = elm.ELM_Network(2,2,2)
net_ELM.from_file(path_file_ELM)
net_WE.from_file(path_file_WE)
betas = np.loadtxt(path_betas, delimiter=';',dtype = float)


""" Normalize the input and target """
inputs,elm_hidden,elm_fo =net_ELM.get_topology()
_,we_hidden,we_fo =net_ELM.get_topology()

patterns,targets       = net_ELM.normalizar(tserie,inputs)
w_patterns,w_targets   = net_WE.training_set_wavelet(tserie,inputs,wavelet,sem)

min_values = net_ELM.get_min()
max_values =  net_ELM.get_max()


""" Portion of the data for training and forecasting """
trn_num        = int(0.7*len(patterns))
tst_num        = len(patterns) - trn_num
elm_trn_patterns    = patterns[0:trn_num]
elm_tst_patterns     = patterns[trn_num:]

we_trn_patterns    = w_patterns[0:trn_num]
we_tst_patterns     = w_patterns[trn_num:]


""" Forecasting with out-of-the-sample data """
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