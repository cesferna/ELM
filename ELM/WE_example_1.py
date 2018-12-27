# -*- coding: utf-8 -*-
import ELM_Network as elm
import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

""" Load the data set from numpy array and CSV files"""
modes = ['periodization','symmetric','reflect','smooth','constant','zero','periodic']
sem = modes[0]
wavelet = 'db1'

stocks = ['IBM','CSCO','HPQ','INTC','MSFT','VZ']
stock  = stocks[5]

path_base = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 
path_data = path_base+'/data/'

tserie = np.load(path_data+stock+'.npy')
path_file = path_base+"/settings/WE/"+stock+"/"+stock+"_"+wavelet+"_network_setting.csv"


net = elm.ELM_Network(2,2,2)
net.from_file(path_file)

inputs,hidden,fo =net.get_topology()

""" Normalize the input and target """
patterns,targets   = net.training_set_wavelet(tserie,inputs,wavelet,sem)


""" Portion of the data for training and forecasting """
trn_num        = int(0.7*len(patterns))
tst_num        = len(patterns) - trn_num
trn_patterns    = patterns[0:trn_num]
trn_targets     = targets[0:trn_num]
tst_patetrn     = patterns[trn_num:]


""" Forecasting with out-of-the-sample data """
forecasting = np.zeros(len(tst_patetrn))
for i in range(len(tst_patetrn)):
    forecasting[i] = net.dnorm_forward_pass(tst_patetrn[i])

"""Forecasting performance assessment"""
print "--------------------------------------------"
metricas = elm.metricas(forecasting,tserie[-len(tst_patetrn):])
print "MAE  : ",round(metricas[0],5)
print "MSE  : ",round(metricas[1],5)
print "RMSE : ",round(metricas[2],5)
print "THEIL: ",round(metricas[4],5)
print "--------------------------------------------"