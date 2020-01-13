# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 14:57:43 2020

@author: Tim Rodgers
"""

#Code for running the FAVs
#LDV = pd.read_excel('LDV_Data_TESTMODIFIED.xlsx', index_col = 2)
LDV = pd.read_excel('LDV_Data.xlsx', index_col = 2)
test = Bayes_FAV(LDV)
#comps = LDV.index.values
comps = {"p,p'- DDE"}
xx = test.FAV_Partition(comps,LDV)