# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:53:47 2019
FAV harmonization. Please see the Jupyter Notebook (_) or the paper () for 
a more detailed explanation. Briefly, 
@author: Tim Rodgers
"""
import pandas as pd
import numpy as np
import math
import uncertainties
from uncertainties import ufloat
from uncertainties import unumpy
import pymc3 as pm
import arviz as az
import pdb
class Bayes_FAV():
    """Bayesian harmonization of physical-chemical properties. 
    estimate final adjusted values (FAVs) for a list of compounds using the 
    Bayesian regression model of Rodgers et al (in prep). FAVs are physical-chemical
    properties adjusted for thermodynamic consistentcy following the "three solubilities"
    approach (Cole and Mackay, 2000; Beyer et al., 2002). This procedure will use a
    Bayesian regression model to determine the likeliest probability distribution
    function for each measured value, based on the uncertainties in the measured 
    values for each property. Rather than assuming that for every property there is a
    "correct" value, we acknowledge that there will always be uncertainty and so use 
    the literature estimates and thermodynamic relationships between properties
    to estimate that uncertainty and provide our degree of belief sbout the range
    of values for each property. 
    
    The properties of interest are:
        Solubilities
        Subcooled liquid vapour pressure (PL/Pa) - converted to subcooled liquid air solubility (SAL/mol m-3) as SAL = PL/RT (Cole & Mackay, 2000)
        Subcooled liquid water solubilty (SWL/mol m-3)
        Subcooled liquid octanol solubility (SOL/mol m-3)
        
        Dimensionless Partition Coefficients
        Octanol/Water partition coefficient (KOW)
        Octanol/Air partition coefficient (KOA)
        Air/Water partition coefficent (KAW) - can be derived from Henry's law constant (H/Pa m3 mol-1) as KAW = H/RT
        
        Internal Energies of Phase Change - one for each solubility & partition system
        dUA
        dUW
        dUO
        dUOW
        dUOA
        dUAW
        
    Attributes:
    ----------
            
            LDVs (df): Literature derived values for all properties. Can include uncertainties using the "uncertainties"
            package or just be nominal values. Index should be the compound name, then you should have each property followed by its
            standard deviation (if they aren't included together)
            (optional) LDV_sigma (df): standard deviations of the LDVs, if not included in the LDVs. 
            Must be positive.
            startcol: First column with an uncertain value. Default is 3 from the example dataset, which 
            has a column for the compound class and the display order
            Colnames (list)- names of columns. Default order is:
                colnames =['dVAPH','dWH','dOH','dOAU','dAWU','dOWU','LogKOA','LogKOW','LogKAW','LogPL','LogSW','LogSO','dfusS','Tm']
            

    """
    colnames =['dVAPH','dWU','dOU','dOAU','dAWU','dOWU','LogKOA','LogKOW','LogKAW','LogPL','LogSW','LogSO','dfusS','Tm']
    def __init__(self,LDVs,startcol = 3,colnames = colnames):
        #Initialize the data. This will take the input dataframe and combine the nominal values
        #and standard deviations using the uncertainties package
        self.startcol = startcol
        self.colnames = colnames
        self.LDVs = self.uLDV_calc(LDVs,startcol, colnames)
        
    def uLDV_calc(self,LDVs,startcol,colnames):
        if type(LDVs.iloc[0,startcol]) != uncertainties.core.Variable: #Check if it is already an uncertain variable
            res = pd.DataFrame(index = LDVs.index) #initialize the dataframe with the same index as LDVs
            i = 0
            for idx,cols in enumerate(LDVs.columns[startcol:]):
                #pdb.set_trace()
                if idx/2 != math.ceil(idx/2): #Skip every other column, which have the standard deviations.
                    pass
                else:
                    #print(idx)
                    colname = colnames[i]
                    res.loc[:,colname] = unumpy.uarray(LDVs.loc[:,cols],LDVs.iloc[:,idx+startcol+1])#adding 5 to get to first SD value
                    i+=1
            #Then, lets convert our vapour pressure to a solubility as Sa = VP/RT
            R = 8.314 #J/molK
            T = 298.15 #K
            res.loc[:,'LogSA'] = unumpy.log10(10**(res.LogPL)/(R*T))#Convert to mol/m³
            res.loc[res.LogPL==0,'LogSA'] = np.nan #Convert zeros to nan
            #Here we will also convert the wet octanol partition coefficient to dry so that we are all on the same page
            res.loc[:,'LogKOWd'] = 1.35*res.LogKOW - 1.58
            res.loc[res.LogKOW==0,'LogKOWd'] = np.nan #Convert zeros to nan
            res = res.replace(0,np.nan) #Convert all zeros to nans for the next part
        else:
            #Convert vapour pressure to solubility and Kow to dry Kow (Kowd)
            R = 8.314 #J/molK
            T = 298.15 #K
            res = LDVs
            res.loc[:,'LogSA'] = unumpy.log10(10**(res.LogPL)/(R*T))#Convert to mol/m³
            res.loc[res.LogPL==0,'LogSA'] = np.nan #Convert zeros to nan
            #Here we will also convert the wet octanol partition coefficient to dry so that we are all on the same page
            res.loc[:,'LogKOWd'] = 1.35*res.LogKOW - 1.58
            #convert to internal energy of phase change (dU) rather than enthalpies (dH) (see Goss 1996 DOI 10.1021/es950508f) using rel.
            #found by Beyer et al. (2002) for PCBs, namely dAU = dvapH - 2,391 J/mol. Like Beyer, we will assume that all enthalpies of
            #phase change in the water phase are actually internal energies, as they were measured volumetrically.
            res.loc[:,'LogdUA'] = res.dVAPH - 2391
            res.loc[res.LogKOW==0,'LogKOWd'] = np.nan #Convert zeros to nan
            res = res.replace(0,np.nan) #Convert all zeros to nans for the next part
            res = LDVs
        return res
    
    def FAV_Partition(self,comps,LDVs,sigma = 1., tune_n = 5000, trace_n = 5000):
        '''
        So, here is the meat of the business for partition coefficients! This will
        set up and solve the Bayesian regression model to harmonize the values, which will
        give us our FAV PDFs at the end. Luckily, overall we just have a system
        of 3 equations and 6 unknowns, if all property values are known. At most 3
        #property values can be missing for a valid system (e.g logKaw - logKow + logKoa = 0)
        Attributes:
            ----------
        comps (list): List of compounds that you want to determine FAVs for. This must match 
        the compound in the index of LDVs
        LDVs: as above
        sigma (float): beta value for the halfcauchy distribution describing the uncertainty of the overall model regression
        '''
        LDVs = self.LDVs #Make sure this form is correct
        #Set up the overall matrix
        basemat = np.array([[1,-1,0,-1,0,0],[1,0,-1,0,0,1],[0,1,-1,0,1,0]]) #Order is: Sa, Sw, So, Kaw, Kow, Koa
        pdb.set_trace()
        #comps = {comps}
        for comp in comps:#Run each compound individually
            #now we are going to go through and set up the model depending on what
            #properties are known. There is probably a more efficient & clever way to do this
            #Define a vector with the properties that are present. This order is fixed as: Sa, Sw, So, Kaw, Kow, Koa
            propsabsent = unumpy.isnan([LDVs.loc[comp,'LogSA'],LDVs.loc[comp,'LogSW']\
                                         ,LDVs.loc[comp,'LogSO'],LDVs.loc[comp,'LogKAW'],\
                                         LDVs.loc[comp,'LogKOW'],LDVs.loc[comp,'LogKOA']])
            if propsabsent.sum() == 0: #All properties are present - best case scenario!
                #Set up the following system of equations
                #LogSA*(1)+ LogSw *(-1) + LogKAW *(-1)= 0
                #LogSA*(1)+ LogSo *(-1) + LogKOA *(1)= 0
                #LogSw*(1)+ LogSo *(-1) + LogKOA *(1)= 0
                #in matrix form:
                X = basemat
                Y = 0
                #y = x*beta
                props_model = pm.Model()
                beta = [0,0,0,0,0,0] #initialize our parameter vector
                #y = x*beta
                #Then, set up our Bayesian model with all compounds!
                with props_model:
                    #Define priors
                    sigma = pm.HalfCauchy('sigma', beta=1., testval=1.) #Model error
                    #alpha = pm.Normal('alpha', 0, sigma=1e-3) #Acceptable misclosure error
                    #For the phys-chem properties, we have some choices of priors depending on our confidence in the data.
                    #For our defaults we can use normal distributions, since we have measured values and have adjusted them to account
                    #for our level of confidence. Of course, th
                    beta[0] = pm.Normal('LogSA', mu = LDVs.loc[comp,'LogSA'].n, sigma=LDVs.loc[comp,'LogSA'].s) #.n = nominal value and .s = std dev.
                    beta[1] = pm.Normal('LogSW', mu = LDVs.loc[comp,'LogSW'].n, sigma=LDVs.loc[comp,'LogSW'].s)
                    beta[2] = pm.Normal('LogSO', mu = LDVs.loc[comp,'LogSO'].n, sigma=LDVs.loc[comp,'LogSO'].s)
                    beta[3] = pm.Normal('LogKAW', mu = LDVs.loc[comp,'LogKAW'].n, sigma=LDVs.loc[comp,'LogKAW'].s) #.n = nominal value and .s = std dev.
                    #Kow can be converted to 
                    #beta[4] = pm.Normal('LogKOWd', mu = LDVs.loc[comp,'LogKOWd'].n, sigma=LDVs.loc[comp,'LogKOWd'].s) 
                    beta[4] = pm.Normal('LogKOW', mu = LDVs.loc[comp,'LogKOW'].n, sigma=LDVs.loc[comp,'LogKOW'].s)
                    beta[5] = pm.Normal('LogKOA', mu = LDVs.loc[comp,'LogKOA'].n, sigma=LDVs.loc[comp,'LogKOA'].s)
                    #Students T distribution has fatter tails, so implies less confidence in the values. We can use this as a way of saying
                    #that we think there might be some bias in the measured values.
                    #beta[3] = pm.StudentT('LogKOW', nu = 1,mu = LDVs.loc[comp,'LogKOW'].n, sd=LDVs.loc[comp,'LogKOW'].s) #.n = nominal value and .s = std dev.
                    #beta[4] = pm.StudentT('LogKAW', nu = 1,mu = LDVs.loc[comp,'LogKAW'].n, sd=LDVs.loc[comp,'LogKAW'].s)
                    #beta[5] = pm.StudentT('LogKOA', nu = 1,mu = LDVs.loc[comp,'LogKOA'].n, sd=LDVs.loc[comp,'LogKOA'].s)
                    #beta[3] = 
                    
                    epsilon = np.dot(X,beta) #This gives us the misclosure error for each of the three equations
                    #The model tries to fit to an observation, in this case that the misclosure should be 0.
                    mu = np.sum(epsilon**2) #Here we are going to use the sum of squares, equivalent to the Schenker method, but Bayesian
                    # Likelihood (sampling distribution) of observations
                    #Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)#Normal
                    Y_obs = pm.StudentT('Y_obs',nu = 3, mu=mu, sigma=sigma, observed=Y)#Students for robust regression (broader tails)
                    res = pm.sample(trace = trace_n, tune=tune_n)
            elif propsabsent.sum() == 1: #Only one property missing - several cases here
                #if SA is missing:
                X = np.array([basemat[0] - basemat[1],basemat[2]])
                X = X[:,1:]
                #Set up the following system of equations
                #LogSA*(1)+ LogSw *(-1) + LogKAW *(-1)= 0
                #LogSA*(1)+ LogSo *(-1) + LogKOA *(1)= 0
                #LogSw*(1)+ LogSo *(-1) + LogKOA *(1)= 0
                #in matrix form:
                Y = 0
                #y = x*beta
                beta = [0,0,0,0,0] #initialize our parameter vector
                props_model = pm.Model()
                #y = x*beta
                #Then, set up our Bayesian model with all compounds!
                with props_model:
                    #Define priors
                    sigma = pm.HalfCauchy('sigma', beta=1., testval=1.) #Model error
                    #alpha = pm.Normal('alpha', 0, sigma=1e-3) #Acceptable misclosure error
                    #For the phys-chem properties, we have some choices of priors depending on our confidence in the data.
                    #For our defaults we can use normal distributions, since we have measured values and have adjusted them to account
                    #for our level of confidence. Of course, th
                    #beta[0] = 0#pm.Normal('LogSA', mu = LDVs.loc[comp,'LogSA'].n, sigma=LDVs.loc[comp,'LogSA'].s) #.n = nominal value and .s = std dev.
                    beta[0] = pm.Normal('LogSW', mu = LDVs.loc[comp,'LogSW'].n, sigma=LDVs.loc[comp,'LogSW'].s)
                    beta[1] = pm.Normal('LogSO', mu = LDVs.loc[comp,'LogSO'].n, sigma=LDVs.loc[comp,'LogSO'].s)
                    beta[2] = pm.Normal('LogKAW', mu = LDVs.loc[comp,'LogKAW'].n, sigma=LDVs.loc[comp,'LogKAW'].s) #.n = nominal value and .s = std dev.
                    #Kow can be converted to 
                    #beta[4] = pm.Normal('LogKOWd', mu = LDVs.loc[comp,'LogKOWd'].n, sigma=LDVs.loc[comp,'LogKOWd'].s) 
                    beta[3] = pm.Normal('LogKOW', mu = LDVs.loc[comp,'LogKOW'].n, sigma=LDVs.loc[comp,'LogKOW'].s)
                    beta[4] = pm.Normal('LogKOA', mu = LDVs.loc[comp,'LogKOA'].n, sigma=LDVs.loc[comp,'LogKOA'].s)
                    #Students T distribution has fatter tails, so implies less confidence in the values. We can use this as a way of saying
                    #that we think there might be some bias in the measured values.
                    #beta[3] = pm.StudentT('LogKOW', nu = 1,mu = LDVs.loc[comp,'LogKOW'].n, sd=LDVs.loc[comp,'LogKOW'].s) #.n = nominal value and .s = std dev.
                    #beta[4] = pm.StudentT('LogKAW', nu = 1,mu = LDVs.loc[comp,'LogKAW'].n, sd=LDVs.loc[comp,'LogKAW'].s)
                    #beta[5] = pm.StudentT('LogKOA', nu = 1,mu = LDVs.loc[comp,'LogKOA'].n, sd=LDVs.loc[comp,'LogKOA'].s)
                    #beta[3] = 
                    
                    epsilon = np.dot(X,beta) #This gives us the misclosure error for each of the three equations
                    #The model tries to fit to an observation, in this case that the misclosure should be 0.
                    mu = np.sum(epsilon**2) #Here we are going to use the sum of squares, equivalent to the Schenker method, but Bayesian
                    
                    # Likelihood (sampling distribution) of observations
                    #Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)#Normal
                    Y_obs = pm.StudentT('Y_obs',nu = 3, mu=mu, sigma=sigma, observed=Y)#Students for robust regression (broader tails)
                    res = pm.sample(trace = trace, tune=tune)
                
        return res
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        