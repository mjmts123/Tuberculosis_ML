# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 06:43:38 2020

@author: admin
"""

# In[1.1 Change working directory and data folder]

import os
import numpy as np
from scipy import stats
import itertools
import pandas as pd
import pymzml
import pickle
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve


import builtins

os.chdir("C:\\Users\\Michael\\Desktop\\Machine_Learning\\Machine_Learning_Project_to_Michael - 7_1_20\\Machine_Learning_Project_to_Michael") # identify the working directionary


#List of all file directories and csv files

values = [["_Italy_train", "Total_Italy_train.csv"],

          ["_liz_train", "Liz_train.csv"]]



classifiers = []


model_save={}


lst = ["Morphological", "Statistical", "M+S"]



results_table = pd.DataFrame(columns = ["Dataset", "Base Model Number", "Features", "Scores", "Models", "values"])






#True if the mzml files have already been transformed and there is already a saved_data.pkl
mzml_files_transformed = True


#True if models have been trained off training dataset, and best models have been selected

models_trained_saved = False



best_index = {}
if models_trained_saved:
    best_index = {'LR1': '2',
    'RF1': '10',
    'GB1': '16',
    'LR2': '1',
    'RF2': '2',
    'GB2': '16',
    'LR3': '1',
    'RF3': '8',
    'GB3': '16'}

    
    

# =============================================================================
# if models_trained_saved:
#     best_index = {'LR1': '3',
#  'RF1': '15',
#  'GB1': '39',
#  'LR2': '3',
#  'RF2': '15',
#  'GB2': '39',
#  'LR3': '3',
#  'RF3': '15',
#  'GB3': '39'}
# =============================================================================

#this loops through the train, test and independent files
for suffix, file_name in values:
    

    
    '''
    suffix:
        _Italy_train
        _Italy_valid
        _CSTB
        
    file name:
        Total_Italy_train
        Total_Italy_valid
        CSTB
    '''
    
    

    
    mzml_path = f"data{suffix}/mzml_data_conversion_SRMspectra/" # folder path to mzml mass spec files
    clinical_info_file = f"data{suffix}/{file_name}" # change the filename

    
    
    from Code.read_clinical_info_file import *
    label_list, sample_id_list = read_clinical_info_file(clinical_info_file, mzml_path)
    

    # In[1.2 Transform mzml files into readable files]
    # Get the list of mzML files
    if not mzml_files_transformed:
        from Code.transform_mzml_files import *
        transform_mzml_files(mzml_path, suffix)
        
    # In[2.1 Training Data]
    from Code.training_data import *
    model_save, best_index = training_data(suffix, sample_id_list, label_list, 
                                           models_trained_saved, model_save, best_index)
            
            
        
                                      
   

# In[3.1 load train best model to make prediction]
    #This for loop tests all the best models saved in the three categories (morphological, statistical, m+s), 
    #and runs them against the correct data saved in Model_save
from Code.prediction import *
classifiers = prediction(values, lst, best_index, model_save, results_table, classifiers)
    # Define a result table as a DataFrame

                
#%%       
#Puts fpr, tpr, and auc values into a results table
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])


from Code.generate_result_table import *
result_table = generate_results_table(result_table, classifiers, model_save)


# result_table.set_index('classifiers', inplace=True)


#%%
#takes results table and generates ROC charts

generate_results_table(result_table, classifiers, model_save)

print("Finished")