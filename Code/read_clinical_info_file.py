# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:00:09 2020

@author: Michael
"""
import pandas as pd
import numpy as np
import os




label_list =[]
sample_id_list =[]

def read_clinical_info_file(clinical_info_file, mzml_path):
    
    
    if clinical_info_file is not None:
        # 1. Read in patient and sample id from clinical info file.
        clinical_info = pd.read_csv(clinical_info_file)
        clinical_info_selected = clinical_info[(clinical_info['Current criteria']=='TB') | (clinical_info['Current criteria']=='Not TB')] 
        global label_list2
        label_list = clinical_info_selected['Current criteria'].values
        print(f"Propotion of positive samples: {np.mean(label_list=='TB')}")
        
        label_list = clinical_info_selected['Current criteria'].values
        label_list = (label_list == 'TB').astype('uint8') # Convert label 'Confirmed TB' and 'Unlikely TB' to 1 and 0
   
   
        
        
        
        def format_sample_id(x):
            return str(x).replace(' ', '').replace(')', '').replace('(', '_')
        clinical_info_selected_sample_id = clinical_info_selected['Sample ID'].values
     
        sample_id_list = [x for x in clinical_info_selected_sample_id]
    
        
    
    else:
        label_list = None
        file_list = os.listdir(mzml_path)
        sample_id_list = [os.path.splitext(file)[0] for file in file_list if os.path.splitext(file)[1].lower() == ".mzml"]
        
    return label_list, sample_id_list