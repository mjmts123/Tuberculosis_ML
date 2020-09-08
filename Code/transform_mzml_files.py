# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 20:23:37 2020

@author: Michael
"""
import pandas as pd
import numpy as np
import os
import pymzml
import math
import pickle


def transform_mzml_files(mzml_path, suffix):
    file_list = os.listdir(mzml_path)
    sample_id_list = [os.path.splitext(file)[0] for file in file_list if os.path.splitext(file)[1].lower() == ".mzml"]
    
    # 3. Extract features from mass spec data. List of SRM M/z: '622', '693', '822', '950', '1021', '1134', '1235'.
    # We look for scantime corresponding to the maximum total ion concentration (TIC) of internal standard.
    # Then, we extract ion intensity, relative abundance and ratio between target/internal standard of different M/z(s)
    # Sample 314622_1.mzml might be mistyped as 314662_1.mzml. Please manually correct the file name before running the below code.
    
    scan_range = 20 # Range around standard_max_index and tartget_max_index to extract feature
    min_scan_minute = 10 # We are only interested in range (min_scan_minute, max_scan_minute)
    max_scan_minute = 25 # We are only interested in range (min_scan_minute, max_scan_minute)
    period_second = 60 # Windows to extract features for non-targeted region
    nb_periods = int(np.ceil((max_scan_minute - min_scan_minute)*60/period_second))
    
    
    maxstd_scanid_list = [] # list to store scan id corresponding to the maximum TIC of internal standard for each sample
    maxstd_scantime_list = [] # list to store scantime corresponding to the maximum TIC of internal standard for each sample
    
    # Next Features are new from TargetQCMS package in R
    features_tic_peakmax_list = []
    features_tic_relmax_list = []
    features_tic_peakarea_list = []
    features_tic_peakratio_list = []
    features_tic_peakshift_list = []
    features_tic_elutionShift_list =  []
    features_tic_similarity_list =  []
    features_tic_symmetry_list = []
    features_tic_jaggedness_list = []
    features_tic_FWHM_list = []
    features_tic_modality_list = []
    
    features_transition_peakmax_list = []
    features_transition_relmax_list = []
    features_transition_peakarea_list = []
    features_transition_peakratio_list = []
    features_transition_peakshift_list = []
    features_transition_elutionShift_list =  []
    features_transition_similarity_list =  []
    features_transition_symmetry_list = []
    features_transition_jaggedness_list = []
    features_transition_FWHM_list = []
    features_transition_modality_list = []
    
    for sample_id in sample_id_list:
    #    sample_id = 'PAVLE' # positive-PAVLE negative-MASIM 
        mzml_file = os.path.join(mzml_path, sample_id + '.mzML')
        if os.path.exists(mzml_file):
            print(f"Processing {sample_id} ...")
            
            # Extract SRM
            run = pymzml.run.Reader(mzml_file)
            standard_scanid = []
            standard_scantime = []
            standard_intensity = []
            standard_tic = []
            target_scanid = []
            target_intensity = []
            target_tic = []
            for spectrum in run:
                if spectrum.ms_level == 2:
                    selected_precursor = spectrum.selected_precursors[0]['mz']
                    if np.round(selected_precursor, 2) == 802.37: # Precursor corresponding to Internal Standard
                        standard_scanid.append(spectrum.ID)
                        standard_scantime.append(float(spectrum.get_element_by_path(['scanList', 'scan', 'cvParam'])[0].get('value')))
                        standard_intensity.append(spectrum.i)
                        standard_tic.append(spectrum.TIC)
                    if np.round(selected_precursor, 2) == 797.38: # Precursor corresponding to Target
                        target_scanid.append(spectrum.ID)
                        target_intensity.append(spectrum.i)
                        target_tic.append(spectrum.TIC)
    
            standard_tic = np.array(standard_tic)
            target_tic = np.array(target_tic)
            standard_scantime = np.array(standard_scantime)
            standard_max_index = np.argmax(standard_tic[standard_scantime < 20], axis=0) # index corresponding to max TIC of internal standard
            target_max_index = np.argmax(target_tic, axis=0) # index corresponding to max TIC of target
          
           
            ## Define the peak boundaries by own calculation, we assume that the standard peak obey normal distribution.
            ## The peak boundaries are the 95% confidence interval's lower and upper thresholds.
         
    
            temp = target_tic[target_max_index - 20:target_max_index + 20]
            temp1 = standard_tic[standard_max_index - 20:standard_max_index + 20]
            temp1 = (np.round(temp1,0)).astype(int)
            new = []
            for i in range(40):
                if i == 1:
                    new = [1]*temp1[1]
                else:
                    new.extend([i]*temp1[i])
            thr = math.floor(np.std(new)*1.96) # if mean scan is 0, the boundaries are (-thr, thr)
        
            ## define normalize function for next use
            def normalize(lst):
                if max(lst) != 0:
                    return [(float(i)-min(lst))/(max(lst)-min(lst)) for i in lst]
                else:
                    return lst
    
            ## 1.1 Target Maximum in peak boundaries
            features_tic_peakmax = []
            tmax = max(target_tic[standard_max_index - thr:standard_max_index + thr])
            features_tic_peakmax.append(tmax)
    
            ## 1.1.2 Target Maximum / Standard Maximum in peak boundaries
            features_tic_relmax = []
            smax = max(standard_tic[standard_max_index - thr:standard_max_index + thr])
            if smax == 0:
                features_tic_relmax.append(0)
            else:
                features_tic_relmax.append(tmax/smax)
            
            ### 1.2 Target transitions Maximum in peak boundaries
            features_transition_peakmax = []
            temp1 = pd.DataFrame(target_intensity[standard_max_index-thr:standard_max_index+thr])
            temp = pd.DataFrame(standard_intensity[standard_max_index-thr:standard_max_index+thr])
            for x in range(7):
                features_transition_peakmax.append(max(temp1[x]))   
            
            ### 1.2.2 Target transitions maximum / standard maximum in peak boundaries
            features_transition_relmax = []            
            for x in range(7):
                features_transition_relmax.append((max(temp1[x]))/(max(temp[x])))         
    
                
            
            ## 2.1 Peak Area between peak boundaries, considered as the summation of each scan's intensity
            features_tic_peakarea = []
            features_tic_peakarea.append(sum(target_tic[standard_max_index - thr:standard_max_index + thr]))
               
            ### 2.2 Peak Area of transitions between peak boundaries
            features_transition_peakarea = []
            for x in range(7):
                features_transition_peakarea.append(sum(temp1[x]))
            
            ## 3.1 Peak ratio: Peak area under the target/Peak area under the standard
            features_tic_peakratio = []
            if max(standard_tic[standard_max_index - thr:standard_max_index + thr]) != 0:
                features_tic_peakratio.append((sum(target_tic[standard_max_index - thr:standard_max_index + thr]))/(sum(standard_tic[standard_max_index - thr:standard_max_index + thr])))
            else:
                features_tic_peakratio.append(0)
            
            ### 3.2 Peak ratio of transitions
            features_transition_peakratio = []
            for x in range(7):
                if max(standard_tic[standard_max_index - thr:standard_max_index + thr]) != 0:
                    features_transition_peakratio.append(sum(temp1[x])/(sum(temp[x])))
                else:
                    features_transition_peakratio.append(0)       
            
            
            ## 4.1 PeakShift: difference between standard and target peak's retention time (scan)
            features_tic_peakshift = []
            if len(temp) != 0:
                temp = target_tic[standard_max_index - thr:standard_max_index + thr]
                features_tic_peakshift.append(abs([n for n,i in enumerate(temp) if i == max(temp)][0] - (len(temp)/2)))
            else:
                features_tic_peakshift.append(999)
            
            ### 4.2 PeakShift of transitions 
            features_transition_peakshift = []
            for x in range(7):
                if len(temp) != 0:
                    temp = target_intensity[standard_max_index - thr:standard_max_index + thr]
                    features_transition_peakshift.append(abs([n for n,i in enumerate(temp[x]) if i == max(temp[x])][0] - (len(temp[x])/2)))
                else:
                    features_transition_peakshift.append(999)           
          
            ## 5.1 ElutionShift features 
            features_tic_elutionShift = []
            target_max_index = standard_max_index-thr+np.argmax(target_tic[standard_max_index-thr:standard_max_index+thr])
            temp = normalize(target_tic[target_max_index - thr:target_max_index + thr])
            features_tic_elutionShift.append(abs(temp[0]-temp[-1]))
    
            
            ### 5.2 ElutionShift_each_transition
            features_transition_elutionShift = []  
            temp = pd.DataFrame(target_intensity[target_max_index - thr:target_max_index + thr])
            for x in range(7):
                num = abs(normalize(temp[x])[len(temp)-1]-normalize(temp[x])[0])
                features_transition_elutionShift.append(num)
            
    
            ## 6.1 PeakShapeSimilarity
            features_tic_similarity = []
            sig1 = pd.Series(normalize(target_tic[target_max_index - thr:target_max_index + thr]))
            sig2 = pd.Series(normalize(standard_tic[standard_max_index - thr:standard_max_index + thr]))
            features_tic_similarity.append(abs(sig1).corr(sig2))
            
            ### 6.2 PeakShapeSimilarity_each_transition
            features_transition_similarity = []
            lst = target_intensity[target_max_index - thr:target_max_index + thr]
            lst = [lst[x].tolist() for x in range(len(lst))]
            for i in range(7):
                sigtr1 = pd.Series(normalize(list(zip(*lst))[i]))
                features_transition_similarity.append(abs(sigtr1).corr(sig2))
      
            ## 7.1 PeakSymmetry
            features_tic_symmetry = []
            pleft = pd.Series(normalize(target_tic[target_max_index-thr: target_max_index-1]))
            pright = pd.Series(normalize(target_tic[target_max_index+1: target_max_index+thr][::-1]))
            features_tic_symmetry.append(abs(pleft.corr(pright)))
    
    
            ### 7.2 PeakSymmetry_each_transition
            features_transition_symmetry = []
            pleft = target_intensity[target_max_index-thr: target_max_index-1] # transition at peaktime left part
            pright = target_intensity[target_max_index+1: target_max_index+thr][::-1] # transition at peaktime right part
            pleft = [pleft[x].tolist() for x in range(len(pleft))]
            pright = [pright[x].tolist() for x in range(len(pright))]
            for i in range(7):
                pltr = pd.Series(normalize(list(zip(*pleft))[i])) # In this code, we should use other list name, 'pltr != pleft'
                prtr = pd.Series(normalize(list(zip(*pright))[i]))
                features_transition_symmetry.append(abs(pltr).corr(prtr))
            features_transition_symmetry = np.nan_to_num(features_transition_symmetry, nan=0) # nan = 0 means this feature at ultra bad situation 
            features_transition_symmetry = features_transition_symmetry.tolist()
       
            ## 8.1 Jaggedness
            features_tic_jaggedness = []
            sig1 = pd.Series(normalize(standard_tic[standard_max_index-thr:standard_max_index+thr]))
            sig2 = pd.Series(normalize(target_tic[standard_max_index-thr:standard_max_index+thr]))
            dif = abs(sig1-sig2)
            new_dif = []
            for x in dif:
                if x<= 0.05*max(dif):
                    new_dif.append(0)
                else:
                    new_dif.append(x)
            features_tic_jaggedness.append((sum(new_dif)-0.05)/(len(new_dif)-0.05))
    
            ### 8.2 Jaggendeniss_each_transition
            features_transition_jaggedness = []
            temp = pd.DataFrame(standard_intensity[standard_max_index-thr:standard_max_index+thr])
            temp1 = pd.DataFrame(target_intensity[standard_max_index-thr:standard_max_index+thr])
            temp = temp.apply(normalize,axis=0)
            temp1 = temp1.apply(normalize,axis=0)
            dif = abs(temp.subtract(temp1))
            for x in range(7):
                features_transition_jaggedness.append((sum(dif[x])-0.05)/(len(dif[x])-0.05))
           
            ## 9.1 FWHM (Full width and half maximum)
            features_tic_FWHM = []
            if max(sig2) == 0:
                features_tic_FWHM.append(0)
            else: 
                pleft = [n for n,i in enumerate(sig2) if i - (max(sig2)/2) > 0][0]
                pright = [n for n,i in enumerate(sig2) if i - (max(sig2)/2) > 0][-1]
                features_tic_FWHM.append(pright-pleft)
            ### 9.2 FWHM_each_transition (Full width and half maximum)
            temp1[0]
            features_transition_FWHM = []
            for x in range(7):
                if max(temp1[x]) == 0:
                    features_transition_FWHM.append(0)
                else:        
                    pleft = [n for n,i in enumerate(temp1[x]) if i - (max(temp1[x])/2) > 0][0]
                    pright = [n for n,i in enumerate(temp1[x]) if i - (max(temp1[x])/2) > 0][-1]
                    features_transition_FWHM.append(pright-pleft)            
                
            ## 10.1 Modality
            features_tic_modality = []
            
            dif = sig1-sig2
            new_dif = []
            rais = []
            fall = []
            for x in dif:
                if abs(x) <= 0.05*max(dif): 
                    new_dif.append(0)
                else: 
                    new_dif.append(x)
            if sum(new_dif) == 0:
                features_tic_modality_list.append(1)
            else:
                rais = [n for n,i in enumerate(new_dif) if i > 0][0]
                fall = [n for n,i in enumerate(new_dif) if i <= 0][-1]
                if rais > fall:
                    features_tic_modality.append(1)
                else:
                    features_tic_modality.append(max(new_dif[rais:fall]))
            
            ### 10.2 Modality_each_transition
            features_transition_modality = []
            dif = temp.subtract(temp1)
            for i in range(7):
                dif[i].apply(lambda x: 0 if abs(x) <= 0.05*max(dif[i]) else x)
            for x in range(7):
                if sum(dif[x]) == 0:
                    features_tic_modality_list.append(1)
                else:
                    rais = [n for n,i in enumerate(dif[x]) if i > 0][0]
                    fall = [n for n,i in enumerate(dif[x]) if i <= 0][-1]
                    if rais > fall:
                        features_transition_modality.append(1)
                    else:
                        features_transition_modality.append(max(dif[x][rais:fall]))           
            
            # New features
            # tic part
            features_tic_peakmax_list.append(features_tic_peakmax[0])
            features_tic_relmax_list.append(features_tic_relmax[0])
            features_tic_peakarea_list.append(features_tic_peakarea[0])
            features_tic_peakratio_list.append(features_tic_peakratio[0])
            features_tic_peakshift_list.append(features_tic_peakshift[0])
            features_tic_elutionShift_list.append(features_tic_elutionShift[0])
            features_tic_similarity_list.append(features_tic_similarity[0])
            features_tic_symmetry_list.append(features_tic_symmetry[0])
            features_tic_jaggedness_list.append(features_tic_jaggedness[0])
            features_tic_FWHM_list.append(features_tic_FWHM[0])
            features_tic_modality_list.append(features_tic_modality[0])
            
            # transition part
            
            features_transition_peakmax_list.append(features_transition_peakmax)
            features_transition_relmax_list.append(features_transition_relmax)
            features_transition_peakarea_list.append(features_transition_peakarea)
            features_transition_peakratio_list.append(features_transition_peakratio)
            features_transition_peakshift_list.append(features_transition_peakshift)
            features_transition_elutionShift_list.append(features_transition_elutionShift)
            features_transition_similarity_list.append(features_transition_similarity)
            features_transition_symmetry_list.append(features_transition_symmetry)
            features_transition_jaggedness_list.append(features_transition_jaggedness)
            features_transition_FWHM_list.append(features_transition_FWHM)
            features_transition_modality_list.append(features_transition_modality)
    
        else:
            print(f"{sample_id} does not exist.")
    
    # In[1.3 Save the data into folder]
          
    
    #### new features start
    features_tic_peakmax_list = np.array(features_tic_peakmax_list)
    features_tic_relmax_list = np.array(features_tic_relmax_list)
    features_tic_peakarea_list = np.array(features_tic_peakarea_list)
    features_tic_peakratio_list = np.array(features_tic_peakratio_list)
    features_tic_peakshift_list = np.array(features_tic_peakshift_list)
    features_tic_elutionShift_list = np.array(features_tic_elutionShift_list)
    features_tic_similarity_list = np.array(features_tic_similarity_list)
    features_tic_symmetry_list = np.array(features_tic_symmetry_list)
    features_tic_jaggedness_list = np.array(features_tic_jaggedness_list)
    features_tic_FWHM_list = np.array(features_tic_FWHM_list)
    features_tic_modality_list = np.array(features_tic_modality_list)
    
    features_transition_peakmax_list = np.array(features_transition_peakmax_list)
    features_transition_relmax_list = np.array(features_transition_relmax_list)
    features_transition_peakarea_list = np.array(features_transition_peakarea_list)
    features_transition_peakratio_list = np.array(features_transition_peakratio_list)
    features_transition_peakshift_list = np.array(features_transition_peakshift_list)
    features_transition_elutionShift_list = np.array(features_transition_elutionShift_list)
    features_transition_similarity_list = np.array(features_transition_similarity_list)
    features_transition_symmetry_list = np.array(features_transition_symmetry_list)
    features_transition_jaggedness_list = np.array(features_transition_jaggedness_list)
    features_transition_FWHM_list = np.array(features_transition_FWHM_list)
    features_transition_modality_list = np.array(features_transition_modality_list)
    
    ### new features over
    
    data = {       
            "sample_id": sample_id_list,
            "features_tic_peakmax_list": features_tic_peakmax_list,
            "features_tic_relmax_list": features_tic_relmax_list,
            "features_tic_peakarea_list": features_tic_peakarea_list,
            "features_tic_peakratio_list": features_tic_peakratio_list,
            "features_tic_peakshift_list": features_tic_peakshift_list,
            "features_tic_elutionShift_list":features_tic_elutionShift_list,
            "features_tic_similarity_list":features_tic_similarity_list,
            "features_tic_symmetry_list":features_tic_symmetry_list,
            "features_tic_jaggedness_list":features_tic_jaggedness_list,
            "features_tic_FWHM_list": features_tic_FWHM_list,
            "features_tic_modality_list":features_tic_modality_list,
            
            "features_transition_peakmax_list": features_transition_peakmax_list,
            "features_transition_relmax_list": features_transition_relmax_list,
            "features_transition_peakarea_list": features_transition_peakarea_list,
            "features_transition_peakratio_list": features_transition_peakratio_list,
            "features_transition_peakshift_list": features_transition_peakshift_list,
            "features_transition_elutionShift_list":features_transition_elutionShift_list,
            "features_transition_similarity_list":features_transition_similarity_list,
            "features_transition_symmetry_list":features_transition_symmetry_list,
            "features_transition_jaggedness_list":features_transition_jaggedness_list,
            "features_transition_FWHM_list":features_transition_FWHM_list,
            "features_transition_modality_list":features_transition_modality_list,
            }
    
    os.makedirs(f"data{suffix}", exist_ok=True)
    datafile_name = f"saved_data.pkl"
    with open(f"data{suffix}/{datafile_name}", "wb") as f:
        pickle.dump(data, f)