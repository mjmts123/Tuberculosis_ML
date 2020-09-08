# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve
import os
import pickle
import pandas as pd




def training_data(suffix, sample_id_list, label_list, models_trained_saved, model_save, best_index):

    datafile_name = f"data{suffix}/saved_data.pkl"
    with open(datafile_name, 'rb') as f:
        data = pickle.load(f)
    
    # Select only features of sample in sample_id_list
    selected_sample_index = [list(data['sample_id']).index(i) for i in sample_id_list]
    #assert(np.array_equal(data['sample_id'][selected_sample_index], sample_id_list))
    
    # features of tic part
    features_tic_peakmax_list = data['features_tic_peakmax_list'][selected_sample_index]
    #features_tic_relmax_list = data['features_tic_relmax_list'][selected_sample_index]
    features_tic_peakarea_list = data['features_tic_peakarea_list'][selected_sample_index]
    features_tic_peakratio_list = data['features_tic_peakratio_list'][selected_sample_index]
    features_tic_peakshift_list = data['features_tic_peakshift_list'][selected_sample_index]
    features_tic_elutionShift_list = data['features_tic_elutionShift_list'][selected_sample_index]
    features_tic_similarity_list = data['features_tic_similarity_list'][selected_sample_index]
    features_tic_symmetry_list = data['features_tic_symmetry_list'][selected_sample_index]
    features_tic_jaggedness_list = data['features_tic_jaggedness_list'][selected_sample_index]
    features_tic_FWHM_list = data['features_tic_FWHM_list'][selected_sample_index]
    features_tic_modality_list = data['features_tic_modality_list'][selected_sample_index]
    
    # features of transition part
    features_transition_peakmax_list = data['features_transition_peakmax_list'][selected_sample_index]
    #features_transition_relmax_list = data['features_transition_relmax_list'][selected_sample_index]
    features_transition_peakarea_list = data['features_transition_peakarea_list'][selected_sample_index]
    features_transition_peakratio_list = data['features_transition_peakratio_list'][selected_sample_index]
    features_transition_peakshift_list = data['features_transition_peakshift_list'][selected_sample_index]
    features_transition_elutionShift_list = data['features_transition_elutionShift_list'][selected_sample_index]
    features_transition_similarity_list = data['features_transition_similarity_list'][selected_sample_index]
    features_transition_symmetry_list = data['features_transition_symmetry_list'][selected_sample_index]
    features_transition_jaggedness_list = data['features_transition_jaggedness_list'][selected_sample_index]
    features_transition_FWHM_list = data['features_transition_FWHM_list'][selected_sample_index]
    features_transition_modality_list = data['features_transition_modality_list'][selected_sample_index]
    
    
    
    intensity_list = [] # list to store intensity features of selected ions
    rel_list = [] # list to store relative abundance features of selected ions
    intensity_sum_list = [] # list to store peak area features of selected ions
    rel_sum_list = [] # list to store relative abundance based on peak area of selected ions
    max_tic_range_list = [] # list to store featured based on Max TIC of target winthin a range around standard_max_index
    std_tic_list = []
    tar_tic_list = []
    
    ## new features
    tic_peakmax_list = []
    tic_relmax_list = []
    tic_peakarea_list = []
    tic_peakratio_list = []
    tic_peakshift_list = []
    tic_elutionShift_list = []
    tic_similarity_list = []
    tic_symmetry_list = []
    tic_jaggedness_list = []
    tic_FWHM_list = []
    tic_modality_list = []
    
    transition_peakmax_list = []
    transition_relmax_list = []
    transition_peakarea_list = []
    transition_peakratio_list = []
    transition_peakshift_list = []
    transition_elutionShift_list = []
    transition_similarity_list = []
    transition_symmetry_list = []
    transition_jaggedness_list = []
    transition_FWHM_list = []
    transition_modality_list = []    
                               
    
    ## We can choose any transition features here.
    for x in range(7):
        peakmax = features_transition_peakmax_list[:,x].reshape(-1,1)
    #    relmax = features_transition_relmax_list[:,x].reshape(-1,1)
        peakarea = features_transition_peakarea_list[:,x].reshape(-1,1)
        peakratio = features_transition_peakratio_list[:,x].reshape(-1,1)
        peakshift = features_transition_peakshift_list[:,x].reshape(-1,1)
        eshift = features_transition_elutionShift_list[:,x].reshape(-1,1)
        similarity = features_transition_similarity_list[:,x].reshape(-1,1)
        symmetry = features_transition_symmetry_list[:,x].reshape(-1,1)
        jaggedness = features_transition_jaggedness_list[:,x].reshape(-1,1)
        FWHM = features_transition_FWHM_list[:,x].reshape(-1,1)
        modality = features_transition_modality_list[:,x].reshape(-1,1)
        
        transition_peakmax_list.append(peakmax)
    #    transition_relmax_list.append(relmax)
        transition_peakarea_list.append(peakarea)
        transition_peakratio_list.append(peakratio)
        transition_peakshift_list.append(peakshift)
        transition_elutionShift_list.append(eshift)
        transition_similarity_list.append(similarity)
        transition_symmetry_list.append(symmetry)
        transition_jaggedness_list.append(jaggedness)
        transition_FWHM_list.append(FWHM)
        transition_modality_list.append(modality)
    
    
    tic_peakmax_list = features_tic_peakmax_list.reshape(-1,1)
    #tic_relmax_list = features_tic_relmax_list.reshape(-1,1)
    tic_peakarea_list = features_tic_peakarea_list.reshape(-1,1)
    tic_peakratio_list = features_tic_peakratio_list.reshape(-1,1)
    tic_peakshift_list = features_tic_peakshift_list.reshape(-1,1)
    tic_elutionShift_list = np.reshape([round(x,8) for x in features_tic_elutionShift_list],(-1,1))
    tic_similarity_list = features_tic_similarity_list.reshape(-1,1)
    tic_symmetry_list = features_tic_symmetry_list.reshape(-1,1)
    tic_jaggedness_list = features_tic_jaggedness_list.reshape(-1,1)
    tic_FWHM_list = features_tic_FWHM_list.reshape(-1,1)
    tic_modality_list = np.reshape(features_tic_modality_list,(-1,1))
    
    transition_peakmax_list = np.concatenate(transition_peakmax_list, axis=1)
    #transition_relmax_list = np.concatenate(transition_relmax_list, axis=1)
    transition_peakarea_list = np.concatenate(transition_peakarea_list, axis=1)
    transition_peakratio_list = np.concatenate(transition_peakratio_list, axis=1)
    transition_peakshift_list = np.concatenate(transition_peakshift_list, axis=1)
    transition_elutionShift_list = np.concatenate(transition_elutionShift_list, axis=1)
    transition_similarity_list = np.concatenate(transition_similarity_list, axis=1)
    transition_symmetry_list = np.concatenate(transition_symmetry_list, axis=1)
    transition_jaggedness_list = np.concatenate(transition_jaggedness_list, axis=1)
    transition_FWHM_list = np.concatenate(transition_FWHM_list, axis=1)
    transition_modality_list = np.concatenate(transition_modality_list, axis=1)
    
    ## We could pick different combination here by changing the 0 and 1, but please notice there must
    ## be only one '1' in each group.
    
    
    feature_values = [[1,0,0,1, "Morphological"], [0,1,0,2, "Statistical"], [0,0,1,3, "M+S"]]
    
    #this for loop goes through all the different combinations of feature_more, 
    #feature_sta, and feature_ms and saves the results in a dataframe (model_save)
    
    for feature1, feature2, feature3, count, feature_type in feature_values:
        
        feature_mor = feature1
        feature_sta = feature2
        feature_ms = feature3
        count = str(count)
        
        
        if feature_mor:
        
            X = np.concatenate((
                            transition_elutionShift_list,
                            transition_similarity_list,
                            transition_symmetry_list,
                            transition_jaggedness_list,
                            transition_modality_list,
                            tic_peakratio_list,
                            tic_peakmax_list,
                            tic_peakarea_list,
                            tic_peakshift_list,                            
                            tic_FWHM_list), axis=1)
            print('morphological features finished')
        if feature_sta:
            X = np.concatenate((
                            transition_peakmax_list,
                            transition_peakarea_list,
                            transition_peakshift_list,                            
                            transition_FWHM_list,
                            transition_peakratio_list,
                            tic_elutionShift_list,
                            tic_similarity_list,
                            tic_symmetry_list,
                            tic_jaggedness_list,
                            tic_modality_list,), axis=1)
            print('statistical features finished')
        if feature_ms:
            X = np.concatenate((
                            tic_peakratio_list,
                            tic_elutionShift_list,
                            tic_similarity_list,
                            tic_symmetry_list,
                            tic_jaggedness_list,
                            tic_modality_list, 
                            transition_peakratio_list,
                            transition_elutionShift_list,
                            transition_similarity_list,
                            transition_symmetry_list,
                            transition_jaggedness_list,
                            transition_modality_list,
                            tic_peakmax_list,
                            tic_peakarea_list,
                            tic_peakshift_list,                            
                            tic_FWHM_list,
                            transition_peakmax_list,
                            transition_peakarea_list,
                            transition_peakshift_list,                            
                            transition_FWHM_list), axis=1)
            print('both mor and sta finished')
        
        X = np.nan_to_num(X, nan=-9999)
        
        y = label_list
        
        z= feature_type
        
        model_save.update({(suffix + feature_type):[X,y,z]})
        
        
        # In[2.2 base modelling and training model]
        # Unless we want to optimize the parameters of models, or this step only need to run once.
        
        #This section only runs once during the training stage
        
        if not models_trained_saved:
            if(suffix == "_Italy_train"):
                
                estimators = [] # list contains different models
                parameters = {'penalty': ['l2'], 'C': [100.0, 10.0, 1.0]}
                for penalty in parameters['penalty']:
                    for C in parameters['C']:
                        clf = LogisticRegression(n_jobs=-1, max_iter=500, penalty=penalty, C=C,
                                                 random_state=0)
                        estimators.append(("LR", clf))
                
                parameters = {'n_estimators': [100, 500], 'max_depth': [2, 3, 5], 'max_features':[None, 'auto']}
                for n_estimators in parameters['n_estimators']:
                    for max_depth in parameters['max_depth']:
                        for max_features in parameters['max_features']: 
                            clf = RandomForestClassifier(n_jobs=-1, max_features=max_features,
                                                         n_estimators=n_estimators, max_depth=max_depth,
                                                         random_state=0)
                            estimators.append(("RF", clf))
                            
                parameters = {'n_estimators': [100, 500], 'max_depth': [2, 3, 5], 'max_features': ['auto', None],
                              'learning_rate': [0.01, 0.001]}
                for n_estimators in parameters['n_estimators']:
                    for max_depth in parameters['max_depth']:
                        for max_features in parameters['max_features']: 
                            for learning_rate in parameters['learning_rate']: 
                                clf = GradientBoostingClassifier(random_state=0, subsample=1.0)
                                estimators.append(("GB", clf))    
                
                # Define dict to store metrics and predicted labels
                accuracy_scores = {}
                f_scores = {}
                sensitivity_scores = {}
                specificity_scores = {}
                auc_roc = {}
                y_prob_final = {}
                
                def setup_scores(i):
                    if i not in accuracy_scores:
                        accuracy_scores[i] = []
                    if i not in f_scores:
                        f_scores[i] = []
                    if i not in sensitivity_scores:
                        sensitivity_scores[i] = []
                    if i not in specificity_scores:
                        specificity_scores[i] = []
                    if i not in auc_roc:
                        auc_roc[i] = []
                    if i not in y_prob_final:
                        y_prob_final[i] = np.ones(len(label_list))*-1
                
                from sklearn.model_selection import StratifiedKFold
                n_repeats =10
                    
                shuffle_split = StratifiedKFold(n_splits=n_repeats, shuffle=True, random_state=0)
                fold = 0
                for train_index, test_index in shuffle_split.split(X=X, y=y):
                    fold += 1
                    print(f"Running fold {fold} of {n_repeats}...")
                    X_train = X[train_index]
                    y_train = y[train_index]
                    X_test = X[test_index]
                    y_test = y[test_index]
                    i = 0
                    # Performance of machine learning models
                    for (model_name, clf) in estimators:
                        i += 1
                        clf.fit(X_train, y_train)
                        y_prob_train = clf.predict_proba(X_train)[:, 1]
                        y_prob_test = clf.predict_proba(X_test)[:, 1]
                        y_pred_train = clf.predict(X_train)
                        y_pred_test = clf.predict(X_test)  
                        setup_scores(i)
                        accuracy_scores[i].append([accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred_test)])
                        f_scores[i].append([f1_score(y_train, y_pred_train), f1_score(y_test, y_pred_test)])
                        sensitivity_scores[i].append([recall_score(y_train, y_pred_train, pos_label=1), recall_score(y_test, y_pred_test, pos_label=1)])
                        specificity_scores[i].append([recall_score(y_train, y_pred_train, pos_label=0), recall_score(y_test, y_pred_test, pos_label=0)])
                        auc_roc[i].append([roc_auc_score(y_train, y_prob_train), roc_auc_score(y_test, y_prob_test)])
                        np.put(y_prob_final[i], test_index, y_prob_test)
                    
                
                # In[2.4 Save the models]
                
                # Save base models for inference
                # If we want to save model generate by different features, we could change the numbers in pathway.
                # However we should remember the number's meaning or using true name instead.
                
                
                os.makedirs(f"model{suffix}{count}/sample", exist_ok=True) # change number 1: Mor; 2: Sta; 3: Mor&Sta features
                
                i = 0
                for (model_name, clf) in estimators:
                    i += 1
                    clf.fit(X, y) # Rerain the models on the whole dataset
                    with open(f"model{suffix}{count}/sample/base_model_{i}.pkl", "wb") as f: # change number
                        pickle.dump(clf, f)
                print("\nBase models saved.")
                
                
            #Determine and save indices of best models    
                test_accuracy_scores = [np.mean(np.array(accuracy_scores[i])[:, 1]) for i in accuracy_scores.keys()]
                LR_loop = "LR" + count
                RF_loop = "RF" + count
                GB_loop = "GB" + count
                best_index_LR = str(np.argmax(test_accuracy_scores[0:2])+1)
                best_index_RF = str(np.argmax(test_accuracy_scores[3:14])+1)
                best_index_GB = str(np.argmax(test_accuracy_scores[15:38])+1)
                #This accounts for if all GB or all RF models have the same accuracy
                if(best_index_RF == "1"):
                    best_index_RF = "4" 
                if(best_index_GB == "1"):
                    best_index_GB = "16" 
                
                
                best_index.update({LR_loop:best_index_LR, RF_loop:best_index_RF, GB_loop:best_index_GB})
                #print(best_index.keys())
              
    return model_save, best_index
        
        