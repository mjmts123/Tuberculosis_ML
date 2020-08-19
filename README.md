# Tuberculosis_ML
Tulane ML project 

Sample text


![](./ROC%20curves.png)
![](./Results%20tables.png)

```
for suf, file_name in values: 
    for index, feature_type in enumerate(lst):
        
        i = index+1
        model_path = f"model_Italy_train{i}/sample/" # Remember to choose the correct model path
        
        LR_loop =  "LR" + str(i)
        RF_loop =  "RF" + str(i)
        GB_loop =  "GB" + str(i)
        model_names = [["base_model_" + str(best_index[LR_loop]), "LR"],["base_model_" + str(best_index[RF_loop]), "RF"],
                           ["base_model_"+ str(best_index[GB_loop]), "GB"]]
        
        #UPDATE THIS WHEN YOU RUN IN ON A NEW FILE - the models saved here are from the Italy train file
        #model_names = [["base_model_3", "LR"],["base_model_15", "RF"],
         #                 ["base_model_39", "GB"]]
        threshold = 0.5
        
        #UPDATE THIS LATER
        test_Dict = {"base_model_" + str(best_index[LR_loop]):"LR", 
                         "base_model_"+ str(best_index[RF_loop]):"RF", "base_model_"+ str(best_index[GB_loop]):"GB"}
        
        
       # test_Dict = {"base_model_3":"LR", 
        #                 "base_model_15":"RF", "base_model_39":"GB"}
        # LR = Logistic Regression; RF = Random Forest; GB = Gradient Boosting
        

        for model_name, model_type in model_names:
            #loops through model types of M, S, M+S
            if "base" in model_name: # The model is base model
                with open(os.path.join(model_path, f"{model_name}.pkl"), 'rb') as f:
                    clf = pickle.load(f)
                    
                    X, y, z = model_save[suf + feature_type]
                    if(suf!="_Italy_train"):
                        classifiers.append([suf+feature_type+model_type ,clf,feature_type])

                    
                    
                    
# =============================================================================
#                     model_save.update({(suffix + feature_type):[X,y,z]})
# =============================================================================
                    
                    
                    
                    y_prob = clf.predict_proba(X)[:, 1]
                    y_pred = (y_prob > threshold).astype('uint8')
                    
                    
                    
                    #Generates excel file of accuracy, sensitivity, F1, and specificity
                    print('Accuracy in ', test_Dict[model_name], accuracy_score(y, y_pred))
                    results_table = results_table.append({"Dataset": suf, "Base Model Number": model_name, "Features": z, "Model": model_type, "Scores":"Accuracy",
                                            "values": accuracy_score(y, y_pred) }, ignore_index = True)
                    
    
                   
                    print('F1_score', test_Dict[model_name], f1_score(y, y_pred))
                    results_table = results_table.append({"Dataset": suf, "Features": z, 
                        "Model": model_type, "Base Model Number": model_name, "Scores":"F1","values": f1_score(y, y_pred)}, ignore_index = True)
                    
                    print('Sensitivity', test_Dict[model_name], recall_score(y, y_pred, pos_label=1))
                    
                    results_table = results_table.append({"Dataset": suf, "Features": z, 
                                    "Model": model_type, "Scores":"Sensitivity", "Base Model Number": model_name,
                                    "values": recall_score(y, y_pred, pos_label=1)}, ignore_index = True)
                                    
                    print('Specificity', test_Dict[model_name], recall_score(y, y_pred, pos_label=0))
                        
                    results_table = results_table.append({"Dataset": suf, "Features": z, 
                                        "Model": model_type, "Scores":"Specificity","Base Model Number": model_name,
                                        "values": recall_score(y, y_pred, pos_label=0)}, ignore_index = True)
```
