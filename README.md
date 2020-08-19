# Tuberculosis_ML

The code repository for results in the paper: XYZ


## Dependecies
- NumPy
- Pandas
- Pymzml
- Pickle
- Matplotlib
- Math
- Logistic 
- Sklearn


## Data sets used in analysis
Code to reproduce the figures in the paper are available in [this notebook](https://github.com/mjmts123/Tuberculosis_ML/blob/MT-branch/scores_organized_8.py)

Four data sets were used within our analysis.

- **Training set:** Italy train 
- **Test data set:** Italy valid
- **Independent data sets:** CSTB, Liz train

The data sets can be swapped out in the following code:

```python
values = [["_Italy_train", "Total_Italy_train.csv"],
          ["_Italy_valid", "Total_Italy_valid.csv"], 
          ["_CSTB", "CSTB.csv"],
          ["_liz_train", "Liz_train.csv"]]
```



The code will transform mzml files into readable files, extracting features from mass spectrometry data. It looks for scantime corresponding to the maximum total ion concentration (TIC) of internal standard, then extracts ion intensity, relative abundance and ratio between target/internal standard of different M/z(s).

Afterwards, it uses the training files to generate models, and tests the models against test and independent data sets.
     
If the code has previously been run, run times can be shortened by setting the following two variables to True:

```python
mzml_files_transformed = True

models_trained_saved = True
```

## Results

### Sensitivity Tables

The following code generates the values for sensitivity tables:

```python
for suf, file_name in values: 
    for index, feature_type in enumerate(lst):
        
        i = index+1
        model_path = f"model_Italy_train{i}/sample/" # Remember to choose the correct model path
        
        LR_loop =  "LR" + str(i)
        RF_loop =  "RF" + str(i)
        GB_loop =  "GB" + str(i)
        model_names = [["base_model_" + str(best_index[LR_loop]), "LR"],["base_model_" + str(best_index[RF_loop]), "RF"],
                           ["base_model_"+ str(best_index[GB_loop]), "GB"]]
        
 
        #model_names = [["base_model_3", "LR"],["base_model_15", "RF"],
         #                 ["base_model_39", "GB"]]
        threshold = 0.5
        

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


![](./Results%20tables.png)

### ROC Curves

The following code generates ROC curves.

```python
count = 1
for identifier in result_table.identifier.unique():
    fig = plt.figure(figsize=(8,8))  
    result_table_segmented = result_table[result_table.identifier.eq(identifier)]
    
    #this generates the ROC curves
    for i in result_table_segmented.index:
        plt.plot(result_table_segmented.loc[i]['fpr'], 
                 result_table_segmented.loc[i]['tpr'], 
                 label="{}, AUC={:.3f}".format(result_table_segmented.loc[i]["model_type"], result_table_segmented.loc[i]['auc']))
    
#generates x and y axis labels
    plt.plot([0,1], [0,1], color='orange', linestyle='--')
    
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("1-Specificity", fontsize=20)
    
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Sensitivity", fontsize=20)
    
    #generates Title
    plt.title(identifier, fontweight='bold', fontsize=30)
    plt.legend(prop={'size':13}, loc='lower right')
    
    #saves ROC curves in folder
    save_name = "test" + str(count)
    fig.savefig(fname=save_name + ".svg",format="svg") # change the figures name here, the 'svg' format is easy to modified by illustrator software.
    plt.show()
    
    count +=1
```

![](./ROC%20curves.png)




