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

## Code output
![](./ROC%20curves.png)








## Sensitivity Tables
![](./Results%20tables.png)
