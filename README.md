# Tuberculosis_ML
Tulane ML project 

Sample text


## Generating ROC charts

Sample text describing code

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
