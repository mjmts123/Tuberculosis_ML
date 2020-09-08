import matplotlib.pyplot as plt
import numpy as np

def generate_ROC(result_table):
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
        title = "{} features in {} dataset".format(result_table_segmented.iloc[0]["feature"], result_table_segmented.iloc[0]["file_name"])
        plt.title(title, fontweight='bold', fontsize=20)
        plt.legend(prop={'size':13}, loc='lower right')
        
        #saves ROC curves in folder
        save_name = title
        fig.savefig(fname=save_name + ".svg",format="svg") # change the figures name here, the 'svg' format is easy to modified by illustrator software.
        plt.show()
        
        count +=1
