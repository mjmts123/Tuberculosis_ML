# -*- coding: utf-8 -*-


from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, roc_curve
def generate_results_table(result_table, classifiers, model_save):
    for key, cls, suf, feature_type, model_type in classifiers:
         X_train, y_train, z_train = model_save["_Italy_train" + feature_type]
         X_test, y_test, z_test = model_save[key[:-2]]
                                 
         model = cls.fit(X_train, y_train)
         yproba = model.predict_proba(X_test)[::,1]
         fpr, tpr, _ = roc_curve(y_test,  yproba)
         auc = roc_auc_score(y_test, yproba)
         result_table = result_table.append({'classifiers':key,
                                             "identifier":str(key)[0:-2],
                                             "file_name":suf,
                                             "feature":feature_type,
                                             "model_type":model_type,
                                             'fpr':fpr, 
                                             'tpr':tpr, 
                                             'auc':auc}, ignore_index=True)
    return result_table
   