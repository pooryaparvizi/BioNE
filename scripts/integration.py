#!/usr/bin/env python3
'''
integration
'''

import argparse
import numpy as np
import pandas as pd
import timeit
from distutils import util
import ast
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif 
import statsmodels.stats.multitest as smt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt  



def parse_args():
    '''
    Parses the integration arguments
    '''
    
    parser = argparse.ArgumentParser(description="Run integration")

    parser.add_argument('--fusion', default= 'late', choices=['early','late','mix'],
            help='The integration type')
    
    parser.add_argument('--annotation', required=True,
            help='The filepath of the annotation file')
    
    parser.add_argument('--annotation-firstcolumn', required=True, 
            help='The filepaths of the node embedding files containing the entities of the first column in the annotation file')
        
    parser.add_argument('--annotation-secondcolumn',  required=False, default= '[]',
            help='The filepaths of the node embedding files containing the entities of the second column in the annotation file')

    parser.add_argument('--cv-type', help='cross-validation method',
                        default= 'split', choices=['kfold', 'stratified', 'split'])

    parser.add_argument('--cv', help='Number of folds', default= 5, type= int)

    parser.add_argument('--cv-shuffle',  action='store_true',
            help='Whether to shuffle each class’s samples before splitting into batches.')


    parser.add_argument('--test-size', help='Percentage of the data to be test-set', default= 0.2, type= float)
    

    parser.add_argument('--imbalance', help='Dealing with imbalance dependent variables',
                        default= None, choices=["equalize","SMOTE", "ADASYN", "None"])
    
    parser.add_argument('--fselection', help='feature selection',
                        default= "None", choices=['qvalue','fvalue','MI', "None"])

    parser.add_argument('--ktop', 
                        help='Select features according to the k highest scores if feature selection is either fvalue or MI', default= 10, type= int)

    parser.add_argument('--model', help='Machine Learning model for Classification',
                        default= 'SVM')

    parser.add_argument('--random_state', default= None, type= int,
            help='Fixing the randomization')
    
    parser.add_argument('--kernel', help='Specifies the kernel type to be used in the algorithm', default= "linear")
    
    parser.add_argument('--C', help='Regularization parameter.', default= 1, type= float)

    parser.add_argument('--ntree', type= int, default= 100,
            help='The number of trees in the forest')
    
    parser.add_argument('--criterion', choices=['gini','entropy'],
            help='The function to measure the quality of a split in random forest', default= "gini")

    parser.add_argument('--njob', type= int, default= 1,
            help='The number of jobs to run in parallel')

    
    parser.add_argument('--output', required=True,
            help='The folder path to save predictions and evaluation results')
    
    return parser.parse_args()



# oversampling
def oversampling(imbalance, X, y, random_state):
    print("Oversampling:", imbalance)
    
    
    y= np.array(y.iloc[:,])
    X= np.array(X)
    mat_size= len(y)
    print('Original dataset shape: %s' % Counter(y))
    
    if imbalance == "equalize":
        Counter_dict= Counter(y)
        min_class= min(Counter_dict, key=Counter_dict.get)
        min_class_size= Counter_dict[min_class]
        
        equalized_mat= pd.DataFrame()
        equalized_y= pd.DataFrame()
        for c in Counter_dict:    
            if c == min_class:    
                minc= np.where(y == c)[0]
                mat_minc= pd.DataFrame(X[minc, :])
                y_minc= pd.DataFrame(y[minc])
                equalized_mat = equalized_mat.append(mat_minc, ignore_index=True)
                equalized_y = equalized_y.append(y_minc, ignore_index=True)
                
            else:
                otherc= np.where(y == c)[0]
                np.random.seed(random_state)
                othercReduce= np.random.choice(otherc, min_class_size, replace=False)
                
                mat_otherReduce= pd.DataFrame(X[othercReduce, :])
                y_otherReduce= pd.DataFrame(y[othercReduce])
                equalized_mat = equalized_mat.append(mat_otherReduce, ignore_index=True)
                equalized_y = equalized_y.append(y_otherReduce, ignore_index=True)
        equalized_y= equalized_y.iloc[:,-1]

    elif imbalance == "SMOTE":
        equalized_mat, equalized_y = SMOTE(random_state= random_state).fit_resample(X, y)

    elif imbalance == "ADASYN":
        equalized_mat, equalized_y = ADASYN(random_state= random_state).fit_resample(X, y)

    elif imbalance == "None":
        equalized_mat, equalized_y = X, y
        

    print('balanced dataset shape: %s' % Counter(equalized_y))
    return equalized_mat, equalized_y


# feature selection
def feature_selection(method, X, y, Xtest, ktop):
    if (np.size(X, 1) < ktop) and (method not in ["None", "qvalue"]):
        raise ValueError("Pick ktop smaller than the matrix column size")
        
    print('Original dataset shape:', X.shape)
    y= y.iloc[:,-1].values
    
    if method == "qvalue":
        fs = SelectKBest(score_func=f_classif, k= 'all')
        X_fs= fs.fit_transform(X, y)
        qvalues= smt.multipletests(fs.pvalues_, method="bonferroni")[1]
        X_fselected= X.iloc[:, (qvalues < 0.1)]
        Xtest_fselected= Xtest.iloc[:, (qvalues < 0.1)]
        
    elif method == "fvalue":
        fs = SelectKBest(score_func=f_classif, k= ktop)
        X_fselected= fs.fit_transform(X, y)
        Xtest_fselected= fs.transform(Xtest)
    elif method == "MI":
        fs = SelectKBest(score_func= mutual_info_classif, k= ktop)
        X_fselected= fs.fit_transform(X, y)
        Xtest_fselected= fs.transform(Xtest)
        
    elif method == "None":
        X_fselected= X
        Xtest_fselected= Xtest
    
    print('feature selected dataset shape:', X_fselected.shape)
    return X_fselected, Xtest_fselected
    

# classification
def classification(model, Xtrain, ytrain, Xtest, ytest, random_state, kernel, C, ntree, criterion, njob):
    print("model:", model)
    if model == "SVM":
        clf = svm.SVC(kernel= kernel, C= C, random_state= random_state, probability= True)
    elif model == "RF":
        clf = RandomForestClassifier(n_estimators= ntree, criterion= criterion, random_state= random_state, n_jobs= njob)
    elif model == "NB":
        clf= GaussianNB()
    elif model == "XGBoost":
        clf= XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        
    ytrain= ytrain.iloc[:,-1].values
    clf.fit(Xtrain, ytrain)
    y_prob= clf.predict_proba(Xtest)
    y_pred= clf.predict(Xtest)
    #plot_roc_curve(clf, Xtest, ytest)
    #plot_precision_recall_curve(clf, Xtest, ytest)

    return y_prob, y_pred



# score report
def score_report(y_test, y_prob, y_pred):
    print("Calculating evaluation metrics")
    mconfusion= confusion_matrix(y_test, y_pred)
    #print(mconfusion)
    tn, fp, fn, tp= confusion_matrix(y_test, y_pred).ravel()
    maccuracy= accuracy_score(y_test, y_pred)
    mprecision= precision_score(y_test, y_pred)
    mrecall= recall_score(y_test, y_pred)
    mspecificity= (tn / (tn + fp))
    mf1= f1_score(y_test, y_pred)

    # ROC and PR
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
    roc_auc = auc(fpr, tpr)
    
    precision, recall, thresholds_pr= precision_recall_curve(y_test, y_prob[:,1])
    pr_auc= auc(recall, precision)
    metric_output= [maccuracy, mprecision, mrecall, mspecificity, mf1, roc_auc, pr_auc]
    return metric_output, mconfusion


# Mutual check
def mutual_check(annotation_file, annotation_firstcolumn, annotation_secondcolumn):
    print("Checking mutual annotations")
        
    for i in annotation_firstcolumn:
        firstcolumn= annotation_file.iloc[:,0]
        xfile= pd.read_csv(i, header= None,index_col= 0,  sep= " ").index.values
        annotation_file= annotation_file.iloc[firstcolumn.isin(xfile).values, :]

    
    for j in annotation_secondcolumn:
        secondcolumn= annotation_file.iloc[:,1]
        xfilej= pd.read_csv(j, header= None,index_col= 0,  sep= " ").index.values
        annotation_file= annotation_file.iloc[secondcolumn.isin(xfilej).values, :]
    annotation_file= annotation_file.reset_index(drop=True, inplace=False)
    return annotation_file



# Separate data for validation
def cv_set(cv_type, test_size, cv, X, y, random_state, shuffle):
    print("Creating validation sets")
    if cv_type == "kfold":
        kf = KFold(n_splits= cv, random_state= random_state, shuffle= shuffle)
        kf_batchs= kf.split(y)
    elif cv_type == "stratified":
        kf = StratifiedKFold(n_splits= cv, random_state= random_state, shuffle= shuffle)
        kf_batchs= kf.split(X, y)
    elif cv_type == 'split':
        train_split, test_split = train_test_split(range(np.size(y,0)), test_size= test_size, random_state= random_state)
        kf_batchs= [[train_split, test_split]]

    return kf_batchs


    
# Merging annotations
def annotation_merge(annotation_file, annotation_firstcolumn, annotation_secondcolumn):    
    print("Merging annotations")
    firstcolumn= annotation_file.iloc[:,0].values
    firstcolumn_embedding= pd.DataFrame()    
    for i in annotation_firstcolumn:
        xfile= pd.read_csv(i, header= None,index_col= 0,  sep= " ")
        
        firstcolumn_embedding= pd.concat([firstcolumn_embedding, xfile], axis=1, join='outer')
    firstcolumn_embedding= firstcolumn_embedding.loc[firstcolumn]
    firstcolumn_embedding= firstcolumn_embedding.reset_index(drop=True, inplace=False)

    if annotation_secondcolumn != []:
        if np.size(annotation_file, 1) == 3:
            secondcolumn= annotation_file.iloc[:,1].values
            secondcolumn_embedding= pd.DataFrame(index= np.unique(secondcolumn)) 
            for j in annotation_secondcolumn:
                x2file= pd.read_csv(j, header= None,index_col= 0,  sep= " ")
                secondcolumn_embedding= pd.concat([secondcolumn_embedding, x2file], axis=1, join='outer')
            secondcolumn_embedding= secondcolumn_embedding.loc[secondcolumn]
            secondcolumn_embedding.index= range(np.size(secondcolumn_embedding,0))
        else:
            raise ValueError ("annotation file have two columns. There is no need for annotation-secondcolumn argument")
            
    result= firstcolumn_embedding
    if annotation_secondcolumn != []:
        result = pd.concat([firstcolumn_embedding, secondcolumn_embedding], axis=1, ignore_index=True)    
    is_NaN = result.notna()
    row_has_NaN = is_NaN.all(axis=1)
    result= result[row_has_NaN]
    result= result.reset_index(drop=True, inplace=False)
    
    annotation_file= annotation_file.reset_index(drop=True, inplace=False)
    annotation_file= annotation_file[row_has_NaN]
    return annotation_file, result




##############
## fusion
def earlyfusion(args):
    
    annotation_firstcolumn= ast.literal_eval(args.annotation_firstcolumn)
    annotation_secondcolumn= ast.literal_eval(args.annotation_secondcolumn)

    classifier= ast.literal_eval(args.model)
    
    score_index= ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
    annotation_file= pd.read_csv(args.annotation, header= None, sep= " ")
    
    anno_new, X= annotation_merge(annotation_file, annotation_firstcolumn, annotation_secondcolumn)
    y= anno_new.iloc[:,-1]
    kf= cv_set(args.cv_type, args.test_size, args.cv, X, y, args.random_state, args.cv_shuffle)
    
    score= []
    if args.cv_type != 'split':
        score_rows= list(map('fold{}'.format, range(1, args.cv + 1)))
    else:
        score_rows= ["split"]
        
    fig, axs = plt.subplots(1, 2, figsize = (15,10))
    y= pd.DataFrame(y)
    X= pd.DataFrame(X)
    counter= 0
    for train_index, test_index in kf:   
        
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]
        

        counter += 1
        print("fold:", counter)
        
        X_train, y_train= oversampling(args.imbalance, X_train, y_train.iloc[:,-1], args.random_state)
        y_train= pd.DataFrame(y_train)
        X_train= pd.DataFrame(X_train)
        
        X_train_fselected, X_test_fselected= feature_selection(args.fselection, X_train, y_train, 
                                                               X_test, args.ktop)
        y_prob, y_pred= classification(classifier[0], X_train_fselected, y_train, X_test_fselected, 
                                       y_test, args.random_state, args.kernel, args.C, 
                                       args.ntree, args.criterion, args.njob)
        metrics, confusion= score_report(y_test, y_prob, y_pred)
        score.append(metrics)
    
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:,1])
        roc_auc = auc(fpr, tpr)
        axs[0].plot(fpr, tpr)
        
        precision, recall, thresholds_pr= precision_recall_curve(y_test, y_prob[:,1])
        pr_auc= auc(recall, precision)
        
        axs[1].plot(recall, precision)

    score= pd.DataFrame(score)
    score.columns= score_index
    score.index= score_rows
    
    score.loc['mean']= score.mean(axis= 0)
    score= score.round(2)
    
    axs[0].title.set_text('ROC')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    roc_auc_text= "average AUC:" +  str(score.loc["mean"]["roc_auc"])
    axs[0].text(0.35, 0.6, roc_auc_text, size=12)
    axs[0].set_xlim([-0.05, 1.05])
    axs[0].set_ylim([-0.05, 1.05])
    
    axs[1].title.set_text('PR')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    pr_auc_text= "average AUC:" +  str(score.loc["mean"]["pr_auc"])
    axs[1].text(0.35, 0.6, pr_auc_text , size=12)
    axs[1].set_xlim([-0.05, 1.05])
    axs[1].set_ylim([-0.05, 1.05])
    plt.savefig(args.output + ".png")
    
    score.to_csv(args.output + ".csv", index=True, header=True)
    return




def latefusion(args):
    
    annotation_firstcolumn= ast.literal_eval(args.annotation_firstcolumn)
    annotation_secondcolumn= ast.literal_eval(args.annotation_secondcolumn)

    classifier= ast.literal_eval(args.model)
    
    if args.cv_type != 'split':
        score_rows= list(map('fold{}'.format, range(1, args.cv + 1)))
    else:
        score_rows= ["split"]
    
    score_index= ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
    annotation_file= pd.read_csv(args.annotation, header= None, sep= " ")

    if len(annotation_firstcolumn) != len(annotation_secondcolumn):
        raise ValueError ("annotation_firstcolumn and annotation_secondcolumn should have the same length")
    
    nfile= len(annotation_firstcolumn)
    anno_new= mutual_check(annotation_file, annotation_firstcolumn, annotation_secondcolumn)
    X= anno_new.iloc[:,:-1]
    y= anno_new.iloc[:,-1]
    

    kf= cv_set(args.cv_type, args.test_size, args.cv, X, y, args.random_state, args.cv_shuffle)
    
    fig, axs = plt.subplots(1, 2, figsize = (15,10))
    score= []      
    counter= 0
    
    for train_index, test_index in kf:    
        counter += 1
        print("fold:", counter)
        for j in range(nfile):
            ann, merged= annotation_merge(anno_new, [annotation_firstcolumn[j]], [annotation_secondcolumn[j]])
            y= pd.DataFrame(ann.iloc[:,-1])
            X= merged

            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            y_train, y_test = y.iloc[train_index,-1], y.iloc[test_index,-1]
        
        
            X_train, y_train= oversampling(args.imbalance, X_train, y_train, args.random_state)
            
            y_train= pd.DataFrame(y_train)
            X_train_fselected, X_test_fselected= feature_selection(args.fselection, X_train, y_train, X_test, args.ktop)
            y_prob, y_pred= classification(classifier[0], X_train_fselected, y_train, X_test_fselected, 
                                           y_test, args.random_state, args.kernel, args.C, args.ntree, args.criterion, args.njob)
            
            globals()["prob_" + str(j)]= y_prob
            globals()["pred_" + str(j)]= y_pred
            
        Weighted_pred= []
        Weighted_prob= []
        for row in range(np.size(y_test,0)):
            Sum0= 0
            Sum1= 0
            for M in range(nfile):
                x= eval("prob_" + str(M))    
                Sum0 += x[row,0]
                Sum1 += x[row,1]
            Weighted_prob.append([Sum0/nfile, Sum1/nfile]) 
            Weighted_pred.append(np.argmax([Sum0, Sum1]))
        Weighted_prob= np.array(Weighted_prob)
        Weighted_pred= np.array(Weighted_pred)
        
        metrics, confusion= score_report(y_test, Weighted_prob, Weighted_pred)
        score.append(metrics)
    
        fpr, tpr, thresholds = roc_curve(y_test, Weighted_prob[:,1])
        roc_auc = auc(fpr, tpr)
        axs[0].plot(fpr, tpr)
        
        precision, recall, thresholds_pr= precision_recall_curve(y_test, Weighted_prob[:,1])
        pr_auc= auc(recall, precision)
        
        axs[1].plot(recall, precision)
        
    score= pd.DataFrame(score)
    score.columns= score_index
    score.index= score_rows
    
    score.loc['mean']= score.mean(axis= 0)
    score= score.round(2)
    
    axs[0].title.set_text('ROC')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    roc_auc_text= "average AUC:" +  str(score.loc["mean"]["roc_auc"])
    axs[0].text(0.35, 0.6, roc_auc_text, size=12)
    axs[0].set_xlim([-0.05, 1.05])
    axs[0].set_ylim([-0.05, 1.05])
    
    axs[1].title.set_text('PR')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    pr_auc_text= "average AUC:" +  str(score.loc["mean"]["pr_auc"])
    axs[1].text(0.35, 0.6, pr_auc_text , size=12)
    axs[1].set_xlim([-0.05, 1.05])
    axs[1].set_ylim([-0.05, 1.05])
    plt.savefig(args.output + ".png")
    
    score.to_csv(args.output + ".csv", index=True, header=True)
    return 




def mixfusion(args):
    
        
    annotation_firstcolumn= ast.literal_eval(args.annotation_firstcolumn)
    annotation_secondcolumn= ast.literal_eval(args.annotation_secondcolumn)

    classifier= ast.literal_eval(args.model)
    #if len(classifier) <= 1:
    #    raise ValueError ("For mix fusion more than one model should given")

    
    score_index= ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
    annotation_file= pd.read_csv(args.annotation, header= None, sep= " ")
    
    anno_new, X= annotation_merge(annotation_file, annotation_firstcolumn, annotation_secondcolumn)
    y= anno_new.iloc[:,-1]
    
    kf= cv_set(args.cv_type, args.test_size, args.cv, X, y, args.random_state, args.cv_shuffle)
    
    score= []
    if args.cv_type != 'split':
        score_rows= list(map('fold{}'.format, range(1, args.cv + 1)))
    else:
        score_rows= ["split"]
        
    fig, axs = plt.subplots(1, 2, figsize = (15,10))
    
    y= pd.DataFrame(y)
    X= pd.DataFrame(X)
    counter= 0
    for train_index, test_index in kf:    
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index,:], y.iloc[test_index,:]

        counter += 1
        print("fold:", counter)
        X_train, y_train= oversampling(args.imbalance, X_train, y_train.iloc[:,-1], args.random_state)
        y_train= pd.DataFrame(y_train)
        X_train= pd.DataFrame(X_train)

        X_train_fselected, X_test_fselected= feature_selection(args.fselection, X_train, y_train, X_test, args.ktop)
        
        
        for cm in classifier:
            print(cm)
            y_prob, y_pred= classification(cm, X_train_fselected, y_train, X_test_fselected, 
                                       y_test, args.random_state, args.kernel, args.C, 
                                       args.ntree, args.criterion, args.njob)    
            
            globals()["prob_" + cm]= y_prob
            globals()["pred_" + cm]= y_pred
            
        Weighted_pred= []
        Weighted_prob= []
        for row in range(np.size(y_test,0)):
            Sum0= 0
            Sum1= 0
            for cm in classifier:
                x= eval("prob_" + cm)    
                Sum0 += x[row,0]
                Sum1 += x[row,1]
            Weighted_prob.append([Sum0/len(classifier), Sum1/len(classifier)]) 
            Weighted_pred.append(np.argmax([Sum0, Sum1]))
        Weighted_prob= np.array(Weighted_prob)
        Weighted_pred= np.array(Weighted_pred)
        
        metrics, confusion= score_report(y_test, Weighted_prob, Weighted_pred)
        score.append(metrics)
    
        fpr, tpr, thresholds = roc_curve(y_test, Weighted_prob[:,1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        axs[0].plot(fpr, tpr)
        
        precision, recall, thresholds_pr= precision_recall_curve(y_test, Weighted_prob[:,1])
        pr_auc= auc(recall, precision)
        
        axs[1].plot(recall, precision)
        
    score= pd.DataFrame(score)
    score.columns= score_index
    score.index= score_rows
    
    score.loc['mean']= score.mean(axis= 0)
    score= score.round(2)
    
    axs[0].title.set_text('ROC')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    roc_auc_text= "average AUC:" +  str(score.loc["mean"]["roc_auc"])
    axs[0].text(0.35, 0.6, roc_auc_text, size=12)
    axs[0].set_xlim([-0.05, 1.05])
    axs[0].set_ylim([-0.05, 1.05])
    
    axs[1].title.set_text('PR')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    pr_auc_text= "average AUC:" +  str(score.loc["mean"]["pr_auc"])
    axs[1].text(0.35, 0.6, pr_auc_text , size=12)
    axs[1].set_xlim([-0.05, 1.05])
    axs[1].set_ylim([-0.05, 1.05])
    plt.savefig(args.output + ".png")
    
    score.to_csv(args.output + ".csv", index=True, header=True)
    return 
            
    
        


def main(args):
    print("**integration**\n")
   
    method= args.fusion
    saved_args = locals()
    saved_args = locals(); print("Arguments:", vars(saved_args["args"]), "\n")
    
    if type(args.random_state) == int:
        args.cv_shuffle = True
        
    print("fusion:", method) 
    if method == "early":
        earlyfusion(args)

    elif method == "late":
        latefusion(args)

    elif method == "mix":
        mixfusion(args)
        
        
        
        
if __name__ == "__main__":
    start = timeit.default_timer()
    main(parse_args())
    stop = timeit.default_timer()
    print('Time:', np.round(stop - start, 3))
