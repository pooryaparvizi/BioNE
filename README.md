# BioNE: a pipeline to integrate the biological network embeddings to use in supervised learning tasks

## Overview

Network embedding approach has provided an effective way to overcome the complexity of large biological network analysis. Network embedding methods convert high-dimensional data to low-dimensional vector representations and in return these representations can be used in supervised machine learning tasks such as link prediction and node classification. However, different network embedding methods follow different approaches and some may learn certain network features that are not learned in other network embedding methods. The BioNE represent a new pipeline to integrate embedding results from different methods to have a more comprehensive knowledge of network and therefore better performance on prediction tasks.

This pipeline consists of three parts; 

&emsp; 1. [Network Preparation](#1-network-preparation)
&emsp; &emsp; 1.1. [Convert Adjacency Matrix to Edge List](#11-convert-adjacency-matrix-to-edge-list)
&emsp; &emsp; 1.2. [Heterogeneous Network Preparation](#12-heterogeneous-network-preparation)
&emsp; 2. [Embedding](#2-embedding)
&emsp; &emsp;2.1. [Merge Embeddings](#21-merge-embeddings)
&emsp; 3. [Predictions Using the Integration of Embeddings](#3-predictions-using-the-integration-of-embeddings)


In order to install packages and create virtual environment, check section [Virtual Environment and Installing Packages](#virtual-environment-and-installing-packages).\
This pipeline will be tested on the Drug-Target Interaction (DTI) as a sample of link prediction task. You can find the scripts of this test in the [Example](#example) section.

&nbsp;

## Virtual Environment and Installing Packages
All of the analysis in this project wrote and tested on virtual environment using python 3.7. The detailed software versions are listed below:
- Python 3.7
- virtualenv 20.4.0
- ubuntu 20.04
- nvidia-driver 460
- cuda 10.0
- cuDNN 7.4.2

To create the virtual environment:
```shell
cd BioNE
virtualenv --python=/usr/bin/python3.7 BioNEvenv
```

To activate and install required [packages](./requirements.txt):
```shell
source BioNEvenv/bin/activate
pip install -r requirements.txt
```
&nbsp;

## 1. Network Preparation
This part consists of two sections. The adjacency matrices are converted to the edge list files in section [1.1. Convert Adjacency Matrix to Edge List](#11-convert-adjacency-matrix-to-edge-list). On the other hand, when required, the user can combine two edge list files to making heterogeneous network using command line is section [1.2. Heterogeneous Network Preparation](#12-heterogeneous-network-preparation).

&nbsp;

### 1.1. Convert Adjacency Matrix to Edge List
In order to conduct network embedding, adjacency matrices should convert to the edge list file format.

```shell
python3 scripts/mat2edgelist.py --input input.txt --directed --keepzero --attribute --output output.txt
```
##### *Arguments*:

&emsp; &emsp; &emsp; **input** &ensp; The filepath of adjacency matrix\
&emsp; &emsp; &emsp; Input adjacency matrix file should be space-delimited file and contains row and column index labels. Click [here](./data/mat_protein_disease.txt) to see a sample file.

&emsp; &emsp; &emsp; **directed** &ensp; Treat the graph as directed\
&emsp; &emsp; &emsp; When directed, row indexes are source nodes and column indexes are target nodes.

&emsp; &emsp; &emsp; **keepzero** &ensp; Adding also negative associations (0s) to the output

&emsp; &emsp; &emsp; **attribute** &ensp; Including the edge attributes to the output file\
&emsp; &emsp; &emsp; If edge attributed are not going to use in embedding as the weights, it is recommended to not use this in order to save memory.

&emsp; &emsp; &emsp; **output** &ensp; The filepath to save edge list file\
&emsp; &emsp; &emsp; The file saves as space-delimited file format. Click [here](./output/edgelist_protein_disease.txt) to see a sample edge list file.

&nbsp;

### 1.2. Heterogeneous Network Preparation
When required, the user can combine two edge lists (e.g. drug-disease and drug-side effect networks) to construct heterogeneous network. The command line below help to combine edge lists. This can be used multiple times to combine more than two edge list files.\
The other way to combine two networks with mutual entities (e.g. drug-disease and drug-side effect networks) is to combine their embeddings (e.g. combine embeddings of drugs from drug-disease network and embedding of drugs from drug-side effect networks) after the [Embedding](#2-embedding) step using [Merge Embeddings](#21-merge-embeddings) command line.

```shell
python3 scripts/merge_edgelist.py --input1 input1.txt --input2 input2.txt --rmduplicate --output output.txt
```
##### *Arguments*:

&emsp; &emsp; &emsp; **input1** &ensp; The filepath of first edge list file\
&emsp; &emsp; &emsp; This file should be an edge list with space-delimited format file. Click [here](./output/edgelist_drug_disease.txt) to see a sample input file.


&emsp; &emsp; &emsp; **input2** &ensp; The filepath of second edge list file\
&emsp; &emsp; &emsp; This file should be an edge list with space-delimited csv format. Click [here](./output/edgelist_drug_se.txt) to see a sample input file.

&emsp; &emsp; &emsp; **rmduplicate** &ensp; Remove duplicated edges

&emsp; &emsp; &emsp; **output** &ensp; The filepath to save the combined edge lists file\
&emsp; &emsp; &emsp; The file saves as space-delimited file format. Click [here](./output/hetero_drugs.txt) to see a sample output file.

&nbsp;

## 2. Embedding
Network embedding methods convert high-dimensional data to low-dimensional vector representations. In this project the user able to conduct these embedding methods:\
[LINE](https://doi.org/10.1145/2736277.2741093), [GraRep](https://doi.org/10.1145/2806416.2806512), [SDNE](https://doi.org/10.1145/2939672.2939753), [LLE](https://doi.org/10.1126/science.290.5500.2323), [HOPE](https://doi.org/10.1145/2939672.2939751), [LaplacianEigenmaps (Lap)](https://dl.acm.org/doi/abs/10.5555/2980539.2980616), [node2vec](https://doi.org/10.1145/2939672.2939754), [DeepWalk](https://doi.org/10.1145/2623330.2623732) and [GF](https://doi.org/10.1145/2488388.2488393).\
Embedding methods in this section are inherited from [OpenNE](https://github.com/thunlp/OpenNE.git) and [OpenNE-PyTorch](https://github.com/thunlp/OpenNE/tree/pytorch) repositories.

```shell
python3 scripts/embedding.py --method lle --input input.txt --directed --weighted --representation_size 128 --output output.txt
```
##### *Arguments*:

&emsp; &emsp; &emsp; **method:** &ensp; Embedding method\
&emsp; &emsp; &emsp; Choices are:\
&emsp; &emsp; &emsp; &emsp; line (parameters: epochs, order, negative_ratio)\
&emsp; &emsp; &emsp; &emsp; grarep (parameters: kstep)\
&emsp; &emsp; &emsp; &emsp; sdne (parameters: alpha, beta, nu1, nu2, bs, lr, epochs, encoder-list)\
&emsp; &emsp; &emsp; &emsp; lle\
&emsp; &emsp; &emsp; &emsp; hope\
&emsp; &emsp; &emsp; &emsp; lap\
&emsp; &emsp; &emsp; &emsp; node2vec (parameters: walk_length, number_walks, workers, p, q, window_size)\
&emsp; &emsp; &emsp; &emsp; deepwalk (parameters: walk_length, number_walks, workers, window_size)\
&emsp; &emsp; &emsp; &emsp; gf (parameters: epochs, lr, weight-decay)

&emsp; &emsp; &emsp; &emsp; Note: *input*, *directed*, *weighted*, *random_state*, *representation_size* and *output* are shared among all methods.

&emsp; &emsp; &emsp; **input**: &ensp; The filepath of edge list file\
&emsp; &emsp; &emsp; This file should be an edge list with space-delimited format. Click [here](./output/edgelist_protein_disease.txt) to see a sample input file.

&emsp; &emsp; &emsp; **directed**: &ensp; Treat the network as directed\
&emsp; &emsp; &emsp; There is no need to use this if you already specified this in section [1.1](#11-convert-adjacency-matrix-to-edge-list).

&emsp; &emsp; &emsp; **weighted**: &ensp; Treat the network as weighted\
&emsp; &emsp; &emsp; To use this, edge attributes should be included in the edge list file. Check *attribute* argument in section [1.1](#11-convert-adjacency-matrix-to-edge-list).

&emsp; &emsp; &emsp; **random_state**: &ensp; Fixing the randomization\
&emsp; &emsp; &emsp; The default value is 1.

&emsp; &emsp; &emsp; **epochs**: &ensp; The number of times that the learning algorithm will work through the entire training data set\
&emsp; &emsp; &emsp; This parameter is used in *line*, *sdne* and *gf*. The default value is 5.

&emsp; &emsp; &emsp; **representation_size**: &ensp; Dimensionality of the output data\
&emsp; &emsp; &emsp; The default value is 128.

&emsp; &emsp; &emsp; **order**: &ensp; Choose the order of *line*\
&emsp; &emsp; &emsp; 1 means first order, 2 means second order, 3 means first order + second order. The default value is 2.

&emsp; &emsp; &emsp; **negative_ratio**: &ensp; Negative sampling ratio\
&emsp; &emsp; &emsp; This parameter is used in *line*. The default is 5.

&emsp; &emsp; &emsp; **kstep**: &ensp; Use k-step transition probability matrix\
&emsp; &emsp; &emsp; This parameter is used in *grarep*. The default value is 4.

&emsp; &emsp; &emsp; **encoder-list**: &ensp; a list of numbers of the neurons at each encoder layer in *sdne*\
&emsp; &emsp; &emsp; The last number is the dimension of the output embeddings. The default is [1000, 128].

&emsp; &emsp; &emsp; **alpha**: &ensp; alpha is a hyperparameter in *sdne*\
&emsp; &emsp; &emsp; The default value is 1e-6.

&emsp; &emsp; &emsp; **beta**: &ensp; beta is a hyperparameter in *sdne*\
&emsp; &emsp; &emsp; The default value is 1e-5.

&emsp; &emsp; &emsp; **nu1**: &ensp; nu1 is a hyperparameter in *sdne*\
&emsp; &emsp; &emsp; The default value is 1e-5.

&emsp; &emsp; &emsp; **nu2**: &ensp; nu2 is a hyperparameter in *sdne*\
&emsp; &emsp; &emsp; The default value is 1e-4.

&emsp; &emsp; &emsp; **bs**: &ensp; batch size in *sdne*\
&emsp; &emsp; &emsp; Number of training samples utilized in one iteration. The default is 200.

&emsp; &emsp; &emsp; **lr**: &ensp; learning rate in *sdne*\
&emsp; &emsp; &emsp; The learning rate controls how quickly the model is adapted to the problem. The default is 0.001.

&emsp; &emsp; &emsp; **walk-length**: &ensp; Length of the random walk started at each node\
&emsp; &emsp; &emsp; This parameter is used in *node2vec* and *deepwalk*. The default value is 80.

&emsp; &emsp; &emsp; **number-walks**: &ensp; Number of random walks to start at each node\
&emsp; &emsp; &emsp; This parameter is used in *node2vec* and *deepwalk*. The default value is 10.

&emsp; &emsp; &emsp; **workers**: &ensp; Number of parallel processes\
&emsp; &emsp; &emsp; This parameter is used in *node2vec* and *deepwalk*. The default value is 8.

&emsp; &emsp; &emsp; **p**: &ensp; Return hyperparameter in *node2vec*\
&emsp; &emsp; &emsp; The default value is 1.

&emsp; &emsp; &emsp; **q**: &ensp; Inout hyperparameter in *node2vec*\
&emsp; &emsp; &emsp; The default value is 1.

&emsp; &emsp; &emsp; **window-size**: &ensp; Window size of skipgram model in *node2vec* and *deepwalk*\
&emsp; &emsp; &emsp; The default value is 10.

&emsp; &emsp; &emsp; **weight-decay**: &ensp; Weight for L2 loss on embedding matrix in *gf*\
&emsp; &emsp; &emsp; The default value is 5e-4.

&emsp; &emsp; &emsp; **output**: &ensp; The filepath to save the embedding results\
&emsp; &emsp; &emsp; The file saves as space-delimited file format. Click [here](./output/hope_6_protein_disease.txt) to see a sample output file.

&nbsp;

### 2.1. Merge Embeddings
When required, the user can combine embedding results of two networks (e.g. drug-disease and drug-side effect networks) with mutual entities (e.g. combining the embeddings of drugs from drug-disease network to embedding of drugs from drug-side effect network).\
Other way, to combine two networks to create heterogeneous network is given in section [1.2. Heterogeneous Network Preparation](#12-heterogeneous-network-preparation).

```shell
python3 scripts/merge_embeddings.py --input1 input1.txt --input2 input2.txt --output output.txt
```
##### *Arguments*:

&emsp; &emsp; &emsp; **input1** &ensp; The filepath of first embedding file\
&emsp; &emsp; &emsp; This file should contain embeddings with space-delimited format file. Click [here](./output/hope_6_drug_disease.txt) to see a sample input file.


&emsp; &emsp; &emsp; **input2** &ensp; The filepath of second embedding file\
&emsp; &emsp; &emsp; This file should contain embeddings with space-delimited format file. Click [here](./output/hope_6_drug_se.txt) to see a sample input file.

&emsp; &emsp; &emsp; **output** &ensp; The filepath to save the combined embedding results for mutual entity\
&emsp; &emsp; &emsp; The file saves as space-delimited file format. Click [here](./output/./output/hope_12_drug_emb_merge.txt) to see a sample output file.

&nbsp;

## 3. Predictions using the integration of embeddings
In this part, we developed three different integration methods (late fusion, early fusion and mix fusion) to integrate embedding results from different methods to have a more comprehensive knowledge of network and therefore better performance on prediction tasks.

```shell
python3 scripts/integration.py --type late --annotation data/annotation.txt 
```
##### *Arguments*:

&emsp; &emsp; &emsp; **fusion** &ensp; The integration type\
&emsp; &emsp; &emsp; Choices are:\
&emsp; &emsp; &emsp; &emsp; early: Merging whole embedding results before putting in to the prediction model\
&emsp; &emsp; &emsp; &emsp; late: Run each embedding results in the prediction model and then add up the achieved prediction probabilities.\
&emsp; &emsp; &emsp; &emsp; mix: Merging whole embedding results, and then add up the prediction probabilities achieved from different prediction models.

&emsp; &emsp; &emsp; **annotation** &ensp; The filepath of the annotation file\
&emsp; &emsp; &emsp; This file should contain either two or three columns. Click [here](./output/annotation.txt) or [here](./output/annotation_2.txt) to see a sample annotation file.\
&emsp; &emsp; &emsp; two column annotation file can only be used in *mix* and *early* fusions.



&emsp; &emsp; &emsp; **annotation-firstcolumn** &ensp; filepaths of the embeddings containing the entities of the **first** column in the annotation file\
&emsp; &emsp; &emsp; The file paths should be given in this format: '[["hope_drugs.txt"](./output/hope_6_hetero_drugs.txt), ["lap_drugs.txt"](./output/lap_6_hetero_drugs.txt)]'.\
&emsp; &emsp; &emsp; When the fusion is late, the annotation-firstcolumn and annotation-secondcolumn should have same length with the same order of embedding methods.

&emsp; &emsp; &emsp; **annotation-secondcolumn** &ensp; filepaths of the embeddings containing the entities of the **second** column in the annotation file\
&emsp; &emsp; &emsp; The file paths should be given in this format: '[["hope_protein.txt"](./output/hope_6_hetero_drugs.txt), ["lap_protein.txt"](./output/lap_6_hetero_drugs.txt)]'.\
&emsp; &emsp; &emsp; When the fusion is late, the annotation-firstcolumn and annotation-secondcolumn should have same length with the same order of embedding methods.

&emsp; &emsp; &emsp; **cv-type** &ensp; Cross-validation method\
&emsp; &emsp; &emsp; Choices are 'kfold', 'stratified' and 'split'. 'split' divide the data regarding the *test-size* size.

&emsp; &emsp; &emsp; **cv** &ensp; Number of folds\
&emsp; &emsp; &emsp; This argument is used when the *cv-type* is either 'kfold' or 'stratified'.

&emsp; &emsp; &emsp; **cv-shuffle** &ensp; Whether to shuffle each class’s samples before splitting into batches\
&emsp; &emsp; &emsp; This argument is used when the *cv-type* is either 'kfold' or 'stratified'.

&emsp; &emsp; &emsp; **test-size** &ensp; Percentage of the data to be test-set\
&emsp; &emsp; &emsp; The value of this argument must be between 0 and 1. This can be used when *cv-type* is 'split'.

&emsp; &emsp; &emsp; **imbalance** &ensp; Dealing with imbalance classes\
&emsp; &emsp; &emsp; Choices are: 'equalize' which equalize the number of majority class to minority class.\
&emsp; &emsp; &emsp; 'SMOTE' and 'ADASYN' are oversampling methods.\
&emsp; &emsp; &emsp; 'None' does not deal with imbalance classes.


&emsp; &emsp; &emsp; **fselection** &ensp; feature selection\
&emsp; &emsp; &emsp; Choices are: 'fvalue', 'qvalue', 'MI' or None.\
&emsp; &emsp; &emsp; ANOVA analyze the differences among the means between classes. The output is either in 'fvalue' or pvalue.\
&emsp; &emsp; &emsp; *ktop* argument helps to select features with K highest 'fvalues'.\
&emsp; &emsp; &emsp; The 'qvalue' is the Bonferroni correction of p-values with values lower than 0.1.\
&emsp; &emsp; &emsp; The MI is based on mutual information. Here *ktop* helps to collect features with K highest MI value.


&emsp; &emsp; &emsp; **ktop** &ensp; Select K highest value features\
&emsp; &emsp; &emsp; Select features according to the k highest scores if feature selection is either fvalue or MI.

&emsp; &emsp; &emsp; **model** &ensp; Machine Learning models\
&emsp; &emsp; &emsp; Choices are 'SVM','RF','NB' and 'XGBoost'.\
&emsp; &emsp; &emsp; The models should be given in this format: '["SVM"]'\
&emsp; &emsp; &emsp; In case it is mix fusion, the models should given in this format: '["SVM","RF", "NB", "XGBoost"]'

&emsp; &emsp; &emsp; **random_state** &ensp; Fixing the randomization\
&emsp; &emsp; &emsp; Default value is None.

&emsp; &emsp; &emsp; **kernel** &ensp; Specifies the kernel type to be used in the algorithm\
&emsp; &emsp; &emsp; This can be used when *classification* is SVM.

&emsp; &emsp; &emsp; **C** &ensp; Regularization parameter\
&emsp; &emsp; &emsp; The default value is 1. This can be used when *model* is SVM.

&emsp; &emsp; &emsp; **ntree** &ensp; The number of trees in the random forest\
&emsp; &emsp; &emsp; Default value is 100.

&emsp; &emsp; &emsp; **criterion** &ensp; The function to measure the quality of a split in random forest\
&emsp; &emsp; &emsp; Choices are 'gini' and 'entropy'

&emsp; &emsp; &emsp; **njob** &ensp; The number of jobs to run in parallel in random forest

&emsp; &emsp; &emsp; **output** &ensp; The folder path to save predictions and evaluation results\
&emsp; &emsp; &emsp; Only provide directory and file prefix. e.g. ./Desktop/DTI_prediction \
&emsp; &emsp; &emsp; Click [here](./output/mix_DTI_prediction.csv) to see a sample prediction output and [here](./output/mix_DTI_prediction.png) for ROC and PR curves.\
&emsp; &emsp; &emsp; In ROC and PR, the label of the positive class is fixed to 1.
    
&nbsp;


## Example
Here you can find the example of Drug-Target interaction link prediction task.\
The toy data provided here are downsized versions of real data and not recommended to use in other scientific study.

```shell
# 1) Network Preparation

# Convert drug-se and drug-disease adjacency matrices to the edge list
python3 scripts/mat2edgelist.py --input ./data/mat_drug_se.txt --output ./output/edgelist_drug_se.txt
python3 scripts/mat2edgelist.py --input ./data/mat_drug_disease.txt --output ./output/edgelist_drug_disease.txt
# Drugs heterogeneous network preparation
python3 scripts/merge_edgelist.py --input1 ./output/edgelist_drug_se.txt --input2 ./output/edgelist_drug_disease.txt --rmduplicate --output ./output/hetero_drugs.txt


# Convert protein-disease adjacency matrix to the edge list
python3 scripts/mat2edgelist.py --input ./data/mat_protein_disease.txt --output ./output/edgelist_protein_disease.txt


# 2) Embedding
# Embedding on drugs heterogeneous network. The hope and lap embeddings conducted
python3 scripts/embedding.py --method hope --input ./output/hetero_drugs.txt --representation_size 6 --output ./output/hope_6_hetero_drugs.txt
python3 scripts/embedding.py --method lap  --input ./output/hetero_drugs.txt --representation_size 6 --output ./output/lap_6_hetero_drugs.txt

# Embedding on protein-disease edge list. The hope and lap embeddings conducted
python3 scripts/embedding.py --method hope --input ./output/edgelist_protein_disease.txt --representation_size 6 --output ./output/hope_6_protein_disease.txt
python3 scripts/embedding.py --method lap  --input ./output/edgelist_protein_disease.txt --representation_size 6 --output ./output/lap_6_protein_disease.txt


# 3) Predictions using the integration of embeddings
# Create annotation file
python3 scripts/mat2edgelist.py --input ./data/mat_drug_protein.txt --directed --keepzero --attribute --output ./output/annotation.txt
# late fusion
python3 scripts/integration.py --fusion late --annotation ./output/annotation.txt --annotation-firstcolumn '["./output/hope_6_hetero_drugs.txt","./output/lap_6_hetero_drugs.txt"]' --annotation-secondcolumn '["./output/hope_6_protein_disease.txt","./output/lap_6_protein_disease.txt"]' --cv-type stratified --cv 10 --imbalance ADASYN --model '["RF"]' --random_state 11 --output ./output/DTI_prediction 
```

&nbsp;

## Citation
Please cite:\
The paper is under processing.

If you only used the embedding part in your research and not the other parts, you should consider citing [OpenNE](https://github.com/thunlp/OpenNE.git) and the articles of embedding methods that you used.

## Contact
If you have any questions, please submit an issue on GitHub or send an email to [poorya.parvizi@ed.ac.uk](mailto:poorya.parvizi@ed.ac.uk).
