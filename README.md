# BioNE: Integration of network embeddings for supervised learning

## Overview

A network embedding approach reduces the complexity of analyzing large biological networks by converting the high-dimensional adjacency matrix representations to low-dimensional vector representations. These lower-dimensional representations can then be used in machine learning prediction tasks such as link prediction and node classification. Several network embedding methods have been proposed with different approaches to obtain network features. Integrating them could offer complementary information about the network, and therefore improve performance on prediction tasks. BioNE is a pipeline that applies a range of network embedding methods following the network preparation step and integrates the vector representations obtained by these methods using three different techniques.

The BioNE pipeline is divided into three steps;

&emsp; 1. [Network Preparation](#1-network-preparation)\
&emsp; &emsp; 1.1. [Convert Adjacency Matrix to Edge List](#11-convert-adjacency-matrix-to-edge-list)\
&emsp; &emsp; 1.2. [Heterogeneous Network Preparation](#12-heterogeneous-network-preparation)\
&emsp; 2. [Network Embedding](#2-network-embedding)\
&emsp; 3. [Predictions Using the Integration of Embeddings](#3-predictions-using-the-integration-of-embeddings)


In order to install packages and create the necessary virtual environment, check section [Virtual Environment and Installing Packages](#virtual-environment-and-installing-packages).\
This pipeline will be tested using Drug-Target Interaction (DTI) data as a link prediction task. You can find the scripts of this test in the [Example](#example) section.

&nbsp;

## Virtual Environment and Installing Packages
All of the analyses are written and tested on virtual environment using python 3.7. The detailed software versions are listed below:
- Python 3.7
- virtualenv 20.4.0
- ubuntu 20.04
- nvidia-driver 460
- cuda 10.0
- cuDNN 7.4.2

To create the virtual environment:
```shell
cd BioNE-main
virtualenv --python=/usr/bin/python3.7 BioNEvenv
```

To activate and install required [packages](./requirements.txt):
```shell
source BioNEvenv/bin/activate
pip install -r requirements.txt
```
&nbsp;

## Input files formats
The input file format for [Network Embedding](#2-network-embedding) is a space-delimited edge list file. If the edge list file is ready in [this](./output/edgelist_protein_disease.txt) format, users can start from the [Network Embedding](#2-network-embedding) step. If the networks are in adjacency matrix format, this pipeline provides the command line to convert adjacency matrices to edge lists in section [1.1. Convert Adjacency Matrix to Edge List](#11-convert-adjacency-matrix-to-edge-list). Adjacency matrices should contain column names and row names, and the format should be space-delimited. Click [here](./data/mat_protein_disease.txt) to see sample adjacency matrix.

&nbsp;

## 1. Network Preparation
This part consists of two sections. Users can convert adjacency matrices to edge list files in section [1.1. Convert Adjacency Matrix to Edge List](#11-convert-adjacency-matrix-to-edge-list). On the other hand, when required, users can combine two edge list files to form a  heterogeneous network using command lines provided in section [1.2. Heterogeneous Network Preparation](#12-heterogeneous-network-preparation).

&nbsp;

### 1.1. Convert Adjacency Matrix to Edge List
In order to conduct network embedding, adjacency matrices should be converted to an edge list file format.

```shell
python3 scripts/mat2edgelist.py --input input.txt --directed --keepzero --attribute --output output.txt
```
##### *Arguments*:

&emsp; &emsp; &emsp; **input** &ensp; The filepath of the adjacency matrix\
&emsp; &emsp; &emsp; Input adjacency matrix file should be space-delimited file and contains row and column index labels.\
&emsp; &emsp; &emsp; Click [here](./data/mat_protein_disease.txt) to see a sample file.

&emsp; &emsp; &emsp; **directed** &ensp; Treat the graph as directed\
&emsp; &emsp; &emsp; When directed, row indexes are source nodes and column indexes are target nodes.

&emsp; &emsp; &emsp; **keepzero** &ensp; Adding negative associations (0s) to the output

&emsp; &emsp; &emsp; **attribute** &ensp; Including the edge attributes to the output file\
&emsp; &emsp; &emsp; If edge attributes are not going to be used as weights in network embedding, removing this line is recommended to save memory.

&emsp; &emsp; &emsp; **output** &ensp; The filepath for the outputted edge list file\
&emsp; &emsp; &emsp; The file will be saved as a space-delimited file. Click [here](./output/edgelist_protein_disease.txt) to see a sample edge list file.

&nbsp;

### 1.2. Heterogeneous Network Preparation
When required, users can combine two edge lists (e.g. drug-disease and drug-side effect networks) to construct a heterogeneous network. The command line below can be used to combine edge lists. This can be used multiple times to combine more than two edge list files.

```shell
python3 scripts/merge_edgelist.py --input1 input1.txt --input2 input2.txt --rmduplicate --output output.txt
```
##### *Arguments*:

&emsp; &emsp; &emsp; **input1** &ensp; The filepath of first edge list file\
&emsp; &emsp; &emsp; This file should be a space-delimited edge list file. Click [here](./output/edgelist_drug_disease.txt) to see a sample input file.


&emsp; &emsp; &emsp; **input2** &ensp; The filepath of second edge list file\
&emsp; &emsp; &emsp; This file should be a space-delimited edge list file. Click [here](./output/edgelist_drug_se.txt) to see a sample input file.

&emsp; &emsp; &emsp; **rmduplicate** &ensp; Removes duplicated edges

&emsp; &emsp; &emsp; **output** &ensp; The filepath for the outputted combined edge list file\
&emsp; &emsp; &emsp; The file will be saved as a space-delimited file. Click [here](./output/hetero_drugs.txt) to see a sample output file.

&nbsp;

## 2. Network Embedding
Network embedding methods convert high-dimensional data to low-dimensional vector representations. In this project users are able to conduct the following embedding methods:\
[LINE](https://doi.org/10.1145/2736277.2741093), [GraRep](https://doi.org/10.1145/2806416.2806512), [SDNE](https://doi.org/10.1145/2939672.2939753), [LLE](https://doi.org/10.1126/science.290.5500.2323), [HOPE](https://doi.org/10.1145/2939672.2939751), [LaplacianEigenmaps (Lap)](https://dl.acm.org/doi/abs/10.5555/2980539.2980616), [node2vec](https://doi.org/10.1145/2939672.2939754), [DeepWalk](https://doi.org/10.1145/2623330.2623732) and [GF](https://doi.org/10.1145/2488388.2488393).\
Embedding methods in this section are inherited from [OpenNE](https://github.com/thunlp/OpenNE.git) and [OpenNE-PyTorch](https://github.com/thunlp/OpenNE/tree/pytorch) repositories.

```shell
python3 scripts/embedding.py --method lle --input input.txt --directed --weighted --representation_size 128 --output output.txt
```
##### *Arguments*:

&emsp; &emsp; &emsp; **method:** &ensp; Network embedding method\
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

&emsp; &emsp; &emsp; &emsp; Note: *input*, *directed*, *weighted*, *random_state* and *representation_size* are shared among all methods.

&emsp; &emsp; &emsp; **input**: &ensp; The filepath of the edge list file\
&emsp; &emsp; &emsp; This file should be an space-delimited edge list. Click [here](./output/edgelist_protein_disease.txt) to see a sample input file.

&emsp; &emsp; &emsp; **directed**: &ensp; Treats the network as directed\
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
&emsp; &emsp; &emsp; This parameter is used in *grarep*. The default value is 2.

&emsp; &emsp; &emsp; **encoder-list**: &ensp; a list of neuron numbers in each encoder layer within *sdne*\
&emsp; &emsp; &emsp; The last number is the dimension of the output embeddings. The default is [1000,128].

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
&emsp; &emsp; &emsp; The learning rate controls how quickly the model adapts to the problem. The default is 0.001.

&emsp; &emsp; &emsp; **walk-length**: &ensp; Length of the random walk started at each node\
&emsp; &emsp; &emsp; This parameter is used in *node2vec* and *deepwalk*. The default value is 20.

&emsp; &emsp; &emsp; **number-walks**: &ensp; Number of random walks to start at each node\
&emsp; &emsp; &emsp; This parameter is used in *node2vec* and *deepwalk*. The default value is 80.

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

&emsp; &emsp; &emsp; **output**: &ensp; The filepath for the embedding results\
&emsp; &emsp; &emsp; The file saves as a space-delimited file. Click [here](./output/hope_6_protein_disease.txt) to see a sample output file.

&nbsp;


## 3. Predictions using the integration of embeddings
For this section, we developed three different integration methods (late fusion, early fusion and mixed fusion) to integrate embedding results from the different methods. This ensures a comprehensive representation of networks and therefore imrpoves performance on prediction tasks.

```shell
python3 scripts/integration.py --fusion late --annotation annotation.txt --annotation-firstcolumn '["hope_x.txt","lap_x.txt"]' --annotation-secondcolumn '["hope_y.txt","lap_y.txt"]' --cv-type stratified --cv 10 --imbalance ADASYN --model '["RF"]' --output ./output
```
##### *Arguments*:

&emsp; &emsp; &emsp; **fusion** &ensp; The integration type\
&emsp; &emsp; &emsp; Choices are:\
&emsp; &emsp; &emsp; &emsp; early: Merging all embedding results before inclusion in the prediction model\
&emsp; &emsp; &emsp; &emsp; late (default): Including each embedding result in the prediction model and then summing up the achieved prediction probabilities.\
&emsp; &emsp; &emsp; &emsp; mix: Merging all embedding results, and then summing up the prediction probabilities achieved from different prediction models.

&emsp; &emsp; &emsp; **annotation** &ensp; The filepath of the annotation file\
&emsp; &emsp; &emsp; This file should contain either two or three columns. Click [here](./output/annotation.txt) or [here](./output/annotation_2.txt) to see a sample annotation file.\
&emsp; &emsp; &emsp; two column annotation file can only be used in *mix* and *early* fusions.



&emsp; &emsp; &emsp; **annotation-firstcolumn** &ensp; filepaths of the embeddings containing the entities of the **first** column in the annotation file\
&emsp; &emsp; &emsp; The file paths should be given in this format: '[["hope_drugs.txt"](./output/hope_6_hetero_drugs.txt), ["lap_drugs.txt"](./output/lap_6_hetero_drugs.txt)]'.\
&emsp; &emsp; &emsp; When late fusion is applied, the annotation-firstcolumn and annotation-secondcolumn should have the same length with the same order of embedding methods.

&emsp; &emsp; &emsp; **annotation-secondcolumn** &ensp; filepaths of the embeddings containing the entities of the **second** column in the annotation file\
&emsp; &emsp; &emsp; The file paths should be given in this format: '[["hope_protein.txt"](./output/hope_6_protein_disease.txt), ["lap_protein.txt"](./output/lap_6_protein_disease.txt)]'.\
&emsp; &emsp; &emsp; When late fusion is applied, the annotation-firstcolumn and annotation-secondcolumn should have same length with the same order of embedding methods.

&emsp; &emsp; &emsp; **cv-type** &ensp; Cross-validation method\
&emsp; &emsp; &emsp; Choices are 'kfold', 'stratified' and 'split' (default). 'split' divides the data according to the *test-size* size.

&emsp; &emsp; &emsp; **cv** &ensp; Number of folds\
&emsp; &emsp; &emsp; This argument is used when the *cv-type* is either 'kfold' or 'stratified'.\
&emsp; &emsp; &emsp; Default value is 5.

&emsp; &emsp; &emsp; **cv-shuffle** &ensp; Whether to shuffle each class’s samples before splitting into batches\
&emsp; &emsp; &emsp; This argument is used when the *cv-type* is either 'kfold' or 'stratified'.

&emsp; &emsp; &emsp; **test-size** &ensp; Percentage of the data to be test-set\
&emsp; &emsp; &emsp; The value of this argument must be between 0 and 1. This can be used when *cv-type* is 'split'.\
&emsp; &emsp; &emsp; Default value is 0.2.

&emsp; &emsp; &emsp; **imbalance** &ensp; Deals with imbalanced classes\
&emsp; &emsp; &emsp; Choices are: 'equalize' which equalizes the number of majority class to minority class.\
&emsp; &emsp; &emsp; 'SMOTE' and 'ADASYN' are oversampling methods.\
&emsp; &emsp; &emsp; 'None' (default) does not deal with imbalanced classes.


&emsp; &emsp; &emsp; **fselection** &ensp; feature selection\
&emsp; &emsp; &emsp; Choices are: 'fvalue', 'qvalue', 'MI' or None.\
&emsp; &emsp; &emsp; ANOVA analyses the differences among the means between classes. The output is either in 'fvalue' or pvalue.\
&emsp; &emsp; &emsp; *ktop* argument helps to select features with K highest 'fvalues'.\
&emsp; &emsp; &emsp; The 'qvalue' is the Bonferroni correction of p-values with values lower than 0.1.\
&emsp; &emsp; &emsp; The MI is based on mutual information. Here *ktop* helps to collect features with K highest MI value.


&emsp; &emsp; &emsp; **ktop** &ensp; Select K highest value features\
&emsp; &emsp; &emsp; Select features according to the k highest scores if feature selection is either fvalue or MI.\
&emsp; &emsp; &emsp; Default value is 10.

&emsp; &emsp; &emsp; **model** &ensp; Machine Learning models\
&emsp; &emsp; &emsp; Choices are 'SVM' (default), 'RF','NB' and 'XGBoost'.\
&emsp; &emsp; &emsp; The models should be given in this format: '["SVM"]'\
&emsp; &emsp; &emsp; In the case where mixed fusion is applied, the models should be given in this format: '["SVM","RF", "NB", "XGBoost"]'

&emsp; &emsp; &emsp; **random_state** &ensp; Fixing the randomization\
&emsp; &emsp; &emsp; Default value is None. 

&emsp; &emsp; &emsp; **kernel** &ensp; Specifies the kernel type to be used in the algorithm\
&emsp; &emsp; &emsp; This can be used when *classification* is SVM.\
&emsp; &emsp; &emsp; Default is 'linear'.

&emsp; &emsp; &emsp; **C** &ensp; Regularization parameter\
&emsp; &emsp; &emsp; The default value is 1. This can be used when *model* is SVM.

&emsp; &emsp; &emsp; **ntree** &ensp; The number of trees in the random forest\
&emsp; &emsp; &emsp; Default value is 100.

&emsp; &emsp; &emsp; **criterion** &ensp; The function to measure the quality of a split in random forest\
&emsp; &emsp; &emsp; Choices are 'gini' (default) and 'entropy'

&emsp; &emsp; &emsp; **njob** &ensp; The number of parallel jobs to run in random forest.

&emsp; &emsp; &emsp; **output** &ensp; The filepath for the predictions and evaluation results\
&emsp; &emsp; &emsp; Only provide directory and file prefix. e.g. ./Desktop/DTI_prediction \
&emsp; &emsp; &emsp; Click [here](./output/mix_DTI_prediction.csv) to see a sample prediction output and [here](./output/mix_DTI_prediction.png) for ROC and PR curves.\
&emsp; &emsp; &emsp; In ROC and PR, the label of the positive class is fixed to 1.
    
&nbsp;


## Example
Here you can find the example of the Drug-Target interaction link prediction task.\
The toy data provided here are downsized versions of real data and not recommended to use in other scientific studies.

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
The paper is under process.

Embedding methods in this section are inherited from OpenNE repository (https://github.com/thunlp/OpenNE). If only network embedding section used in your research and not the other parts, you should consider citing [OpenNE](https://github.com/thunlp/OpenNE.git) and the articles of embedding methods that you used.

## Contact
If you have any questions, please submit an issue on GitHub or send an email to [poorya.parvizi@ed.ac.uk](mailto:poorya.parvizi@ed.ac.uk).

## License
Licensed under [GPLv3](./LICENCE) license
