#!/usr/bin/env python3
'''
embedding
'''

import sys
import timeit
import tensorflow as tf
import numpy as np
import pandas as pd
import random
from sklearn.metrics import confusion_matrix
from sklearn import svm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
sys.path.append('./scripts/openNE')
from graph import *
import lle
from classify import Classifier, read_node_label
import line
import grarep
import time
import ast
import sdne
import hope
import node2vec
import lap
import gf


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--method', required=True, choices=['node2vec','deepwalk','line','grarep','lle','hope','lap','gf','sdne'], help='Embedding method')    
    parser.add_argument('--input', required=True,
                        help='The filepath of edge list file')
    parser.add_argument('--random_state', type= int, default= 1,
            help='Fixing the randomization')
    parser.add_argument('--directed', action='store_true',
                        help='Treat the network as directed')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat the network as weighted')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The number of times that the learning algorithm will work through the entire training data set')
    parser.add_argument('--representation_size', default=128, type=int,
                        help='Dimensionality of the output data')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of line. 1 means first order, 2 means second order, 3 means first order + second order')    
    parser.add_argument('--negative_ratio', default=5, type=int,
                        help='Negative sampling ratio')
    parser.add_argument('--kstep', default=2, type=int,
                        help='Use k-step transition probability matrix')
    parser.add_argument('--encoder_list', default='[1000, 128]', type=str,
                        help='a list of numbers of the neurons at each encoder layer in sdne')    
    parser.add_argument('--alpha', default=1e-6, type=float,
                        help='alhpa is a hyperparameter in sdne')
    parser.add_argument('--beta', default=5., type=float,
                        help='beta is a hyperparameter in sdne')
    parser.add_argument('--nu1', default=1e-5, type=float,
                        help='nu1 is a hyperparameter in sdne')
    parser.add_argument('--nu2', default=1e-4, type=float,
                        help='nu2 is a hyperparameter in sdne')
    parser.add_argument('--bs', default=200, type=int,
                        help='batch size in sdne')    
    parser.add_argument('--lr', default=0.01, type=float,
                        help='learning rate in sdne')    
    parser.add_argument('--walk-length', default=20, type=int,
                        help='Length of the random walk started at each node')
    parser.add_argument('--number-walks', default=80, type=int,
                        help='Number of random walks to start at each node')
    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes')
    parser.add_argument('--p', default=1.0, type=float, help= 'Return hyperparameter in node2vec')
    parser.add_argument('--q', default=1.0, type=float, help= 'Inout hyperparameter in node2vec')
    parser.add_argument('--window-size', default=10, type=int,
                        help='Window size of skipgram model in node2vec and deepwalk')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight for L2 loss on embedding matrix in gf')
    parser.add_argument('--output',
                        help='The filepath to save the embedding results')

    return parser.parse_args()


def main(args):
    print("**embedding**\n")

    saved_args = locals()
    saved_args = locals(); print("Arguments:", vars(saved_args["args"]), "\n")

    g = Graph()
    tf.compat.v1.reset_default_graph()
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    
    g.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)
    

    
    if args.method == 'line':

        model = line.LINE(g, epoch= args.epochs, rep_size= args.representation_size, 
                           order=args.order, negative_ratio= args.negative_ratio)
        
        
    elif args.method == 'grarep':
        model = grarep.GraRep(graph=g, Kstep=args.kstep, dim=args.representation_size)

        
    elif args.method == 'sdne':
        encoder_layer_list = ast.literal_eval(args.encoder_list)
        model = sdne.SDNE(g, encoder_layer_list=encoder_layer_list,
                          alpha=args.alpha, beta=args.beta, nu1=args.nu1, nu2=args.nu2,
                          batch_size=args.bs, epoch=args.epochs, learning_rate=args.lr)

    elif args.method == 'lle':
        model = lle.LLE(graph=g, d=args.representation_size)
        
    
    elif args.method == 'hope':
        model = hope.HOPE(graph=g, d=args.representation_size)
    
    elif args.method == 'lap':
        model= lap.LaplacianEigenmaps(g, rep_size=args.representation_size)
    
    elif args.method == 'node2vec':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, p=args.p, q=args.q, window=args.window_size)

    elif args.method == 'deepwalk':
        model = node2vec.Node2vec(graph=g, path_length=args.walk_length,
                                  num_paths=args.number_walks, dim=args.representation_size,
                                  workers=args.workers, window=args.window_size, dw=True)
        
    elif args.method == 'gf':
        model = gf.GraphFactorization(g, rep_size=args.representation_size,
                                      epoch=args.epochs, learning_rate=args.lr, weight_decay=args.weight_decay)

    ## Save embeddings    
    model_df= pd.DataFrame.from_dict(model.vectors, orient='index')
    model_df.to_csv(args.output, sep=' ', index=True, header=False)
    print("output file size:", model_df.shape)

if __name__ == "__main__":
    start = timeit.default_timer()
    main(parse_args())
    stop = timeit.default_timer()
    print('Time:', np.round(stop - start, 3))
