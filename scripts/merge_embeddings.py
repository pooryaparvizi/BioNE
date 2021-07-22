#!/usr/bin/env python3
'''
merge_embeddings
'''

import argparse
import numpy as np
import pandas as pd
from distutils import util

def parse_args():
    '''
    Parses the merge_embeddings arguments
    '''
    
    parser = argparse.ArgumentParser(description="Run merge_embeddings")
    
    parser.add_argument('--input1', required=True,
            help='The filepath of first embedding file')
    
    parser.add_argument('--input2', required=True,
            help='The filepath of second embedding file')
        
    parser.add_argument('--output', required=True,
            help='The filepath to save the combined embedding results for mutual entity')
    
    return parser.parse_args()


def main(args):
    print("**merge_edgelist**")
    df1 = pd.read_csv(args.input1, index_col= 0, header= None, sep= " ")
    print("input1 size:", df1.shape)
    df2 = pd.read_csv(args.input2, index_col= 0, header= None, sep= " ")
    print("input2 size:", df2.shape)
    result = pd.concat([df1, df2], axis=1, join='inner')

    if np.size(result, 0) > 0:
        print("mutual entities:", np.size(result, 0))
    else:
        raise ValueError("No mutual entites")
    
    print("output file size:", result.shape)
    result.to_csv(args.output, index= True, header=False, sep= " ")
    
if __name__ == "__main__":
	args = parse_args()
	main(args)
