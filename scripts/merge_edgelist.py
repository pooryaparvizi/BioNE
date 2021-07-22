#!/usr/bin/env python3
'''
merge_edgelist
'''

import argparse
import numpy as np
import pandas as pd
from distutils import util

def parse_args():
    '''
    Parses the merge_edgelist arguments
    '''
    
    parser = argparse.ArgumentParser(description="Run merge_edgelist")
    
    parser.add_argument('--input1', required=True,
            help='The filepath of first edge list file')
    
    parser.add_argument('--input2', required=True,
            help='The filepath of second edge list file')
        
    parser.add_argument('--rmduplicate',  action='store_true',
            help='Remove duplicate interactions')

    parser.add_argument('--output', required=True,
            help='The filepath to save the combined edge lists file')
    
    return parser.parse_args()


def main(args):
    print("**merge_edgelist**")
    df1 = pd.read_csv(args.input1, header= None, sep= " ")
    print("input1 size:", df1.shape)
    df2 = pd.read_csv(args.input2, header= None, sep= " ")
    print("input2 size:", df2.shape)
    if np.size(df1,1) == np.size(df2, 1):
        merge_df1_df2= pd.concat([df1, df2])
    else:
        raise ValueError("input1 and input2 must have same column size\n")
    
    if args.rmduplicate:
        initial= np.size(merge_df1_df2, 0)
        merge_df1_df2= merge_df1_df2.drop_duplicates()
        dif= initial - np.size(merge_df1_df2, 0)
        print("rmduplicate: " , str(dif) , "removed")

    else:
        print("rmduplicate: False")
    
        
    print("Merged file size: " , merge_df1_df2.shape , "\n")
    merge_df1_df2.to_csv(args.output, index= False, header=False, sep= " ")
    
if __name__ == "__main__":
	args = parse_args()
	main(args)
