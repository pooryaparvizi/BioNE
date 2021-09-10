#!/usr/bin/env python3


#    BioNE: Integration of network embeddings for supervised learning
#    Copyright (C) 2021  Poorya Parvizi
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


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
