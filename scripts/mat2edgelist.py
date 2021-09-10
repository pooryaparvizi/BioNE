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
mat2edgelist
'''

import argparse
import numpy as np
import pandas as pd
from distutils import util

def parse_args():
    '''
    Parses the mat2edgelist arguments
    '''
    
    parser = argparse.ArgumentParser(description="Run mat2edgelist")
    
    parser.add_argument('--input', required=True,
            help='The filepath of adjacency matrix')
         
    parser.add_argument('--directed',  action='store_true',
            help='Treat the graph as directed')
    
    parser.add_argument('--keepzero',  action='store_true',
            help='Adding also non-interactive nodes to the output')
    
    parser.add_argument('--attribute',  action='store_true',
                        help='Including the edge attributes to the output file') 
    
    parser.add_argument('--output', required=True,
            help='The filepath to save edge list file')
    
    return parser.parse_args()


def edgelist(mat, directed, keepzero, attribute):
    
    print("Adjacency matrix size:", mat.shape)
    mat_values = mat.values
    
    column_index = mat.columns
    number_column = len(column_index)
    
    row_index = mat.index
    number_row = len(row_index)
        
    row_index_array = np.array(row_index)
    column_index_array = np.array(column_index)
        
    empty_array = np.empty((number_row, number_column, 2), dtype="<U10")
    empty_array[...,0] = row_index_array[:,None]
    empty_array[...,1] = column_index_array
    mask = np.full((number_row,number_column), True, dtype=bool)
    mat_edgelist = pd.DataFrame(empty_array[mask], columns=[['Source','Target']])
    mat_edgelist['attribute'] = mat_values[mask]
    
    if (keepzero == False):
        nonZero= np.array(mat_edgelist[["attribute"]] > 0)
        mat_edgelist= mat_edgelist.iloc[nonZero]
        print("keepzero: False")
    else:
        print("keepzero: True")


    if(directed == False):
        otherside= mat_edgelist[['Target', 'Source', 'attribute']]
        otherside2= otherside.rename(columns={"Target": "Source", "Source": "Target"})
        df_dir_conc= pd.concat([mat_edgelist, otherside2])
        mat_edgelist= df_dir_conc.drop_duplicates()
        mat_edgelist= mat_edgelist.reset_index(drop=True)
        print("directed: False")
    else:
        print("directed: True")
        
    
    if attribute == False:
        mat_edgelist= mat_edgelist[['Source', 'Target']]
        print("attributes: not included")
    else:
        print("attributes: included")
    
    print("Edge list size:", mat_edgelist.shape, "\n")
    return mat_edgelist


def main(args):
    print("**mat2edgelist**")
    df = pd.read_csv(args.input, header= 0, sep= " ")
    edgelist_file= edgelist(mat= df, directed= args.directed, keepzero= args.keepzero, attribute= args.attribute)
    edgelist_file.to_csv(args.output, index= False, sep= " ", header= False)
    
if __name__ == "__main__":
	args = parse_args()
	main(args)
