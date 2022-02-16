import argparse
from cmath import e
import os
import pandas as pd
from benchtools.src.datatools import ascii_column

parser = argparse.ArgumentParser()
parser.add_argument("--boxn", type=int, default=1, help="Number of the black box to parse [default: 1]")
parser.add_argument('--dir', type=str, default="../../", help='Path of h5 files')
parser.add_argument("--out", type=str, default="../h5/", help="Folder to save output files")
parser.add_argument("--chunksize", type=int, default=100000, help="Number of the black box to parse [default: 100,000]")
parser.add_argument("--datasize", type=int, default=1000000, help="Number of the black box to parse [default: 1,000,000]")


flags = parser.parse_args()

samples_path = flags.dir
bb = flags.boxn
save_path = flags.out

# Getting the name of the files
sample = 'events_LHCO2020_BlackBox{}.h5'.format(bb)
key = 'events_LHCO2020_BlackBox{}.masterkey'.format(bb)

start = 0
chunksize = flags.chunksize
dataset_size = flags.datasize

# doing it with a loop
while start is not dataset_size:
    print(start)
    # Reading the files for the data and the labels
    df_bb = pd.read_hdf(os.path.join(samples_path,sample), start=start, stop=start+chunksize)
    df_key = ascii_column(os.path.join(samples_path,key))

    # Joining the df's
    bb_with_key = df_bb.assign(label=pd.Series(df_key.iloc[start:start+chunksize,0]).values)
    bb_with_key.rename(columns={'label': 2100}, inplace=True)

    # Saving as h5
    bb_with_key.to_hdf(os.path.join(save_path,'BlackBox{}_with_key.h5'.format(bb)), key='bb') 
    
    start+=chunksize 

print('Succesfully joined')
