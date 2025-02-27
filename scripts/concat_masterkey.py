import h5py
import argparse
from cmath import e
import os
import pandas as pd
from benchtools.src.datatools import ascii_column

parser = argparse.ArgumentParser()
parser.add_argument("--boxn", type=int, default=1, help="Number of the black box to parse [default: 1]")
parser.add_argument('--dir', type=str, default="../../", help='Path of h5 files')
parser.add_argument("--out", type=str, default="../h5/", help="Folder to save output files")
parser.add_argument("--chunksize", type=int, default=333333, help="Number of the black box to parse [default: 100,000]")
parser.add_argument("--datasize", type=int, default=1000000, help="Number of the events in the black box [default: 1,000,000]")


flags = parser.parse_args()

SAMPLES_PATH = flags.dir
BB = flags.boxn
SAVE_PATH = flags.out

# Getting the name of the files
sample = 'events_LHCO2020_BlackBox{}.h5'.format(BB)
key = 'events_LHCO2020_BlackBox{}.masterkey'.format(BB)

start = 0
CHUNKSIZE = flags.chunksize
N_EVENTS = flags.datasize

# Reading the label
df_key = ascii_column(os.path.join(SAMPLES_PATH,key))

list_df = []
ii=0

# looping to concat with key
while start < N_EVENTS:
    print(start)
    # Reading the files for the data
    if (start+CHUNKSIZE) > N_EVENTS:
        df_bb = pd.read_hdf(os.path.join(SAMPLES_PATH,sample), start=start)
    else:
        df_bb = pd.read_hdf(os.path.join(SAMPLES_PATH,sample), start=start, stop=start+CHUNKSIZE)
    
    # Joining the df's
    bb_with_key = df_bb.assign(label=pd.Series(df_key.iloc[start:start+CHUNKSIZE,0]).values)
    bb_with_key.rename(columns={'label': 2100}, inplace=True)
    

    bb_with_key.to_hdf(os.path.join(SAVE_PATH,'BlackBox{}_with_key_{}.h5'.format(BB,ii)), key='bb')

    start += CHUNKSIZE
    ii+=1
    

print('Succesfully joined')
