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
parser.add_argument("--chunksize", type=int, default=100000, help="Number of the black box to parse [default: 100,000]")
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

f = h5py.File(os.path.join(SAVE_PATH,'BlackBox{}_with_key.h5'.format(BB)), 'a')

# doing it with a loop
for ii in range(int(N_EVENTS/CHUNKSIZE)):
    print(start)
    # Reading the files for the data
    df_bb = pd.read_hdf(os.path.join(SAMPLES_PATH,sample), start=start, stop=start+CHUNKSIZE)
    # Joining the df's
    bb_with_key = df_bb.assign(label=pd.Series(df_key.iloc[start:start+CHUNKSIZE,0]).values)
    bb_with_key.rename(columns={'label': 2100}, inplace=True)
    
    bb_array = bb_with_key.to_numpy()

    if ii == 0:
        f.create_dataset('bb', data=bb_array, compression="gzip", chunks=True, maxshape=(None, None))
        print(f['bb'].shape)
    else:
        # Append new data to it
        print(f['bb'].shape[0] , bb_array.shape[0])
        f['bb'].resize((f['bb'].shape[0] + bb_array.shape[0]), axis=1)
        f['bb'][-bb_array.shape[0]:] = bb_array
        
    start+=CHUNKSIZE 

'''

while start != N_EVENTS:
    print(start)
    # Reading the files for the data
    df_bb = pd.read_hdf(os.path.join(SAMPLES_PATH,sample), start=start, stop=start+CHUNKSIZE)
    # Joining the df's
    bb_with_key = df_bb.assign(label=pd.Series(df_key.iloc[start:start+CHUNKSIZE,0]).values)
    bb_with_key.rename(columns={'label': 2100}, inplace=True)

    list_df.append(bb_with_key)

'''
print(f['bb'].shape)
    

# Saving as h5
#df_final = pd.concat(list_df, ignore_index=True)
#df_final.to_hdf(os.path.join(SAVE_PATH,'BlackBox{}_with_key.h5'.format(bb)), key='bb')

print('Succesfully joined')
