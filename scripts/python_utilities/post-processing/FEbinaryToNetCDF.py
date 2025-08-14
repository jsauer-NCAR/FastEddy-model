import os, sys
import struct
import numpy as np
import numpy.matlib
import xarray as xr
import pandas as pd
import time 
import warnings
import gc
import json
import argparse
from mpi4py import MPI

def field3dTranspose(fld,extents):
    fld=fld.reshape(extents)
    fldFinal=np.transpose(fld,axes=[2,1,0])
    del fld
    return fldFinal[np.newaxis,Nh:-Nh,Nh:-Nh,Nh:-Nh]

def field2dTranspose(fld,extents):
    fld=fld.reshape(extents)
    fldFinal=np.transpose(fld,axes=[1,0])
    del fld
    return fldFinal[np.newaxis,Nh:-Nh,Nh:-Nh]

def readBinary(numOutRanks,outpath,theseFiles):
    
    if 'fulldata_dict' in locals():
        del fulldata_dict
    if 'subdata_dict' in locals():
        del subdata_dict
    gc.collect()

    verboseLogging=False #True 
    print(theseFiles)
    fulldata_dict = {}

    for thatFile in theseFiles:
       subdata_dict = {}
       print(thatFile)
       try:
          thisFile='{:s}{:s}'.format(outpath,thatFile)
          flength = os.stat(thisFile).st_size
          with open(thisFile, mode='rb') as f:
               while(f.tell() < flength): #while the filepointer is not at the end of the binary file
                 ## Read and parse the binary representation of 
                 ## the current variable "name" (a string of integer length)
                 nameLen=struct.unpack("i", f.read(4))
                 if verboseLogging:
                     print(len(nameLen),nameLen[0])
                 fieldName=f.read(nameLen[0]).rstrip(b'\x00').decode()
                 if verboseLogging:
                     print(fieldName)
                 ## Read and parse the binary representation of 
                 ## the current variable "type" (a string of integer length)
                 typeLen=struct.unpack("i", f.read(4))
                 if verboseLogging:
                     print(len(typeLen),typeLen[0])
                 fieldType=f.read(typeLen[0]).rstrip(b'\x00').decode()
                 ## Read and parse the binary representation of 
                 ## the current variable "number-of-dimensions" (integer) 
                 nDims=struct.unpack("i", f.read(4))
                 if verboseLogging:
                     print(nDims)
                 ## Read and parse the binary representation of 
                 ## the current variable "dimension extents" (1-d integer array) 
                 extents=np.array([],dtype=np.int32)
                 fmtStr='{:d}i'.format(nDims[0])
                 extents=np.asarray(struct.unpack(fmtStr,f.read(nDims[0]*4)),dtype=np.int32)
                 if verboseLogging:
                     print(extents)
                 ## Read, parse, reshape, transpose the binary field based on the type and extents  
                 if fieldType == 'float':
                     fmtStr='{:d}f'.format(np.prod(extents))
                     if(len(extents)==3):
                         fld3dfloat=np.frombuffer(f.read(np.prod(extents)*4),dtype=np.float32).reshape(extents)
                         fldFinal=field3dTranspose(fld3dfloat,extents)
                         if verboseLogging:
                             print(fldFinal.shape)
                     elif(len(extents)==2):
                         fld2dfloat=np.frombuffer(f.read(np.prod(extents)*4),dtype=np.float32).reshape(extents)
                         fldFinal=field2dTranspose(fld2dfloat,extents)
                         if verboseLogging:
                             print(fldFinal.shape)
                     elif(len(extents)==1):
                         fld1dfloat=np.frombuffer(f.read(np.prod(extents)*4),dtype=np.float32)
                         fldFinal=fld1dfloat
                         if verboseLogging:
                             print(fldFinal.shape)
                 elif fieldType == 'int':
                     fmtStr='{:d}i'.format(np.prod(extents))
                     if(len(extents)==1):
                         fld1dint=np.frombuffer(f.read(np.prod(extents)*4),dtype=np.int32)
                         fldFinal=fld1dint
                         if verboseLogging:
                             print(fldFinal.shape)
                 ### Add the named field to a dictionary as a key-value pair 
                 subdata_dict[fieldName]=fldFinal
       except IOError:
         print('Error While Opening the file: {:s}'.format(thisFile))
       finally:
        if f:  # Check if f was successfully assigned a file object
            f.close()
            # The file is closed here
       ### If this is the first time through allocate all the full data arrays 
       ### necessary for a full data dictionary
       if len(fulldata_dict) == 0:
           rank_cnt=0
           for key in subdata_dict.keys():
               subextents = subdata_dict[key].shape
               fullextents = subextents
               if len(subextents) > 1:  ## Note tuples are immutable so using tuple(list(blah_tuple)) as workaround 
                                        ## to compute xIndex value in fullextents 
                   list_extents = list(subextents)
                   list_extents[-1] = list_extents[-1]*numOutRanks
                   fullextents = tuple(list_extents)
               if verboseLogging:
                   print(f"rank_cnt = {rank_cnt}, {key}: subextents = {subextents}, fullextents={fullextents}")
               subtype = subdata_dict[key].dtype
               fulldata_dict[key] = np.zeros(fullextents,dtype=subtype)
               if verboseLogging:
                    print(f"rank_cnt = {rank_cnt}, {key}:  fulldata_dict[key].shape= {fulldata_dict[key].shape}, fullextents={fullextents}")
               ## NOTE for now this assumes concatenation is always on the xIndex 
               ## (just like the original converter script)!!!
               if len(subextents) == 1:    #[time]
                   fulldata_dict[key] = np.copy(subdata_dict[key])
               elif len(subextents) == 3:  #[time, yIndex, xIndex]
                   fulldata_dict[key][:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]] = np.copy(subdata_dict[key])
               elif len(subextents) == 4:  #[time, zIndex, yIndex, xIndex]
                   if verboseLogging:
                       print(f"rank_cnt = {rank_cnt}, {key}:  slice.shape= {fulldata_dict[key][:,:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]].shape}, fullextents={fullextents}")
                   fulldata_dict[key][:,:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]] = np.copy(subdata_dict[key])
           rank_cnt += 1
       else:  ## This is the second or later file in the per-rank sequence so just fill fullData array segments
           for key in subdata_dict.keys():
               subextents = subdata_dict[key].shape
               fullextents = fulldata_dict[key].shape
               if verboseLogging:
                   print(f"rank_cnt = {rank_cnt}, {key}: subextents = {subextents}, fullextents={fullextents}")
               subtype = subdata_dict[key].dtype
               if len(subextents) == 1: 
                   if verboseLogging:
                      print(f"rank_cnt = {rank_cnt}, skipping {key} since no spatial dimensions involved...")
               elif len(subextents) == 3:
                   fulldata_dict[key][:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]] = np.copy(subdata_dict[key])
               elif len(subextents) == 4:
                   fulldata_dict[key][:,:,:,rank_cnt*subextents[-1]:(rank_cnt+1)*subextents[-1]] = np.copy(subdata_dict[key])
           rank_cnt += 1
              
    #Create the full-domain single xarray dataset
    dsFull=xr.Dataset()
    for key in fulldata_dict.keys():
        if verboseLogging:
             print(f"rank_cnt = {rank_cnt}, creating dataset field {key} with shape {fulldata_dict[key].shape}...")
        if len(fulldata_dict[key].shape) == 4:
            dsFull[key]=xr.DataArray(fulldata_dict[key],dims=['time','zIndex','yIndex','xIndex'])
        if len(fulldata_dict[key].shape) == 3:
            dsFull[key]=xr.DataArray(fulldata_dict[key],dims=['time','yIndex','xIndex'])
        if len(fulldata_dict[key].shape) == 1:
            dsFull[key]=xr.DataArray(fulldata_dict[key],dims=['time'])
   
    ## Clean up memory 
    del subdata_dict
    del fulldata_dict
    del fldFinal 
    gc.collect()

    return dsFull

###

def parse_args():
    """ parse the command line arguments """

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="JSON file with converter parameter settings")
    args = parser.parse_args()
    return args

################## main() ################################################################################
print("Hello performing first MPI calls.")

mpi_size = MPI.COMM_WORLD.Get_size()
mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_name = MPI.Get_processor_name()

########################################
### Parse the command line arguments ###
########################################
args = parse_args()
#########################################################
### Read the json file of converter script parameters ###
#########################################################
with open(args.file) as file:
  params = json.loads(file.read())

outpath = params["outpath"]
FEoutBase = params["FEoutBase"]
numOutRanks = params["numOutRanks"] 
fileSetSize = params["fileSetSize"]
tstart = params["tstart"]
tstep = params["tstep"]
netCDFpath = params["netCDFpath"]
removeBinaries = params["removeBinaries"]

if mpi_size <= fileSetSize:
  fileBatchsize = np.int32(fileSetSize/mpi_size)
else:
  print('mpi_size of {:d} is > fileSetSize = {:d}. Please ensure mpi_size <= fileSetSize.*'.format(mpi_size,fileSetSize))
  exit()
tstop=tstart+tstep*fileSetSize   
print("{:d}/{:d}: Hello World! on {:s}.".format(mpi_rank, mpi_size, mpi_name))
print('Converting binary FE outputs {:s}{:s}_rank_{:d}-{:d}.*'.format(outpath,FEoutBase,0,numOutRanks))
print('In batches of {:d} files per rank beginning from timestep {:d} to timestep {:d} every {:d} timesteps.'.format(fileBatchsize,tstart,tstop,tstep))
print('Writing full netCDF files to {:s}/{:s}.*'.format(netCDFpath,FEoutBase))

Nh=3

if(mpi_rank == 0):
  if not os.path.exists(netCDFpath):
    os.makedirs(netCDFpath)

ts_list=np.arange(tstart,tstop+1,tstep,dtype=np.int32)

#setup mpi task decomposition over the set of output file timesteps
list_len = len(ts_list)
elems_perRank = np.int64(np.floor(list_len/mpi_size))
extra_elems = np.int64(list_len % elems_perRank)
if mpi_rank == 0:
       print("{:d}/{:d}: len(ts_list) = {:d}".format(mpi_rank, mpi_size,list_len))
       print("{:d}/{:d}: elems_perRank = {:d}".format(mpi_rank, mpi_size,elems_perRank))
       print("{:d}/{:d}: extra_elems = {:d}".format(mpi_rank, mpi_size,extra_elems))
for iRank in range(mpi_size):
    if mpi_rank == iRank:
       mystart = (iRank)*elems_perRank
       myend = (iRank+1)*elems_perRank
       if iRank is (mpi_size-1):
          myend = myend+(list_len-mpi_size*elems_perRank) ###Catch straggler files with the last rank
       mytslist = ts_list[mystart:myend]
       print("{:d}/{:d}: mytslist = ts_list({:d}:{:d})".format(mpi_rank, mpi_size, mystart, myend))
       print("{:d}/{:d}: Converting from {:s}.{:d} to {:s}.{:d}".format(mpi_rank, mpi_size, FEoutBase, mytslist[0], FEoutBase, mytslist[-1]))
       print("{:d}/{:d}: len(myfileslist) = {:d}".format(mpi_rank, mpi_size,len(mytslist)))
    MPI.COMM_WORLD.Barrier()

#Each rank can now loop over a subset of the timesteps to concatenate and create a single netCDF file per timestep
for timeStep in mytslist:
   theseFiles=[]
   for outRank in range(numOutRanks):
       theseFiles.append('{:s}_rank_{:d}.{:d}'.format(FEoutBase,outRank,timeStep))
   parseProceed=False
   goodCnt=0
   for thatFile in theseFiles:
       #print('Checking {:s} '.format(thatFile))
       thisFile='{:s}{:s}'.format(outpath,thatFile)
       if os.path.exists(thisFile):
           goodCnt+=1
   #print(goodCnt)
   if(goodCnt==numOutRanks):
       parseProceed=True
   else:
       print('{:d} specified binary files are missing. Skipping timestep: {:d}...'.format(numOutRanks-goodCnt,timeStep))
   if parseProceed:
     dsFull=readBinary(numOutRanks,outpath,theseFiles)
     #write the full  domain datatset to netcdf file
     if False:
        dsFull.to_netcdf('{:s}NETCDF/{:s}.{:d}'.format(outpath,FEoutBase,timeStep),format='NETCDF4')
     else:
        dsFull.to_netcdf('{:s}/{:s}.{:d}'.format(netCDFpath,FEoutBase,timeStep),format='NETCDF4')
     del dsFull
     gc.collect()
     if os.path.exists('{:s}/{:s}.{:d}'.format(netCDFpath,FEoutBase,timeStep)):
       if removeBinaries:
         for thatFile in theseFiles:
           thisFile='{:s}{:s}'.format(outpath,thatFile)
           os.remove(thisFile)

MPI.COMM_WORLD.Barrier()
print("{:d}/{:d}: Conversions complete on {:s}.".format(mpi_rank, mpi_size, mpi_name))
print("{:d}/{:d}: Goodbye World! on {:s}.".format(mpi_rank, mpi_size, mpi_name))
MPI.Finalize()
