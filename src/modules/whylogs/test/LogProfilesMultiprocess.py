import os
import random
from multiprocessing import Process, Queue
from timeit import default_timer as timer
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wilds import get_dataset
from whylogs.core.view.dataset_profile_view import DatasetProfileView

import torchvision.transforms as T
import torch
import torchvision
import PIL

# import sys
# sys.path.append('/home/ubuntu/image-drift-monitoring/src')
from whylogs_logger import Whylogs_Logger
def main():

    CAMELYON_ROOT_PATH = '{}/data/camelyon17_v1.0'.format(os.getcwd())
    GLOBALWHEAT_ROOT_PATH = '{}/data/global_wheat_v1.1'.format(os.getcwd())
    IWILDCAM_ROOT_PATH = '{}/data/iwildcam_v2.0'.format(os.getcwd())
    POVERTY_ROOT_PATH =  '{}/data/poverty_v1.1'.format(os.getcwd())
    RXRX1_ROOT_PATH = '/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0'
    CSV_PATH = '{}/data/whylogs_output/profile_compare/'.format(os.getcwd())

    
    w_logger = Whylogs_Logger() # init whylogs logger 
    
    t = timer() # setup timer
    
    """ WILDS CAMELYON DATASET """
    # dataset = get_dataset(dataset="camelyon17", download=False)
    """ WILDS GLOBALWHEAT DATASET """
    # dataset = get_dataset(dataset="globalwheat", download=False)
    """ WILDS IWILDCAM DATASET """
    # dataset = get_dataset(dataset="iwildcam", download=False)
    """ WILDS RXRX1 DATASET """
    dataset = get_dataset(dataset="rxrx1",download=False)
    


    """ GET SPLITS """
    # Get the training set (in distribution)

    train_data = dataset.get_subset("train")

    # Get the validation set (in distribution)

    # val_data = dataset.get_subset("val")

    # Get the test set (out of distribution)

    # test_data = dataset.get_subset("test")


    

    """ LOG TRAIN SPLIT TO BINARY USING MULTIPROCESSING"""
    for i in range(5,105,5):    
        log_profile_to_bin_multiple_processes(
            w_logger=w_logger,
            percentage=i,
            indices=train_data.indices,
            dataset=train_data.dataset,
            num_processes=30,
            split='train',
            dataset_name='rxrx1',
            dataset_dir_path=RXRX1_ROOT_PATH)

    """ LOG VAL SPLIT TO BINARY USING MULTIPROCESSING"""
    # for j in range(5,105,5):    
    #     log_profile_to_bin_multiple_processes(
    #         w_logger=w_logger,
    #         percentage=j,
    #         indices=val_data.indices,
    #         dataset=val_data.dataset,
    #         num_processes=30,
    #         split='val',
    #         dataset_name='rxrx1',
    #         dataset_dir_path=RXRX1_ROOT_PATH)
    
    """ LOG TEST SPLIT TO BINARY USING MULTIPROCESSING"""
    # for k in range(5,105,5):    
    #     log_profile_to_bin_multiple_processes(
    #         w_logger=w_logger,
    #         percentage=k,
    #         indices=test_data.indices,
    #         dataset=test_data.dataset,
    #         num_processes=30,
    #         split='test',
    #         dataset_name='rxrx1',
    #         dataset_dir_path=RXRX1_ROOT_PATH)

    dt = timer() - t
    print(f'Time (s) {dt:.3f}')



# logging split (x) from dataset (y) with percentage (z) to binary x_y_profile_z.bin
def log_profile_to_bin_multiple_processes(w_logger : Whylogs_Logger, percentage: int, indices : List[int], dataset: any, num_processes: int, split: str, dataset_name: str, dataset_dir_path : str):
    
    count = int(len(indices) * (percentage / 100)) # calculate logging amout from percentage

    indices_arr = indices # get the indices to array

    random.shuffle(indices_arr) # shuffle array in place 

    indices_arr = indices_arr[:count]
    split_array = np.array_split(indices_arr,num_processes) # split array into x arrays for x processes
    
    
    queue = Queue() # create queue to gather process output
    profiles = [] # create array to store profiles
    processes = [] # array to check running processes

    # define processes with task and args
    for i in range(num_processes):
        tmp_process = Process(target=logging_task,args=(split_array[i],queue,dataset,w_logger))
        processes.append(tmp_process)

    # start processes
    for p in processes:
        p.start()

    # prevent deadlock for queue
    while 1:
        running = any(p.is_alive() for p in processes)
        while not queue.empty():
            profiles.append(queue.get())
        if not running:
            break
    
    # wait for all processes to finish
    for p in processes:
        p.join()


    # debug print 
    for p in profiles:
        print(p)
    
    
    merged_profiles = w_logger.merge_profiles(profiles=profiles) # merge all profiles from processes
    print('merged Profiles')

    print(w_logger.serialize_profile(profile=merged_profiles,binary_name='{}_{}_profile_{}'.format(split,dataset_name,percentage),data_directory_path=dataset_dir_path)) # serialize the merged profiles to output bin

   

# needed for logging (multiprocess)
def logging_task(indices : any,queue : Queue, dataset: any,w_logger : Whylogs_Logger):
    first_profile = w_logger.log_pil_image_to_profile(image=dataset[indices[0]][0],profile=None)

    for idx in indices[1:]:
        first_profile = w_logger.log_pil_image_to_profile(image=dataset[idx][0],profile=first_profile)

    queue.put(first_profile)
    print('process finished task')

if __name__ == "__main__":
    main()