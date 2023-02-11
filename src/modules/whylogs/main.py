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
    """ WILDS POVERTY DATASET """
    # dataset = get_dataset(dataset="poverty",download=False)
    """ WILDS RXRX1 DATASET """
    dataset = get_dataset(dataset="rxrx1",download=False)
    


    """ GET SPLITS """
    # Get the training set (in distribution)

    # train_data = dataset.get_subset("train")

    # Get the validation set (in distribution)

    # val_data = dataset.get_subset("val")

    # Get the test set (out of distribution)

    test_data = dataset.get_subset("test")

    """ MY TESTING """
   
    
    # tensor =train_data.dataset[train_data.indices[0]][0]
    
    # print(tensor)

    
    

    """ LOG TRAIN SPLIT TO BINARY USING MULTIPROCESSING"""
    # for i in range(5,105,5):    
    #     log_profile_to_bin_multiple_processes(
    #         w_logger=w_logger,
    #         percentage=i,
    #         indices=train_data.indices,
    #         dataset=train_data.dataset,
    #         num_processes=30,
    #         split='train',
    #         dataset_name='rxrx1',
    #         dataset_dir_path=RXRX1_ROOT_PATH)

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
    for k in range(5,105,5):    
        log_profile_to_bin_multiple_processes(
            w_logger=w_logger,
            percentage=k,
            indices=test_data.indices,
            dataset=test_data.dataset,
            num_processes=30,
            split='test',
            dataset_name='rxrx1',
            dataset_dir_path=RXRX1_ROOT_PATH)

    dt = timer() - t
    print(f'Time (s) {dt:.3f}')

    # profile_1 = w_logger.deserialize_profile('/home/jinglewsl/evoila/projects/image-drift-monitoring/data','train_camelyon_profile_100')
    # # w_logger.create_profile_summary_json(profile=profile_1,data_directory_path='/home/jinglewsl/evoila/projects/image-drift-monitoring/data/',summary_name='train_camelyon_profile_100')
    # profile_2 = w_logger.deserialize_profile('/home/jinglewsl/evoila/projects/image-drift-monitoring/data/','test_camelyon_profile_100')
    # w_logger.create_profile_compare_summary_json(target_profile=profile_2,
    #                                 ref_profile=profile_1,
    #                                 data_directory_path='/home/jinglewsl/evoila/projects/image-drift-monitoring/data/',
    #                                 compare_summary_name='test_camelyon_100_vs_train_camelyon_100')


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


# load dataframe from comparison csv
def read_csv(csv_path :str, csv_name : str, percentage : int) -> pd.DataFrame:
    df =  [pd.read_csv('{}{}.csv'.format(csv_path,csv_name)).drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'{}'.format(percentage)})]
    return df



# needed ?
def plot_difference(dfs : List[pd.DataFrame],output_path:str, target: str,target_p : int, ref:str, ref_p :int):
    df_joined = dfs[0]
    for i in range(1,20,1):
        df_joined = df_joined.join(dfs[i].set_index('metric'),on='metric')
    data= {
    'metric' :           [ str(out) for out in range(5,105,5)],
    'Brightness.mean':   df_joined[df_joined['metric'] == 'image.Brightness.mean' ].to_numpy()[0][1:],
    'Brightness.stddev' :df_joined[df_joined['metric'] == 'image.Brightness.stddev' ].to_numpy()[0][1:],
    'Hue.mean' :         df_joined[df_joined['metric'] == 'image.Hue.mean' ].to_numpy()[0][1:],
    'Hue.stddev' :       df_joined[df_joined['metric'] == 'image.Hue.stddev' ].to_numpy()[0][1:],
    'ImagePixelHeight':  df_joined[df_joined['metric'] == 'image.ImagePixelHeight' ].to_numpy()[0][1:],
    'ImagePixelWidth':   df_joined[df_joined['metric'] == 'image.ImagePixelWidth' ].to_numpy()[0][1:],
    'Saturation.mean':   df_joined[df_joined['metric'] == 'image.Saturation.mean' ].to_numpy()[0][1:],
    'Saturation.stddev': df_joined[df_joined['metric'] == 'image.Saturation.stddev' ].to_numpy()[0][1:],
    'Drift Threshold' :  np.full(20,0.05),
    }
    
    
    converted_df = pd.DataFrame(data)
    my_yticks = np.arange(0,1,0.05)
    print(converted_df)
    ax =converted_df.plot(
        kind='line',
        y=['Brightness.mean', 'Brightness.stddev', 'Hue.mean','Hue.stddev','ImagePixelHeight','ImagePixelWidth','Saturation.mean','Saturation.stddev'],
        x='metric',
        figsize=(12,6), 
        title='{}{} vs {} {}'.format(target,target_p,ref,ref_p),
        xlabel='Batch size percentage',
        yticks=my_yticks
        )
    converted_df.plot(kind='line',y='Drift Threshold',color='black',ax=ax)


    plt.savefig( '{}{}_{}_v_{}_{}.png'.format(output_path,target,target_p,ref,ref_p))

# load binaries to profiles & create drift metric to compare
def create_profile_comparison(w_logger : Whylogs_Logger, target_binary_name: str,ref_binary_name : str,data_directory_path :str,compare_summary_name: str):
    
    target = w_logger.deserialize_profile(data_directory_path=data_directory_path,binary_name=target_binary_name)
    ref = w_logger.deserialize_profile(data_directory_path=data_directory_path,binary_name=ref_binary_name)

    w_logger.create_profile_compare_summary_json(
        target_profile=target,
        ref_profile=ref,
        data_directory_path=data_directory_path,
        compare_summary_name=compare_summary_name)


    print(w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=data_directory_path,compare_summary_name=compare_summary_name))


def misc():
    """ COMPARE DISTRIBUTIONS """
    # import pandas as pd
    # run train vs train comp
    # for i in range(5,60,5):
    #     if i == 25:
    #         continue
    #     train_profile_1 =  w_logger.deserialize_profile(data_directory_path=CAMELYON_ROOT_PATH,binary_name='{}_camelyon_profile_{}'.format('train',i))
    #     train_profile_2 =  w_logger.deserialize_profile(data_directory_path=CAMELYON_ROOT_PATH,binary_name='{}_camelyon_profile_{}'.format('train',100))
    #     log_compare(target=train_profile_1,ref=train_profile_2,data_directory_path=CAMELYON_ROOT_PATH,compare_summary_name='train_{}_vs_train_{}'.format(i,100))

    # dfs =[]
    # for j in range(5,60,5):
    #     if j == 25:
    #         continue
    #     dfs += read_csv(csv_path=CSV_PATH,csv_name='train_{}_vs_train_100'.format(j),percentage=j)
        
    

    # df_joined = dfs[0]
    # for i in range(1,10,1):
    #     df_joined = df_joined.join(dfs[i].set_index('metric'),on='metric')

    # df_joined.to_csv('tmp.csv')

    
    
    # converted_df = pd.DataFrame(data)

    # print(converted_df)

    # plot_difference(dfs=dfs,output_path=CSV_PATH,target='test',target_p=5,ref='train',ref_p=100)   
    

    # run train vs test comp --> .csv
    # for i in range(5,55,5):
    #     for j in range(5,105,5):
    #         test_profile =  w_logger.deserialize_profile(data_directory_path=CAMELYON_ROOT_PATH,binary_name='{}_camelyon_profile_{}'.format('test',j))
    #         train_profile = w_logger.deserialize_profile(data_directory_path=CAMELYON_ROOT_PATH,binary_name='{}_camelyon_profile_{}'.format('train',i))
    #         log_compare(target=test_profile,ref=train_profile,data_directory_path=CAMELYON_ROOT_PATH,compare_summary_name='train_{}_vs_test_{}'.format(i,j))
    
    # run train vs val comp --> .csv
    # for i in range(5,105,5):
    #     for j in range(5,105,5):
    #         val_profile =  w_logger.deserialize_profile(data_directory_path=CAMELYON_ROOT_PATH,binary_name='{}_camelyon_profile_{}'.format('val',j))
    #         train_profile = w_logger.deserialize_profile(data_directory_path=CAMELYON_ROOT_PATH,binary_name='{}_camelyon_profile_{}'.format('train',i))
    #         log_compare(target=val_profile,ref=train_profile,data_directory_path=CAMELYON_ROOT_PATH,compare_summary_name='train_{}_vs_val_{}'.format(i,j))
 
    """ PLOT DIFFERENCE test x vs train 95"""
    # CSV_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/ml-image-drift-monitoring/src/modules/whylogs/data/whylogs_output/profile_compare/'
    # OUT = '/home/jinglewsl/evoila/sandbox/whylogs_v1/ml-image-drift-monitoring/src/modules/whylogs/data/whylogs_output/profile_compare/'
    # dfs = []
    # for i in range(5,100,5):
    #     dfs += read_csv(csv_path=CSV_PATH,csv_name='train_95_vs_test_{}'.format(i))
    # for i in range(5,100,5):
    #     plot_difference(percentage=i,dfs=dfs,output_path=OUT,target='test',ref='train',ref_p=95)
    """ CREATE VIZ FROM COMPARE """
    
    # w_logger.create_visualization(data_directory_path=CAMELYON_ROOT_PATH,viz_name='25_train_camelyon_v_train_camelyon',target_profile=train_camelyon_profile_25,referece_profile=train_camelyon_profile)
    # w_logger.create_visualization(data_directory_path=CAMELYON_ROOT_PATH,viz_name='first_25_train_camelyon_v_next_25_train_camelyon',target_profile=train_camelyon_profile_next_25,referece_profile=train_camelyon_profile_25)

# ======================================================================================
# call
if __name__ == "__main__":
    main()