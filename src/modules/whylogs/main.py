from typing import List
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
from wilds.datasets.wilds_dataset import WILDSSubset
from progressbar import progressbar
from whylogs_logger import whylogs_logger
w_logger = whylogs_logger()
import random
from PIL.Image import Image 
import pandas as pd
import numpy as np
import os
CAMELYON_ROOT_PATH = '{}/data'.format(os.getcwd())
CSV_PATH = '{}/data/whylogs_output/profile_compare/'.format(os.getcwd())
def main():

    
    """ WILDS CAMELYON DATASET """
    # dataset = get_dataset(dataset="camelyon17", download=True)

    """ GET DATASETS """
    # Get the training set (in distribution)

    # train_data = dataset.get_subset("train")


    # Get the validation set (in distribution)

    # val_data = dataset.get_subset("val")

    # Get the test set (out of distribution)

    # test_data = dataset.get_subset("test")


    """ LOG DATASETS TO PROFILES AND SERIALIZE TO BINARIES"""

    # pil_images_train = [ train_data.dataset[idx][0] for idx in progressbar(train_data.indices)]
    # pil_images_val = [ val_data.dataset[idx][0] for idx in progressbar(val_data.indices)]
    # pil_images_test =  [ test_data.dataset[idx][0] for idx in progressbar(test_data.indices)]


    
    # size = int(len(train_data.indices))
    for i in range(80,100,5):
        log_percentage_of_dataset(percentage=i,pil_images=pil_images_train,size=int(len(train_data.indices)),split='train')
        # log_percentage_of_dataset(percentage=i,pil_images=pil_images_val,size=int(len(val_data.indices)),split='val')
        # log_percentage_of_dataset(percentage=i,pil_images=pil_images_test,size=int(len(test_data.indices)),split='test')


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

import matplotlib.pyplot as plt
def read_csv(csv_path :str, csv_name : str, percentage : int) -> pd.DataFrame:
    df =  [pd.read_csv('{}{}.csv'.format(csv_path,csv_name)).drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'{}'.format(percentage)})]
    return df

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


def log_compare(target,ref,data_directory_path,compare_summary_name):
       

    w_logger.create_profile_compare_summary_json(
        target_profile=target,
        ref_profile=ref,
        data_directory_path=data_directory_path,
        compare_summary_name=compare_summary_name)


    print(w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=data_directory_path,compare_summary_name=compare_summary_name))




def log_percentage_of_dataset(percentage: int, pil_images : List[Image], size : int, split: str):
   
    random.shuffle(pil_images)

    count = int(size * (percentage / 100))
    tmp_profile= w_logger.log_pil_images_data_from_list(
        data_directory_path=CAMELYON_ROOT_PATH,
        pil_data_arr=pil_images[:count],
    )
    print(w_logger.serialize_profile(profile=tmp_profile,binary_name='{}_camelyon_profile_{}'.format(split,percentage),data_directory_path=CAMELYON_ROOT_PATH))


# ======================================================================================
# call
if __name__ == "__main__":
    main()