import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
pd.set_option("display.precision", 20)
pd.set_option('max_colwidth', 800)

# PROFILE_COMP_CSV_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_comps/' 
LANDSCAPE_COMP_CSV_PATH = '/home/jingle/evoila/ml-image-drift-monitoring/landscape_v_camera/whylogs_output/profile_compare/'

def read_csv(baseline_batch_size: int):
    df_15 =  [pd.read_csv('{}camera_15_v_baseline_{}.csv'.format(LANDSCAPE_COMP_CSV_PATH,baseline_batch_size)).drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'15'})]
    df_25 =  [pd.read_csv('{}camera_25_v_baseline_{}.csv'.format(LANDSCAPE_COMP_CSV_PATH,baseline_batch_size)).drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'25'})]
    df_50 =  [pd.read_csv('{}camera_50_v_baseline_{}.csv'.format(LANDSCAPE_COMP_CSV_PATH,baseline_batch_size)).drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'50'})]
    df_100 = [pd.read_csv('{}camera_100_v_baseline_{}.csv'.format(LANDSCAPE_COMP_CSV_PATH,baseline_batch_size)).drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'100'})]
    df_150 = [pd.read_csv('{}camera_150_v_baseline_{}.csv'.format(LANDSCAPE_COMP_CSV_PATH,baseline_batch_size)).drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'150'})]
    df_200 = [pd.read_csv('{}camera_200_v_baseline_{}.csv'.format(LANDSCAPE_COMP_CSV_PATH,baseline_batch_size)).drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'200'})]
    dfs = df_15 + df_25 + df_50 + df_100 + df_150 + df_200
    return dfs


def plot_difference(baseline_batch_size : int):
    dfs = read_csv(baseline_batch_size=baseline_batch_size)
    df_joined = dfs[0].join(
                dfs[1].set_index('metric'), on='metric').join(
                    dfs[2].set_index('metric'), on='metric').join(
                        dfs[3].set_index('metric'),on='metric').join(
                            dfs[4].set_index('metric'),on='metric').join(
                                dfs[5].set_index('metric'),on='metric')
    data= {
    'metric' :           ['15', '25', '50','100','150','200'],
    'Brightness.mean':   df_joined[df_joined['metric'] == 'image.Brightness.mean' ].to_numpy()[0][1:],
    'Brightness.stddev' : df_joined[df_joined['metric'] == 'image.Brightness.stddev' ].to_numpy()[0][1:],
    'Hue.mean' :         df_joined[df_joined['metric'] == 'image.Hue.mean' ].to_numpy()[0][1:],
    'Hue.stddev' :       df_joined[df_joined['metric'] == 'image.Hue.stddev' ].to_numpy()[0][1:],
    'ImagePixelHeight':  df_joined[df_joined['metric'] == 'image.ImagePixelHeight' ].to_numpy()[0][1:],
    'ImagePixelWidth':   df_joined[df_joined['metric'] == 'image.ImagePixelWidth' ].to_numpy()[0][1:],
    'Saturation.mean':   df_joined[df_joined['metric'] == 'image.Saturation.mean' ].to_numpy()[0][1:],
    'Saturation.stddev': df_joined[df_joined['metric'] == 'image.Saturation.stddev' ].to_numpy()[0][1:],
    'Drift Threshold' :  [0.05,0.05,0.05,0.05,0.05,0.05],
    }
    converted_df = pd.DataFrame(data)
    my_yticks = np.arange(0,1,0.05)

    ax =converted_df.plot.line(
        y=['Brightness.mean', 'Brightness.stddev', 'Hue.mean','Hue.stddev','ImagePixelHeight','ImagePixelWidth','Saturation.mean','Saturation.stddev'],
        x='metric',
        figsize=(12,6), 
        title='camera vs baseline {}'.format(baseline_batch_size),
        xlabel='Batch size',
        yticks=my_yticks,
        )
    converted_df.plot(kind='line',y='Drift Threshold',color='black',ax=ax)


    plt.savefig('camera_v_baseline_{}.png'.format(baseline_batch_size))
