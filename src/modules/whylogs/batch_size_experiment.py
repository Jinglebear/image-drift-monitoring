import pandas as pd
import os
pd.set_option("display.precision", 20)
pd.set_option('max_colwidth', 800)

df_15 = pd.read_csv('/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_comps/camera_15_v_baseline_15.csv').drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'15'})
df_25 = pd.read_csv('/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_comps/camera_25_v_baseline_15.csv').drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'25'})
df_50 = pd.read_csv('/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_comps/camera_50_v_baseline_15.csv').drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'50'})
df_100 = pd.read_csv('/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_comps/camera_100_v_baseline_15.csv').drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'100'})
df_150 = pd.read_csv('/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_comps/camera_150_v_baseline_15.csv').drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'150'})
df_200 = pd.read_csv('/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_comps/camera_200_v_baseline_15.csv').drop("Unnamed: 0",axis=1).drop('rating',axis=1).rename(columns={'p_val':'200'})


df_joined = df_15.join(
                df_25.set_index('metric'), on='metric').join(
                    df_50.set_index('metric'), on='metric').join(
                        df_100.set_index('metric'),on='metric').join(
                            df_150.set_index('metric'),on='metric').join(
                                df_200.set_index('metric'),on='metric')

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


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
my_yticks = np.arange(0,1,0.05)

ax =converted_df.plot.line(
    y=['Brightness.mean', 'Brightness.stddev', 'Hue.mean','Hue.stddev','ImagePixelHeight','ImagePixelWidth','Saturation.mean','Saturation.stddev'],
    x='metric',
    figsize=(12,6), 
    title='camera vs baseline 15',
    xlabel='Batch size',
    yticks=my_yticks,
     )
converted_df.plot(kind='line',y='Drift Threshold',color='black',ax=ax)


plt.savefig('camera_v_baseline_15.png')