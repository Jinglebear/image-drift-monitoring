
def main():
    from whylogs_logger import whylogs_logger
    w_logger = whylogs_logger()

    """ TRAIN ANTS VS TRAIN BEES LOG """
    TRAIN_ANTS_PATH = '/home/jinglewsl/evoila/data/image_data/ants_vs_bees/train/ants/'
    TRAIN_BEES_PATH = '/home/jinglewsl/evoila/data/image_data/ants_vs_bees/train/bees/'
    ANTS_VS_BEES_PATH = '/home/jinglewsl/evoila/data/image_data/ants_vs_bees/train/'

    # train_ants_profile = w_logger.log_data(data_directory_path=ANTS_VS_BEES_PATH,sub_dir_path=TRAIN_ANTS_PATH)
    # train_bees_profile = w_logger.log_data(data_directory_path=ANTS_VS_BEES_PATH,sub_dir_path=TRAIN_BEES_PATH)

    # w_logger.serialize_profile(profile=train_ants_profile,binary_name='train_ants',data_directory_path=ANTS_VS_BEES_PATH)
    # w_logger.serialize_profile(profile=train_bees_profile,binary_name='train_bees',data_directory_path=ANTS_VS_BEES_PATH)
   
    # train_ants_profile=w_logger.deserialize_profile(data_directory_path=ANTS_VS_BEES_PATH,binary_name='train_ants')
    # train_bees_profile=w_logger.deserialize_profile(data_directory_path=ANTS_VS_BEES_PATH,binary_name='train_bees')

    # w_logger.create_profile_compare_summary_json(target_profile=train_ants_profile,ref_profile=train_bees_profile,data_directory_path=ANTS_VS_BEES_PATH,compare_summary_name='ants_vs_bees_comp')

    # df =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=ANTS_VS_BEES_PATH,compare_summary_name='ants_vs_bees_comp')
    # print(df , '\n\n')

    
    # w_logger.create_visualization( data_directory_path=ANTS_VS_BEES_PATH,viz_name='ants_vs_bees_comp',target_profile=train_ants_profile,referece_profile=train_bees_profile)
    
    """ LANDSCAPE BASELINE vs CAMERA """
    LANDSCAPE_DATA_PATH = '/home/jingle/evoila/ml-image-drift-monitoring/landscape_v_camera/'
    LANDSCAPE_DATA_BASELINE = '/home/jingle/evoila/ml-image-drift-monitoring/landscape_v_camera/landscape_baseline/baseline/'
    # baseline_1800  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=1800,shuffle=True)
    # baseline_1200  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=1200,shuffle=True)
    # baseline_1000  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=1000,shuffle=True)
    # baseline_900   = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=900,shuffle=True)
    # baseline_450  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=450,shuffle=True)
    # baseline_200  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=200,shuffle=True)
    # baseline_150  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=150,shuffle=True)
    # baseline_100  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=100,shuffle=True)
    # baseline_50  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=50,shuffle=True)
    # baseline_25  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_BASELINE,batch_size=25,shuffle=True)

    LANDSCAPE_DATA_CAMERA = '/home/jingle/evoila/ml-image-drift-monitoring/landscape_v_camera/camera_images/camera/'
    # camera_200 = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_CAMERA,batch_size=200,shuffle=True)
    # camera_150 = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_CAMERA,batch_size=150,shuffle=True)
    # camera_100 = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_CAMERA,batch_size=100,shuffle=True)
    # camera_50  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_CAMERA,batch_size=50,shuffle=True)
    # camera_25  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_CAMERA,batch_size=25,shuffle=True)
    # camera_15  = w_logger.log_data(data_directory_path=LANDSCAPE_DATA_PATH,sub_dir_path=LANDSCAPE_DATA_CAMERA,batch_size=15,shuffle=True)


    """ SERIALIZE """

    # w_logger.serialize_profile(baseline_1800,'baseline_1800',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(baseline_1200,'baseline_1200',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(baseline_1000,'baseline_1000',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(baseline_900,'baseline_900',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(baseline_450,'baseline_450',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(baseline_200,'baseline_200',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(baseline_150,'baseline_150',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(baseline_100,'baseline_100',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(baseline_50,'baseline_50',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(baseline_25,'baseline_25',LANDSCAPE_DATA_PATH)

    # w_logger.serialize_profile(camera_200,'camera_200',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(camera_150,'camera_150',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(camera_100,'camera_100',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(camera_50,'camera_50',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(camera_25,'camera_25',LANDSCAPE_DATA_PATH)
    # w_logger.serialize_profile(camera_15,'camera_15',LANDSCAPE_DATA_PATH)


    """ DESERIALIZE """
    # baseline_1800 = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_1800')
    # baseline_1200 = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_1200')
    # baseline_1000 = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_1000')
    # baseline_900  = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_900') 
    # baseline_450  = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_450')
    # baseline_200  = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_200')
    # baseline_150  = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_150')
    # baseline_100  = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_100')
    # baseline_50   = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_50')
    # baseline_25   = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='baseline_25')

    # camera_200 = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='camera_200')
    # camera_150 = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='camera_150')
    # camera_100 = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='camera_100')
    # camera_50  = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='camera_50')
    # camera_25  = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='camera_25')
    # camera_15  = w_logger.deserialize_profile(data_directory_path=LANDSCAPE_DATA_PATH,binary_name='camera_15')

    """ PROFILE COMPARE SUMMARY """
    # camera 200 vs baseline x
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_1800,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_1800')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_1200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_1200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_1000,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_1000')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_900,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_900')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_450,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_450')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_150,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_150')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_100,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_100')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_50,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_50')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_200,ref_profile=baseline_25,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_25')
    # # camera 150 vs baseline x
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_1800,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_1800')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_1200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_1200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_1000,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_1000')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_900,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_900')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_450,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_450')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_150,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_150')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_100,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_100')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_50,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_50')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_150,ref_profile=baseline_25,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_25')
    # # camera 100 vs baseline x
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_1800,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_1800')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_1200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_1200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_1000,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_1000')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_900,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_900')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_450,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_450')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_150,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_150')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_100,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_100')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_50,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_50')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_100,ref_profile=baseline_25,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_25')
    # # camera 50 vs baseline x
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_1800,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_1800')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_1200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_1200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_1000,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_1000')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_900,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_900')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_450,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_450')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_150,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_150')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_100,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_100')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_50,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_50')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_50,ref_profile=baseline_25,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_25')
    # # camera 25 vs baseline x
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_1800,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_1800')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_1200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_1200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_1000,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_1000')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_900,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_900')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_450,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_450')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_150,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_150')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_100,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_100')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_50,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_50')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_25,ref_profile=baseline_25,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_25')
    # # camera 15 vs baseline x
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_1800,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_1800')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_1200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_1200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_1000,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_1000')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_900,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_900')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_450,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_450')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_200,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_200')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_150,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_150')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_100,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_100')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_50,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_50')
    # w_logger.create_profile_compare_summary_json(target_profile=camera_15,ref_profile=baseline_25,data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_25')
    """ DRIFT METRIC CSV & DF """
    # camera 200 vs baseline x drift metric csv & df 
    df1 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_1800')
    df2 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_1200')
    df3 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_1000')
    df4 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_900')
    df5 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_450')
    df6 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_200')
    df7 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_150')
    df8 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_100')
    df9 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_50')
    df10 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_200_v_baseline_25')
    # camera 150 vs baseline x drift metric csv & df 
    df11 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_1800')
    df12 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_1200')
    df13 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_1000')
    df14 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_900')
    df15 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_450')
    df16 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_200')
    df17 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_150')
    df18 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_100')
    df19 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_50')
    df20 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_150_v_baseline_25')
    # camera 100 vs baseline x drift metric csv & df 
    df21 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_1800')
    df22 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_1200')
    df23 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_1000')
    df24 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_900')
    df25 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_450')
    df26 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_200')
    df27 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_150')
    df28 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_100')
    df29 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_50')
    df30 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_100_v_baseline_25')
    # camera 50 vs baseline x drift metric csv & df 
    df31 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_1800')
    df32 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_1200')
    df33 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_1000')
    df34 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_900')
    df35 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_450')
    df36 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_200')
    df37 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_150')
    df38 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_100')
    df39 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_50')
    df40 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_50_v_baseline_25')
    # camera 25 vs baseline x drift metric csv & df 
    df41 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_1800')
    df42 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_1200')
    df43 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_1000')
    df44 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_900')
    df45 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_450')
    df46 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_200')
    df47 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_150')
    df48 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_100')
    df49 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_50')
    df50 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_25_v_baseline_25')
    # camera 15 vs baseline x drift metric csv & df 
    df51 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_1800')
    df52 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_1200')
    df53 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_1000')
    df54 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_900')
    df55 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_450')
    df56 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_200')
    df57 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_150')
    df58 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_100')
    df59 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_50')
    df60 =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=LANDSCAPE_DATA_PATH,compare_summary_name='camera_15_v_baseline_25')

    dfs10 = []
    dfs10.append(df1)
    dfs10.append(df2)
    dfs10.append(df3)
    dfs10.append(df4)
    dfs10.append(df5)
    dfs10.append(df6)
    dfs10.append(df7)
    dfs10.append(df8)
    dfs10.append(df9)
    dfs10.append(df10)
    
    
    

    # dfs20 = df11 + df12 + df13 + df14 + df15 + df16 + df17 + df18 + df19 + df20 
    # dfs30 = df21 + df22 + df23 + df24 + df25 + df26 + df27 + df28 + df29 + df30
    # dfs40 = df31 + df32 + df33 + df34 + df35 + df36 + df37 + df38 + df39 + df40
    # dfs50 = df41 + df42 + df43 + df44 + df45 + df46 + df47 + df48 + df49 + df50
    # dfs60 = df51 + df52 + df53 + df54 + df55 + df56 + df57 + df58 + df59 + df60

    # dfs_all = dfs10 + dfs20 + dfs30 + dfs40 + dfs50 +dfs60

    print(type(dfs10))
    i =0
    for df in dfs10:
        i+=i
        print('\nRun Num: {} \n'.format(i))
        print(df)



    """ PLOT DIFFERENCE """

    import plot_difference as plot_diff

    # plot_diff.plot_difference(baseline_batch_size=1800)
    # plot_diff.plot_difference(baseline_batch_size=1200)
    # plot_diff.plot_difference(baseline_batch_size=1000)
    # plot_diff.plot_difference(baseline_batch_size=900)
    # plot_diff.plot_difference(baseline_batch_size=450)
    # plot_diff.plot_difference(baseline_batch_size=200)
    # plot_diff.plot_difference(baseline_batch_size=100)
    # plot_diff.plot_difference(baseline_batch_size=50)
    # plot_diff.plot_difference(baseline_batch_size=25)


    

    


# ======================================================================================
# call
if __name__ == "__main__":
    main()