
def main():
    from whylogs_logger import whylogs_logger
    w_logger = whylogs_logger()

    """ ANTS VS BEES LOG """
    TRAIN_ANTS_PATH = '/home/jinglewsl/evoila/data/image_data/ants_vs_bees/train/ants/'
    TRAIN_BEES_PATH = '/home/jinglewsl/evoila/data/image_data/ants_vs_bees/train/bees/'
    ANTS_VS_BEES_PATH = '/home/jinglewsl/evoila/data/image_data/ants_vs_bees/train/'

    # train_ants_profile = w_logger.log_data(data_directory_path=ANTS_VS_BEES_PATH,sub_dir_path=TRAIN_ANTS_PATH)
    # train_bees_profile = w_logger.log_data(data_directory_path=ANTS_VS_BEES_PATH,sub_dir_path=TRAIN_BEES_PATH)

    # w_logger.serialize_profile(profile=train_ants_profile,binary_name='train_ants',data_directory_path=ANTS_VS_BEES_PATH)
    # w_logger.serialize_profile(profile=train_bees_profile,binary_name='train_bees',data_directory_path=ANTS_VS_BEES_PATH)
   
    train_ants_profile=w_logger.deserialize_profile(data_directory_path=ANTS_VS_BEES_PATH,binary_name='train_ants')
    train_bees_profile=w_logger.deserialize_profile(data_directory_path=ANTS_VS_BEES_PATH,binary_name='train_bees')

    w_logger.create_profile_compare_summary_json(target_profile=train_ants_profile,ref_profile=train_bees_profile,data_directory_path=ANTS_VS_BEES_PATH,compare_summary_name='ants_vs_bees_comp')

    df =w_logger.create_drift_metric_df_from_comp_summary_json(data_directory_path=ANTS_VS_BEES_PATH,compare_summary_name='ants_vs_bees_comp')
    print(df , '\n\n')

    
    w_logger.create_visualization( data_directory_path=ANTS_VS_BEES_PATH,viz_name='ants_vs_bees_comp',target_profile=train_ants_profile,referece_profile=train_bees_profile)
    
    
    

# ======================================================================================
# call
if __name__ == "__main__":
    main()