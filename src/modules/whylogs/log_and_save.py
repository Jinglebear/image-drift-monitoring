import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import pytz
from PIL import Image
from progressbar import progressbar
from whylogs.core.view.dataset_profile_view import DatasetProfileView
from whylogs.extras.image_metric import log_image
from whylogs.migration import uncompound
from whylogs.viz import NotebookProfileVisualizer
from whylogs.viz.utils.profile_viz_calculations import (
    OverallStats, add_feature_statistics, add_overall_statistics,
    frequent_items_from_view, generate_profile_summary, generate_summaries,
    histogram_from_view)
import random
""" log data from directory to a profile, set logging date """
""" ****************************************************** """


def log_data(data_directory: str, datetime: datetime, logger: logging.Logger, batch_size: int, shuffle: bool) -> DatasetProfileView:
    try:
        logger.info(
            'Started logging image data from: {}'.format(data_directory))
        profile = None
        data_directory_list = os.listdir(data_directory)
        if shuffle:
            random.shuffle(data_directory_list)  # shuffle images
        data_directory_list = data_directory_list[0:batch_size]
        for img_name in progressbar(data_directory_list):
            img = Image.open(data_directory + img_name)  # read in image
            _ = log_image(img).profile()  # generate profile
            # optionally set dataset_timestamp
            _.set_dataset_timestamp(datetime)
            _view = _.view()  # extract mergeable profile view
            # merge each profile while looping
            if profile is None:
                profile = _view
            else:
                profile = profile.merge(_view)
        logger.info('Logged image data from: {} with batch size {}'.format(
            data_directory, profile.to_pandas()['image/Brightness.mean:counts/n']['image']))
        return profile
    except Exception as e:
        logger.exception('Exception in log_data(): {}'.format(e))


""" serialize a profile to a .bin """
""" ****************************************************** """


def serialize_profile(profile: DatasetProfileView, binary_name: str, logger: logging.Logger) -> str:
    try:
        return profile.write('{}.bin'.format(binary_name))
    except Exception as e:
        logger.exception('Exception in serialize_profile(): {}'.format(e))


""" deserialize a profile from a .bin """
""" ****************************************************** """


def deserialize_profile(path: str, logger: logging.Logger) -> DatasetProfileView:
    try:
        return DatasetProfileView.read(path)
    except Exception as e:
        logger.exception('Exception in deserialize_profile(): {}'.format(e))


""" create profile viz """
""" ****************** """


def create_metrics(target_profile: DatasetProfileView, referece_profile: DatasetProfileView, logger: logging.Logger) -> NotebookProfileVisualizer:
    try:
        _viz = NotebookProfileVisualizer
        _viz.set_profiles(target_profile_view=target_profile,
                          reference_profile_view=referece_profile)
        return _viz
    except Exception as e:
        logger.exception('Exception in create_metrics(): {}'.format(e))


""" create profile summary json"""
""" ************************** """


def create_profile_summary_json(profile: DatasetProfileView, path: str, logger: logging.Logger) -> str:
    try:
        with open('{}.json'.format(path), 'w') as output_file:
            output_file.write(str(generate_profile_summary(
                target_view=uncompound._uncompound_dataset_profile(profile), config=None)['profile_from_whylogs']))
        logger.info('Created profile summary json for {}'.format(path))
    except Exception as e:
        logger.exception('Exception in create_profile_summary(): {}'.format(e))


""" create profile comparison json """
""" ****************************** """


def create_profile_compare_summary_json(target_profile: DatasetProfileView, ref_profile: DatasetProfileView, path: str, logger: logging.Logger) -> str:
    try:
        with open('{}.json'.format(path), 'w') as output_file:
            target_uncompound = uncompound._uncompound_dataset_profile(
                target_profile)
            ref_uncompound = uncompound._uncompound_dataset_profile(
                ref_profile)
            output_file.write(str(generate_summaries(target_view=target_uncompound,
                              ref_view=ref_uncompound, config=None)['reference_profile_from_whylogs']))
    except Exception as e:
        logger.exception(
            'Exception in create_profile_compare_summary_json(): {}'.format(e))


def create_drift_metric_df_from_comp_summary_json(path: str, logger: logging.Logger) -> pd.DataFrame:
    """ create metric and p_val dataframes """
    """ ********************************** """
    try:
        with open(path, 'r') as input:
            data = json.load(input)
        tmp = {"metric": [], "p_val": [], "rating": []}
        for metric in data['columns']:
            if metric != 'image.Colorspace':
                tmp['metric'].append(metric)
                p_val = data['columns'][metric]['drift_from_ref']
                tmp['p_val'].append(p_val)
                if p_val <= 0.05:
                    tmp["rating"].append('Detected Drift (0.00 - 0.05)')
                elif p_val <= 0.15:
                    tmp["rating"].append('Possible Drift (0.05 - 0.15)')
                elif p_val <= 1.0:
                    tmp['rating'].append('No Drift (0.15 - 1.00)')
        df = pd.DataFrame.from_dict(tmp)
        df.sort_values(by=['metric'], inplace=True)
        df.to_csv('{}.csv'.format(path[0:-5]))
        
        return df
    except Exception as e:
        logger.exception(
            'Exception in create_drift_metric_df_from_comp_summary_json(): {}'.format(e))


def main():
    """ init datetime, logger """
    my_datetime = datetime.now(pytz.timezone('Europe/Berlin'))
    my_format = logging.Formatter('%(asctime)s [%(levelname)s]  %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(my_format)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    LANDSCAPE_DATA_RAW_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/landscape_data_raw'

    LANDSCAPE_BINS_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/bins'

    LANDSCAPE_JSON_PROFILES_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_summaries'

    LANDSCAPE_JSON_COMP_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_comps'

    LANDSCAPE_DATA_RAW_BASELINE_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/landscape_data_raw/landscape_baseline/baseline/'
    LANDSCAPE_DATA_RAW_CAMERA_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/landscape_data_raw/camera_images/camera/'

    """ comparison jsons to dataframe """
    pd.set_option("display.precision", 20)
    pd.set_option('max_colwidth', 800)

    """ log  baseline profiles """
    profile_baseline_1800 = log_data(LANDSCAPE_DATA_RAW_BASELINE_PATH, my_datetime, logger, 1800, True)
    profile_baseline_900 = log_data(LANDSCAPE_DATA_RAW_BASELINE_PATH, my_datetime, logger, 900, True)
    profile_baseline_450 = log_data(LANDSCAPE_DATA_RAW_BASELINE_PATH, my_datetime, logger, 450, True)
    profile_baseline_200 = log_data(LANDSCAPE_DATA_RAW_BASELINE_PATH, my_datetime, logger, 200, True)
    profile_baseline_150 = log_data(LANDSCAPE_DATA_RAW_BASELINE_PATH, my_datetime, logger, 150, True)
    profile_baseline_100 = log_data(LANDSCAPE_DATA_RAW_BASELINE_PATH, my_datetime, logger, 100, True)
    profile_baseline_50 = log_data(LANDSCAPE_DATA_RAW_BASELINE_PATH, my_datetime, logger, 50, True)
    profile_baseline_25 = log_data(LANDSCAPE_DATA_RAW_BASELINE_PATH, my_datetime, logger, 25, True)
    profile_baseline_15 = log_data(LANDSCAPE_DATA_RAW_BASELINE_PATH, my_datetime, logger, 15, True)

    """ serialize baseline profiles """
    serialize_profile(profile_baseline_15,'{}/baseline_15'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_baseline_25,'{}/baseline_25'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_baseline_50,'{}/baseline_50'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_baseline_100,'{}/baseline_100'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_baseline_150,'{}/baseline_150'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_baseline_200,'{}/baseline_200'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_baseline_450,'{}/baseline_450'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_baseline_900,'{}/baseline_900'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_baseline_1800,'{}/baseline_1800'.format(LANDSCAPE_BINS_PATH), logger)

    """ log camera profiles """
    profile_camera_200 = log_data(LANDSCAPE_DATA_RAW_CAMERA_PATH, my_datetime, logger, 200, True)
    profile_camera_150 = log_data(LANDSCAPE_DATA_RAW_CAMERA_PATH, my_datetime, logger,150,True)
    profile_camera_100 = log_data(LANDSCAPE_DATA_RAW_CAMERA_PATH, my_datetime, logger, 100, True)
    profile_camera_50 = log_data(LANDSCAPE_DATA_RAW_CAMERA_PATH, my_datetime, logger, 50, True)
    profile_camera_25 = log_data(LANDSCAPE_DATA_RAW_CAMERA_PATH, my_datetime, logger, 25, True)
    profile_camera_15 = log_data(LANDSCAPE_DATA_RAW_CAMERA_PATH, my_datetime, logger, 15, True)

    """ serialize camera profiles """
    serialize_profile(profile_camera_15, '{}/camera_15'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_camera_25, '{}/camera_25'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_camera_50, '{}/camera_50'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_camera_100, '{}/camera_100'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_camera_150, '{}/camera_150'.format(LANDSCAPE_BINS_PATH), logger)
    serialize_profile(profile_camera_200, '{}/camera_200'.format(LANDSCAPE_BINS_PATH), logger)

    """ load baseline profiles from bin """
    baseline_1800 = deserialize_profile('{}/baseline_1800.bin'.format(LANDSCAPE_BINS_PATH), logger)
    baseline_900 = deserialize_profile('{}/baseline_900.bin'.format(LANDSCAPE_BINS_PATH), logger)
    baseline_450 = deserialize_profile('{}/baseline_450.bin'.format(LANDSCAPE_BINS_PATH), logger)
    baseline_200 = deserialize_profile('{}/baseline_200.bin'.format(LANDSCAPE_BINS_PATH), logger)
    baseline_150 = deserialize_profile('{}/baseline_150.bin'.format(LANDSCAPE_BINS_PATH), logger)
    baseline_100 = deserialize_profile('{}/baseline_100.bin'.format(LANDSCAPE_BINS_PATH), logger)
    baseline_50 = deserialize_profile('{}/baseline_50.bin'.format(LANDSCAPE_BINS_PATH), logger)
    baseline_25 = deserialize_profile('{}/baseline_25.bin'.format(LANDSCAPE_BINS_PATH), logger)
    baseline_15 = deserialize_profile('{}/baseline_15.bin'.format(LANDSCAPE_BINS_PATH), logger)
    
    """ load camera  profiles from bin """
    camera_200 = deserialize_profile('{}/camera_200.bin'.format(LANDSCAPE_BINS_PATH), logger)
    camera_150 = deserialize_profile('{}/camera_150.bin'.format(LANDSCAPE_BINS_PATH), logger)
    camera_100 = deserialize_profile('{}/camera_100.bin'.format(LANDSCAPE_BINS_PATH), logger)
    camera_50 = deserialize_profile('{}/camera_50.bin'.format(LANDSCAPE_BINS_PATH), logger)
    camera_25 = deserialize_profile('{}/camera_25.bin'.format(LANDSCAPE_BINS_PATH), logger)
    camera_15 = deserialize_profile('{}/camera_15.bin'.format(LANDSCAPE_BINS_PATH), logger)




    """ create comparisons camera v baseline 1800 """
    create_profile_compare_summary_json(camera_15, baseline_1800, '{}/camera_15_v_baseline_1800'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_25, baseline_1800, '{}/camera_25_v_baseline_1800'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_50, baseline_1800, '{}/camera_50_v_baseline_1800'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_100, baseline_1800, '{}/camera_100_v_baseline_1800'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_150, baseline_1800, '{}/camera_150_v_baseline_1800'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_200, baseline_1800, '{}/camera_200_v_baseline_1800'.format(LANDSCAPE_JSON_COMP_PATH), logger)

    """ create comparisons camera v baseline 900 """
    create_profile_compare_summary_json(camera_15, baseline_900, '{}/camera_15_v_baseline_900'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_25, baseline_900, '{}/camera_25_v_baseline_900'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_50, baseline_900, '{}/camera_50_v_baseline_900'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_100, baseline_900, '{}/camera_100_v_baseline_900'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_150, baseline_900, '{}/camera_150_v_baseline_900'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_200, baseline_900, '{}/camera_200_v_baseline_900'.format(LANDSCAPE_JSON_COMP_PATH), logger)

    """ create comparisons camera v baseline 450 """
    create_profile_compare_summary_json(camera_15,  baseline_450, '{}/camera_15_v_baseline_450'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_25,  baseline_450, '{}/camera_25_v_baseline_450'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_50,  baseline_450, '{}/camera_50_v_baseline_450'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_150, baseline_450, '{}/camera_150_v_baseline_450'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_100, baseline_450, '{}/camera_100_v_baseline_450'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    create_profile_compare_summary_json(camera_200, baseline_450, '{}/camera_200_v_baseline_450'.format(LANDSCAPE_JSON_COMP_PATH), logger)


    """ camera vs baseline 1800 """
    df_camera_15_v_baseline_1800 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_15_v_baseline_1800.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_25_v_baseline_1800 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_25_v_baseline_1800.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_50_v_baseline_1800 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_50_v_baseline_1800.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_100_v_baseline_1800 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_100_v_baseline_1800.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_150_v_baseline_1800 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_150_v_baseline_1800.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_200_v_baseline_1800 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_200_v_baseline_1800.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)

    """ camera vs baseline 900 """
    df_camera_15_v_baseline_900 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_15_v_baseline_900.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_25_v_baseline_900 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_25_v_baseline_900.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_50_v_baseline_900 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_50_v_baseline_900.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_100_v_baseline_900 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_100_v_baseline_900.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_150_v_baseline_900 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_150_v_baseline_900.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_200_v_baseline_900 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_200_v_baseline_900.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)

    """ camera vs baseline 450 """
    df_camera_15_v_baseline_450 =  create_drift_metric_df_from_comp_summary_json(path='{}/camera_15_v_baseline_450.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_25_v_baseline_450 =  create_drift_metric_df_from_comp_summary_json(path='{}/camera_25_v_baseline_450.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_50_v_baseline_450 =  create_drift_metric_df_from_comp_summary_json(path='{}/camera_50_v_baseline_450.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_100_v_baseline_450 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_100_v_baseline_450.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_150_v_baseline_450 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_150_v_baseline_450.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    df_camera_200_v_baseline_450 = create_drift_metric_df_from_comp_summary_json(path='{}/camera_200_v_baseline_450.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)



if __name__ == "__main__":
    main()
