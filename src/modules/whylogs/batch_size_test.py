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


def log_data(data_directory: str, datetime: datetime, logger: logging.Logger, batch_size : int, shuffle : bool) -> DatasetProfileView:
    try:
        logger.info(
            'Started logging image data from: {}'.format(data_directory))
        profile = None
        data_directory_list =  os.listdir(data_directory)
        if shuffle:
            random.shuffle(data_directory_list) # shuffle images 
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

    """ log  & serialize landscape dataset  """
    # profile_baseline = log_data(
    #     '{}/landscape_baseline/baseline/'.format(LANDSCAPE_DATA_RAW_PATH), my_datetime, logger)
    # serialize_profile(profile_baseline,
    #                   '{}/baseline'.format(LANDSCAPE_BINS_PATH), logger)

    # profile_landscape = log_data(
    #     '{}/landscape_images/landscape/'.format(LANDSCAPE_DATA_RAW_PATH), my_datetime, logger)
    # serialize_profile(profile_landscape,
    #                   '{}/landscape'.format(LANDSCAPE_BINS_PATH), logger)

    # profile_camera = log_data(
    #     '{}/camera_images/camera/'.format(LANDSCAPE_DATA_RAW_PATH), my_datetime, logger)
    # serialize_profile(
    #     profile_camera, '{}/camera'.format(LANDSCAPE_BINS_PATH), logger)

    profile_baseline_200 = log_data(
        '{}/landscape_baseline/baseline/'.format(LANDSCAPE_DATA_RAW_PATH), my_datetime, logger, 200, True)
    serialize_profile(
        profile_baseline_200, '{}/baseline_200'.format(LANDSCAPE_BINS_PATH), logger)

    """ load & deserialize landscape dataset """
    # profile_baseline = deserialize_profile(
    #     '{}/baseline.bin'.format(LANDSCAPE_BINS_PATH), logger)
    # profile_landscape = deserialize_profile(
    #     '{}/landscape.bin'.format(LANDSCAPE_BINS_PATH), logger)
    # profile_camera = deserialize_profile(
    #     '{}/camera.bin'.format(LANDSCAPE_BINS_PATH), logger)

    """ create summary json's """
    # create_profile_summary_json(
    #     profile_baseline, '{}/baseline'.format(LANDSCAPE_JSON_PROFILES_PATH), logger)
    # create_profile_summary_json(
    #     profile_landscape, '{}/landscape'.format(LANDSCAPE_JSON_PROFILES_PATH), logger)
    # create_profile_summary_json(
    #     profile_camera, '{}/camera'.format(LANDSCAPE_JSON_PROFILES_PATH), logger)

    """ crate profile comp json's """
    # create_profile_compare_summary_json(
    #     profile_landscape, profile_baseline, '{}/landscape_v_baseline'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    # create_profile_compare_summary_json(
    #     profile_camera, profile_baseline, '{}/camera_v_baseline'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    # create_profile_compare_summary_json(
    #     profile_landscape, profile_landscape, '{}/landscape_v_landscape'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    # create_profile_compare_summary_json(
    #     profile_camera, profile_landscape, '{}/camera_v_landscape'.format(LANDSCAPE_JSON_COMP_PATH), logger)
    
    """ comparison jsons to dataframe """
    pd.set_option("display.precision", 20)
    pd.set_option('max_colwidth', 800)

    # df_landscape_v_baseline = create_drift_metric_df_from_comp_summary_json(
    #     path='{}/landscape_v_baseline.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    # print(f'{"landscape_v_baseline":.^50}',
    #       "\n", df_landscape_v_baseline, "\n")

    # df_camera_v_baseline = create_drift_metric_df_from_comp_summary_json(
    #     path='{}/camera_v_baseline.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    # print(f'{"camera (200) v baseline (1800)":.^50}', "\n", df_camera_v_baseline, "\n")

    # df_landscape_v_landscape = create_drift_metric_df_from_comp_summary_json(
    #     path='{}/landscape_v_landscape.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    # print(f'{"landscape_v_landscape":.^50}',
    #       "\n", df_landscape_v_landscape, "\n")

    # df_camera_v_landscape = create_drift_metric_df_from_comp_summary_json(
    #     path='{}/camera_v_landscape.json'.format(LANDSCAPE_JSON_COMP_PATH), logger=logger)
    # print(f'{"camera (200) v landscape (200)":.^50}',
    #       "\n", df_camera_v_landscape, "\n")

    
if __name__ == "__main__":
    main()
