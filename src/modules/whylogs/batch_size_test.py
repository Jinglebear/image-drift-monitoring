import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

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

""" log data from directory to a profile, set logging date """
""" ****************************************************** """
def log_data(data_directory : str,datetime : datetime, logger : logging.Logger) -> DatasetProfileView:
    try:
        logger.info('Started logging image data from: {}'.format(data_directory))
        profile = None

        for img_name in progressbar(os.listdir(data_directory)):

            img = Image.open(data_directory + img_name) # read in image
            _ = log_image(img).profile() # generate profile  
            _.set_dataset_timestamp(datetime) # optionally set dataset_timestamp
            _view = _.view() # extract mergeable profile view
            # merge each profile while looping
            if profile is None:
                profile = _view
            else:
                profile = profile.merge(_view)
        logger.info('Logged image data from: {} with batch size {}'.format(data_directory,profile.to_pandas()['image/Brightness.mean:counts/n']['image']))
        return profile
    except Exception as e:
        logger.exception('Exception in log_data(): {}'.format(e))
""" serialize a profile to a .bin """
""" ****************************************************** """
def serialize_profile(profile : DatasetProfileView, binary_name : str,logger : logging.Logger) -> str:
    try:
        return profile.write('{}.bin'.format(binary_name))
    except Exception as e:
        logger.exception('Exception in serialize_profile(): {}'.format(e))

""" deserialize a profile from a .bin """
""" ****************************************************** """
def deserialize_profile(path : str,logger : logging.Logger) -> DatasetProfileView:
    try:
        return DatasetProfileView.read(path)
    except Exception as e:
        logger.exception('Exception in deserialize_profile(): {}'.format(e))
        
        

""" create profile viz """
""" ****************** """
def create_metrics(target_profile: DatasetProfileView, referece_profile: DatasetProfileView,logger : logging.Logger) -> NotebookProfileVisualizer:
    try:
        _viz = NotebookProfileVisualizer
        _viz.set_profiles(target_profile_view=target_profile,reference_profile_view=referece_profile)
        return _viz
    except Exception as e:
        logger.exception('Exception in create_metrics(): {}'.format(e))

""" create profile summary """
""" ********************** """
def create_profile_summary(profile: DatasetProfileView,logger : logging.Logger) ->  Optional[Dict[str, Any]]:
    try:
        return generate_profile_summary(target_view=profile,config=None)
    except Exception as e:
        logger.exception('Exception in create_profile_summary(): {}'.format(e))

        

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

    """ log baseline """
    profile_baseline = log_data('/home/jingle/image-drift/landscape_data/landscape_baseline/baseline/',my_datetime,logger)
    serialize_profile(profile_baseline,'baseline')
    """ log landscape """
    profile_landscape = log_data('/home/jingle/image-drift/landscape_data/landscape_images/landscape/',my_datetime,logger)
    serialize_profile(profile_landscape,'landscape')
    """ log camera """
    profile_camera = log_data('/home/jingle/image-drift/landscape_data/camera_images/camera/',my_datetime,logger)
    serialize_profile(profile_camera,'camera')


    # """ load baseline """
    # profile_baseline = deserialize_profile('baseline.bin')
    # """ load landscape """
    # profile_landscape = deserialize_profile('landscape.bin')
    # """ load camera """
    # profile_camera = deserialize_profile('camera.bin')


    # _viz = NotebookProfileVisualizer()
    # _viz.set_profiles(profile_baseline)
    # tmp = _viz.profile_summary()
    # # print(type(tmp))
    # # with open("data.html", "w") as file:
    # #     file.write(tmp.data)

    # """ 
    # Generate Profile Summary without Vizualizer    

    # """
    # profile_baseline_uncompound = uncompound._uncompound_dataset_profile(profile_baseline)
    # tmp2 = generate_profile_summary(target_view=profile_baseline_uncompound,config=None)
    # with open('output.json', 'w') as output_file:
    #     output_file.write(str(tmp2['profile_from_whylogs']))

    
    

if __name__ == "__main__":
    main()
