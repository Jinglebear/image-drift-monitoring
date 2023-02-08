# whylogs import
from modules.whylogs.whylogs_logger import Whylogs_Logger

# MISC imports
import json


def main():
    
    # Load Configuration
    with open('config/common/drift_detection_config.json') as config_file:
        drift_detection_config = json.load(config_file)
    

    # init whylogs logger
    my_whylogs_logger = Whylogs_Logger(drift_detection_config)
    print(my_whylogs_logger.config['VERSION'])
    

    


    










# ======================================================================================
# call
if __name__ == "__main__":
    main()