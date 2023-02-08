

# MISC imports
import json


def main():
    
   

    # Load Configuration
    with open('config/drift_detection_config.json') as config_file:
        drift_detection_config = json.load(config_file)
    
    print(drift_detection_config['VERSION'])
    

   


    










# ======================================================================================
# call
if __name__ == "__main__":
    main()