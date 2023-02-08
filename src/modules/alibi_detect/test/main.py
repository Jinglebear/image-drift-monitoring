
# my class
from untrained_encoder import UntrainedAutoencoder

# wilds import
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds import get_dataset

# torchvision transforms
import torchvision.transforms as transforms

# misc
import numpy as np
import os
import json
def main():
    CAMELYON_UAE_DETECTOR_PATH =  '/home/jinglewsl/evoila/projects/image-drift-monitoring/src/modules/alibi_detect/uae_ks_detector_camelyon'
    


    camelyon_train_comp = np.load('camelyon_train_ds.npz')

    camelyon_train = camelyon_train_comp['arr_0']
   
    print(type(camelyon_train))

    my_uae = UntrainedAutoencoder()

    encoder_fn = my_uae.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train.shape[1:],batch_size=512)

    # # my_uae.init_detector(reference_data=train_ds,encoder_fn=encoder_fn, p_val= 0.05, path=CAMELYON_UAE_DETECTOR_PATH, save_dec=True)
    
    # my_uae.import_detector(path=CAMELYON_UAE_DETECTOR_PATH)




    # res_val_ds = my_uae.make_prediction(target_data=val_ds)

    # print('H_0"\n\n')

    # res_train_ds = my_uae.make_prediction(target_data=train_ds)

    # print('res_val:', res_val_ds)
    # print('res_train:', res_train_ds)
    
    # dumped_val = json.dumps(res_val_ds, cls=NumpyEncoder)
    # dumped_train = json.dumps(res_train_ds, cls=NumpyEncoder)
    
    # with open("res_val_ds.json","w")as outfile:
    #     json.dump(dumped_val,outfile)
    # with open("res_train_ds.json","w")as outfile:
    #     json.dump(dumped_train,outfile)
    

   

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# ======================================================================================
# call
if __name__ == "__main__":
    main()