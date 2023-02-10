import numpy as np

def main():


    camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
    camelyon_train = camelyon_train_comp['arr_0']

    np.random.shuffle(camelyon_train)

    camelyon_train_75 = camelyon_train[:int(len(camelyon_train)*0.75)]
    np.savez_compressed('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_75_ds.npz',camelyon_train_75)
    np.random.shuffle(camelyon_train)
    camelyon_train_80 = camelyon_train[:int(len(camelyon_train)*0.8)]
    np.savez_compressed('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_80_ds.npz',camelyon_train_80)
    np.random.shuffle(camelyon_train)
    camelyon_train_85 = camelyon_train[:int(len(camelyon_train)*0.85)]
    np.savez_compressed('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_85_ds.npz',camelyon_train_85)
    np.random.shuffle(camelyon_train)
    camelyon_train_90 = camelyon_train[:int(len(camelyon_train)*0.9)]
    np.savez_compressed('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_90_ds.npz',camelyon_train_90)
    np.random.shuffle(camelyon_train)
    camelyon_train_95 = camelyon_train[:int(len(camelyon_train)*0.95)]
    np.savez_compressed('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_95_ds.npz',camelyon_train_95)
    np.random.shuffle(camelyon_train)



# ======================================================================================
# call
if __name__ == "__main__":
    main()