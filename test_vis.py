import pickle

metadata = pickle.load(open("./data/metadata.pkl", "rb"))
# print("metadata", metadata)

raw_data_path = "/userdata/kerasData/data/new_data/raw_images/"
raw_labels_path = "/userdata/kerasData/data/new_data/drive_clone/"
labels_path = "/root/pytorch_lightning_data/drive_clone_numpy/"

example = "20180603_FIRE_smer-tcs8-mobo-c/1528058095_+00840"
