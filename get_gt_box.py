import pickle

metadata = pickle.load(open("./data/metadata.pkl", "rb"))

# print(metadata)

# Initialize metadata.pkl
# metadata = {}

# metadata['fire_to_images'] = util_fns.generate_fire_to_images(raw_data_path)

# metadata['omit_no_xml'] = []
# metadata['omit_no_contour'] = []
# metadata['omit_no_contour_or_bbox'] = []
# metadata['omit_mislabeled'] = np.loadtxt('./data/omit_mislabeled.txt', dtype=str)

# metadata['labeled_fires'] = [folder.stem for folder in filter(Path.is_dir, Path(labels_path).iterdir())]
# metadata['unlabeled_fires'] = []
# metadata['train_only_fires'] = []
# metadata['eligible_fires'] = []
# metadata['monochrome_fires'] = []
# metadata['night_fires'] = np.loadtxt('./data/night_fires.txt', dtype=str)
# metadata['mislabeled_fires'] = np.loadtxt('./data/mislabeled_fires.txt', dtype=str)

# metadata['bbox_labels'] = {}

labeled_fires = metadata["labeled_fires"]
labeled_fires.sort()
print("labeled_fires", len(labeled_fires))

bbox_labels = metadata["bbox_labels"]
print("bbox_labels", len(bbox_labels))

unlabeled_fires = metadata["unlabeled_fires"]
print("unlabeled_fires", len(unlabeled_fires))

train_only_fires = metadata["train_only_fires"]
print("train_only_fires", len(train_only_fires))

monochrome_fires = metadata["monochrome_fires"]
print("monochrome_fires", len(monochrome_fires))

night_fires = metadata["night_fires"]
print("night_fires", len(night_fires))

mislabeled_fires = metadata["mislabeled_fires"]
print("mislabeled_fires", len(mislabeled_fires))

fire_to_images = metadata["fire_to_images"]
print("fire_to_images", len(fire_to_images))

# gt_bboxes = metadata["bbox_labels"][image_name]
# Model      # Fires # Images
# Train        144   11.3 K
# Validation    64   4.9 K
# Test          62   4.9 K
# Omitted       45   3.7 K
# Total         315  24.8 K
