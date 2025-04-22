dataset = CarConceptDataset(df, root_dir="/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main")

# İlk örneği test edelim
image, concepts, label = dataset[0]

print("Image shape:", image.shape)
print("Concepts:", concepts)
print("Class label:", label)
