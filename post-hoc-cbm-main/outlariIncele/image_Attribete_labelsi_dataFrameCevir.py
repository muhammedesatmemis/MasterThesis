import pandas as pd

column_names = [
    "label", "image_path",
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large"
]

df = pd.read_csv("/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/data/image_attribute_labels.txt", sep="\s+", header=None, names=column_names)

print(df.head())
print(f"\nToplam örnek sayısı: {len(df)}")
print(f"Etiketler: {df['label'].unique()}")
