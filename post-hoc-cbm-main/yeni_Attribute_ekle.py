import os

# Dataset klasörünün yolu
dataset_root = "/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/dataset"
output_file = "/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/data/image_attribute_labels.txt"

# Attribute başlıkları (bilgi amaçlı, dosyaya yazılmaz)
attributes = [
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large",
    "compact", "boxy", "sloped_rear", "roof_rails", "offroad", "flat_front", "luxury"
]

# Attribute tanımları (20 adet)
default_attributes = {
    "Hatchback":     [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    "StationWagon":  [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    "Pickup":        [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    "Coupe":         [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    "Minivan":       [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    "SUV":           [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
    "Crossover":     [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1],
    "Sedan":         [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    "Cabrio":        [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0]
}

lines = []

# Görselleri gez
for class_folder in sorted(os.listdir(dataset_root)):
    class_path = os.path.join(dataset_root, class_folder)
    if os.path.isdir(class_path):
        for image_file in sorted(os.listdir(class_path)):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                relative_path = f"{class_folder}/{image_file}"
                attributes = default_attributes.get(class_folder, [0] * 20)
                attr_string = " ".join(str(x) for x in attributes)
                line = f"{class_folder} {relative_path} {attr_string}"
                lines.append(line)

# Dosyaya yaz
with open(output_file, "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"✅ {len(lines)} satır yazıldı: {output_file}")
