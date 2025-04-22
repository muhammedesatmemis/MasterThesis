import os

dataset_root = "/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/dataset"
output_file = "/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/data/image_attribute_labels.txt"

concept_profiles = {
    "Cabrio":       [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    "Coupe":        [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    "Crossover":    [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    "Hatchback":    [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    "Minivan":      [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    "Pickup":       [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
    "Sedan":        [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    "StationWagon": [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    "SUV":          [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
}

with open(output_file, "w") as f:
    for class_name in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, class_name)
        if not os.path.isdir(class_path):
            continue
        if class_name not in concept_profiles:
            print(f"❌ Uyarı: {class_name} için kavram profili tanımlanmamış!")
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                rel_path = f"{class_name}/{img_name}"
                attrs = concept_profiles[class_name]
                attr_str = " ".join(map(str, attrs))
                f.write(f"{class_name} {rel_path} {attr_str}\n")

print("✅ Otomatik etiketleme tamamlandı!")
