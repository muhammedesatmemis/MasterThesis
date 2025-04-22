import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Dataset sınıfı
class CarConceptDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.data = dataframe
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = f"{self.root_dir}/{row['image_path']}"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        concept_labels = torch.tensor(row[2:].astype(float).values, dtype=torch.float32)
        class_label = torch.tensor(self.label_to_idx[row['label']], dtype=torch.long)
        return image, concept_labels, class_label

# Sütun isimleri
column_names = [
    "label", "image_path",
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large",
    "compact", "boxy", "sloped_rear", "roof_rails", "offroad", "flat_front", "luxury"
]

# DataFrame yükle
df = pd.read_csv("/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/data/image_attribute_labels.txt",
                 sep="\s+", header=None, names=column_names)

# İlk 5 satırı hızlıca gözden geçir
print("📄 İlk 5 satır (hızlı kontrol):")
print(df.head())

# Dataset nesnesini oluştur
dataset = CarConceptDataset(df, root_dir="/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/dataset")

# Bir örnek al ve göster
image, concepts, label = dataset[0]

print("✅ Test Başarılı")
print("Image shape:", image.shape)
print("Concepts:", concepts)
print("Class label:", label)
