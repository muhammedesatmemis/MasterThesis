import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 🔹 Label Predictor modeli
class LabelPredictor(nn.Module):
    def __init__(self, input_dim=13, num_classes=9):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# 🔹 Etiket dosyasını oku
column_names = [
    "label", "image_path",
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large"
]

df = pd.read_csv(
    "/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/data/image_attribute_labels.txt",
    sep="\s+", header=None, names=column_names
)

# 🔹 Etiketleri sayısal hale getir
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# 🔹 Kavramlar ve etiketleri ayır
concepts = torch.tensor(df.iloc[:, 2:15].values, dtype=torch.float32)
labels = torch.tensor(df["label_encoded"].values, dtype=torch.long)

# 🔹 Modeli yükle
model = LabelPredictor()
model.load_state_dict(torch.load("label_predictor.pt"))  # Eğer eğitimde kaydettiysen
model.eval()

# 🔹 Test
correct = 0
total = len(concepts)

with torch.no_grad():
    outputs = model(concepts)
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == labels).sum().item()

accuracy = correct / total
print(f"\n🎯 True Concepts ile Sınıf Tahmini Doğruluğu: {accuracy:.2f} ({correct}/{total})")
