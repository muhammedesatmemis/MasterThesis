import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 🔹 Dataset
class ConceptToClassDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(dataframe['label'].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        concepts = torch.tensor(row[2:].astype(float).values, dtype=torch.float32)
        class_label = torch.tensor(self.label_to_idx[row['label']], dtype=torch.long)
        return concepts, class_label

# 🔹 Model
class ClassPredictor(nn.Module):
    def __init__(self, input_size=20, num_classes=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 🔹 Veri Yükle
column_names = [
    "label", "image_path",
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large",
    "compact", "boxy", "sloped_rear", "roof_rails", "offroad", "flat_front", "luxury"
]

df = pd.read_csv(
    "/Users/esatsmac/Desktop/MasterThesis/post-hoc-cbm-main/my_dataset/data/image_attribute_labels.txt",
    sep="\s+", header=None, names=column_names
)

# 🔹 DataLoader
dataset = ConceptToClassDataset(df)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 🔹 Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassPredictor(input_size=20, num_classes=len(dataset.label_to_idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 🔁 Eğitim
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for concepts, class_labels in dataloader:
        concepts, class_labels = concepts.to(device), class_labels.to(device)

        optimizer.zero_grad()
        outputs = model(concepts)
        loss = criterion(outputs, class_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"🏷️ Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "label_model.pth")
print("✅ Sınıf tahmin modeli eğitimi tamamlandı.")
