import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from concept_train import ConceptPredictor
from label_predictor_train import ClassPredictor

# 🔹 Parametreler
image_path = "my_dataset/test_images/img_1.png"  # ← tahmin edilecek görselin yolu
num_concepts = 20
num_classes = 8

# 🔹 Model yükleme
concept_model = ConceptPredictor(num_concepts=num_concepts)
label_model = ClassPredictor(input_size=num_concepts, num_classes=num_classes)

concept_model.load_state_dict(torch.load("concept_model.pth", map_location="cpu"))
label_model.load_state_dict(torch.load("label_model.pth", map_location="cpu"))

concept_model.eval()
label_model.eval()

# 🔹 Görsel ön işleme
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]

# 🔹 Tahmin
with torch.no_grad():
    concepts = concept_model(image_tensor)
    concept_binary = (concepts > 0.5).int().squeeze().tolist()

    class_logits = label_model(concepts)
    predicted_class = torch.argmax(class_logits, dim=1).item()

# 🔹 Kavram ve sınıf isimleri
attribute_names = [
    "2door", "4door", "tall_roof", "short_roof", "long_trunk", "short_trunk",
    "high_body", "low_body", "sporty", "open_back", "sunroof", "small", "large",
    "compact", "boxy", "sloped_rear", "roof_rails", "offroad", "flat_front", "luxury"
]

class_names = ["Cabrio", "Coupe", "Crossover", "Hatchback", "Pickup", "SUV", "Sedan", "StationWagon"]

# 🔹 Sonuç yazdır
print(f"📷 Görsel: {os.path.basename(image_path)}")
print("\n🔢 Tahmin Edilen Conceptler:")
for attr, val in zip(attribute_names, concept_binary):
    if val == 1:
        print(f"  ✅ {attr}")

print(f"\n🎯 Tahmin Edilen Sınıf: {class_names[predicted_class]}")
