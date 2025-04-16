import os
import shutil
import random

dataset_path = "my_dataset/dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Train ve Test klasörlerini oluştur
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Her sınıftaki resimleri train-test olarak böl
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)

    if os.path.isdir(class_dir):  # Eğer klasörse
        images = os.listdir(class_dir)
        random.shuffle(images)  # Karıştır

        # Eğer sadece 2 resim varsa, birini train, diğerini test'e koy
        if len(images) >= 2:
            train_images = images[:1]  # İlk resmi train'e
            test_images = images[1:]  # İkinci resmi test'e

            # Sınıf klasörlerini oluştur
            os.makedirs(os.path.join(train_path, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_path, class_name), exist_ok=True)

            # Dosyaları taşı
            for img in train_images:
                shutil.move(os.path.join(class_dir, img), os.path.join(train_path, class_name, img))

            for img in test_images:
                shutil.move(os.path.join(class_dir, img), os.path.join(test_path, class_name, img))

print("✅ Train-Test bölme işlemi tamamlandı! 🚀")
