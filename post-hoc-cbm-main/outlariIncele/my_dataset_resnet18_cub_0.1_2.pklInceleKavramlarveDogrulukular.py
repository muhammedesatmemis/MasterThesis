import pickle

# .pkl dosyasının yolu
pkl_file_path = '../output/my_dataset_resnet18_cub_0.1_2.pkl'

# Dosyayı aç
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

print("\n📊 Öğrenilen Kavramlar (Concepts) ve Detayları:\n")

for key in data.keys():
    concept_tuple = data[key]
    vector = concept_tuple[0]  # Kavram vektörü
    accuracy = concept_tuple[1]  # Doğruluk
    threshold = concept_tuple[2]  # Threshold
    extra_score = concept_tuple[3]  # Ek skor
    stats = concept_tuple[4]  # İstatistikler

    print(f"🔹 Kavram: {key}")
    print(f"   🎯 Doğruluk: {accuracy:.2f}")
    print(f"   🚧 Threshold: {threshold:.2f}")
    print(f"   📐 Vektör Boyutu: {vector.shape}")
    print(f"   📈 Ek Skor (muhtemelen AUC): {extra_score[0]:.4f}")
    print(f"   🔺 Max: {stats['max']:.4f}, 🔻 Min: {stats['min']:.4f}")
    print("-" * 50)
