with open("image_attribute_labels.txt") as f:
    for i, line in enumerate(f, 1):
        count = len(line.strip().split())
        if count != 22:
            print(f"⚠️ Satır {i} sütun sayısı: {count} (hatalı): {line.strip()}")
