import os
import shutil
import random
from pathlib import Path

from google.colab import drive
drive.mount('/content/drive')

# Ruta a tu dataset original (donde están todas las imágenes por clase)
original_dataset_dir = '/content/drive/MyDrive/maize-leaf-disease/Data'

# Nueva ruta para almacenar el dataset dividido
base_dir = '/content/drive/MyDrive/maize-leaf-disease/Data2'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Crear carpetas de salida
for folder in [train_dir, val_dir]:
    os.makedirs(folder, exist_ok=True)

# Parámetro de división
split_ratio = 0.8  # 80% train, 20% val

# Procesar cada clase
classes = os.listdir(original_dataset_dir)

for class_name in classes:
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    images = [img for img in images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Crear carpetas de clase
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Copiar imágenes
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

    print(f'Clase {class_name}: {len(train_images)} entrenamiento, {len(val_images)} validación')

print("✅ División completa.")