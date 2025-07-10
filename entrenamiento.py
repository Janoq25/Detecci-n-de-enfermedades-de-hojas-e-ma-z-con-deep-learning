import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2, ResNet50, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_pre
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from google.colab import drive

from google.colab import drive
drive.mount('/content/drive')

# ------------------------
# 📂 Rutas y configuración general
# ------------------------
drive.mount('/content/drive')
base_path = "/content/drive/MyDrive/maize-leaf-disease"
train_dir = f"{base_path}/Data2/train"
val_dir = f"{base_path}/Data2/val"
report_path = f"{base_path}/Reports"
model_path = f"{base_path}/Models"
os.makedirs(report_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# ------------------------
# ⚙️ Carga de datos
# ------------------------
img_size = 128
batch_size = 32

# Cargar nombres de clases
temp_gen = ImageDataGenerator(preprocessing_function=mobilenet_pre)
temp_data = temp_gen.flow_from_directory(train_dir, target_size=(img_size, img_size))
class_names = list(temp_data.class_indices.keys())
num_classes = len(class_names)

# ------------------------
# 🔄 Generadores de datos
# ------------------------
def get_data_generators(preprocess_fn):
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
        fill_mode='nearest'
    )
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_data = train_gen.flow_from_directory(train_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')
    val_data = val_gen.flow_from_directory(val_dir, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical', shuffle=False)

    return train_data, val_data

# ------------------------
# ⚖️ Construcción de modelos
# ------------------------
def build_model(base_model_fn, preprocess_fn, name):
    base_model = base_model_fn(include_top=False, input_shape=(img_size, img_size, 3), weights='imagenet')
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output, name=name)
    return model

# ------------------------
# 🔢 Entrenamiento y evaluación
# ------------------------
def train_and_evaluate(model_fn, preprocess_fn, model_name):
    print(f"\n======== Entrenando {model_name} ========")
    model = build_model(model_fn, preprocess_fn, model_name)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    train_data, val_data = get_data_generators(preprocess_fn)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3, factor=0.3, verbose=1)
    ]

    start = time.time()
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=callbacks
    )
    duration = time.time() - start
    print(f"\u23f1 Tiempo de entrenamiento: {duration:.2f} segundos")

    # Guardar modelo
    model.save(os.path.join(model_path, f"{model_name}.h5"))

    # Evaluación
    val_data.reset()
    preds = model.predict(val_data)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_data.classes

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Guardar reporte
    with open(os.path.join(report_path, f"{model_name}_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=class_names))

    # Guardar matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title(f"Matriz de Confusión - {model_name}")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.savefig(os.path.join(report_path, f"{model_name}_confusion_matrix.png"))
    plt.close()

# ------------------------
# 📈 Entrenar todos los modelos
# ------------------------
modelos = [
    (MobileNetV2, mobilenet_pre, "MobileNetV2"),
    (ResNet50, resnet_pre, "ResNet50"),
    (EfficientNetB0, efficientnet_pre, "EfficientNetB0")
]

for model_fn, preprocess_fn, name in modelos:
    train_and_evaluate(model_fn, preprocess_fn, name)

print("\n📅 Entrenamiento y evaluación completados.")

