import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
import os
import pickle
from itertools import combinations

from google.colab import drive
drive.mount('/content/drive')

# Configuración
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 4

# Definir las enfermedades (ajusta según tu dataset)
DISEASES = {
    'Blight': 'Tizón',
    'Common_Rust': 'Roya Común',
    'Gray_Leaf_Spot': 'Mancha Gris',
    'Healthy': 'Saludable'
}

# Rutas
models_dir = '/content/drive/MyDrive/maize-leaf-disease/Models'
reports_dir = '/content/drive/MyDrive/maize-leaf-disease/Reports2'
test_dir = '/content/drive/MyDrive/maize-leaf-disease/Data2/val'  # Usando val como test

# Crear directorio de reportes
os.makedirs(reports_dir, exist_ok=True)

# Coeficiente de Matthews
def matthews_corrcoef(cm):
    """Calcular el coeficiente de Matthews para matriz de confusión multiclase"""
    if cm.shape[0] == 2:  # Caso binario
        tp, fp, fn, tn = cm[1][1], cm[0][1], cm[1][0], cm[0][0]
        numerator = (tp * tn) - (fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numerator / denominator if denominator != 0 else 0
    else:  # Caso multiclase
        # Implementación para multiclase basada en la definición original
        n = cm.shape[0]
        sum_p = np.sum(cm, axis=1)  # Suma por filas (predicciones)
        sum_t = np.sum(cm, axis=0)  # Suma por columnas (verdaderos)
        c = np.trace(cm)  # Suma diagonal (correctas)
        s = np.sum(cm)  # Total de muestras

        numerator = c * s - np.dot(sum_p, sum_t)
        denominator = np.sqrt((s**2 - np.dot(sum_p, sum_p)) * (s**2 - np.dot(sum_t, sum_t)))

        return numerator / denominator if denominator != 0 else 0

# Prueba de McNemar
def mcnemar_test(y_true, y_model1, y_model2):
    """Prueba de McNemar para comparar dos modelos"""
    # Crear tabla de contingencia
    table = np.zeros((2, 2))
    for true, pred1, pred2 in zip(y_true, y_model1, y_model2):
        if pred1 == true and pred2 != true:
            table[0][1] += 1  # Modelo 1 correcto, Modelo 2 incorrecto
        elif pred1 != true and pred2 == true:
            table[1][0] += 1  # Modelo 1 incorrecto, Modelo 2 correcto
        elif pred1 == true and pred2 == true:
            table[0][0] += 1  # Ambos correctos
        else:
            table[1][1] += 1  # Ambos incorrectos

    # Calcular estadístico de McNemar
    if table[0][1] + table[1][0] > 25:  # Corrección de Yates para muestras grandes
        statistic = (np.abs(table[0][1] - table[1][0]) - 1)**2 / (table[0][1] + table[1][0])
    else:
        statistic = (np.abs(table[0][1] - table[1][0]))**2 / (table[0][1] + table[1][0])

    if table[0][1] + table[1][0] == 0:
        p_value = 1.0  # No hay diferencias
    else:
        p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return statistic, p_value, table

def load_saved_models():
    """Cargar todos los modelos guardados"""
    models = {}

    # Buscar archivos de modelos
    model_files = {
        'MobileNetV2': 'MobileNetV2.h5',
        'ResNet50': 'ResNet50.h5',
        'EfficientNetB0': 'EfficientNetB0.h5'
    }

    # Intentar también con otros nombres posibles
    alternative_names = {
        'MobileNetV2': ['mobilenetv2_checkpoint.h5', 'MobileNetV2_best.h5'],
        'ResNet50': ['resnet50_checkpoint.h5', 'ResNet50_best.h5'],
        'EfficientNetB0': ['efficientnetb0_checkpoint.h5', 'EfficientNetB0_best.h5']
    }

    print("🔍 Buscando modelos guardados...")

    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)

        if os.path.exists(model_path):
            print(f"✅ Encontrado: {filename}")
            models[model_name] = tf.keras.models.load_model(model_path)
        else:
            # Buscar nombres alternativos
            found = False
            for alt_name in alternative_names[model_name]:
                alt_path = os.path.join(models_dir, alt_name)
                if os.path.exists(alt_path):
                    print(f"✅ Encontrado: {alt_name}")
                    models[model_name] = tf.keras.models.load_model(alt_path)
                    found = True
                    break

            if not found:
                print(f"❌ No encontrado: {model_name}")

    return models

def create_test_generators():
    """Crear generadores de datos para evaluación"""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    generators = {}

    # Generadores para cada tipo de preprocesamiento
    datagens = {
        "MobileNetV2": ImageDataGenerator(preprocessing_function=mobilenet_preprocess),
        "ResNet50": ImageDataGenerator(preprocessing_function=resnet_preprocess),
        "EfficientNetB0": ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
    }

    for model_name, datagen in datagens.items():
        test_gen = datagen.flow_from_directory(
            test_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False  # Importante para mantener orden
        )
        generators[model_name] = test_gen

    return generators

def evaluate_models(models, generators):
    """Evaluar todos los modelos y obtener métricas"""
    results = {}
    all_predictions = {}  # Para almacenar predicciones de todos los modelos

    print("\n📊 Evaluando modelos...")

    for model_name, model in models.items():
        print(f"\n🔄 Evaluando {model_name}...")

        test_gen = generators[model_name]
        test_gen.reset()  # Reiniciar generador

        # Evaluación básica
        test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)

        # Predicciones
        test_gen.reset()
        predictions = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(predictions, axis=1)

        # Etiquetas verdaderas
        y_true = test_gen.classes

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)

        # Coeficiente de Matthews
        mcc = matthews_corrcoef(cm)

        # Reporte de clasificación
        class_names = list(test_gen.class_indices.keys())
        report = classification_report(y_true, y_pred,
                                     target_names=class_names,
                                     output_dict=True)

        results[model_name] = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'confusion_matrix': cm,
            'classification_report': report,
            'class_names': class_names,
            'matthews_corrcoef': mcc,
            'y_true': y_true,
            'y_pred': y_pred
        }

        all_predictions[model_name] = y_pred

        print(f"✅ {model_name}: Accuracy = {test_accuracy:.4f}, MCC = {mcc:.4f}")

    # Realizar pruebas de McNemar entre todos los pares de modelos
    mcnemar_results = {}
    model_names = list(models.keys())

    if len(model_names) > 1:
        print("\n🔬 Realizando pruebas de McNemar...")

        for model1, model2 in combinations(model_names, 2):
            y_true = results[model1]['y_true']  # Mismo y_true para todos
            y_pred1 = results[model1]['y_pred']
            y_pred2 = results[model2]['y_pred']

            statistic, p_value, table = mcnemar_test(y_true, y_pred1, y_pred2)

            mcnemar_results[f"{model1}_vs_{model2}"] = {
                'statistic': statistic,
                'p_value': p_value,
                'contingency_table': table,
                'model1': model1,
                'model2': model2
            }

            print(f"📊 {model1} vs {model2}: p-value = {p_value:.4f}")

    return results, mcnemar_results

def generate_comprehensive_reports(results, mcnemar_results):
    """Generar todos los reportes visuales y de texto"""

    # 1. Comparación general de modelos (incluyendo MCC)
    plt.figure(figsize=(20, 8))

    # Subplot 1: Precisión
    plt.subplot(2, 2, 1)
    model_names = list(results.keys())
    accuracies = [results[name]['test_accuracy'] for name in model_names]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726', '#AB47BC']
    bars = plt.bar(model_names, accuracies, color=colors[:len(model_names)])
    plt.title('Comparación de Precisión en Test', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # Añadir valores sobre las barras
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Subplot 2: Coeficiente de Matthews
    plt.subplot(2, 2, 2)
    mcc_values = [results[name]['matthews_corrcoef'] for name in model_names]
    bars = plt.bar(model_names, mcc_values, color=colors[:len(model_names)])
    plt.title('Coeficiente de Matthews', fontsize=14, fontweight='bold')
    plt.ylabel('MCC')
    plt.ylim(-1, 1)

    # Añadir valores sobre las barras
    for bar, mcc in zip(bars, mcc_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mcc:.3f}', ha='center', va='bottom', fontweight='bold')

    # Subplot 3: Pérdida
    plt.subplot(2, 2, 3)
    losses = [results[name]['test_loss'] for name in model_names]
    bars = plt.bar(model_names, losses, color=colors[:len(model_names)])
    plt.title('Comparación de Pérdida en Test', fontsize=14, fontweight='bold')
    plt.ylabel('Loss')

    # Añadir valores sobre las barras
    for bar, loss in zip(bars, losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')

    # Subplot 4: Resumen de métricas
    plt.subplot(2, 2, 4)
    plt.axis('off')

    # Crear tabla de resumen
    summary_text = "RESUMEN DE RESULTADOS\n" + "="*25 + "\n\n"
    best_model = max(results, key=lambda x: results[x]['test_accuracy'])

    for i, (name, result) in enumerate(results.items()):
        marker = "🏆 " if name == best_model else f"{i+1}. "
        summary_text += f"{marker}{name}:\n"
        summary_text += f"   Precisión: {result['test_accuracy']:.4f}\n"
        summary_text += f"   MCC: {result['matthews_corrcoef']:.4f}\n"
        summary_text += f"   Pérdida: {result['test_loss']:.4f}\n\n"

    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(f'{reports_dir}/modelos_comparacion_completa.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. Gráfico de pruebas de McNemar
    if mcnemar_results:
        # Crear figura para tablas de contingencia
        n_comparisons = len(mcnemar_results)
        fig_height = max(8, n_comparisons * 3)
        plt.figure(figsize=(18, fig_height))

        # Subplot 1: P-values de McNemar
        plt.subplot(2, 3, 1)
        comparisons = list(mcnemar_results.keys())
        p_values = [mcnemar_results[comp]['p_value'] for comp in comparisons]

        bars = plt.bar(range(len(comparisons)), p_values, color='lightcoral')
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        plt.title('Pruebas de McNemar - P-values', fontsize=14, fontweight='bold')
        plt.ylabel('P-value')
        plt.xlabel('Comparaciones')
        plt.xticks(range(len(comparisons)), [comp.replace('_vs_', '\nvs\n') for comp in comparisons])
        plt.legend()

        # Añadir valores sobre las barras
        for bar, p_val in zip(bars, p_values):
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{p_val:.3f}\n{significance}', ha='center', va='bottom', fontweight='bold')

        # Subplot 2: Tabla de significancia
        plt.subplot(2, 3, 2)
        plt.axis('off')

        significance_text = "SIGNIFICANCIA ESTADÍSTICA\n" + "="*30 + "\n\n"
        significance_text += "*** p < 0.001 (muy significativo)\n"
        significance_text += "**  p < 0.01  (significativo)\n"
        significance_text += "*   p < 0.05  (marginalmente significativo)\n"
        significance_text += "ns  p ≥ 0.05  (no significativo)\n\n"

        significance_text += "RESULTADOS:\n" + "-"*20 + "\n"
        for comp, result in mcnemar_results.items():
            model1, model2 = result['model1'], result['model2']
            p_val = result['p_value']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            # Determinar cuál modelo es mejor
            acc1 = results[model1]['test_accuracy']
            acc2 = results[model2]['test_accuracy']
            better = model1 if acc1 > acc2 else model2

            significance_text += f"{model1} vs {model2}: {significance}\n"
            if p_val < 0.05:
                significance_text += f"  → {better} es significativamente mejor\n"
            else:
                significance_text += f"  → No hay diferencia significativa\n"
            significance_text += "\n"

        plt.text(0.1, 0.9, significance_text, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')

        # Subplots 3-6: Tablas de contingencia individuales
        subplot_positions = [(2, 3, 3), (2, 3, 4), (2, 3, 5), (2, 3, 6)]

        for idx, (comp, result) in enumerate(mcnemar_results.items()):
            if idx < len(subplot_positions):
                plt.subplot(*subplot_positions[idx])

                table = result['contingency_table']
                model1, model2 = result['model1'], result['model2']

                # Crear heatmap de la tabla de contingencia
                sns.heatmap(table, annot=True, fmt='.0f', cmap='Blues',
                           xticklabels=[f'{model2} Incorrecto', f'{model2} Correcto'],
                           yticklabels=[f'{model1} Correcto', f'{model1} Incorrecto'],
                           cbar=False)

                plt.title(f'Tabla de Contingencia\n{model1} vs {model2}',
                         fontsize=11, fontweight='bold')
                plt.xlabel(f'{model2}')
                plt.ylabel(f'{model1}')

                # Añadir información adicional
                p_val = result['p_value']
                statistic = result['statistic']
                plt.text(0.5, -0.15, f'Estadístico: {statistic:.3f}\nP-value: {p_val:.3f}',
                        transform=plt.gca().transAxes, ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(f'{reports_dir}/mcnemar_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Crear figura separada solo para las tablas de contingencia (más grande y clara)
        n_comparisons = len(mcnemar_results)
        cols = min(3, n_comparisons)
        rows = (n_comparisons + cols - 1) // cols

        plt.figure(figsize=(6*cols, 5*rows))

        for idx, (comp, result) in enumerate(mcnemar_results.items()):
            plt.subplot(rows, cols, idx + 1)

            table = result['contingency_table']
            model1, model2 = result['model1'], result['model2']

            # Crear heatmap más detallado
            ax = sns.heatmap(table, annot=True, fmt='.0f', cmap='RdYlBu_r',
                           xticklabels=[f'{model2} Incorrecto', f'{model2} Correcto'],
                           yticklabels=[f'{model1} Correcto', f'{model1} Incorrecto'],
                           cbar=True, square=True, linewidths=1)

            plt.title(f'Tabla de Contingencia McNemar\n{model1} vs {model2}',
                     fontsize=12, fontweight='bold', pad=20)
            plt.xlabel(f'{model2}', fontsize=11)
            plt.ylabel(f'{model1}', fontsize=11)

            # Añadir información estadística
            p_val = result['p_value']
            statistic = result['statistic']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            info_text = f'Estadístico χ²: {statistic:.3f}\nP-value: {p_val:.3f} {significance}'
            if p_val < 0.05:
                acc1 = results[model1]['test_accuracy']
                acc2 = results[model2]['test_accuracy']
                better = model1 if acc1 > acc2 else model2
                info_text += f'\n{better} es significativamente mejor'
            else:
                info_text += f'\nNo hay diferencia significativa'

            plt.text(0.5, -0.25, info_text, transform=plt.gca().transAxes,
                    ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="lightgray", alpha=0.7))

        plt.tight_layout()
        plt.savefig(f'{reports_dir}/mcnemar_contingency_tables.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 3. Matrices de confusión individuales
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for idx, (name, result) in enumerate(results.items()):
        class_names = result['class_names']
        sns.heatmap(result['confusion_matrix'],
                   annot=True, fmt='d',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cmap='Blues',
                   ax=axes[idx])
        axes[idx].set_title(f'Matriz de Confusión - {name}\nMCC: {result["matthews_corrcoef"]:.3f}',
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predicción')
        axes[idx].set_ylabel('Valor Real')

    plt.tight_layout()
    plt.savefig(f'{reports_dir}/matrices_confusion_todos.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 4. Matrices de confusión individuales (archivos separados)
    for name, result in results.items():
        plt.figure(figsize=(8, 6))
        class_names = result['class_names']
        sns.heatmap(result['confusion_matrix'],
                   annot=True, fmt='d',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cmap='Blues')
        plt.title(f'Matriz de Confusión - {name}\nMCC: {result["matthews_corrcoef"]:.3f}',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')

        # Guardar matriz individual
        filename = name.replace(' ', '_').lower()
        plt.savefig(f'{reports_dir}/matriz_confusion_{filename}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 5. Reporte detallado de clasificación
    plt.figure(figsize=(15, 10))

    # Crear métricas por clase para cada modelo
    classes = results[list(results.keys())[0]]['class_names']

    # Crear subplots para métricas por clase
    metrics = ['precision', 'recall', 'f1-score']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Gráfico por métrica
    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Preparar datos para el gráfico
        model_names = list(results.keys())
        x = np.arange(len(classes))
        width = 0.25

        for i, model_name in enumerate(model_names):
            values = []
            for class_name in classes:
                report = results[model_name]['classification_report']
                if class_name in report:
                    values.append(report[class_name][metric])
                else:
                    values.append(0)

            ax.bar(x + i*width, values, width, label=model_name,
                  color=colors[i] if i < len(colors) else '#999999')

        ax.set_xlabel('Clases de Enfermedades')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Comparación de {metric.capitalize()} por Clase')
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Tabla resumen en el cuarto subplot
    axes[3].axis('off')

    # Crear tabla de métricas generales
    table_data = []
    for name, result in results.items():
        report = result['classification_report']
        table_data.append([
            name,
            f"{result['test_accuracy']:.4f}",
            f"{result['matthews_corrcoef']:.4f}",
            f"{report['macro avg']['precision']:.4f}",
            f"{report['macro avg']['recall']:.4f}",
            f"{report['macro avg']['f1-score']:.4f}"
        ])

    # Crear tabla
    table = axes[3].table(cellText=table_data,
                         colLabels=['Modelo', 'Accuracy', 'MCC', 'Precision Avg', 'Recall Avg', 'F1-Score Avg'],
                         cellLoc='center',
                         loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 2)

    # Colorear la fila del mejor modelo
    best_model = max(results, key=lambda x: results[x]['test_accuracy'])
    best_idx = model_names.index(best_model)
    for col in range(6):
        table[(best_idx + 1, col)].set_facecolor('#90EE90')  # Verde claro

    axes[3].set_title('Resumen de Métricas por Modelo', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'{reports_dir}/metricas_detalladas_por_clase.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. Guardar reporte de texto
    with open(f'{reports_dir}/reporte_completo.txt', 'w', encoding='utf-8') as f:
        f.write("REPORTE COMPLETO DE CLASIFICACIÓN DE ENFERMEDADES EN MAÍZ\n")
        f.write("="*60 + "\n\n")

        best_model = max(results, key=lambda x: results[x]['test_accuracy'])
        f.write(f"MEJOR MODELO: {best_model}\n")
        f.write(f"Precisión: {results[best_model]['test_accuracy']:.4f}\n")
        f.write(f"Coeficiente de Matthews: {results[best_model]['matthews_corrcoef']:.4f}\n\n")

        for name, result in results.items():
            f.write(f"\n{name.upper()}\n")
            f.write("-" * len(name) + "\n")
            f.write(f"Precisión en Test: {result['test_accuracy']:.4f}\n")
            f.write(f"Pérdida en Test: {result['test_loss']:.4f}\n")
            f.write(f"Coeficiente de Matthews: {result['matthews_corrcoef']:.4f}\n\n")

            f.write("Reporte de clasificación:\n")
            report = result['classification_report']
            classes = result['class_names']
            for class_name in classes:
                if class_name in report:
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Precision: {report[class_name]['precision']:.4f}\n")
                    f.write(f"    Recall: {report[class_name]['recall']:.4f}\n")
                    f.write(f"    F1-Score: {report[class_name]['f1-score']:.4f}\n")
            f.write("\n")

        # Agregar resultados de McNemar
        if mcnemar_results:
            f.write("\nPRUEBAS DE MCNEMAR\n")
            f.write("="*20 + "\n\n")
            f.write("Comparación estadística entre modelos:\n")
            f.write("(p < 0.05 indica diferencia significativa)\n\n")

            for comp, result in mcnemar_results.items():
                model1, model2 = result['model1'], result['model2']
                p_val = result['p_value']
                statistic = result['statistic']

                f.write(f"{model1} vs {model2}:\n")
                f.write(f"  Estadístico: {statistic:.4f}\n")
                f.write(f"  P-value: {p_val:.4f}\n")

                if p_val < 0.05:
                    acc1 = results[model1]['test_accuracy']
                    acc2 = results[model2]['test_accuracy']
                    better = model1 if acc1 > acc2 else model2
                    f.write(f"  Resultado: {better} es significativamente mejor\n")
                else:
                    f.write(f"  Resultado: No hay diferencia significativa\n")
                f.write("\n")

    print("📈 Reportes generados exitosamente:")
    print(f"   - Comparación completa: {reports_dir}/modelos_comparacion_completa.png")
    print(f"   - Análisis McNemar: {reports_dir}/mcnemar_analysis.png")
    print(f"   - Tablas de contingencia McNemar: {reports_dir}/mcnemar_contingency_tables.png")
    print(f"   - Matrices de confusión: {reports_dir}/matrices_confusion_todos.png")
    print(f"   - Métricas detalladas: {reports_dir}/metricas_detalladas_por_clase.png")
    print(f"   - Reporte de texto: {reports_dir}/reporte_completo.txt")

    for name in results.keys():
        filename = name.replace(' ', '_').lower()
        print(f"   - Matriz individual {name}: {reports_dir}/matriz_confusion_{filename}.png")

    return results

# EJECUTAR TODO EL PROCESO
def main():
    print("🚀 Generando reportes desde modelos guardados...")

    # 1. Cargar modelos
    models = load_saved_models()

    if not models:
        print("❌ No se encontraron modelos guardados.")
        print("Verifica la ruta:", models_dir)
        print("Archivos disponibles:")
        if os.path.exists(models_dir):
            print(os.listdir(models_dir))
        return

    # 2. Crear generadores de test
    generators = create_test_generators()

    # 3. Evaluar modelos
    results, mcnemar_results = evaluate_models(models, generators)

    # 4. Generar reportes
    final_results = generate_comprehensive_reports(results, mcnemar_results)

    # 5. Resumen final
    print("\n" + "="*60)
    print("🎯 RESUMEN FINAL")
    print("="*60)

    sorted_results = sorted(final_results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)

    for i, (name, result) in enumerate(sorted_results):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"{rank} {name}: {result['test_accuracy']:.4f} de precisión, MCC: {result['matthews_corrcoef']:.4f}")

    # Resumen de significancia estadística
    if mcnemar_results:
        print("\n📊 SIGNIFICANCIA ESTADÍSTICA:")
        print("-" * 30)
        for comp, result in mcnemar_results.items():
            model1, model2 = result['model1'], result['model2']
            p_val = result['p_value']
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            if p_val < 0.05:
                acc1 = results[model1]['test_accuracy']
                acc2 = results[model2]['test_accuracy']
                better = model1 if acc1 > acc2 else model2
                print(f"🔬 {model1} vs {model2}: {better} es significativamente mejor ({significance})")
            else:
                print(f"🔬 {model1} vs {model2}: No hay diferencia significativa ({significance})")

    print(f"\n✅ Proceso completado exitosamente!")
    print(f"💡 El mejor modelo es: {sorted_results[0][0]} con {sorted_results[0][1]['test_accuracy']:.4f} de precisión")

# Ejecutar
if __name__ == "__main__":
    main()      