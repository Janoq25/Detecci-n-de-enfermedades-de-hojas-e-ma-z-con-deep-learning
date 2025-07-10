import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path
import time
import base64
from fpdf import FPDF
import io
from datetime import datetime
import pytz

# Configuración de la página
st.set_page_config(
    page_title="🌽 Detector de Enfermedades en Hojas de Maíz",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #2E8B57;
    }
    .prediction-result {
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .healthy {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .diseased {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .nav-tab {
        background-color: #f8f9fa;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.2rem;
        cursor: pointer;
    }
    .nav-tab:hover {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Configuración de rutas - AJUSTA ESTAS RUTAS SEGÚN TU ESTRUCTURA
MODEL_PATH = "/content/drive/MyDrive/maize-leaf-disease/Models"  # Cambia por tu ruta de modelos
REPORTS_PATH = "/content/drive/MyDrive/maize-leaf-disease/Reports2"  # Cambia por tu ruta de reportes
IMG_SIZE = 128

# Nombres de clases (ajusta según tus clases reales)
CLASS_NAMES = [
    "Mancha gris",
    "Roña común",
    "Tizón del norte",
    "Sano"
]

@st.cache_resource
def load_models():
    """Carga los modelos entrenados"""
    models = {}
    model_files = {
        "MobileNetV2": "MobileNetV2.h5",
        "ResNet50": "ResNet50.h5",
        "EfficientNetB0": "EfficientNetB0.h5"
    }

    for name, filename in model_files.items():
        model_path = os.path.join(MODEL_PATH, filename)
        if os.path.exists(model_path):
            try:
                models[name] = load_model(model_path)
                st.success(f"✅ Modelo {name} cargado exitosamente")
            except Exception as e:
                st.error(f"❌ Error cargando {name}: {str(e)}")
        else:
            st.warning(f"⚠️ No se encontró el archivo: {model_path}")

    return models

def check_report_files():
    """Verifica la existencia de archivos de reportes"""
    reports_path = Path(REPORTS_PATH)

    expected_files = {
        "Comparación General": "modelos_comparacion_completa.png",
        "Matrices Combinadas": "matrices_confusion_todos.png",
        "Matriz MobileNetV2": "matriz_confusion_mobilenetv2.png",
        "Matriz ResNet50": "matriz_confusion_resnet50.png",
        "Matriz EfficientNetB0": "matriz_confusion_efficientnetb0.png",
        "Métricas Detalladas": "metricas_detalladas_por_clase.png",
        "Análisis McNemar": "mcnemar_analysis.png",
        "Reporte Completo": "reporte_completo.txt"
    }

    existing_files = {}
    for name, filename in expected_files.items():
        file_path = reports_path / filename
        existing_files[name] = file_path.exists()

    return existing_files, reports_path

def preprocess_image(image, model_name):
    """Preprocesa la imagen según el modelo"""
    # Redimensionar imagen
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convertir a array y expandir dimensiones
    image_array = np.array(image_resized, dtype=np.float32)
    image_expanded = np.expand_dims(image_array, axis=0)

    # Aplicar preprocesamiento específico del modelo
    if model_name == "MobileNetV2":
        return mobilenet_preprocess(image_expanded)
    elif model_name == "ResNet50":
        return resnet_preprocess(image_expanded)
    elif model_name == "EfficientNetB0":
        return efficientnet_preprocess(image_expanded)
    else:
        return image_expanded / 255.0

def predict_disease(image, models):
    """Realiza predicciones con todos los modelos"""
    predictions = {}

    for model_name, model in models.items():
        # Preprocesar imagen
        processed_image = preprocess_image(image, model_name)

        # Realizar predicción
        pred = model.predict(processed_image, verbose=0)
        pred_class_idx = np.argmax(pred[0])
        pred_class = CLASS_NAMES[pred_class_idx]
        confidence = float(pred[0][pred_class_idx])

        predictions[model_name] = {
            'class': pred_class,
            'confidence': confidence,
            'probabilities': pred[0]
        }

    return predictions

def get_peru_time():
    """Obtiene la fecha y hora actual en zona horaria de Perú"""
    peru_tz = pytz.timezone('America/Lima')
    peru_time = datetime.now(peru_tz)
    return peru_time

def clean_text_for_pdf(text):
    """Limpia el texto eliminando caracteres especiales incompatibles con latin-1"""
    import unicodedata

    # Reemplazos específicos
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N',
        'ü': 'u', 'Ü': 'U',
        '°': ' grados', '–': '-', '—': '-',
        ''': "'", ''': "'", '"': '"', '"': '"',
        '€': 'EUR', '£': 'GBP', '¥': 'YEN',
        '©': '(c)', '®': '(R)', '™': '(TM)',
        # Emojis comunes por si quedan algunos
        '🌽': '[MAIZ]', '📊': '[GRAFICO]', '📋': '[INFO]',
        '🔍': '[BUSCAR]', '⚠️': '[ALERTA]', '✅': '[OK]',
        '❌': '[ERROR]', '🟢': '[VERDE]', '🔴': '[ROJO]',
        '💡': '[IDEA]', '📷': '[IMAGEN]', '🤖': '[ROBOT]'
    }

    # Aplicar reemplazos
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Normalizar y convertir a ASCII
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text


def generate_pdf_report(image, predictions, uploaded_filename, consensus_reached, consensus_diagnosis):
    """Genera un reporte PDF mejorado con gráficas y mejor estructura"""

    peru_time = get_peru_time()

    # Limpiar texto de entrada
    uploaded_filename = clean_text_for_pdf(uploaded_filename)
    if consensus_diagnosis:
        consensus_diagnosis = clean_text_for_pdf(consensus_diagnosis)

    class PDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=15)

        def header(self):
            # Logo o símbolo (puedes personalizar)
            self.set_font('Arial', 'B', 20)
            self.set_text_color(46, 139, 87)  # Verde
            self.cell(0, 15, 'DIAGNOSTICO FITOSANITARIO - MAIZ', 0, 1, 'C')

            self.set_font('Arial', 'I', 12)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, 'Sistema de Deteccion Automatica de Enfermedades', 0, 1, 'C')

            # Línea separadora
            self.set_draw_color(46, 139, 87)
            self.line(10, 35, 200, 35)
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f'Pagina {self.page_no()} | Generado el {peru_time.strftime("%Y-%m-%d %H:%M:%S")} (Hora Peru)', 0, 0, 'C')

        def chapter_title(self, title, icon=""):
            self.ln(5)
            self.set_font('Arial', 'B', 16)
            self.set_text_color(46, 139, 87)
            self.cell(0, 12, f'{icon} {title}', 0, 1, 'L')
            self.set_draw_color(46, 139, 87)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(8)

        def section_title(self, title, icon=""):
            self.ln(3)
            self.set_font('Arial', 'B', 12)
            self.set_text_color(70, 70, 70)
            self.cell(0, 8, f'{icon} {title}', 0, 1, 'L')
            self.ln(2)

        def normal_text(self, text, bold=False):
            self.set_font('Arial', 'B' if bold else '', 10)
            self.set_text_color(0, 0, 0)
            self.cell(0, 6, text, 0, 1, 'L')

        def info_box(self, title, content, bg_color=(240, 248, 255)):
            # Guardar posición actual
            x, y = self.get_x(), self.get_y()

            # Dibujar fondo
            self.set_fill_color(*bg_color)
            self.rect(x, y, 190, len(content.split('\n')) * 5 + 15, 'F')

            # Título del box
            self.set_font('Arial', 'B', 11)
            self.set_text_color(25, 25, 112)
            self.cell(0, 8, title, 0, 1, 'L')

            # Contenido
            self.set_font('Arial', '', 9)
            self.set_text_color(0, 0, 0)
            for line in content.split('\n'):
                if line.strip():
                    self.cell(0, 5, f"  {line.strip()}", 0, 1, 'L')
            self.ln(5)

        def add_consensus_result(self, consensus_reached, consensus_diagnosis):
            if consensus_reached:
                if consensus_diagnosis == "Sano":
                    bg_color = (212, 237, 218)  # Verde claro
                    title = "[OK] DIAGNOSTICO: HOJA SALUDABLE"
                else:
                    bg_color = (248, 215, 218)  # Rojo claro
                    title = f"[!] DIAGNOSTICO: {consensus_diagnosis.upper()}"
            else:
                bg_color = (255, 243, 205)  # Amarillo claro
                title = "[?] SIN CONSENSO ENTRE MODELOS"

            self.set_fill_color(*bg_color)
            self.rect(10, self.get_y(), 190, 15, 'F')

            self.set_font('Arial', 'B', 14)
            self.set_text_color(0, 0, 0)
            self.cell(0, 15, title, 0, 1, 'C')
            self.ln(5)

    # Crear PDF
    pdf = PDF()
    pdf.add_page()

    # 1. INFORMACIÓN GENERAL
    pdf.chapter_title("INFORMACION DEL ANALISIS", "[INFO]")

    pdf.normal_text(f"Archivo: {uploaded_filename}", bold=True)
    pdf.normal_text(f"Fecha y hora: {peru_time.strftime('%Y-%m-%d %H:%M:%S')} (Hora Peru)")
    pdf.normal_text(f"Modelos utilizados: MobileNetV2, ResNet50, EfficientNetB0")
    pdf.normal_text(f"Resolucion de procesamiento: {IMG_SIZE}x{IMG_SIZE} pixeles")

    # 2. DIAGNÓSTICO PRINCIPAL
    pdf.chapter_title("DIAGNOSTICO PRINCIPAL", "[DIAG]")
    pdf.add_consensus_result(consensus_reached, consensus_diagnosis)

    # 3. IMAGEN ANALIZADA
    pdf.chapter_title("IMAGEN ANALIZADA", "[IMG]")

    try:
        # Guardar imagen temporalmente
        image_pil = Image.fromarray(image)
        temp_img_path = f"temp_analysis_img_{int(peru_time.timestamp())}.png"
        image_pil.save(temp_img_path, format='PNG')

        # Calcular dimensiones para centrar la imagen
        img_width = 80
        page_width = 190
        x_position = (page_width - img_width) / 2 + 10

        pdf.image(temp_img_path, x=x_position, w=img_width)
        pdf.ln(60)

        # Información de la imagen
        pdf.section_title("Detalles de la imagen:", "[i]")
        pdf.normal_text(f"- Tamano original: {image_pil.size[0]}x{image_pil.size[1]} pixeles")
        pdf.normal_text(f"- Formato: {image_pil.format if hasattr(image_pil, 'format') else 'Unknown'}")
        pdf.normal_text(f"- Canales de color: RGB")

        # Limpiar archivo temporal de imagen
        try:
            os.remove(temp_img_path)
        except:
            pass

    except Exception as e:
        pdf.normal_text(f"[Error al procesar la imagen: {e}]")
        pdf.ln(10)

    # 4. RESULTADOS DETALLADOS POR MODELO
    pdf.add_page()
    pdf.chapter_title("RESULTADOS DETALLADOS", "[MODELS]")

    # Crear gráficas para cada modelo
    temp_graph_paths = []

    try:
        for i, (model_name, pred) in enumerate(predictions.items()):
            # Crear gráfica individual para cada modelo
            fig, ax = plt.subplots(figsize=(8, 5))

            # Configurar colores
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#2E8B57']
            bars = ax.bar(CLASS_NAMES, pred['probabilities'], color=colors, alpha=0.8)

            # Personalizar gráfica
            ax.set_title(f'Predicciones del Modelo {model_name}', fontsize=14, fontweight='bold', pad=20)
            ax.set_ylabel('Probabilidad', fontsize=12)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')

            # Resaltar la predicción más alta
            max_idx = np.argmax(pred['probabilities'])
            bars[max_idx].set_color('#2E8B57')
            bars[max_idx].set_alpha(1.0)

            # Añadir valores en las barras
            for j, v in enumerate(pred['probabilities']):
                ax.text(j, v + 0.02, f'{v:.1%}', ha='center', va='bottom',
                       fontsize=10, fontweight='bold')

            # Rotar etiquetas del eje x
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Guardar gráfica temporalmente
            temp_graph_path = f"temp_graph_{model_name}_{int(peru_time.timestamp())}.png"
            plt.savefig(temp_graph_path, dpi=150, bbox_inches='tight')
            temp_graph_paths.append(temp_graph_path)
            plt.close()

        # Añadir las gráficas al PDF
        for i, (model_name, pred) in enumerate(predictions.items()):
            pdf.section_title(f"Modelo {model_name}", "[M]")

            # Información del modelo
            confidence_level = "ALTA" if pred['confidence'] > 0.8 else "MEDIA" if pred['confidence'] > 0.6 else "BAJA"

            pdf.info_box(
                f"Resultado del Modelo {model_name}",
                f"Prediccion: {clean_text_for_pdf(pred['class'])}\n"
                f"Confianza: {pred['confidence']:.2%} ({confidence_level})\n"
                f"Estado: {'[OK] Saludable' if pred['class'] == 'Sano' else '[!] Enfermedad detectada'}"
            )

            # Añadir gráfica
            if i < len(temp_graph_paths) and os.path.exists(temp_graph_paths[i]):
                pdf.image(temp_graph_paths[i], x=25, w=160)
                pdf.ln(80)

            # Tabla de probabilidades
            pdf.section_title("Probabilidades por clase:", "[DATA]")
            for j, class_name in enumerate(CLASS_NAMES):
                prob = pred['probabilities'][j]
                marker = "=>" if j == np.argmax(pred['probabilities']) else "  "
                pdf.normal_text(f"{marker} {clean_text_for_pdf(class_name)}: {prob:.2%}")

            pdf.ln(5)

            # Añadir nueva página si no es el último modelo
            if i < len(predictions) - 1:
                pdf.add_page()

    except Exception as e:
        pdf.normal_text(f"Error generando gráficas: {e}")

    finally:
        # Limpiar archivos temporales de gráficas
        for temp_path in temp_graph_paths:
            try:
                os.remove(temp_path)
            except:
                pass

    # 5. ANÁLISIS COMPARATIVO
    pdf.add_page()
    pdf.chapter_title("ANALISIS COMPARATIVO", "[COMP]")

    # Tabla resumen
    pdf.section_title("Resumen de predicciones:", "[SUM]")
    pdf.normal_text("Modelo                Prediccion           Confianza    Estado")
    pdf.normal_text("-" * 65)

    for model_name, pred in predictions.items():
        status = "[OK] Sana" if pred['class'] == 'Sano' else "[!] Enferma"
        clean_class = clean_text_for_pdf(pred['class'])
        line = f"{model_name:<15} {clean_class:<15} {pred['confidence']:>8.1%}    {status}"
        pdf.normal_text(line)

    pdf.ln(8)

    # Análisis de consenso
    if consensus_reached:
        pdf.info_box(
            "[OK] Consenso Alcanzado",
            f"Los tres modelos coinciden en el diagnostico: {consensus_diagnosis}\n"
            f"Esto indica alta confiabilidad en el resultado.\n"
            f"Nivel de acuerdo: 100% (3/3 modelos)"
        )
    else:
        predictions_list = [pred['class'] for pred in predictions.values()]
        unique_predictions = list(set(predictions_list))

        consensus_text = "Los modelos presentan diferentes diagnosticos:\n"
        for pred in unique_predictions:
            count = predictions_list.count(pred)
            clean_pred = clean_text_for_pdf(pred)
            consensus_text += f"- {clean_pred}: {count} modelo(s)\n"
        consensus_text += "Se recomienda analisis adicional para confirmar."

        pdf.info_box("[!] Sin Consenso", consensus_text)

    # 6. RECOMENDACIONES
    pdf.chapter_title("RECOMENDACIONES", "[REC]")

    if consensus_reached:
        if consensus_diagnosis == "Sano":
            recommendations = [
                "- Continuar con las practicas de manejo actuales",
                "- Realizar monitoreos preventivos regulares cada 7-10 dias",
                "- Mantener condiciones optimas de cultivo (riego, fertilizacion)",
                "- Implementar rotacion de cultivos para prevenir enfermedades",
                "- Vigilar plantas circundantes por posibles sintomas"
            ]
        else:
            recommendations = [
                "- Consultar inmediatamente con un especialista en fitopatologia",
                "- Aislar las plantas afectadas si es posible",
                "- Implementar medidas de control especificas para la enfermedad",
                "- Monitorear la extension de la enfermedad en el cultivo",
                "- Considerar tratamientos preventivos en plantas cercanas",
                "- Documentar la evolucion con fotografias regulares",
                "- Revisar condiciones ambientales que favorecen la enfermedad"
            ]
    else:
        recommendations = [
            "- Tomar una nueva imagen con mejor calidad e iluminacion",
            "- Asegurar que la hoja este bien centrada y enfocada",
            "- Consultar con un especialista para confirmacion visual",
            "- Realizar analisis de laboratorio si persisten sintomas",
            "- Considerar multiples muestras de diferentes partes de la planta"
        ]

    for rec in recommendations:
        pdf.normal_text(rec)

    # 7. INFORMACIÓN SOBRE ENFERMEDADES
    if consensus_reached and consensus_diagnosis != "Sano":
        pdf.add_page()
        pdf.chapter_title("INFORMACION ESPECIFICA", "[DISEASE]")

        disease_details = {
            "Tizon del norte": {
                "descripcion": "Enfermedad fungica causada por Exserohilum turcicum que afecta principalmente las hojas del maiz.",
                "sintomas": [
                    "- Lesiones alargadas en forma de cigarro",
                    "- Color marron grisaceo con bordes definidos",
                    "- Pueden alcanzar varios centimetros de longitud",
                    "- Amarillamiento prematuro de hojas",
                    "- En casos severos, marchitez de la planta"
                ],
                "condiciones": "Favorecido por alta humedad (>90%) y temperaturas de 18-27C",
                "tratamiento": [
                    "- Aplicacion de fungicidas especificos (azoles, estrobilurinas)",
                    "- Uso de variedades resistentes",
                    "- Rotacion de cultivos con especies no susceptibles",
                    "- Manejo de residuos de cosecha",
                    "- Espaciamiento adecuado para mejorar ventilacion"
                ]
            },
            "Rona común": {
                "descripcion": "Enfermedad fungica causada por Puccinia sorghi que produce pustulas caracteristicas en las hojas.",
                "sintomas": [
                    "- Pustulas pequenas y circulares de color marron-rojizo",
                    "- Aparecen en ambas caras de la hoja",
                    "- Pueden coalescer formando areas grandes",
                    "- Amarillamiento prematuro del follaje",
                    "- Reduccion en el vigor de la planta"
                ],
                "condiciones": "Temperaturas moderadas (16-25C) y presencia de rocio matutino",
                "tratamiento": [
                    "- Fungicidas preventivos antes de la aparicion de sintomas",
                    "- Variedades con genes de resistencia",
                    "- Eliminacion de hospederos alternativos",
                    "- Monitoreo temprano y control oportuno",
                    "- Aplicacion foliar de productos cupricos"
                ]
            },
            "Mancha gris": {
                "descripcion": "Enfermedad fungica causada por Cercospora zeae-maydis que produce manchas caracteristicas en las hojas.",
                "sintomas": [
                    "- Manchas rectangulares de color gris a marron",
                    "- Delimitadas por las venas de las hojas",
                    "- Pueden desarrollar un halo amarillento",
                    "- Coalescencia causa muerte de tejido foliar",
                    "- Afecta principalmente hojas inferiores"
                ],
                "condiciones": "Alta humedad relativa y temperaturas calidas (25-30C)",
                "tratamiento": [
                    "- Rotacion con cultivos no gramineas",
                    "- Aplicacion de fungicidas sistemicos",
                    "- Manejo de densidad de siembra",
                    "- Eliminacion de residuos infectados",
                    "- Mejoramiento de drenaje del suelo"
                ]
            }
        }

        if consensus_diagnosis in disease_details:
            details = disease_details[consensus_diagnosis]

            pdf.section_title(f"Enfermedad: {consensus_diagnosis}", "[PATHOGEN]")
            pdf.normal_text(details['descripcion'])
            pdf.ln(3)

            pdf.section_title("Sintomas caracteristicos:", "[SYMP]")
            for sintoma in details['sintomas']:
                pdf.normal_text(sintoma)
            pdf.ln(3)

            pdf.section_title("Condiciones favorables:", "[ENV]")
            pdf.normal_text(details['condiciones'])
            pdf.ln(3)

            pdf.section_title("Estrategias de manejo:", "[TREAT]")
            for tratamiento in details['tratamiento']:
                pdf.normal_text(tratamiento)

    # 8. INFORMACIÓN TÉCNICA Y DISCLAIMER
    pdf.add_page()
    pdf.chapter_title("INFORMACION TECNICA", "[TECH]")

    pdf.section_title("Especificaciones del sistema:", "[SPEC]")
    tech_info = [
        "- Modelos basados en transfer learning con redes neuronales convolucionales",
        "- Dataset de entrenamiento: PlantVillage Corn Leaf Disease",
        "- Arquitecturas: MobileNetV2, ResNet50, EfficientNetB0",
        "- Precision promedio en validacion: >95%",
        "- Resolucion de procesamiento: 128x128 pixeles",
        "- Preprocesamiento especifico por modelo aplicado",
        "- Analisis basado en caracteristicas visuales de la hoja"
    ]

    for info in tech_info:
        pdf.normal_text(info)

    pdf.ln(8)
    pdf.info_box(
        "[!] IMPORTANTE - LIMITACIONES Y DISCLAIMER",
        "- Este analisis automatizado debe ser validado por un profesional\n"
        "- La precision del diagnostico depende de la calidad de la imagen\n"
        "- Se recomienda tomar multiples muestras para mayor certeza\n"
        "- Este sistema es una herramienta de apoyo, no un sustituto del diagnostico profesional\n"
        "- En caso de dudas, consulte con un fitopatologo certificado\n"
        "- Los resultados pueden variar segun condiciones de iluminacion y enfoque"
    )

    # 9. PIE DE PÁGINA CON INFORMACIÓN DE CONTACTO
    pdf.ln(10)
    pdf.section_title("Informacion del sistema:", "[SYS]")
    pdf.normal_text("Sistema de Deteccion Automatica de Enfermedades en Maiz")
    pdf.normal_text(f"Version: 2.0 | Fecha de generacion: {peru_time.strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.normal_text("Desarrollado con tecnologia de Deep Learning")

    # Generar PDF final
    try:
        pdf_output = pdf.output(dest='S')
        return pdf_output.encode('latin-1') if isinstance(pdf_output, str) else pdf_output
    except Exception as e:
        # Método alternativo para versiones más nuevas
        temp_pdf_path = f"temp_report_{int(peru_time.timestamp())}.pdf"
        pdf.output(temp_pdf_path)

        with open(temp_pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        try:
            os.remove(temp_pdf_path)
        except:
            pass

        return pdf_bytes


def plot_predictions(predictions):
    """Crea gráficos de las predicciones"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (model_name, pred) in enumerate(predictions.items()):
        ax = axes[idx]

        # Crear gráfico de barras
        bars = ax.bar(CLASS_NAMES, pred['probabilities'])
        ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probabilidad')
        ax.set_ylim(0, 1)

        # Colorear la barra de la predicción más alta
        max_idx = np.argmax(pred['probabilities'])
        bars[max_idx].set_color('#2E8B57')

        # Rotar etiquetas del eje x
        ax.tick_params(axis='x', rotation=45)

        # Añadir valores en las barras
        for i, v in enumerate(pred['probabilities']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig

def show_prediction_interface(models):
    """Muestra la interfaz de predicción"""
    st.markdown("## 📤 Cargar Imagen")
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de una hoja de maíz",
        type=['png', 'jpg', 'jpeg'],
        help="Formatos soportados: PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        # Mostrar imagen cargada
        col1, col2 = st.columns([1, 2])

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)

            # Información de la imagen
            st.markdown("### 📋 Información de la imagen")
            st.write(f"**Nombre:** {uploaded_file.name}")
            st.write(f"**Tamaño:** {image.size}")
            st.write(f"**Formato:** {image.format}")

        with col2:
            # Convertir a array numpy para procesamiento
            image_array = np.array(image.convert('RGB'))

            # Realizar predicciones
            st.markdown("## 🔍 Realizando Predicciones...")

            with st.spinner('Procesando imagen con los modelos...'):
                predictions = predict_disease(image_array, models)

            # Mostrar resultados
            st.markdown("## 📊 Resultados de Predicción")

            # Crear tarjetas de resultados
            for model_name, pred in predictions.items():
                is_healthy = pred['class'] == 'Sano'
                card_class = "healthy" if is_healthy else "diseased"

                st.markdown(f"""
                <div class="model-card">
                    <h3>🤖 {model_name}</h3>
                    <div class="prediction-result {card_class}">
                        Predicción: {pred['class']} ({pred['confidence']:.2%} confianza)
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Gráficos de probabilidades
            st.markdown("## 📈 Probabilidades por Modelo")
            fig = plot_predictions(predictions)
            st.pyplot(fig)

            # Tabla resumen
            st.markdown("## 📋 Resumen de Resultados")
            summary_data = []
            for model_name, pred in predictions.items():
                summary_data.append({
                    'Modelo': model_name,
                    'Predicción': pred['class'],
                    'Confianza': f"{pred['confidence']:.2%}",
                    'Estado': '🟢 Sana' if pred['class'] == 'Sano' else '🔴 Enferma'
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)

            # Consenso de modelos
            st.markdown("## 🎯 Consenso de Modelos")
            predictions_list = [pred['class'] for pred in predictions.values()]
            unique_predictions = list(set(predictions_list))

            consensus_reached = len(unique_predictions) == 1
            consensus_diagnosis = unique_predictions[0] if consensus_reached else None

            if consensus_reached:
                st.success(f"✅ **Consenso alcanzado:** Todos los modelos predicen '{consensus_diagnosis}'")
            else:
                st.warning("⚠️ **Sin consenso:** Los modelos tienen predicciones diferentes")
                for pred in unique_predictions:
                    count = predictions_list.count(pred)
                    st.write(f"- {pred}: {count} modelo(s)")

            # Botón para generar reporte PDF
            st.markdown("## 📄 Generar Reporte")

            col_pdf1, col_pdf2 = st.columns([1, 2])

            with col_pdf1:
                if st.button("📥 Generar Reporte PDF", type="primary", use_container_width=True):
                    with st.spinner("Generando reporte PDF..."):
                        try:
                            pdf_bytes = generate_pdf_report(
                                image=image_array,
                                predictions=predictions,
                                uploaded_filename=uploaded_file.name,
                                consensus_reached=consensus_reached,
                                consensus_diagnosis=consensus_diagnosis
                            )

                            # Generar nombre del archivo con timestamp de Perú
                            peru_time = get_peru_time()
                            timestamp = peru_time.strftime("%Y%m%d_%H%M%S")
                            pdf_filename = f"reporte_maiz_{timestamp}.pdf"

                            # Botón de descarga
                            st.download_button(
                                label="📥 Descargar Reporte PDF",
                                data=pdf_bytes,
                                file_name=pdf_filename,
                                mime="application/pdf",
                                use_container_width=True
                            )

                            st.success(f"✅ Reporte PDF generado exitosamente! ({peru_time.strftime('%H:%M:%S')} - Hora Perú)")

                        except Exception as e:
                            st.error(f"❌ Error al generar el PDF: {str(e)}")
                            st.info("💡 Asegúrate de que las librerías estén instaladas: `pip install fpdf2 pytz`")

            with col_pdf2:
                st.info("""
                **📋 El reporte PDF incluye:**
                - Imagen analizada
                - Diagnóstico de cada modelo
                - Probabilidades detalladas
                - Consenso entre modelos
                - Recomendaciones específicas
                - Información técnica
                """)

            # Información adicional sobre el diagnóstico
            if consensus_reached:
                st.markdown("## 💡 Información sobre el Diagnóstico")

                if consensus_diagnosis == "Sano":
                    st.success("""
                    **Hoja Saludable Detectada**

                    La hoja analizada no presenta signos visibles de enfermedad.
                    Continúe con las prácticas de manejo actuales y mantenga
                    un monitoreo preventivo regular.
                    """)
                else:
                    disease_info = {
                        "Tizón del norte": {
                            "description": "Enfermedad fúngica que causa lesiones alargadas de color marrón.",
                            "recommendations": "Aplicar fungicidas, mejorar ventilación, evitar humedad excesiva."
                        },
                        "Roña común": {
                            "description": "Enfermedad fúngica que produce pústulas de color marrón-rojizo.",
                            "recommendations": "Usar variedades resistentes, aplicar fungicidas preventivos."
                        },
                        "Mancha gris": {
                            "description": "Enfermedad que causa manchas grises rectangulares en las hojas.",
                            "recommendations": "Rotación de cultivos, manejo de residuos, fungicidas específicos."
                        }
                    }

                    if consensus_diagnosis in disease_info:
                        info = disease_info[consensus_diagnosis]
                        st.warning(f"""
                        **{consensus_diagnosis} Detectado**

                        **Descripción:** {info['description']}

                        **Recomendaciones:** {info['recommendations']}

                        ⚠️ *Consulte con un especialista en fitopatología para confirmar el diagnóstico y obtener un plan de tratamiento específico.*
                        """)
            else:
                st.info("""
                **🔍 Análisis Adicional Recomendado**

                Los modelos no alcanzaron consenso. Esto puede deberse a:
                - Calidad de la imagen
                - Estadio temprano de la enfermedad
                - Condiciones de iluminación

                Recomendamos tomar una nueva fotografía con mejor iluminación
                o consultar con un especialista.
                """)

def show_training_reports():
    """Muestra los reportes de entrenamiento"""
    st.header("📊 Reportes de Entrenamiento")
    st.markdown("Visualización completa de todos los reportes generados durante el entrenamiento y evaluación de los modelos.")

    existing_files, reports_path = check_report_files()

    # Mostrar estado de archivos
    with st.expander("📁 Estado de Archivos de Reportes"):
        for file, exists in existing_files.items():
            if exists:
                st.success(f"✅ {file}")
            else:
                st.error(f"❌ {file} - No encontrado")

    # Sección de tiempos de entrenamiento
    st.subheader("⏱️ Tiempos de Entrenamiento")

    # Crear métricas de tiempo
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🚀 MobileNetV2",
            value="46.97 min",
            delta="Más rápido",
            delta_color="normal"
        )
        st.caption("2,818 segundos total")

    with col2:
        st.metric(
            label="⚡ EfficientNetB0",
            value="55.61 min",
            delta="Moderado",
            delta_color="normal"
        )
        st.caption("3,337 segundos total")

    with col3:
        st.metric(
            label="🎯 ResNet50",
            value="162.8 min",
            delta="Más lento",
            delta_color="inverse"
        )
        st.caption("9,768 segundos total")

    # Gráfico de tiempos
    st.markdown("#### 📊 Comparación Visual de Tiempos")

    # Datos de tiempo
    time_data = {
        'Modelo': ['MobileNetV2', 'EfficientNetB0', 'ResNet50'],
        'Tiempo (min)': [46.97, 55.61, 162.8],
        'Eficiencia': ['Alta', 'Media-Alta', 'Baja']
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E8B57', '#FFA500', '#DC143C']  # Verde, naranja, rojo
    bars = ax.bar(time_data['Modelo'], time_data['Tiempo (min)'], color=colors, alpha=0.7)

    ax.set_ylabel('Tiempo de Entrenamiento (minutos)')
    ax.set_title('Tiempo de Entrenamiento por Modelo')
    ax.grid(True, alpha=0.3, axis='y')

    # Añadir valores en las barras
    for bar, tiempo in zip(bars, time_data['Tiempo (min)']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{tiempo:.1f} min', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    # Análisis de eficiencia
    st.markdown("""
    **📋 Análisis de Eficiencia:**
    - **MobileNetV2**: Entrenamiento más rápido, ideal para prototipado
    - **EfficientNetB0**: Buen balance tiempo/rendimiento
    - **ResNet50**: Entrenamiento más lento pero mayor precisión final
    """)

    # 1. Comparación General de Modelos
    st.subheader("🏆 Comparación General de Modelos")
    comparison_file = reports_path / "modelos_comparacion_completa.png"
    if comparison_file.exists():
        st.image(str(comparison_file), caption="Comparación de precisión y pérdida entre los tres modelos")
    else:
        st.error("Archivo de comparación no encontrado")

    # 2. Matrices de Confusión Combinadas
    st.subheader("🔍 Matrices de Confusión - Vista Comparativa")
    matrices_file = reports_path / "matrices_confusion_todos.png"
    if matrices_file.exists():
        st.image(str(matrices_file), caption="Matrices de confusión de los tres modelos lado a lado")
    else:
        st.error("Archivo de matrices combinadas no encontrado")

    # 3. Matrices Individuales
    st.subheader("🔎 Matrices de Confusión - Detalle Individual")

    matrix_files = {
        "MobileNetV2": "matriz_confusion_mobilenetv2.png",
        "CNN ResNet50": "matriz_confusion_resnet50.png",
        "CNN EfficientNetB0": "matriz_confusion_efficientnetb0.png"
    }

    cols = st.columns(3)
    for idx, (model_name, filename) in enumerate(matrix_files.items()):
        with cols[idx]:
            matrix_path = reports_path / filename
            if matrix_path.exists():
                st.image(str(matrix_path), caption=f"Matriz - {model_name}")
            else:
                st.error(f"Matriz de {model_name} no encontrada")

    # 4. Métricas Detalladas
    st.subheader("📈 Métricas Detalladas por Clase")
    metrics_file = reports_path / "metricas_detalladas_por_clase.png"
    if metrics_file.exists():
        st.image(str(metrics_file), caption="Análisis detallado de Precision, Recall y F1-Score por clase y modelo")
    else:
        st.error("Archivo de métricas detalladas no encontrado")

    # 4. Métricas Detalladas
    st.subheader("📈 Pruebas de McNemar")
    metrics_file = reports_path / "mcnemar_analysis.png"
    if metrics_file.exists():
        st.image(str(metrics_file), caption="Pruebas de mcnemar y tablas de contingencia")
    else:
        st.error("Archivo de analisis de mcnemar no encontrado")

    # 5. Reporte de Texto
    st.subheader("📄 Reporte Detallado en Texto")
    text_report_path = reports_path / "reporte_completo.txt"
    if text_report_path.exists():
        with open(text_report_path, 'r', encoding='utf-8') as f:
            report_content = f.read()
        st.text_area("Reporte Completo", report_content, height=400)

        # Botón de descarga
        st.download_button(
            label="📥 Descargar Reporte Completo",
            data=report_content,
            file_name="reporte_maiz_completo.txt",
            mime="text/plain"
        )
    else:
        st.error("Archivo de reporte de texto no encontrado")

def show_model_comparison():
    """Muestra la comparación entre modelos"""
    st.header("🔬 Comparación de Modelos")

    # Información general sobre los modelos
    st.markdown("""
    ### 🤖 Modelos Implementados

    **MobileNetV2:**
    - Arquitectura optimizada para dispositivos móviles
    - Menos parámetros y mayor velocidad
    - Ideal para aplicaciones en tiempo real

    **ResNet50:**
    - Arquitectura con conexiones residuales
    - Excelente para tareas de clasificación complejas
    - Mayor precisión en datasets desafiantes

    **EfficientNetB0:**
    - Arquitectura optimizada para eficiencia
    - Balance entre precisión y velocidad
    - Escalamiento uniforme de ancho, profundidad y resolución
    """)

    # Crear tabla comparativa con tiempos de entrenamiento
    st.subheader("📊 Tabla Comparativa")
    comparison_data = {
        "Característica": [
            "Parámetros (aprox.)",
            "Tiempo de entrenamiento",
            "Velocidad de inferencia",
            "Precisión final",
            "Val Accuracy final",
            "Uso de memoria",
            "Mejor para"
        ],
        "MobileNetV2": [
            "3.5M",
            "46.97 min (2,818 seg)",
            "Muy rápida",
            "99.21%",
            "93.52%",
            "Bajo",
            "Aplicaciones móviles"
        ],
        "ResNet50": [
            "25M",
            "162.8 min (9,768 seg)",
            "Moderada",
            "99.54%",
            "98.83%",
            "Alto",
            "Precisión máxima"
        ],
        "EfficientNetB0": [
            "5.3M",
            "55.61 min (3,337 seg)",
            "Rápida",
            "98.42%",
            "98.19%",
            "Moderado",
            "Balance eficiencia/precisión"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)

    # Análisis de rendimiento por tiempo
    st.subheader("⏱️ Análisis de Eficiencia Temporal")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🥇 Mejor eficiencia tiempo/precisión:**
        - **MobileNetV2**: Entrenamiento más rápido con buena precisión
        - Ideal para desarrollo iterativo rápido

        **🏆 Mejor precisión absoluta:**
        - **ResNet50**: Máxima precisión de validación (98.83%)
        - Tiempo considerable pero resultados superiores
        """)

    with col2:
        st.markdown("""
        **⚖️ Mejor balance:**
        - **EfficientNetB0**: Buen balance tiempo/precisión
        - Precisión alta con tiempo moderado

        **📊 Ratio eficiencia:**
        - MobileNetV2: 33.2% precisión/minuto
        - EfficientNetB0: 17.7% precisión/minuto
        - ResNet50: 6.1% precisión/minuto
        """)

    # Gráfico de tiempo vs precisión
    st.subheader("📈 Tiempo de Entrenamiento vs Precisión")

    # Datos para el gráfico
    models_data = {
        'Modelo': ['MobileNetV2', 'EfficientNetB0', 'ResNet50'],
        'Tiempo (minutos)': [46.97, 55.61, 162.8],
        'Val Accuracy (%)': [93.52, 98.19, 98.83],
        'Training Accuracy (%)': [99.21, 98.42, 99.54]
    }

    # Crear gráfico con matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico 1: Tiempo vs Val Accuracy
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    ax1.scatter(models_data['Tiempo (minutos)'], models_data['Val Accuracy (%)'],
               c=colors, s=200, alpha=0.7)
    ax1.set_xlabel('Tiempo de Entrenamiento (minutos)')
    ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('Tiempo vs Precisión de Validación')
    ax1.grid(True, alpha=0.3)

    # Añadir etiquetas
    for i, model in enumerate(models_data['Modelo']):
        ax1.annotate(model,
                    (models_data['Tiempo (minutos)'][i], models_data['Val Accuracy (%)'][i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Gráfico 2: Comparación de barras
    x = np.arange(len(models_data['Modelo']))
    width = 0.35

    ax2.bar(x - width/2, models_data['Training Accuracy (%)'], width,
           label='Training Accuracy', color='lightcoral', alpha=0.8)
    ax2.bar(x + width/2, models_data['Val Accuracy (%)'], width,
           label='Validation Accuracy', color='skyblue', alpha=0.8)

    ax2.set_xlabel('Modelos')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Comparación de Precisiones')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models_data['Modelo'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

def main():
    # Encabezado principal
    st.markdown('<h1 class="main-header">🌽 Detector de Enfermedades en Hojas de Maíz</h1>',
                unsafe_allow_html=True)

    # Navegación con tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Predicción", "📊 Reportes de Entrenamiento", "🔬 Comparación de Modelos"])

    with tab1:
        st.markdown("""
        Esta aplicación utiliza tres modelos de deep learning para detectar enfermedades en hojas de maíz:
        - **MobileNetV2**: Modelo eficiente y rápido
        - **ResNet50**: Modelo robusto con conexiones residuales
        - **EfficientNetB0**: Modelo optimizado para eficiencia
        """)

        # Cargar modelos
        st.markdown("## 🤖 Cargando Modelos...")
        models = load_models()

        if not models:
            st.error("❌ No se pudieron cargar los modelos. Verifica las rutas.")
        else:
            st.success(f"✅ {len(models)} modelo(s) cargado(s) exitosamente")
            show_prediction_interface(models)

    with tab2:
        show_training_reports()

    with tab3:
        show_model_comparison()

    # Sidebar con información
    st.sidebar.markdown("## 📊 Información de la App")
    st.sidebar.markdown("""
    **Clases detectables:**
    - 🟢 Sano
    - 🔴 Tizón del norte
    - 🟠 Roña común
    - 🟡 Mancha gris
    """)

    st.sidebar.markdown("## 📋 Instrucciones")
    st.sidebar.markdown("""
    1. Ve a la pestaña "Predicción"
    2. Carga una imagen de una hoja de maíz
    3. Espera a que se procese
    4. Revisa las predicciones de los tres modelos
    5. Analiza las probabilidades
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Configuración")
    if st.sidebar.button("🔄 Recargar Modelos"):
        st.cache_resource.clear()
        st.rerun()

    if st.sidebar.button("📁 Verificar Archivos"):
        existing_files, _ = check_report_files()
        files_found = sum(existing_files.values())
        total_files = len(existing_files)
        st.sidebar.success(f"Archivos encontrados: {files_found}/{total_files}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Notas")
    st.sidebar.markdown("""
    - Sube imágenes claras de hojas
    - Resolución recomendada: 224x224px
    - Formatos: JPG, PNG
    - Para mejores resultados, centra la hoja en la imagen
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        🌽 Desarrollado para el análisis de enfermedades en cultivos de maíz<br>
        Utiliza modelos de deep learning entrenados con transfer learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()  