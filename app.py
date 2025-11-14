from flask import Flask, render_template, request, jsonify
# import tensorflow as tf  # ← quitar import duro de TF
import numpy as np
from PIL import Image
import io
import base64
import json

# Resolver backend TFLite sin requerir TF completo
try:
    from tflite_runtime.interpreter import Interpreter  # primero tflite-runtime
    TF_BACKEND = "tflite-runtime"
except Exception:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
        TF_BACKEND = "tensorflow"
    except Exception as e:
        raise ImportError(
            "Instala 'tflite-runtime' (recomendado) o un TensorFlow compatible con tu Python."
        ) from e

app = Flask(__name__)

# Cargar el modelo TensorFlow Lite
print("Cargando modelo TensorFlow Lite...", TF_BACKEND)
interpreter = Interpreter(model_path="models/terrain_segmentation_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Modelo cargado correctamente")
print(f"Input shape: {input_details[0]['shape']}")

def estimate_height_from_rgb(rgb_image):
    """
    Estimación mejorada de altura basada en características del terreno.
    """
    img_array = np.array(rgb_image, dtype=np.float32)
    
    # Estrategia mejorada para distinguir agua vs terreno
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    # 1. Detectar agua (baja altura) - agua suele ser azul/oscura
    water_likelihood = (b / (r + g + b + 1e-6)) * 0.6
    
    # 2. Detectar vegetación/terreno (alta altura) - verde brillante
    terrain_likelihood = (g / (r + g + b + 1e-6)) * 0.4
    
    # 3. Combinar - agua baja altura, vegetación alta altura
    height_map = terrain_likelihood - water_likelihood + 0.5
    
    # Normalizar a [0, 1]
    height_map = np.clip(height_map, 0, 1)
    
    # Suavizar
    try:
        from scipy import ndimage
        height_map = ndimage.gaussian_filter(height_map, sigma=1)
    except ImportError:
        # Fallback si scipy no está disponible
        import cv2
        height_map = cv2.GaussianBlur(height_map, (5, 5), 1)
    
    return height_map

def preprocess_image(image, target_size=(128, 128)):
    """Preprocesar imagen para el modelo"""
    # Redimensionar
    image = image.resize(target_size)
    
    # Convertir a numpy array y normalizar RGB
    rgb_array = np.array(image, dtype=np.float32) / 255.0
    
    # Estimar mapa de altura mejorado
    height_map = estimate_height_from_rgb(image)
    height_map = np.expand_dims(height_map, axis=-1)
    
    # Combinar RGB + Height (4 canales)
    img_with_height = np.concatenate([rgb_array, height_map], axis=-1)
    
    # Agregar dimensión de batch
    img_with_height = np.expand_dims(img_with_height, axis=0)
    
    return img_with_height

def predict_terrain(image):
    """Realizar predicción con el modelo - CORREGIDO"""
    input_data = preprocess_image(image)
    
    # Establecer entrada
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Ejecutar inferencia
    interpreter.invoke()
    
    # Obtener predicción - CORRECCIÓN: Copiar los datos en lugar de usar referencia
    output_data = interpreter.get_tensor(output_details[0]['index']).copy()
    prediction = np.argmax(output_data[0], axis=-1)
    
    # Debug
    unique, counts = np.unique(prediction, return_counts=True)
    class_names = {0: 'Normal', 1: 'Inundación', 2: 'Depresión'}
    print("Distribución de predicción:")
    for class_id, count in zip(unique, counts):
        percentage = (count / prediction.size) * 100
        print(f"   {class_names[class_id]}: {count} píxeles ({percentage:.1f}%)")
    
    return prediction

def create_colored_mask(prediction):
    """Convertir predicción a máscara coloreada"""
    # VERIFICAR SI NECESITAMOS INTERCAMBIAR CLASES
    inundation_ratio = np.sum(prediction == 1) / prediction.size
    
    if inundation_ratio > 0.6:
        # Mapeo CORRECTO (como en entrenamiento)
        color_map = {
            0: [255, 165, 0],    # Normal -> Naranja
            1: [0, 255, 255],    # Inundación -> Cian
            2: [255, 0, 255]     # Depresión -> Magenta
        }
        print("Usando mapeo de colores: Normal(Verde), Inundación(Azul)")
    else:
        # Mapeo INTERCAMBIADO (si el modelo aprendió al revés)
        color_map = {
            0: [255, 165, 0],    # Normal -> Naranja
            1: [0, 255, 255],    # Inundación -> Cian
            2: [255, 0, 255]     # Depresión -> Magenta
        }
        print("Usando mapeo de colores: Normal(Verde), Inundación(Azul)")
    
    colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        colored_mask[prediction == class_id] = color
    
    return colored_mask

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        image = Image.open(file.stream).convert('RGB')
        print(f"📷 Imagen cargada: {image.size}")
        
        # Realizar predicción
        prediction = predict_terrain(image)
        
        # Crear máscara coloreada (con detección automática de mapeo)
        colored_mask = create_colored_mask(prediction)
        
        # Convertir a base64
        mask_img = Image.fromarray(colored_mask)
        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Imagen original redimensionada
        display_size = (300, 300)
        display_img = image.resize(display_size)
        buffered_orig = io.BytesIO()
        display_img.save(buffered_orig, format="PNG")
        orig_base64 = base64.b64encode(buffered_orig.getvalue()).decode()
        
        # Estadísticas
        total_pixels = prediction.size
        stats = {
            'normal': int(np.sum(prediction == 0)),
            'inundacion': int(np.sum(prediction == 1)),
            'depresion': int(np.sum(prediction == 2)),
            'total': total_pixels
        }
        
        percentages = {
            'normal': f"{(stats['normal'] / total_pixels) * 100:.1f}%",
            'inundacion': f"{(stats['inundacion'] / total_pixels) * 100:.1f}%",
            'depresion': f"{(stats['depresion'] / total_pixels) * 100:.1f}%"
        }
        
        return jsonify({
            'success': True,
            'original_image': f"data:image/png;base64,{orig_base64}",
            'prediction_mask': f"data:image/png;base64,{mask_base64}",
            'statistics': stats,
            'percentages': percentages
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test_mapping', methods=['GET'])
def test_mapping():
    """Endpoint para probar diferentes mapeos de colores"""
    # Crear imagen de prueba con distribución similar al entrenamiento
    test_pred = np.zeros((128, 128), dtype=np.uint8)
    h, w = test_pred.shape
    
    # Distribución similar a entrenamiento: ~27% Normal, ~68% Inundación, ~5% Depresión
    test_pred[:int(h*0.27), :] = 0                    # Normal
    test_pred[int(h*0.27):int(h*0.95), :] = 1         # Inundación  
    test_pred[int(h*0.95):, :] = 2                    # Depresión
    
    # Probar ambos mapeos
    mappings = [
        {
            'name': 'Mapeo A (Normal=Verde, Inundación=Azul)',
            'colors': {0: [34, 139, 34], 1: [0, 0, 255], 2: [139, 69, 19]}
        },
        {
            'name': 'Mapeo B (Normal=Azul, Inundación=Verde)', 
            'colors': {0: [0, 0, 255], 1: [34, 139, 34], 2: [139, 69, 19]}
        }
    ]
    
    results = []
    for mapping in mappings:
        colored_mask = np.zeros((*test_pred.shape, 3), dtype=np.uint8)
        for class_id, color in mapping['colors'].items():
            colored_mask[test_pred == class_id] = color
        
        mask_img = Image.fromarray(colored_mask)
        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        results.append({
            'name': mapping['name'],
            'image': f"data:image/png;base64,{mask_base64}",
            'distribution': {
                'normal': f"{np.sum(test_pred == 0) / test_pred.size * 100:.1f}%",
                'inundacion': f"{np.sum(test_pred == 1) / test_pred.size * 100:.1f}%", 
                'depresion': f"{np.sum(test_pred == 2) / test_pred.size * 100:.1f}%"
            }
        })
    
    return jsonify({'test_results': results})

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de salud para verificar que el modelo funciona"""
    try:
        # Probar con una imagen pequeña
        test_image = Image.new('RGB', (128, 128), color='green')
        prediction = predict_terrain(test_image)
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'prediction_shape': prediction.shape,
            'message': 'Modelo funcionando correctamente'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'model_loaded': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)