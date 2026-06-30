# =================================================================
# PROYECTO AISSA - DETECCIÓN HUMANA EN FOTOGRAFÍA (TEST)
# =================================================================
# Autor: 
# Fecha: 12 de febrero de 2026
# Institución: Universidad Popular Autónoma del Estado de Puebla
# Descripción: Procesamiento de imágenes estáticas utilizando el 
#              SDK de Roboflow para validación de modelos YOLOv8.
# =================================================================

# 1. Importación de librerías
import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient
import os

# 2. Configuración de conexión al servidor de Roboflow
# Se utiliza el cliente HTTP para comunicación Serverless
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="KSxvi4i7RLOpoqV20nnc" # API Key de entorno de pruebas
)

# 3. Definición de recursos y rutas
image_path = "test_image_1.jpg" # Imagen de prueba con 5 personas

try:
    # 4. Ejecución del Workflow en la nube
    # Se envía la imagen al espacio de trabajo drones-npsoz
    results = client.run_workflow(
        workspace_name="drones-npsoz",
        workflow_id="aissadetection",
        images={"image": image_path}
    )

    # 5. Extracción y validación de predicciones
    # Accedemos directamente a la llave 'predictions' definida en el JSON de Roboflow
    predictions = results[0].get('predictions', [])
    print(f"--- Análisis de Imagen Completado ---")
    print(f"Objetos detectados por el sensor: {len(predictions)}")

    # 6. Procesamiento visual de resultados
    # Cargamos la imagen original para realizar el dibujo de cuadros (Anclaje NumPy)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error crítico: No se pudo localizar el archivo {image_path}")
        exit()

    # Solo se procede al dibujo si existen detecciones confirmadas
    if len(predictions) > 0:
        # Conversión de formato Roboflow a Supervision para renderizado
        detections = sv.Detections.from_inference({"predictions": predictions})
        
        # Inicialización de anotadores profesionales
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        # Aplicación de capas visuales sobre la imagen original
        image = box_annotator.annotate(scene=image, detections=detections)
        image = label_annotator.annotate(scene=image, detections=detections)
    else:
        print("Aviso: El motor YOLOv8 no encontró patrones humanos en esta imagen.")

    # 7. Visualización final de la prueba
    cv2.imshow("Deteccion UPAEP - Validacion de Fotografia", image)
    
    # El sistema espera la pulsación de cualquier tecla para cerrar la ventana
    print("Presione cualquier tecla en la ventana de imagen para finalizar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error crítico durante la inferencia: {e}")