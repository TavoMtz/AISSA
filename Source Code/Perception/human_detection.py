# =================================================================
# PROYECTO AISSA - DETECCIÓN HUMANA EN FOTOGRAFÍA (TEST)
# =================================================================
# Autor: 
# Fecha: 12 de febrero de 2026
# Institución: Universidad Popular Autónoma del Estado de Puebla
# Descripción: Procesamiento de imágenes estáticas mediante el uso 
#              de Workflows de Roboflow y la librería Supervision.
# =================================================================

# 1. Importación de librerías
import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient
import os

# 2. Configuración del entorno
# Crear el directorio de salida para las imágenes procesadas si no existe
if not os.path.exists('imagen_procesada'): 
    os.makedirs('imagen_procesada')

# 3. Conexión al motor de inferencia (Roboflow Workflow)
# Se utiliza el cliente HTTP para comunicación Serverless con la nube
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="KSxvi4i7RLOpoqV20nnc" # API Key del entorno de pruebas
)

# 4. Definición de recursos
# Ruta de la imagen de prueba
image_path = "test_image_1.jpg"

try:
    # 5. Ejecución del flujo de trabajo (Workflow)
    # Se envía la imagen al workspace 'drones-npsoz' para su análisis
    results = client.run_workflow(
        workspace_name="drones-npsoz",
        workflow_id="aissadetection",
        images={"image": image_path},
        use_cache=True
    )
    
    # 6. Extracción y validación de resultados
    # Obtenemos el diccionario de salidas para localizar los datos de la IA
    output_dict = results[0].get('outputs', {})

    # Búsqueda dinámica del campo 'predictions' dentro de los nodos del flujo
    if 'predictions' in output_dict:
        predictions = output_dict['predictions']
        print("¡Datos de predicción encontrados exitosamente!")
    else:
        # Fallback: búsqueda en el primer nodo disponible si el nombre no coincide
        node_name = list(output_dict.keys())[0] if output_dict else None
        print(f"Buscando datos en el nodo: {node_name}")
        predictions = output_dict[node_name].get('predictions', []) if node_name else []

    # 7. Conversión a formato de Supervision
    # Transformamos las coordenadas de la IA en un objeto detectable por la librería
    detections = sv.Detections.from_inference({"predictions": predictions})

    # 8. Carga y renderizado de la imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error crítico: No se pudo cargar la imagen {image_path}")
        exit()

    # Inicialización de herramientas de anotación profesional
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # Dibujo de cuadros delimitadores y etiquetas sobre la imagen
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # 9. Almacenamiento y despliegue de resultados
    output_path = "./imagen_procesada/test_image_processed.jpg"
    cv2.imwrite(output_path, annotated_frame)
    print(f"\nProcesamiento finalizado. Imagen guardada en: {output_path}")

    # Mostrar la ventana con las detecciones
    cv2.imshow("Human Detection - AISSA UPAEP", annotated_frame)
    print("Presione cualquier tecla para cerrar la ventana.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f"Sistema: Procesamiento completado exitosamente.")

except Exception as e:
    print(f"Error crítico ejecutando el workflow: {e}")