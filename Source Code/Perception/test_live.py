# =================================================================
# PROYECTO AISSA - SISTEMA DE DETECCIÓN HUMANA EN TIEMPO REAL (Test)
# =================================================================
# Autor: Enrique Alfonso Gracián Castro
# Fecha: 12 de febrero de 2026
# Institución: Universidad Popular Autónoma del Estado de Puebla
# Descripción: Implementación de inferencia en tiempo real utilizando 
#              el motor YOLOv8 a través de Roboflow Inference Pipeline.
#Modificacion: Miguel Hernandez Camacho 
#Fecha  Modificacion: 13-02-2026
# =================================================================

# 1. Importación de librerías
import cv2
from inference import InferencePipeline
import os
import supervision as sv
import numpy as np


# 2. Definición de la lógica de procesamiento (Callback)
# Modificación Mike: Cambio del nodo box para tener control del dibujo
def procesar_cuadro(result, video_frame):
    """
    Función que se ejecuta por cada cuadro de video procesado.
    Extrae las predicciones y actualiza la salida visual.
    """

    res_dict = result[0] if isinstance(result, list) else result
    detections = res_dict.get('predictions')

    if not isinstance(detections, sv.Detections):
        # Fallback: intentar convertir el resultado crudo a supervision
        try:
            detections = sv.Detections.from_inference(res_dict)
        except:
            print(f"Error en conversión: {e}")
            return

    frame_final = video_frame.image.copy() # Se toma cuadro limpio de la camara

    if detections is not None and len(detections) > 0:
        # Ordenamos por confianza y tomamos la mejor
        idx_max = np.argmax(detections.confidence)  

        best_det = detections[idx_max : idx_max + 1]
        
        # 3. Extraer coordenadas del objeto Supervision
        # best_detection.xyxy devuelve [x_min, y_min, x_max, y_max]
        xyxy = best_det.xyxy[0] 
        conf = best_det.confidence[0]
        
        # Dibujado directo con OpenCV
        p1 = (int(xyxy[0]), int(xyxy[1]))
        p2 = (int(xyxy[2]), int(xyxy[3]))

        cv2.rectangle(frame_final, p1, p2, (0, 0, 255), 3)
        cv2.putText(frame_final, f"PERSONA: {conf:.2%}", (p1[0], p1[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Limpieza de la terminal para visualización de logs en vivo
    #os.system('cls' if os.name == 'nt' else 'clear')
    print("================================")
    print(f"¡SISTEMA AISSA ACTIVO!")
    print(f"Objetivo fijado: {'SI' if len(detections) > 0 else 'NO'}")
    print("================================")

    # 4. Despliegue de la interfaz de usuario
    cv2.imshow("PROYECTO AISSA - Monitoreo UPAEP", frame_final)

    # El comando waitKey permite el refresco de la ventana y el cierre seguro con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nFinalizando conexión con el servidor...")
        os._exit(0)

# 5. Configuración e Inicialización del Pipeline de Inferencia
# Se establecen las credenciales y rutas de conexión verificadas
pipeline = InferencePipeline.init_with_workflow(
    api_key="2e7GrlyT7n1CQRKLRJJi",  # Private API Key del proyecto drones-npsoz
    workspace_name="drones-npsoz",   # Workspace verificado en Roboflow
    workflow_id="aissadetection",    # Workflow ID configurado en la nube
    video_reference=0,               # Referencia a la webcam principal
    on_prediction=procesar_cuadro    # Asignación de la función callback
)

# 6. Ejecución del ciclo principal de monitoreo
try:
    print("Conectando con el sistema de Reconocimiento")
    pipeline.start()
    pipeline.join()
except Exception as e:
    print(f"Error durante la ejecución del sistema: {e}")