import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
import cv2

from coco_reader import COCOReader

def visualize_coco_ground_truth(coco_file, image_folder, output_video_path, fps=30):
    """
    Visualiza el ground truth de un archivo COCO y lo guarda como video usando matplotlib.
    
    Args:
        coco_file: Ruta al archivo JSON de anotaciones COCO
        image_folder: Carpeta que contiene las imágenes
        output_video_path: Ruta donde guardar el video resultante
        fps: Frames por segundo del video resultante
    """
    # Cargar el lector COCO
    reader = COCOReader(
        coco_file=Path(coco_file),
        image_folder=Path(image_folder),
        start_time=datetime.now(),
        time_step=timedelta(seconds=1.0/fps)
    )
    
    # Cargar datos COCO para obtener información de imágenes
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Crear mapeo de ID de imagen a nombre de archivo
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Colores para diferentes categorías (hasta 10 categorías)
    colors = [
        'blue',
        'green',
        'red',
        'cyan',
        'magenta',
        'yellow',
        'black',
        'purple',
        'orange',
        'brown'
    ]
    
    # Mapeo de ID de imagen a detecciones
    image_to_detections = {}
    
    # Recopilar todas las detecciones primero
    for timestamp, detections in reader.detections_gen():
        for detection in detections:
            image_id = detection.metadata['image_id']
            if image_id not in image_to_detections:
                image_to_detections[image_id] = []
            image_to_detections[image_id].append(detection)
    
    # Ordenar las IDs de imágenes para procesarlas en orden
    image_ids = sorted(image_to_detections.keys())
    
    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Lista para almacenar los artistas de cada frame
    artists = []
    
    # Procesar cada imagen
    print(f"Procesando imágenes y generando video en {output_video_path}...")
    
    for image_id in image_ids:
        # Cargar imagen
        file_name = image_id_to_filename[image_id]
        image_path = os.path.join(image_folder, file_name)
        
        # Leer imagen con OpenCV y convertir de BGR a RGB
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print(f"¡Advertencia! No se pudo cargar la imagen: {image_path}")
            continue
        
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Artistas para este frame
        frame_artists = []
        
        # Mostrar imagen
        img_artist = ax.imshow(rgb_image, animated=True)
        ax.set_title(f"Frame: {file_name}")
        ax.set_axis_off()
        
        frame_artists.append(img_artist)
        
        # Dibujar las detecciones para esta imagen
        for detection in image_to_detections[image_id]:
            # Obtener valores de la caja
            x, y, w, h = detection.state_vector
            
            # Obtener información de categoría para elegir un color
            category_id = detection.metadata.get('category_id', 0)
            color = colors[category_id % len(colors)]
            
            # Dibujar caja
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            frame_artists.append(rect)
            
            # Añadir etiqueta
            category_name = detection.metadata.get('category_name', 'unknown')
            object_id = detection.metadata.get('object_id', '')
            label = f"{category_name} ({object_id})"
            text = ax.text(x, y - 10, label, color=color, fontsize=8, 
                          bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            frame_artists.append(text)
        
        # Añadir artistas de este frame a la lista
        artists.append(frame_artists)
        
        # Mostrar progreso
        print(f"Procesada imagen ID: {image_id} - {file_name}")
    
    # Crear animación
    ani = animation.ArtistAnimation(fig, artists, interval=1000/fps, blit=True, repeat_delay=1000)
    
    # Guardar video
    ani.save(output_video_path, writer='ffmpeg', fps=fps)
    
    plt.close(fig)
    print(f"¡Video guardado en {output_video_path}!")

if __name__ == "__main__":
    coco_file = "/ws/Frutillas/annotations/annotations_bbox_1_l_week1_30_70.json"
    image_folder = "/ws/Frutillas/images/"
    output_video = "/ws/Frutillas/results/videos/ground_truth_visualization.mp4"
    
    # Generar video
    visualize_coco_ground_truth(coco_file, image_folder, output_video, fps=5)

