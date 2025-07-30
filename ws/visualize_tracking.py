import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import cv2
import pickle
import os
from pathlib import Path
import json
from datetime import datetime

# Necesitamos definir la misma clase que usamos en coco_tracking.py
class TrackInfo:
    """Clase auxiliar para almacenar información sobre tracks"""
    def __init__(self, track_id):
        self.id = track_id
        self.positions = []  # Lista de (timestamp, x, y, w, h)
        self.category_id = None
        self.category_name = None
    
    def add_position(self, timestamp, x, y, w, h):
        self.positions.append((timestamp, x, y, w, h))
    
    def set_category(self, category_id, category_name="unknown"):
        self.category_id = category_id
        self.category_name = category_name
    
    def get_movement_stats(self):
        """Calcula estadísticas de movimiento para este track"""
        if len(self.positions) <= 1:
            return 0.0, 0.0
        
        # Calcular movimiento neto (distancia entre primera y última posición)
        first_pos = np.array([self.positions[0][1], self.positions[0][2]])
        last_pos = np.array([self.positions[-1][1], self.positions[-1][2]])
        net_movement = np.linalg.norm(last_pos - first_pos)
        
        # Calcular distancia total recorrida
        total_distance = 0.0
        for i in range(1, len(self.positions)):
            prev_pos = np.array([self.positions[i-1][1], self.positions[i-1][2]])
            curr_pos = np.array([self.positions[i][1], self.positions[i][2]])
            total_distance += np.linalg.norm(curr_pos - prev_pos)
        
        return net_movement, total_distance

def load_tracking_results(results_file):
    """Carga los resultados de tracking desde un archivo pickle."""
    print(f"Cargando resultados de tracking desde {results_file}...")
    with open(results_file, 'rb') as f:
        result_data = pickle.load(f)
    
    # Compatibilidad con versiones anteriores
    if isinstance(result_data, dict) and 'all_tracks' in result_data:
        return result_data['all_tracks'], result_data.get('track_info', {})
    else:
        return result_data, {}

def load_coco_data(coco_file):
    """Carga los datos COCO para mapear image_id a filename."""
    print(f"Cargando datos COCO desde {coco_file}...")
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Crear mapeo de ID a filename
    image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Extraer categorías
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    return coco_data, image_id_to_filename, categories

def visualize_tracking(tracking_results, track_info, image_folder, coco_file, output_video, fps=5, debug=False):
    """
    Visualiza los resultados de tracking sobre las imágenes originales.
    
    Args:
        tracking_results: Lista de tuplas (timestamp, tracks)
        track_info: Diccionario con información adicional de tracks
        image_folder: Carpeta que contiene las imágenes originales
        coco_file: Archivo de anotaciones COCO
        output_video: Ruta del archivo de video a generar
        fps: Frames por segundo para el video
        debug: Si se deben imprimir mensajes de depuración
    """
    # Cargar datos COCO
    coco_data, image_id_to_filename, categories = load_coco_data(coco_file)
    
    # Ordenar imágenes por ID para asegurar el orden correcto
    images = sorted(coco_data['images'], key=lambda x: x['id'])
    
    # Definir colores por categoría
    # Crear un mapa de colores para categorías
    category_colors = {
        1: 'red',       # Frutilla madura
        2: 'green',     # Frutilla verde
        3: 'orange',    # Frutilla semi-madura
        4: 'purple',    # Frutilla muy madura
        5: 'yellow',    # Otra categoría
        6: 'blue',      # Otra categoría
        7: 'cyan',      # Otra categoría
        8: 'magenta',   # Otra categoría
        9: 'brown',     # Otra categoría
        10: 'pink'      # Otra categoría
    }
    
    # Mapeo de nombre de categoría a color
    category_name_to_color = {
        'fruit_ripe': 'red',
        'fruit_unripe': 'green',
        'fruit_semiripe': 'orange',
        'fruit_overripe': 'purple',
        'flower': 'blue'
    }
    
    # Color por defecto para categorías desconocidas
    default_color = 'gray'
    
    # Preparar la figura
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Lista para almacenar los artistas de cada frame
    artists = []
    
    print(f"Generando visualización con {len(tracking_results)} frames...")
    
    # Procesar cada frame
    for i, (timestamp, tracks) in enumerate(tracking_results):
        # Mostrar progreso
        if i % 10 == 0:
            print(f"Procesando frame {i+1}/{len(tracking_results)}")
        
        # Obtener la imagen correspondiente a este frame
        if i < len(images):
            image_info = images[i]
            image_filename = image_info['file_name']
            image_path = os.path.join(image_folder, image_filename)
        else:
            print(f"¡Advertencia! No hay suficientes imágenes para el frame {i}")
            continue
        
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            print(f"¡Error al cargar imagen {image_path}!")
            continue
        
        # Convertir de BGR a RGB para matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Artistas para este frame
        frame_artists = []
        
        # Mostrar imagen (debe ser el primer artista)
        img_artist = ax.imshow(img_rgb, animated=True)
        frame_artists.append(img_artist)
        
        # Añadir información de frame
        title = ax.text(10, 20, f"Frame: {i+1}/{len(tracking_results)} - Tracks: {len(tracks)}", 
                       color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))
        frame_artists.append(title)
        
        # Dibujar tracks SÓLO para este frame
        for track_idx, track in enumerate(tracks):
            track_id = str(track.id)
            
            # Obtener información de categoría
            category_id = None
            category_name = "Unknown"
            
            # Intentar obtener categoría de track_info
            if track_id in track_info:
                info = track_info[track_id]
                category_id = info.category_id
                category_name = info.category_name
            # Si no, intentar obtenerla de los metadatos del track
            elif hasattr(track, 'metadata'):
                if 'category_id' in track.metadata:
                    category_id = track.metadata['category_id']
                    category_name = categories.get(category_id, f"Category {category_id}")
                elif 'class' in track.metadata and 'id' in track.metadata['class']:
                    category_id = track.metadata['class']['id']
                    category_name = track.metadata['class'].get('name', f"Category {category_id}")
            
            # Determinar color basado en categoría
            if category_name in category_name_to_color:
                color = category_name_to_color[category_name]
            elif category_id in category_colors:
                color = category_colors[category_id]
            else:
                color = default_color
            
            # Obtener el estado actual del track
            if len(track.states) == 0:
                continue
                
            state = track.states[-1]
            
            # Extraer bounding box del estado
            try:
                x, vx, y, vy, w, h = state.state_vector.flatten()
            except ValueError:
                # Si el vector de estado tiene un formato diferente
                state_vector = state.state_vector.flatten()
                if len(state_vector) == 4:  # [x, y, w, h]
                    x, y, w, h = state_vector
                else:
                    print(f"Error: Vector de estado inesperado: {state_vector}")
                    continue
            
            # Dibujar bounding box
            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=2, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            frame_artists.append(rect)
            
            # Dibujar ID del track y categoría
            text = ax.text(
                x, y-10, f"ID:{track_id[:8]} - {category_name}", 
                color='black', fontsize=8, 
                bbox=dict(facecolor=color, alpha=0.7, pad=1)
            )
            frame_artists.append(text)
            
            '''
            # Dibujar trayectoria SOLO con los estados hasta este frame
            if track_id in track_info:
                positions = track_info[track_id].positions
                valid_positions = [p for p in positions if p[0] <= timestamp]
                
                if len(valid_positions) > 1:
                    # Extraer los puntos de la trayectoria (centros de los bboxes)
                    traj_x = []
                    traj_y = []
                    
                    for _, px, py, pw, ph in valid_positions[-20:]:  # Últimos 20 estados válidos
                        # Usar el centro del bounding box
                        center_x = px + pw/2
                        center_y = py + ph/2
                        traj_x.append(center_x)
                        traj_y.append(center_y)
                    
                    # Dibujar línea de trayectoria
                    if len(traj_x) > 1:
                        line = ax.plot(traj_x, traj_y, color=color, linewidth=1.5, alpha=0.7)[0]
                        frame_artists.append(line)
            '''
        
        # Añadir timestamp
        time_text = ax.text(
            img_rgb.shape[1] - 200, 20, 
            f"Timestamp: {timestamp.strftime('%H:%M:%S.%f')[:-3]}", 
            color='white', fontsize=8, 
            bbox=dict(facecolor='black', alpha=0.7)
        )
        frame_artists.append(time_text)
        
        # # Añadir firma
        # signature = ax.text(
        #     10, img_rgb.shape[0] - 10,
        #     f"Generado por: MatiSick - {datetime.now().strftime('%Y-%m-%d')}", 
        #     color='white', fontsize=8, 
        #     bbox=dict(facecolor='black', alpha=0.5)
        # )
        # frame_artists.append(signature)
        
        # Añadir artistas de este frame a la lista
        artists.append(frame_artists)
    
    # Crear animación
    print("Generando animación...")
    ani = animation.ArtistAnimation(fig, artists, interval=1000/fps, blit=True, repeat_delay=1000)
    
    # Guardar video
    print(f"Guardando video en {output_video}...")
    ani.save(output_video, writer='ffmpeg', fps=fps)
    
    plt.close(fig)
    print(f"¡Video guardado en {output_video}!")

if __name__ == "__main__":
    # Configurar rutas
    results_file = "/ws/Frutillas/results/tracking/tracking_results.pkl"
    image_folder = "/ws/Frutillas/images/"
    coco_file = "/ws/Frutillas/annotations/annotations_bbox_1_l_week1_30_70.json"
    output_video = "/ws/Frutillas/results/videos/tracking_visualization.mp4"
    
    # Cargar resultados de tracking
    tracking_results, track_info = load_tracking_results(results_file)
    
    # Generar visualización
    visualize_tracking(tracking_results, track_info, image_folder, coco_file, output_video)