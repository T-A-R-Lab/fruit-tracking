import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from copy import deepcopy

# Importar nuestro COCOReader
from coco_reader import COCOReader

# Importaciones de Stone Soup para tracking
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity, RandomWalk
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.types.state import GaussianState
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.tracker.simple import MultiTargetTracker

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

def export_to_mot(tracking_results, output_file, default_class_id=1):
    """
    Exporta resultados de tracking a formato MOT Challenge.
    tracking_results: lista de (timestamp, tracks)
    """
    uuid_to_int = {}
    next_id = 1
    with open(output_file, "w") as f:
        for frame_idx, (timestamp, tracks) in enumerate(tracking_results, 1):
            for track in tracks:
                if len(track.states) == 0:
                    continue
                
                state = track.states[-1]
                # Normalmente [x, vx, y, vy, w, h]
                try:
                    x, vx, y, vy, w, h = state.state_vector.flatten()
                except ValueError:
                    x, y, w, h = state.state_vector.flatten()
                
                # Reasignar ID del track para MOT
                # Si se desea, se podria usar UUID en vez de ints en los MOT
                str_id = str(track.id)
                if str_id not in uuid_to_int:
                    uuid_to_int[str_id] = next_id
                    next_id += 1
                mot_id = uuid_to_int[str_id]
                
                # Obtener class_id si está disponible en track_info
                class_id = default_class_id
                if track.metadata["category_id"] is not None:
                    class_id = track.metadata["category_id"]
                
                not_ignored = 1        # 1 = no ignorar (evaluar siempre)
                visibility = 1         # 1 = completamente visible

                # Formato MOT: frame, id, x, y, w, h, not_ignored, class_id, visibility, -1
                line = f"{frame_idx},{mot_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{not_ignored},{class_id},{visibility},-1\n"
                f.write(line)
    print(f"Archivo MOT exportado en: {output_file}")

def run_tracking(coco_file, image_folder):
    """
    Ejecuta el tracking en datos COCO y retorna los resultados.
    
    Args:
        coco_file: Ruta al archivo JSON de anotaciones COCO
        image_folder: Carpeta que contiene las imágenes
    
    Returns:
        Lista de tuplas (timestamp, tracks)
    """
    print(f"Iniciando tracking con datos COCO...")
    print(f"Archivo COCO: {coco_file}")
    print(f"Carpeta de imágenes: {image_folder}")
    
    # Cargar datos COCO para obtener información de categorías
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)
    
    # Mapear category_id a category_name
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Crear el lector COCO
    reader = COCOReader(
        coco_file=Path(coco_file),
        image_folder=Path(image_folder),
        start_time=datetime.now(),
        time_step=timedelta(seconds=0.1)
    )
    
    # AJUSTE 1: Aumentar el ruido del modelo de transición para permitir más movimiento
    t_models = [
        ConstantVelocity(50**2),  # x, vx - Aumentado para permitir más movimiento
        ConstantVelocity(50**2),  # y, vy
        RandomWalk(30**2),        # w
        RandomWalk(30**2)         # h
    ]
    transition_model = CombinedLinearGaussianTransitionModel(t_models)
    
    # AJUSTE 2: Aumentar el ruido de medición
    measurement_model = LinearGaussian(
        ndim_state=6,             # [x, vx, y, vy, w, h]
        mapping=[0, 2, 4, 5],     # mapeo a [x, y, w, h]
        noise_covar=np.diag([5**2, 5**2, 7**2, 7**2])
    )
    
    # Filtrado Kalman
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    
    # AJUSTE 3: Asociación de datos más flexible
    hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), 30)  # Umbral aumentado
    data_associator = GNNWith2DAssignment(hypothesiser)
    
    # AJUSTE 4: Inicialización de tracks
    prior_state = GaussianState(
        StateVector(np.zeros((6, 1))),
        CovarianceMatrix(np.diag([100**2, 50**2, 100**2, 50**2, 100**2, 100**2]))
    )
    
    # AJUSTE 5: Parámetros de inicialización y eliminación más apropiados
    deleter_init = UpdateTimeStepsDeleter(time_steps_since_update=2)
    initiator = MultiMeasurementInitiator(
        prior_state, 
        deleter_init, 
        data_associator, 
        updater,
        measurement_model, 
        min_points=4  # Menos puntos para iniciar tracks más rápido
    )
    
    # AJUSTE 6: Eliminación de tracks más persistente
    deleter = UpdateTimeStepsDeleter(time_steps_since_update=4)
    
    # Construir el tracker
    tracker = MultiTargetTracker(
        initiator=initiator,
        deleter=deleter,
        detector=reader,
        data_associator=data_associator,
        updater=updater,
    )
    
    # Ejecutar el tracker y almacenar resultados
    all_tracks = []
    all_track_info = {}  # Diccionario para almacenar información de todos los tracks
    total_frames = 0
    
    print("Procesando frames...")
    for timestamp, tracks in tracker:
        total_frames += 1
        
        # Crear copia profunda de los tracks para este frame
        frame_tracks = []
        for track in tracks:
            # Guardar la información de este track
            track_id = str(track.id)
            if track_id not in all_track_info:
                all_track_info[track_id] = TrackInfo(track_id)
                
                # Extraer información de categoría si está disponible
                if hasattr(track, 'metadata'):
                    if 'category_id' in track.metadata:
                        cat_id = track.metadata['category_id']
                        cat_name = categories.get(cat_id, f"Category {cat_id}")
                        all_track_info[track_id].set_category(cat_id, cat_name)
                    elif 'class' in track.metadata and 'id' in track.metadata['class']:
                        cat_id = track.metadata['class']['id']
                        cat_name = track.metadata['class'].get('name', f"Category {cat_id}")
                        all_track_info[track_id].set_category(cat_id, cat_name)
            
            # Registrar la posición actual
            if len(track.states) > 0:
                state = track.states[-1]
                state_vector = state.state_vector
                x, _, y, _, w, h = state_vector.flatten()
                all_track_info[track_id].add_position(timestamp, x, y, w, h)
            
            # Guardar una copia del track para este frame
            frame_tracks.append(deepcopy(track))
        
        # Mostrar progreso
        if total_frames % 10 == 0 or total_frames <= 5:
            print(f"Frame {total_frames}: {len(tracks)} tracks activos")
            if len(tracks) > 0:
                print("  Posiciones actuales:")
                for i, track in enumerate(tracks):
                    if len(track.states) > 0:
                        pos = track.states[-1].state_vector
                        print(f"    Track {track.id}: x={pos[0,0]:.1f}, y={pos[2,0]:.1f}, w={pos[4,0]:.1f}, h={pos[5,0]:.1f}")
        
        all_tracks.append((timestamp, frame_tracks))
    
    print(f"\nTracking completado. Total de frames procesados: {len(all_tracks)}")
    
    # Análisis de resultados
    max_tracks = max(len(tracks) for _, tracks in all_tracks) if all_tracks else 0
    print(f"Número máximo de tracks simultáneos: {max_tracks}")
    print(f"Número total de tracks únicos: {len(all_track_info)}")
    
    # Verificación de movimiento de tracks
    print("\nVerificando movimiento de tracks:")
    for track_id, track_info in all_track_info.items():
        if len(track_info.positions) <= 1:
            continue
            
        first_time = track_info.positions[0][0]
        last_time = track_info.positions[-1][0]
        net_movement, total_distance = track_info.get_movement_stats()
        
        print(f"Track {track_id}: {len(track_info.positions)} frames")
        print(f"  Categoría: {track_info.category_name}")
        print(f"  Movimiento neto: {net_movement:.1f} pixels")
        print(f"  Distancia total: {total_distance:.1f} pixels")
        print(f"  Frames de vida: {first_time} a {last_time}")
    
    return all_tracks, all_track_info

if __name__ == "__main__":
    # Configurar rutas - ajusta estas según tu sistema
    coco_file = "/ws/Frutillas/annotations/annotations_bbox_1_l_week1_30_70.json"
    image_folder = "/ws/Frutillas/images/"
    output_mot_file = "/ws/Frutillas/results/tracking/tracking_results_mot.txt"
    
    # Ejecutar tracking
    tracking_results, _ = run_tracking(coco_file, image_folder)
    
    # Exportar a MOT
    export_to_mot(tracking_results, output_mot_file)
    
    print("Proceso de tracking completado. Archivo MOT generado.")