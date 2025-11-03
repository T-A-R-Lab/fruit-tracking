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
from stonesoup.functions import compute_box_ious  # Función para calcular IoU

class TrackInfo:
    """Clase auxiliar para almacenar información sobre tracks"""
    def __init__(self, track_id):
        self.id = track_id
        self.positions = []  # Lista de (timestamp, x, y, w, h)
        self.category_id = None
        self.category_name = None
        self.gt_positions = []  # Lista de posiciones ground truth
        self.ious = []  # IoU entre las bboxes predichas y las ground truth
    
    def add_position(self, timestamp, x, y, w, h):
        self.positions.append((timestamp, x, y, w, h))
    
    def add_gt_position(self, timestamp, x, y, w, h):
        self.gt_positions.append((timestamp, x, y, w, h))
    
    def add_iou(self, iou):
        self.ious.append(iou)
    
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

    def get_mean_iou(self):
        """Retorna el IoU promedio para este track"""
        if not self.ious:
            return 0.0
        return sum(self.ious) / len(self.ious)


def export_to_mot(tracking_results, output_file, default_class_id=1, include_gt=False):
    """
    Exporta resultados de tracking a formato MOT Challenge.
    tracking_results: lista de (timestamp, tracks)
    include_gt: Si es True, incluye ground truth en archivo separado
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
                str_id = str(track.id)
                if str_id not in uuid_to_int:
                    uuid_to_int[str_id] = next_id
                    next_id += 1
                mot_id = uuid_to_int[str_id]
                
                # Obtener class_id si está disponible en track_info
                class_id = default_class_id
                if hasattr(track, 'metadata') and 'category_id' in track.metadata:
                    class_id = track.metadata['category_id']
                
                not_ignored = 1        # 1 = no ignorar (evaluar siempre)
                visibility = 1         # 1 = completamente visible

                # Formato MOT: frame, id, x, y, w, h, not_ignored, class_id, visibility, -1
                line = f"{frame_idx},{mot_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{not_ignored},{class_id},{visibility},-1\n"
                f.write(line)
    
    # Si se solicita, exportar ground truth a un archivo separado
    if include_gt:
        gt_output_file = output_file.replace('.txt', '_gt.txt')
        with open(gt_output_file, "w") as f:
            for frame_idx, (timestamp, gt_tracks) in enumerate(tracking_results, 1):
                for track in gt_tracks:
                    if not hasattr(track, 'gt_state') or track.gt_state is None:
                        continue
                    
                    x, y, w, h = track.gt_state.flatten()
                    mot_id = uuid_to_int.get(str(track.id), 0)
                    
                    class_id = default_class_id
                    if hasattr(track, 'metadata') and 'category_id' in track.metadata:
                        class_id = track.metadata['category_id']
                    
                    line = f"{frame_idx},{mot_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,{class_id},1,-1\n"
                    f.write(line)
        print(f"Archivo MOT de ground truth exportado en: {gt_output_file}")
    
    print(f"Archivo MOT exportado en: {output_file}")


def calculate_iou(box1, box2):
    """
    Calcula IoU entre dos bounding boxes en formato [x, y, width, height]
    Donde (x,y) es la esquina superior izquierda
    """
    # Convertir a formato [x1, y1, x2, y2] (esquina sup. izq. a inf. der.)
    b1_x1, b1_y1 = box1[0], box1[1]
    b1_x2, b1_y2 = box1[0] + box1[2], box1[1] + box1[3]
    
    b2_x1, b2_y1 = box2[0], box2[1]
    b2_x2, b2_y2 = box2[0] + box2[2], box2[1] + box2[3]
    
    # Intersección
    x_left = max(b1_x1, b2_x1)
    y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2)
    y_bottom = min(b1_y2, b2_y2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Áreas de cada bbox
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # IoU
    iou = intersection_area / float(b1_area + b2_area - intersection_area)
    
    return iou


def find_matching_gt(timestamp, track_id, gt_paths):
    """
    Encuentra el estado de ground truth correspondiente a un track en un timestamp específico
    """
    # Primero intentar encontrar por track_id si existe
    for path in gt_paths:
        if str(path.id) == str(track_id):
            # Buscar el estado con el timestamp más cercano
            closest_state = None
            min_time_diff = float('inf')
            
            for state in path.states:
                time_diff = abs((state.timestamp - timestamp).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_state = state
            
            # Si encontramos un estado y está suficientemente cerca en tiempo
            if closest_state is not None and min_time_diff < 0.5:  # Umbral de 0.5 segundos
                return closest_state
    
    # Si no encontramos por ID, buscar por proximidad espacial y temporal
    # Esto podría implementarse si se necesita
    
    return None


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
    
    # Obtener ground truth paths
    gt_paths = list(reader.ground_truth_paths_gen())
    print(f"Obtenidos {len(gt_paths)} caminos de ground truth")
    
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
                
                # Obtener coordenadas de la bbox del tracker
                try:
                    x, _, y, _, w, h = state_vector.flatten()
                except ValueError:
                    # Manejar casos donde el vector de estado tiene diferente tamaño
                    x, y, w, h = state_vector.flatten()
                
                # Registrar posición
                all_track_info[track_id].add_position(timestamp, x, y, w, h)
                
                # Buscar ground truth correspondiente
                gt_state = find_matching_gt(timestamp, track.id, gt_paths)
                
                # Agregar gt_state al track para exportación
                track.gt_state = None
                
                # Si encontramos un ground truth correspondiente
                if gt_state is not None:
                    gt_vector = gt_state.state_vector
                    gt_x, gt_y, gt_w, gt_h = gt_vector.flatten()
                    
                    # Guardar posición de ground truth
                    all_track_info[track_id].add_gt_position(timestamp, gt_x, gt_y, gt_w, gt_h)
                    
                    # Calcular IoU entre bbox predecida y ground truth
                    iou = calculate_iou([x, y, w, h], [gt_x, gt_y, gt_w, gt_h])
                    all_track_info[track_id].add_iou(iou)
                    
                    # Guardar gt_state para exportación
                    track.gt_state = gt_vector
            
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
    
    # Verificación de movimiento de tracks y comparación con ground truth
    print("\nVerificando movimiento de tracks y comparación con ground truth:")
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
        
        # Mostrar información de ground truth si está disponible
        if track_info.gt_positions:
            print(f"  Ground Truth disponible: {len(track_info.gt_positions)} frames")
            if track_info.ious:
                print(f"  IoU promedio: {track_info.get_mean_iou():.4f}")
    
    # Generar estadísticas generales de IoU
    all_ious = [iou for track_info in all_track_info.values() for iou in track_info.ious]
    if all_ious:
        print(f"\nEstadísticas de IoU global:")
        print(f"  IoU promedio: {sum(all_ious) / len(all_ious):.4f}")
        print(f"  IoU mínimo: {min(all_ious):.4f}")
        print(f"  IoU máximo: {max(all_ious):.4f}")
    else:
        print("\nNo se encontraron correspondencias con ground truth para calcular IoU")
    
    return all_tracks, all_track_info


def export_results_with_gt(tracking_results, track_info, output_file):
    """
    Exporta resultados de tracking junto con ground truth a un archivo CSV
    """
    with open(output_file, "w") as f:
        # Escribir encabezado
        f.write("frame,track_id,pred_x,pred_y,pred_w,pred_h,gt_x,gt_y,gt_w,gt_h,iou,category\n")
        
        for frame_idx, (timestamp, tracks) in enumerate(tracking_results, 1):
            for track in tracks:
                track_id = str(track.id)
                if track_id not in track_info:
                    continue
                
                info = track_info[track_id]
                
                # Encontrar posición en este frame
                pred_pos = None
                for pos in info.positions:
                    if pos[0] == timestamp:
                        pred_pos = pos
                        break
                
                if pred_pos is None:
                    continue
                
                # Encontrar ground truth en este frame
                gt_pos = None
                for pos in info.gt_positions:
                    if pos[0] == timestamp:
                        gt_pos = pos
                        break
                
                # Valores predichos
                pred_x, pred_y, pred_w, pred_h = pred_pos[1], pred_pos[2], pred_pos[3], pred_pos[4]
                
                # Valores de ground truth (si están disponibles)
                if gt_pos:
                    gt_x, gt_y, gt_w, gt_h = gt_pos[1], gt_pos[2], gt_pos[3], gt_pos[4]
                    iou = calculate_iou([pred_x, pred_y, pred_w, pred_h], [gt_x, gt_y, gt_w, gt_h])
                else:
                    gt_x, gt_y, gt_w, gt_h = "", "", "", ""
                    iou = ""
                
                # Escribir línea
                f.write(f"{frame_idx},{track_id},{pred_x},{pred_y},{pred_w},{pred_h},")
                f.write(f"{gt_x},{gt_y},{gt_w},{gt_h},{iou},{info.category_name}\n")
    
    print(f"Resultados con ground truth exportados en: {output_file}")


if __name__ == "__main__":
    # Configurar rutas - ajusta estas según tu sistema
    coco_file = "/ws/Frutillas/annotations/annotations_bbox_1_l_week1_30_70.json"
    image_folder = "/ws/Frutillas/images/"
    output_mot_file = "/ws/Frutillas/results/tracking/tracking_results_mot.txt"
    output_csv_file = "/ws/Frutillas/results/tracking/tracking_with_gt.csv"
    
    # Ejecutar tracking
    tracking_results, track_info = run_tracking(coco_file, image_folder)
    
    # Exportar a MOT
    export_to_mot(tracking_results, output_mot_file, include_gt=True)
    
    # Exportar resultados con ground truth
    export_results_with_gt(tracking_results, track_info, output_csv_file)
    
    print("Proceso de tracking completado. Archivos generados.")