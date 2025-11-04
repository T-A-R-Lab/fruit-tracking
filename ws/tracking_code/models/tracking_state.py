"""
Estado del tracking - Mantiene el estado actual del seguimiento de objetos
IMPORTANTE: Las bounding boxes SIEMPRE provienen del ground truth (COCO)
El filtro de Kalman se usa SOLO para predecir posición y asociar IDs, NO para modificar bboxes
"""
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

# Stone Soup imports
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.state import GaussianState
from stonesoup.types.detection import Detection as StoneSoupDetection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.types.array import StateVector, CovarianceMatrix

from models.coco_reader import COCOReader
from models.id_manager import IDManager
from config import TRACKER_CONFIG, MOT_DEFAULTS


@dataclass
class Detection:
    """Representa una detección en un frame"""
    bbox: List[float]  # [x, y, width, height] - SIEMPRE del ground truth
    track_id: int
    category_id: int
    category_name: str
    visibility: int = 1  # 0=oculto, 1=visible
    annotation_id: int = -1


@dataclass
class KalmanTrack:
    """
    Track con filtro de Kalman para predicción de posición usando Stone Soup
    IMPORTANTE: La bbox mostrada es SIEMPRE del GT, Kalman solo para asociación
    """
    track_id: int
    category_id: int
    
    # Estado de Stone Soup: GaussianState con [x, vx, y, vy]
    state: GaussianState = None
    
    # Última bbox del ground truth asociada
    last_gt_bbox: List[float] = field(default_factory=list)
    
    # Frames desde la última actualización
    frames_since_update: int = 0


class KalmanFilter:
    """Filtro de Kalman usando Stone Soup con modelo 2D completo para tracking"""
    
    def __init__(self):
        # Modelo de transición 2D: [x, vx, y, vy] con velocidad constante
        # CombinedLinearGaussianTransitionModel combina dos modelos ConstantVelocity
        self.transition_model = CombinedLinearGaussianTransitionModel([
            ConstantVelocity(5.0),  # Para x con menos ruido
            ConstantVelocity(5.0)   # Para y con menos ruido
        ])
        
        # Modelo de medición: observamos [x, y] 
        # Estado es [x, vx, y, vy], observamos posiciones (índices 0 y 2)
        self.measurement_model = LinearGaussian(
            ndim_state=4,
            mapping=[0, 2],  # mapeo a x (índice 0) e y (índice 2)
            noise_covar=np.diag([2.0, 2.0])  # Bajo ruido de medición
        )
        
        # Predictor y actualizador de Kalman de Stone Soup
        self.predictor = KalmanPredictor(self.transition_model)
        self.updater = KalmanUpdater(self.measurement_model)
        
        # Hypothesiser para asociación de datos usando distancia de Mahalanobis
        self.hypothesiser = DistanceHypothesiser(
            predictor=self.predictor,
            updater=self.updater,
            measure=Mahalanobis(),
            missed_distance=30.0  # Distancia máxima para asociación
        )
        
        # Data associator usando GNN (Global Nearest Neighbor)
        self.data_associator = GNNWith2DAssignment(self.hypothesiser)
    
    def predict(self, track: KalmanTrack) -> np.ndarray:
        """
        Predice el siguiente estado del track usando Stone Soup
        
        Returns:
            Estado predicho [x, vx, y, vy]
        """
        from datetime import timedelta, datetime
        
        # Si el track no tiene timestamp, asignar uno
        if track.state.timestamp is None:
            track.state = GaussianState(
                track.state.state_vector,
                track.state.covar,
                timestamp=datetime.now()
            )
        
        # Calcular nuevo timestamp (1 segundo después del anterior)
        new_timestamp = track.state.timestamp + timedelta(seconds=1)
        
        # Predicción usando Stone Soup con timestamp nuevo
        predicted_state = self.predictor.predict(track.state, timestamp=new_timestamp)
        track.state = predicted_state
        
        return track.state.state_vector.flatten()
    
    def update(self, track: KalmanTrack, measurement: np.ndarray):
        """
        Actualiza el track con una nueva medición (bbox del GT) usando Stone Soup
        
        Args:
            measurement: [x_center, y_center] (solo posición)
        """
        from datetime import datetime
        
        # Asegurar que el track tenga timestamp
        if track.state.timestamp is None:
            track.state = GaussianState(
                track.state.state_vector,
                track.state.covar,
                timestamp=datetime.now()
            )
        
        # Crear detección de Stone Soup (debe ser vector columna) con timestamp
        detection = StoneSoupDetection(
            measurement.reshape(-1, 1),
            timestamp=track.state.timestamp
        )
        
        # Crear hipótesis (asociación entre predicción y detección)
        hypothesis = SingleHypothesis(track.state, detection)
        
        # Actualizar usando Stone Soup
        updated_state = self.updater.update(hypothesis)
        track.state = updated_state
        
        # Guardar bbox del ground truth (con ancho y alto)
        track.frames_since_update = 0
    
    def initialize_track(self, bbox: List[float], track_id: int, category_id: int) -> KalmanTrack:
        """
        Inicializa un nuevo track con una bbox del GT usando Stone Soup
        
        Args:
            bbox: [x, y, width, height]
            track_id: ID del track
            category_id: Categoría del objeto
            
        Returns:
            Nuevo KalmanTrack
        """
        from datetime import datetime
        
        x, y, w, h = bbox
        
        # Centro del bbox
        cx = x + w / 2
        cy = y + h / 2
        
        # Estado inicial: [x, vx=0, y, vy=0]
        # Stone Soup usa vectores columna
        state_vector = StateVector([[cx], [0], [cy], [0]])
        
        # Covarianza inicial (baja incertidumbre en posición, alta en velocidad)
        covariance = CovarianceMatrix(np.diag([10.0, 50.0, 10.0, 50.0]))
        
        # Crear GaussianState de Stone Soup CON TIMESTAMP
        gaussian_state = GaussianState(state_vector, covariance, timestamp=datetime.now())
        
        track = KalmanTrack(
            track_id=track_id,
            category_id=category_id,
            state=gaussian_state,
            last_gt_bbox=bbox,
            frames_since_update=0
        )
        
        return track

def bbox_to_measurement(bbox: List[float]) -> np.ndarray:
    """Convierte bbox [x,y,w,h] a medición [cx,cy] para Kalman con Stone Soup"""
    x, y, w, h = bbox
    return np.array([x + w/2, y + h/2])


def calculate_distance(predicted_state: GaussianState, bbox: List[float]) -> float:
    """
    Calcula distancia entre estado predicho de Stone Soup y bbox del GT
    
    Args:
        predicted_state: GaussianState de Stone Soup con [x, vx, y, vy]
        bbox: [x, y, width, height]
        
    Returns:
        Distancia normalizada
    """
    # Extraer estado predicho (Stone Soup usa vectores columna)
    state_vector = predicted_state.state_vector.flatten()
    pred_cx = state_vector[0]
    pred_cy = state_vector[2]
    
    # Centro de la bbox
    bbox_cx = bbox[0] + bbox[2] / 2
    bbox_cy = bbox[1] + bbox[3] / 2
    
    # Distancia euclidiana normalizada por tamaño de la bbox
    dx = (pred_cx - bbox_cx) / max(bbox[2], 1)
    dy = (pred_cy - bbox_cy) / max(bbox[3], 1)
    
    distance = np.sqrt(dx**2 + dy**2)
    
    return distance


class TrackingState:
    """
    Mantiene el estado del tracking a través de los frames.
    
    Usa filtro de Kalman para:
    - Predecir dónde debería estar cada objeto
    - Asociar detecciones del GT con tracks existentes
    
    Las bboxes SIEMPRE son del ground truth, nunca del Kalman.
    """
    
    def __init__(self, coco_reader: COCOReader):
        """
        Inicializa el estado de tracking
        
        Args:
            coco_reader: Lector de anotaciones COCO
        """
        self.coco_reader = coco_reader
        self.id_manager = IDManager()
        self.kalman_filter = KalmanFilter()
        
        # Estado actual
        self.current_frame = 0
        self.max_processed_frame = 0
        
        # Almacenamiento de detecciones por frame: {frame: [Detection]}
        self.detections_by_frame: Dict[int, List[Detection]] = {}
        
        # Tracks activos con sus filtros de Kalman
        self.active_tracks: Dict[int, KalmanTrack] = {}  # {track_id: KalmanTrack}
        
        # Cambios de visibilidad: {frame: {track_id: visibility}}
        self.visibility_changes: Dict[int, Dict[int, int]] = {}
        
        # Threshold de distancia para asociación (más estricto ahora que el modelo es mejor)
        self.max_distance_threshold = 3.0  # Distancia normalizada máxima
        self.max_frames_lost = 5  # Frames máximos sin detección antes de eliminar track
    
    def _associate_detections(self, annotations: List[Dict], frame: int) -> List[Detection]:
        """
        Asocia detecciones del GT con tracks existentes usando Kalman
        
        Args:
            annotations: Anotaciones del COCO
            frame: Número de frame
            
        Returns:
            Lista de detecciones con track_ids asignados
        """
        detections = []
        
        # Si no hay tracks activos, crear todos nuevos
        if not self.active_tracks:
            for ann in annotations:
                track_id = self.id_manager.get_new_id()
                
                # Inicializar track con Kalman
                kalman_track = self.kalman_filter.initialize_track(
                    ann['bbox'], track_id, ann['category_id']
                )
                self.active_tracks[track_id] = kalman_track
                
                detection = Detection(
                    bbox=ann['bbox'],  # Bbox del GT
                    track_id=track_id,
                    category_id=ann['category_id'],
                    category_name=ann['category_name'],
                    visibility=MOT_DEFAULTS['visibility'],
                    annotation_id=ann['annotation_id']
                )
                detections.append(detection)
            
            return detections
        
        # Predecir posiciones de todos los tracks activos
        for track in self.active_tracks.values():
            self.kalman_filter.predict(track)
            track.frames_since_update += 1
        
        # Crear matriz de costos (distancia entre predicciones y detecciones)
        n_detections = len(annotations)
        n_tracks = len(self.active_tracks)
        
        track_ids = list(self.active_tracks.keys())
        cost_matrix = np.full((n_detections, n_tracks), 1e6)  # Alto costo por defecto
        
        for i, ann in enumerate(annotations):
            for j, track_id in enumerate(track_ids):
                track = self.active_tracks[track_id]
                
                # Solo asociar si es la misma categoría
                if ann['category_id'] != track.category_id:
                    continue
                
                # Calcular distancia entre predicción y detección
                distance = calculate_distance(track.state, ann['bbox'])
                cost_matrix[i, j] = distance
        
        # Resolver asociación usando algoritmo húngaro
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Procesar asociaciones
        associated_detections = set()
        associated_tracks = set()
        
        for det_idx, track_idx in zip(row_indices, col_indices):
            cost = cost_matrix[det_idx, track_idx]
            
            # Solo asociar si la distancia es razonable
            if cost < self.max_distance_threshold:
                ann = annotations[det_idx]
                track_id = track_ids[track_idx]
                track = self.active_tracks[track_id]
                
                # Actualizar Kalman con la medición del GT
                measurement = bbox_to_measurement(ann['bbox'])
                self.kalman_filter.update(track, measurement)
                
                # Guardar bbox del GT
                track.last_gt_bbox = ann['bbox']
                
                # Aplicar correcciones manuales de ID
                final_track_id = self.id_manager.get_corrected_id(track_id, frame)
                
                # Obtener visibilidad
                visibility = MOT_DEFAULTS['visibility']
                if frame in self.visibility_changes:
                    if final_track_id in self.visibility_changes[frame]:
                        visibility = self.visibility_changes[frame][final_track_id]
                
                detection = Detection(
                    bbox=ann['bbox'],  # IMPORTANTE: bbox del GT, no del Kalman
                    track_id=final_track_id,
                    category_id=ann['category_id'],
                    category_name=ann['category_name'],
                    visibility=visibility,
                    annotation_id=ann['annotation_id']
                )
                detections.append(detection)
                
                associated_detections.add(det_idx)
                associated_tracks.add(track_id)
        
        # Crear tracks nuevos para detecciones no asociadas
        for i, ann in enumerate(annotations):
            if i not in associated_detections:
                track_id = self.id_manager.get_new_id()
                
                # Inicializar nuevo track con Kalman
                kalman_track = self.kalman_filter.initialize_track(
                    ann['bbox'], track_id, ann['category_id']
                )
                self.active_tracks[track_id] = kalman_track
                
                # Aplicar correcciones manuales
                final_track_id = self.id_manager.get_corrected_id(track_id, frame)
                
                # Obtener visibilidad
                visibility = MOT_DEFAULTS['visibility']
                if frame in self.visibility_changes:
                    if final_track_id in self.visibility_changes[frame]:
                        visibility = self.visibility_changes[frame][final_track_id]
                
                detection = Detection(
                    bbox=ann['bbox'],  # Bbox del GT
                    track_id=final_track_id,
                    category_id=ann['category_id'],
                    category_name=ann['category_name'],
                    visibility=visibility,
                    annotation_id=ann['annotation_id']
                )
                detections.append(detection)
        
        # Eliminar tracks que no se han actualizado en mucho tiempo
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if track.frames_since_update > self.max_frames_lost:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
        
        return detections
    
    def process_next_frame(self) -> bool:
        """
        Procesa el siguiente frame.
        Solo avanza si no hemos llegado al final.
        
        Returns:
            True si se procesó exitosamente, False si no hay más frames
        """
        # Verificar si ya estamos en el último frame
        if self.current_frame >= self.coco_reader.get_total_frames():
            return False
        
        next_frame = self.current_frame + 1
        
        # Si ya fue procesado, solo avanzar el puntero
        if next_frame in self.detections_by_frame:
            self.current_frame = next_frame
            return True
        
        # Procesar nuevo frame: obtener anotaciones del ground truth
        annotations = self.coco_reader.get_annotations(self.current_frame)
        
        # Asignar IDs de tracking usando Kalman
        detections = self._associate_detections(annotations, next_frame)
        
        # Guardar detecciones
        self.detections_by_frame[next_frame] = detections
        
        # Avanzar frame
        self.current_frame = next_frame
        self.max_processed_frame = max(self.max_processed_frame, next_frame)
        
        return True
    
    def go_to_frame(self, frame: int) -> bool:
        """
        Va a un frame específico (debe estar ya procesado)
        
        Args:
            frame: Número de frame (1-based)
            
        Returns:
            True si se pudo ir al frame, False si no está procesado
        """
        if frame < 1 or frame > self.max_processed_frame:
            return False
        
        self.current_frame = frame
        return True
    
    def get_current_detections(self) -> List[Detection]:
        """
        Obtiene las detecciones del frame actual
        
        Returns:
            Lista de detecciones del frame actual
        """
        if self.current_frame == 0:
            return []
        return self.detections_by_frame.get(self.current_frame, [])
    
    def get_detections_for_frame(self, frame: int) -> List[Detection]:
        """
        Obtiene las detecciones de un frame específico
        
        Args:
            frame: Número de frame
            
        Returns:
            Lista de detecciones
        """
        return self.detections_by_frame.get(frame, [])
    
    def update_track_id(self, frame: int, old_id: int, new_id: int) -> bool:
        """
        Actualiza el ID de un track en un frame específico
        
        Args:
            frame: Frame donde hacer el cambio
            old_id: ID actual
            new_id: Nuevo ID
            
        Returns:
            True si se pudo actualizar, False si hay colisión de IDs
        """
        # Verificar que no haya colisión en el frame actual
        detections = self.detections_by_frame.get(frame, [])
        existing_ids = {d.track_id for d in detections if d.track_id != old_id}
        
        if not self.id_manager.is_id_available(new_id, frame, existing_ids):
            return False
        
        # Registrar corrección
        self.id_manager.register_manual_correction(frame, old_id, new_id)
        
        # Actualizar en el frame actual
        for detection in detections:
            if detection.track_id == old_id:
                detection.track_id = new_id
        
        # MEJORADO: Transferir el track de Kalman si existe
        if old_id in self.active_tracks:
            kalman_track = self.active_tracks[old_id]
            kalman_track.track_id = new_id  # Cambiar el ID del track
            
            # Mover el track a la nueva key
            self.active_tracks[new_id] = kalman_track
            del self.active_tracks[old_id]
        
        # Invalidar solo frames posteriores para reprocesarlos con el nuevo ID
        frames_to_remove = [f for f in self.detections_by_frame.keys() if f > frame]
        for f in frames_to_remove:
            del self.detections_by_frame[f]
        
        self.max_processed_frame = frame
        
        # MEJORADO: Si hay varias correcciones consecutivas del mismo ID,
        # re-entrenar el Kalman para aprender la trayectoria real
        if new_id in self.active_tracks:
            # Buscar frames consecutivos editados con este ID
            start_frame = frame
            while start_frame > 1 and start_frame - 1 in self.detections_by_frame:
                has_id = any(d.track_id == new_id for d in self.detections_by_frame[start_frame - 1])
                if has_id:
                    start_frame -= 1
                else:
                    break
            
            # Si hay al menos 3 frames consecutivos, reentrenar
            if frame - start_frame >= 2:
                self.retrain_kalman_from_history(new_id, start_frame, frame)
        
        return True
    

    def retrain_kalman_from_history(self, track_id: int, start_frame: int, end_frame: int):
        """
        Re-entrena el filtro de Kalman con el historial de detecciones manuales
        Útil después de corregir manualmente varios frames seguidos
        
        Args:
            track_id: ID del track a reentrenar
            start_frame: Frame inicial
            end_frame: Frame final
        """
        if track_id not in self.active_tracks:
            return
        
        track = self.active_tracks[track_id]
        
        # Recopilar todas las detecciones de este track en el rango
        training_detections = []
        for frame in range(start_frame, end_frame + 1):
            if frame in self.detections_by_frame:
                for det in self.detections_by_frame[frame]:
                    if det.track_id == track_id:
                        training_detections.append((frame, det.bbox))
        
        # Re-entrenar el Kalman con estas detecciones
        if len(training_detections) >= 2:
            # Usar las últimas N detecciones para actualizar velocidad
            for i in range(len(training_detections) - 1):
                frame1, bbox1 = training_detections[i]
                frame2, bbox2 = training_detections[i + 1]
                
                # Calcular centros
                cx1 = bbox1[0] + bbox1[2] / 2
                cy1 = bbox1[1] + bbox1[3] / 2
                cx2 = bbox2[0] + bbox2[2] / 2
                cy2 = bbox2[1] + bbox2[3] / 2
                
                # Predecir y actualizar para ajustar la velocidad
                self.kalman_filter.predict(track)
                measurement = bbox_to_measurement(bbox2)
                self.kalman_filter.update(track, measurement)

    def update_visibility(self, frame: int, track_id: int, visibility: int):
        """
        Actualiza la visibilidad de un track en un frame específico
        
        Args:
            frame: Frame donde hacer el cambio
            track_id: ID del track
            visibility: Nueva visibilidad (0=oculto, 1=visible)
        """
        if frame not in self.visibility_changes:
            self.visibility_changes[frame] = {}
        
        self.visibility_changes[frame][track_id] = visibility
        
        # Actualizar en el frame actual si está cargado
        if frame in self.detections_by_frame:
            for detection in self.detections_by_frame[frame]:
                if detection.track_id == track_id:
                    detection.visibility = visibility
    
    def get_existing_ids_in_frame(self, frame: int) -> Set[int]:
        """
        Obtiene todos los IDs que existen en un frame
        
        Args:
            frame: Número de frame
            
        Returns:
            Set de IDs en ese frame
        """
        detections = self.detections_by_frame.get(frame, [])
        return {d.track_id for d in detections}
    
    def suggest_available_id(self, frame: int) -> int:
        """
        Sugiere un ID disponible para el frame actual
        
        Args:
            frame: Número de frame
            
        Returns:
            ID disponible sugerido
        """
        existing_ids = self.get_existing_ids_in_frame(frame)
        return self.id_manager.suggest_available_id(existing_ids)
    
    def reset(self):
        """Reinicia todo el tracking desde el principio"""
        self.current_frame = 0
        self.max_processed_frame = 0
        self.detections_by_frame = {}
        self.active_tracks = {}
        self.visibility_changes = {}
        self.id_manager.reset()
    
    def reset_from_current(self):
        """Reinicia el tracking desde el frame actual en adelante"""
        if self.current_frame <= 0:
            self.reset()
            return
        
        # Eliminar frames posteriores
        frames_to_remove = [f for f in self.detections_by_frame.keys() 
                           if f > self.current_frame]
        for f in frames_to_remove:
            del self.detections_by_frame[f]
        
        # Reiniciar tracks de Kalman
        self.active_tracks = {}
        
        # Limpiar correcciones y cambios de visibilidad posteriores
        self.id_manager.reset_from_frame(self.current_frame)
        
        visibility_frames_to_remove = [f for f in self.visibility_changes.keys() 
                                       if f > self.current_frame]
        for f in visibility_frames_to_remove:
            del self.visibility_changes[f]
        
        self.max_processed_frame = self.current_frame
