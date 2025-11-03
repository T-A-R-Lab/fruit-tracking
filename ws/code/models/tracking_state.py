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
    Track con filtro de Kalman para predicción de posición
    IMPORTANTE: La bbox mostrada es SIEMPRE del GT, Kalman solo para asociación
    """
    track_id: int
    category_id: int
    
    # Estado del Kalman: [x, vx, y, vy, w, h]
    # x, y: centro del bbox
    # vx, vy: velocidad
    # w, h: dimensiones
    state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6) * 1000)
    
    # Última bbox del ground truth asociada
    last_gt_bbox: List[float] = field(default_factory=list)
    
    # Frames desde la última actualización
    frames_since_update: int = 0


class KalmanFilter:
    """Filtro de Kalman para tracking de posición (solo para asociación de IDs)"""
    
    def __init__(self):
        # Modelo de transición: [x, vx, y, vy, w, h]
        # Asumimos velocidad constante para x, y
        # w, h son constantes (con ruido)
        self.dt = 1.0  # Delta tiempo entre frames
        
        # Matriz de transición
        self.F = np.array([
            [1, self.dt, 0, 0,       0, 0],  # x = x + vx*dt
            [0, 1,       0, 0,       0, 0],  # vx = vx
            [0, 0,       1, self.dt, 0, 0],  # y = y + vy*dt
            [0, 0,       0, 1,       0, 0],  # vy = vy
            [0, 0,       0, 0,       1, 0],  # w = w
            [0, 0,       0, 0,       0, 1],  # h = h
        ])
        
        # Ruido del proceso
        self.Q = np.diag([50**2, 50**2, 50**2, 50**2, 30**2, 30**2])
        
        # Matriz de observación: medimos [x, y, w, h]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],  # observamos x
            [0, 0, 1, 0, 0, 0],  # observamos y
            [0, 0, 0, 0, 1, 0],  # observamos w
            [0, 0, 0, 0, 0, 1],  # observamos h
        ])
        
        # Ruido de medición
        self.R = np.diag([5**2, 5**2, 7**2, 7**2])
    
    def predict(self, track: KalmanTrack) -> np.ndarray:
        """
        Predice el siguiente estado del track
        
        Returns:
            Estado predicho [x, y, w, h, vx, vy]
        """
        # Predicción del estado
        track.state = self.F @ track.state
        
        # Predicción de la covarianza
        track.covariance = self.F @ track.covariance @ self.F.T + self.Q
        
        return track.state
    
    def update(self, track: KalmanTrack, measurement: np.ndarray):
        """
        Actualiza el track con una nueva medición (bbox del GT)
        
        Args:
            measurement: [x_center, y_center, w, h]
        """
        # Innovación (diferencia entre medición y predicción)
        y = measurement - self.H @ track.state
        
        # Covarianza de la innovación
        S = self.H @ track.covariance @ self.H.T + self.R
        
        # Ganancia de Kalman
        K = track.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Actualizar estado
        track.state = track.state + K @ y
        
        # Actualizar covarianza
        I = np.eye(len(track.state))
        track.covariance = (I - K @ self.H) @ track.covariance
        
        track.frames_since_update = 0
    
    def initialize_track(self, bbox: List[float], track_id: int, category_id: int) -> KalmanTrack:
        """
        Inicializa un nuevo track con una bbox del GT
        
        Args:
            bbox: [x, y, width, height]
            track_id: ID del track
            category_id: Categoría del objeto
            
        Returns:
            Nuevo KalmanTrack
        """
        x, y, w, h = bbox
        
        # Centro del bbox
        cx = x + w / 2
        cy = y + h / 2
        
        # Estado inicial: [cx, vx=0, cy, vy=0, w, h]
        state = np.array([cx, 0, cy, 0, w, h])
        
        # Covarianza inicial (alta incertidumbre en velocidad)
        covariance = np.diag([100**2, 50**2, 100**2, 50**2, 100**2, 100**2])
        
        track = KalmanTrack(
            track_id=track_id,
            category_id=category_id,
            state=state,
            covariance=covariance,
            last_gt_bbox=bbox,
            frames_since_update=0
        )
        
        return track


def bbox_to_measurement(bbox: List[float]) -> np.ndarray:
    """Convierte bbox [x,y,w,h] a medición [cx,cy,w,h] para Kalman"""
    x, y, w, h = bbox
    return np.array([x + w/2, y + h/2, w, h])


def calculate_distance(predicted_state: np.ndarray, bbox: List[float]) -> float:
    """
    Calcula distancia entre estado predicho y bbox del GT
    
    Args:
        predicted_state: [cx, vx, cy, vy, w, h]
        bbox: [x, y, width, height]
        
    Returns:
        Distancia normalizada
    """
    # Centro predicho
    pred_cx = predicted_state[0]
    pred_cy = predicted_state[2]
    pred_w = predicted_state[4]
    pred_h = predicted_state[5]
    
    # Centro de la bbox
    bbox_cx = bbox[0] + bbox[2] / 2
    bbox_cy = bbox[1] + bbox[3] / 2
    
    # Distancia euclidiana normalizada por tamaño
    dx = (pred_cx - bbox_cx) / max(pred_w, bbox[2], 1)
    dy = (pred_cy - bbox_cy) / max(pred_h, bbox[3], 1)
    
    distance = np.sqrt(dx**2 + dy**2)
    
    # Penalizar si hay diferencia grande en tamaño
    size_diff = abs(pred_w - bbox[2]) / max(pred_w, 1) + abs(pred_h - bbox[3]) / max(pred_h, 1)
    
    return distance + 0.2 * size_diff


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
        
        # Threshold de distancia para asociación
        self.max_distance_threshold = 2.0  # Distancia normalizada máxima
    
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
        max_frames_lost = 3
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if track.frames_since_update > max_frames_lost:
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
        
        # Invalidar frames posteriores para reprocesarlos con el nuevo ID
        frames_to_remove = [f for f in self.detections_by_frame.keys() if f > frame]
        for f in frames_to_remove:
            del self.detections_by_frame[f]
        
        # También invalidar tracks activos de Kalman
        self.active_tracks = {}
        
        self.max_processed_frame = frame
        
        return True
    
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
