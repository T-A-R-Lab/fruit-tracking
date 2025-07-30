import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from stonesoup.reader import DetectionReader
from stonesoup.types.detection import Detection
from stonesoup.types.state import StateVector
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.base import Property
from stonesoup.buffered_generator import BufferedGenerator


class COCOReader(DetectionReader, BufferedGenerator):
    """Reader for COCO format ground truth data."""
    
    coco_file = Property(
        Path,
        doc="Path to COCO annotation file"
    )
    
    image_folder = Property(
        Path,
        doc="Path to folder containing images"
    )
    
    start_time = Property(
        datetime,
        default=None,
        doc="Start time for the first frame. Default is current time."
    )
    
    time_step = Property(
        timedelta,
        default=None,
        doc="Time between frames. Default is 0.1 seconds."
    )
    
    def __init__(self, *args, **kwargs):
        # Llamar al constructor de la clase padre con los argumentos
        super().__init__(*args, **kwargs)
        
        # Establecer valores predeterminados si no se proporcionaron
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.time_step is None:
            self.time_step = timedelta(seconds=0.1)
        
        # Cargar datos COCO
        with open(self.coco_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Inicializar contador de frames
        self.frame_counter = 0
        
        # Crear mapeo de image_id a anotaciones
        self.image_id_to_anns = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_anns:
                self.image_id_to_anns[image_id] = []
            self.image_id_to_anns[image_id].append(ann)
        
        # Obtener todas las imágenes y ordenarlas por id
        self.images = sorted(self.coco_data['images'], key=lambda x: x['id'])
        
        # Mapear category_id a nombre de categoría
        self.category_map = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Inicializar seguimiento interno de objetos
        self.object_paths = {}
    
    @BufferedGenerator.generator_method
    def detections_gen(self):
        """Generate detections from COCO annotations."""
        for image in self.images:
            image_id = image['id']
            timestamp = self.start_time + self.time_step * self.frame_counter
            self.frame_counter += 1
            
            # Get annotations for this image
            annotations = self.image_id_to_anns.get(image_id, [])
            
            # Importante: usar set() en lugar de list() para las detecciones
            detections = set()
            
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                
                # Create detection with state vector [x, y, width, height]
                state_vector = StateVector([bbox[0], bbox[1], bbox[2], bbox[3]])
                
                # Create detection
                detection = Detection(
                    state_vector,
                    timestamp=timestamp,
                    metadata={
                        'id': ann['id'],
                        'image_id': image_id,
                        'category_id': ann['category_id'],
                        'category_name': self.category_map.get(ann['category_id'], 'unknown'),
                        'object_id': ann.get('track_id', ann['id'])  # Use track_id if available, else use ann id
                    }
                )
                # Usar add() en lugar de append() para conjuntos
                detections.add(detection)
                
                # Update object paths for ground truth tracking
                object_id = ann.get('track_id', ann['id'])
                if object_id not in self.object_paths:
                    self.object_paths[object_id] = GroundTruthPath(
                        states=[],
                        id=object_id
                    )
                
                # Add state to path
                gt_state = GroundTruthState(
                    state_vector=state_vector,
                    timestamp=timestamp,
                    metadata={
                        'image_id': image_id,
                        'category_id': ann['category_id'],
                        'category_name': self.category_map.get(ann['category_id'], 'unknown')
                    }
                )
                self.object_paths[object_id].states.append(gt_state)
            
            yield timestamp, detections
    
    def ground_truth_paths_gen(self):
        """Generate ground truth paths after processing all detections."""
        # Ensure all detections have been processed
        for _ in self.detections_gen():
            pass
        
        # Return all paths
        for path in self.object_paths.values():
            yield path