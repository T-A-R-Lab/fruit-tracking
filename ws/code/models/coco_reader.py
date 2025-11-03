"""
Lector simplificado de anotaciones COCO para tracking frame por frame
"""
import json
from typing import List, Dict, Optional


class COCOReader:
    """Lee y proporciona acceso a anotaciones COCO organizadas por frames"""
    
    def __init__(self, coco_file: str):
        """
        Inicializa el lector de COCO
        
        Args:
            coco_file: Ruta al archivo JSON de anotaciones COCO
        """
        with open(coco_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Mapear category_id a nombre de categoría
        self.category_map = {cat['id']: cat['name'] 
                            for cat in self.coco_data['categories']}
        
        # Obtener todas las imágenes y ordenarlas por id
        self.images = sorted(self.coco_data['images'], key=lambda x: x['id'])
        
        # Crear mapeo de image_id a anotaciones
        self.image_to_anns = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_to_anns:
                self.image_to_anns[image_id] = []
            self.image_to_anns[image_id].append(ann)
    
    def get_image_filename(self, frame_idx: int) -> Optional[str]:
        """
        Obtiene el nombre del archivo de imagen para un frame
        
        Args:
            frame_idx: Índice del frame (0-based)
            
        Returns:
            Nombre del archivo o None si el índice es inválido
        """
        if frame_idx < 0 or frame_idx >= len(self.images):
            return None
        return self.images[frame_idx]['file_name']
    
    def get_annotations(self, frame_idx: int) -> List[Dict]:
        """
        Obtiene las anotaciones (ground truth) para un frame específico
        
        Args:
            frame_idx: Índice del frame (0-based)
            
        Returns:
            Lista de anotaciones con formato:
            {
                'bbox': [x, y, width, height],
                'category_id': int,
                'category_name': str,
                'annotation_id': int,
                'track_id': int (si está disponible)
            }
        """
        if frame_idx < 0 or frame_idx >= len(self.images):
            return []
        
        image_id = self.images[frame_idx]['id']
        annotations = self.image_to_anns.get(image_id, [])
        
        # Convertir a formato simplificado
        result = []
        for ann in annotations:
            result.append({
                'bbox': ann['bbox'],  # [x, y, width, height]
                'category_id': ann['category_id'],
                'category_name': self.category_map.get(ann['category_id'], 'unknown'),
                'annotation_id': ann['id'],
                'track_id': ann.get('track_id', ann['id'])  # Usar track_id si existe
            })
        
        return result
    
    def get_total_frames(self) -> int:
        """Retorna el número total de frames"""
        return len(self.images)
