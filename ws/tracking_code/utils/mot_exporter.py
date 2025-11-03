"""
Exportador de tracking a formato MOT (Multiple Object Tracking)
Formato: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility, -1
"""
from typing import List, Dict
from models.tracking_state import Detection
from config import MOT_DEFAULTS


class MOTExporter:
    """Exporta datos de tracking al formato MOT"""
    
    @staticmethod
    def detection_to_mot_line(frame: int, detection: Detection) -> str:
        """
        Convierte una detección a una línea en formato MOT
        
        Args:
            frame: Número de frame
            detection: Detección a exportar
            
        Returns:
            Línea en formato MOT (CSV)
        """
        x, y, w, h = detection.bbox
        
        # Formato MOT: frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility,world_coord
        values = [
            frame,
            detection.track_id,
            x,
            y,
            w,
            h,
            MOT_DEFAULTS['confidence'],
            detection.category_id,
            detection.visibility,
            MOT_DEFAULTS['world_coord']
        ]
        
        return ",".join(str(v) for v in values)
    
    @staticmethod
    def export_frame(frame: int, detections: List[Detection]) -> List[str]:
        """
        Exporta todas las detecciones de un frame
        
        Args:
            frame: Número de frame
            detections: Lista de detecciones
            
        Returns:
            Lista de líneas en formato MOT
        """
        lines = []
        for detection in detections:
            line = MOTExporter.detection_to_mot_line(frame, detection)
            lines.append(line)
        
        return lines
    
    @staticmethod
    def export_all_frames(detections_by_frame: Dict[int, List[Detection]]) -> List[str]:
        """
        Exporta todas las detecciones de todos los frames
        
        Args:
            detections_by_frame: Diccionario {frame: [detections]}
            
        Returns:
            Lista de todas las líneas en formato MOT
        """
        all_lines = []
        
        for frame in sorted(detections_by_frame.keys()):
            detections = detections_by_frame[frame]
            lines = MOTExporter.export_frame(frame, detections)
            all_lines.extend(lines)
        
        return all_lines
    
    @staticmethod
    def save_to_file(filepath: str, detections_by_frame: Dict[int, List[Detection]]):
        """
        Guarda tracking a un archivo MOT
        
        Args:
            filepath: Ruta del archivo de salida
            detections_by_frame: Diccionario {frame: [detections]}
        """
        lines = MOTExporter.export_all_frames(detections_by_frame)
        
        with open(filepath, 'w') as f:
            for line in lines:
                f.write(line + '\n')
