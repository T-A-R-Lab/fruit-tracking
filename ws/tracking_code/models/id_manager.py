"""
Gestor de IDs únicos para tracking
Maneja la asignación de IDs, detección de colisiones y correcciones manuales
"""
from typing import Dict, Set, Optional


class IDManager:
    """
    Gestiona IDs de tracks de forma única y consistente.
    
    Estrategia:
    - Cada track nuevo obtiene un ID único autoincremental
    - Al editar manualmente un ID, se verifica que no exista en el frame actual
    - Se mantiene un registro de todos los IDs usados para evitar colisiones
    - Las correcciones manuales se propagan hacia adelante (frames futuros)
    """
    
    def __init__(self, start_id: int = 1):
        """
        Inicializa el gestor de IDs
        
        Args:
            start_id: ID inicial para auto-asignación
        """
        self.next_id = start_id
        self.used_ids: Set[int] = set()  # Todos los IDs que se han usado alguna vez
        
        # Mapeo de correcciones: {frame: {old_id: new_id}}
        self.corrections: Dict[int, Dict[int, int]] = {}
        
    def get_new_id(self) -> int:
        """
        Genera un nuevo ID único
        
        Returns:
            Nuevo ID único
        """
        new_id = self.next_id
        self.used_ids.add(new_id)
        self.next_id += 1
        return new_id
    
    def is_id_available(self, track_id: int, frame: int, 
                       existing_ids_in_frame: Set[int]) -> bool:
        """
        Verifica si un ID está disponible para usar en un frame específico
        
        Args:
            track_id: ID a verificar
            frame: Número de frame
            existing_ids_in_frame: IDs que ya existen en ese frame
            
        Returns:
            True si el ID está disponible (no hay colisión)
        """
        # No puede estar ya en uso en el frame actual
        if track_id in existing_ids_in_frame:
            return False
        return True
    
    def suggest_available_id(self, existing_ids_in_frame: Set[int]) -> int:
        """
        Sugiere un ID disponible que no esté en el frame actual
        
        Args:
            existing_ids_in_frame: IDs que ya existen en ese frame
            
        Returns:
            ID disponible sugerido
        """
        # Empezar desde next_id y buscar el primero disponible
        candidate = self.next_id
        while candidate in existing_ids_in_frame:
            candidate += 1
        
        # Actualizar next_id si es necesario
        if candidate >= self.next_id:
            self.next_id = candidate + 1
        
        self.used_ids.add(candidate)
        return candidate
    
    def register_manual_correction(self, frame: int, old_id: int, new_id: int):
        """
        Registra una corrección manual de ID
        
        Args:
            frame: Frame donde se hizo la corrección
            old_id: ID original
            new_id: Nuevo ID asignado
        """
        if frame not in self.corrections:
            self.corrections[frame] = {}
        
        self.corrections[frame][old_id] = new_id
        self.used_ids.add(new_id)
        
        # Actualizar next_id si el nuevo ID es mayor
        if new_id >= self.next_id:
            self.next_id = new_id + 1
    
    def get_corrected_id(self, track_id: int, frame: int) -> int:
        """
        Obtiene el ID corregido para un track en un frame específico
        Aplica todas las correcciones desde el inicio hasta ese frame
        
        Args:
            track_id: ID original del track
            frame: Frame actual
            
        Returns:
            ID corregido (o el original si no hay correcciones)
        """
        current_id = track_id
        
        # Aplicar correcciones en orden cronológico
        for f in sorted(self.corrections.keys()):
            if f > frame:
                break
            if current_id in self.corrections[f]:
                current_id = self.corrections[f][current_id]
        
        return current_id
    
    def get_corrections_up_to_frame(self, frame: int) -> Dict[int, int]:
        """
        Obtiene todas las correcciones acumuladas hasta un frame
        
        Args:
            frame: Frame hasta el cual obtener correcciones
            
        Returns:
            Diccionario {old_id: final_id} con todas las correcciones aplicadas
        """
        accumulated = {}
        
        for f in sorted(self.corrections.keys()):
            if f > frame:
                break
            for old_id, new_id in self.corrections[f].items():
                # Aplicar corrección en cadena
                # Si old_id ya fue corregido antes, actualizar el mapeo
                final_old_id = old_id
                for prev_old, prev_new in accumulated.items():
                    if prev_new == old_id:
                        final_old_id = prev_old
                        break
                accumulated[final_old_id] = new_id
        
        return accumulated
    
    def reset(self, start_id: int = 1):
        """Reinicia el gestor de IDs"""
        self.next_id = start_id
        self.used_ids = set()
        self.corrections = {}
    
    def reset_from_frame(self, frame: int):
        """
        Reinicia correcciones desde un frame específico en adelante
        
        Args:
            frame: Frame desde el cual eliminar correcciones
        """
        # Eliminar correcciones posteriores al frame
        frames_to_remove = [f for f in self.corrections.keys() if f > frame]
        for f in frames_to_remove:
            del self.corrections[f]
