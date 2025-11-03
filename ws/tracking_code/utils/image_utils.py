"""
Utilidades para manejo de imágenes y transformaciones de coordenadas
"""
import cv2
import numpy as np
from typing import Tuple
from config import CANVAS_WIDTH, CANVAS_HEIGHT


class ImageTransform:
    """
    Maneja transformaciones de imágenes para visualización
    Calcula escalado y padding para mantener aspect ratio
    """
    
    def __init__(self, orig_width: int, orig_height: int, 
                 canvas_width: int = CANVAS_WIDTH, 
                 canvas_height: int = CANVAS_HEIGHT):
        """
        Inicializa las transformaciones
        
        Args:
            orig_width: Ancho original de la imagen
            orig_height: Alto original de la imagen
            canvas_width: Ancho del canvas
            canvas_height: Alto del canvas
        """
        self.orig_width = orig_width
        self.orig_height = orig_height
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        
        # Calcular escala para mantener aspect ratio
        self.scale = min(
            canvas_width / orig_width,
            canvas_height / orig_height
        )
        
        # Calcular dimensiones escaladas
        self.scaled_width = int(orig_width * self.scale)
        self.scaled_height = int(orig_height * self.scale)
        
        # Calcular padding para centrar
        self.pad_x = (canvas_width - self.scaled_width) // 2
        self.pad_y = (canvas_height - self.scaled_height) // 2
    
    def transform_bbox(self, x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
        """
        Transforma coordenadas de bbox de espacio original a canvas
        
        Args:
            x, y, w, h: Coordenadas originales [x, y, width, height]
            
        Returns:
            Tupla (x, y, w, h) en coordenadas del canvas
        """
        x_canvas = int(x * self.scale) + self.pad_x
        y_canvas = int(y * self.scale) + self.pad_y
        w_canvas = int(w * self.scale)
        h_canvas = int(h * self.scale)
        
        return x_canvas, y_canvas, w_canvas, h_canvas
    
    def point_in_bbox(self, px: int, py: int, 
                     bbox_x: int, bbox_y: int, 
                     bbox_w: int, bbox_h: int) -> bool:
        """
        Verifica si un punto está dentro de una bounding box
        
        Args:
            px, py: Coordenadas del punto
            bbox_x, bbox_y, bbox_w, bbox_h: Coordenadas de la bbox
            
        Returns:
            True si el punto está dentro de la bbox
        """
        return (bbox_x <= px <= bbox_x + bbox_w and 
                bbox_y <= py <= bbox_y + bbox_h)
    
    def prepare_image_for_canvas(self, image: np.ndarray) -> np.ndarray:
        """
        Prepara una imagen para mostrar en el canvas
        (redimensiona y añade padding)
        
        Args:
            image: Imagen original en formato RGB
            
        Returns:
            Imagen preparada para el canvas
        """
        # Redimensionar
        resized = cv2.resize(image, (self.scaled_width, self.scaled_height))
        
        # Crear canvas con padding
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        canvas[self.pad_y:self.pad_y + self.scaled_height, 
               self.pad_x:self.pad_x + self.scaled_width] = resized
        
        return canvas


def load_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    """
    Carga una imagen y retorna la imagen en RGB más sus dimensiones
    
    Args:
        image_path: Ruta a la imagen
        
    Returns:
        Tupla (imagen_rgb, ancho, alto)
        Si hay error, retorna una imagen negra con mensaje de error
    """
    img = cv2.imread(image_path)
    
    if img is None:
        # Crear imagen de error
        img = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(img, f"Error loading: {image_path}", 
                   (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        height, width = img.shape[:2]
        return img, width, height
    
    # Convertir BGR a RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]
    
    return img_rgb, width, height


def create_placeholder_image(text: str = "No image") -> np.ndarray:
    """
    Crea una imagen placeholder
    
    Args:
        text: Texto a mostrar
        
    Returns:
        Imagen placeholder en RGB
    """
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(img, text, (250, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return img


def get_color_for_id(track_id: int) -> str:
    """
    Obtiene un color para un ID de track
    
    Args:
        track_id: ID del track
        
    Returns:
        String de color en formato hex
    """
    from config import COLORS
    return COLORS[track_id % len(COLORS)]
