"""
Configuración global de la aplicación
"""

# Configuración de Canvas
CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 900

# Configuración de colores para IDs de tracking
COLORS = [
    "#ff4040", "#40ff40", "#4040ff", "#ffbf00", "#00bfbf",
    "#bf40ff", "#ff40bf", "#bfff40", "#40bfff", "#ff8040"
]

# Configuración de tracking
TRACKER_CONFIG = {
    "min_states_confirmed": 2,  # Mínimo de estados para considerar un track confirmado
    "time_step_seconds": 0.1,   # Paso de tiempo entre frames
    "iou_threshold": 0.3,       # IoU mínimo para asociar detecciones (0.0 - 1.0)
}

# Configuración de visualización
UI_CONFIG = {
    "bbox_width_normal": 3,
    "bbox_width_selected": 6,
    "text_font": ("Arial", 16),
    "text_offset_x": 5,
    "text_offset_y": 15,
}

# Valores por defecto MOT
MOT_DEFAULTS = {
    "confidence": 1,      # Confianza por defecto
    "class_id": 1,        # ID de clase por defecto
    "visibility": 1,      # Visibilidad por defecto (1=visible, 0=oculto)
    "world_coord": -1,    # Coordenada mundial por defecto
}
