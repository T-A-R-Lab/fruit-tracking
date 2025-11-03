"""
Ventana principal de la aplicación de tracking
"""
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import os
from typing import Optional

from models.tracking_state import TrackingState
from utils.image_utils import (
    ImageTransform, load_image, create_placeholder_image, get_color_for_id
)
from utils.mot_exporter import MOTExporter
from config import CANVAS_WIDTH, CANVAS_HEIGHT, UI_CONFIG


class TrackingEditorWindow:
    """Ventana principal del editor de tracking"""
    
    def __init__(self, root: tk.Tk, tracking_state: TrackingState, 
                 image_folder: str, save_file: str):
        """
        Inicializa la ventana
        
        Args:
            root: Ventana raíz de tkinter
            tracking_state: Estado del tracking
            image_folder: Carpeta con las imágenes
            save_file: Archivo donde guardar los resultados
        """
        self.root = root
        self.tracking_state = tracking_state
        self.image_folder = image_folder
        self.save_file = save_file
        
        # Estado de la UI
        self.selected_bbox_idx: Optional[int] = None
        self.current_transform: Optional[ImageTransform] = None
        self.tk_img = None  # Mantener referencia a la imagen
        
        self._setup_ui()
        self._process_first_frame()
    
    def _setup_ui(self):
        """Configura la interfaz de usuario"""
        self.root.title("Tracking Editor")
        
        # Canvas para mostrar imagen y bounding boxes
        self.canvas = tk.Canvas(
            self.root, 
            width=CANVAS_WIDTH, 
            height=CANVAS_HEIGHT,
            bg='black'
        )
        self.canvas.pack()
        
        # Frame de botones
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        # Botones de navegación
        self.prev_btn = tk.Button(
            btn_frame, 
            text="<< Prev", 
            command=self.prev_frame,
            width=10
        )
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(
            btn_frame, 
            text="Next >>", 
            command=self.next_frame,
            width=10
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Botón guardar
        self.save_btn = tk.Button(
            btn_frame, 
            text="Save", 
            command=self.save,
            width=10,
            bg='#4CAF50',
            fg='white'
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Botones de reset
        self.reset_btn = tk.Button(
            btn_frame, 
            text="Reset All", 
            command=self.reset_all,
            width=10,
            bg='#f44336',
            fg='white'
        )
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_from_btn = tk.Button(
            btn_frame, 
            text="Reset From Here", 
            command=self.reset_from_here,
            width=15,
            bg='#FF9800',
            fg='white'
        )
        self.reset_from_btn.pack(side=tk.LEFT, padx=5)
        
        # Label de información
        self.info_label = tk.Label(
            btn_frame, 
            text="", 
            font=("Arial", 10)
        )
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        # Eventos
        self.canvas.bind("<Button-1>", self.on_canvas_click)
    
    def _process_first_frame(self):
        """Procesa el primer frame al iniciar"""
        try:
            success = self.tracking_state.process_next_frame()
            if success:
                self.update_view()
            else:
                messagebox.showerror("Error", "No se pudo procesar el primer frame")
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar primer frame: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_view(self):
        """Actualiza la visualización del frame actual"""
        # Obtener nombre de imagen
        frame_idx = self.tracking_state.current_frame - 1  # 0-based para COCO reader
        img_filename = self.tracking_state.coco_reader.get_image_filename(frame_idx)
        
        # Cargar imagen
        if img_filename:
            img_path = os.path.join(self.image_folder, img_filename)
            if os.path.exists(img_path):
                img, width, height = load_image(img_path)
            else:
                img = create_placeholder_image(f"File not found: {img_filename}")
                height, width = img.shape[:2]
        else:
            img = create_placeholder_image("No image available")
            height, width = img.shape[:2]
        
        # Crear transformación
        self.current_transform = ImageTransform(width, height)
        
        # Preparar imagen para canvas
        canvas_img = self.current_transform.prepare_image_for_canvas(img)
        
        # Convertir a formato PIL y Tkinter
        pil_img = Image.fromarray(canvas_img)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        # Limpiar canvas y mostrar imagen
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        # Dibujar bounding boxes
        self._draw_bboxes()
        
        # Actualizar información
        self._update_info_label()
    
    def _draw_bboxes(self):
        """Dibuja las bounding boxes en el canvas"""
        detections = self.tracking_state.get_current_detections()
        
        for idx, detection in enumerate(detections):
            # Solo dibujar si es visible
            if detection.visibility == 0:
                continue
            
            # Transformar coordenadas
            x, y, w, h = self.current_transform.transform_bbox(*detection.bbox)
            
            # Color basado en el ID
            color = get_color_for_id(detection.track_id)
            
            # Ancho de línea
            width = (UI_CONFIG['bbox_width_selected'] 
                    if idx == self.selected_bbox_idx 
                    else UI_CONFIG['bbox_width_normal'])
            
            # Dibujar rectángulo
            tag = f"bbox_{idx}"
            self.canvas.create_rectangle(
                x, y, x + w, y + h,
                outline=color,
                width=width,
                tags=tag
            )
            
            # Dibujar texto con ID
            text_x = x + UI_CONFIG['text_offset_x']
            text_y = y + UI_CONFIG['text_offset_y']
            self.canvas.create_text(
                text_x, text_y,
                anchor=tk.NW,
                text=f"ID:{detection.track_id}",
                fill=color,
                font=UI_CONFIG['text_font'],
                tags=tag
            )
    
    def _update_info_label(self):
        """Actualiza el label de información"""
        current = self.tracking_state.current_frame
        total = self.tracking_state.coco_reader.get_total_frames()
        max_proc = self.tracking_state.max_processed_frame
        
        detections = self.tracking_state.get_current_detections()
        visible = sum(1 for d in detections if d.visibility == 1)
        total_dets = len(detections)
        
        info_text = (f"Frame {current}/{total} | "
                    f"Max procesado: {max_proc} | "
                    f"Visible: {visible}/{total_dets} tracks")
        
        self.info_label.config(text=info_text)
    
    def on_canvas_click(self, event):
        """Maneja clics en el canvas"""
        if not self.current_transform:
            return
        
        detections = self.tracking_state.get_current_detections()
        
        # Buscar bbox clickeada (incluyendo invisibles para poder editarlas)
        for idx, detection in enumerate(detections):
            x, y, w, h = self.current_transform.transform_bbox(*detection.bbox)
            
            if self.current_transform.point_in_bbox(event.x, event.y, x, y, w, h):
                self.selected_bbox_idx = idx
                self.update_view()
                self._edit_detection(detection, idx)
                return
        
        # No se clickeó ninguna bbox
        self.selected_bbox_idx = None
        self.update_view()
    
    def _edit_detection(self, detection, idx):
        """Abre diálogo para editar una detección"""
        frame = self.tracking_state.current_frame
        
        # Editar ID
        new_id = simpledialog.askinteger(
            "Editar ID",
            f"Track ID actual: {detection.track_id}\n"
            f"Nuevo track ID (entero):",
            initialvalue=detection.track_id,
            minvalue=1
        )
        
        if new_id is not None and new_id != detection.track_id:
            # Intentar actualizar ID
            success = self.tracking_state.update_track_id(
                frame, detection.track_id, new_id
            )
            
            if not success:
                # Hay colisión, sugerir un ID alternativo
                suggested_id = self.tracking_state.suggest_available_id(frame)
                retry = messagebox.askyesno(
                    "ID duplicado",
                    f"El ID {new_id} ya existe en este frame.\n"
                    f"¿Usar ID {suggested_id} en su lugar?"
                )
                
                if retry:
                    self.tracking_state.update_track_id(
                        frame, detection.track_id, suggested_id
                    )
                    self.update_view()
                    return
            else:
                # Actualización exitosa
                self.update_view()
        
        # Editar visibilidad
        new_visibility = simpledialog.askinteger(
            "Editar Visibilidad",
            f"Visibilidad actual: {detection.visibility}\n"
            f"Nueva visibilidad (0=oculto, 1=visible):",
            initialvalue=detection.visibility,
            minvalue=0,
            maxvalue=1
        )
        
        if new_visibility is not None and new_visibility != detection.visibility:
            self.tracking_state.update_visibility(
                frame, detection.track_id, new_visibility
            )
            self.update_view()
    
    def next_frame(self):
        """Avanza al siguiente frame"""
        self.selected_bbox_idx = None
        
        success = self.tracking_state.process_next_frame()
        if success:
            self.update_view()
        else:
            messagebox.showinfo("Fin", "No hay más frames para procesar")
    
    def prev_frame(self):
        """Retrocede al frame anterior"""
        if self.tracking_state.current_frame <= 1:
            messagebox.showinfo("Inicio", "Ya estás en el primer frame")
            return
        
        self.selected_bbox_idx = None
        prev_frame = self.tracking_state.current_frame - 1
        
        success = self.tracking_state.go_to_frame(prev_frame)
        if success:
            self.update_view()
        else:
            messagebox.showerror("Error", "No se pudo ir al frame anterior")
    
    def save(self):
        """Guarda los resultados en formato MOT"""
        try:
            MOTExporter.save_to_file(
                self.save_file,
                self.tracking_state.detections_by_frame
            )
            messagebox.showinfo("Guardado", f"Archivo guardado en:\n{self.save_file}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def reset_all(self):
        """Reinicia todo el tracking"""
        confirm = messagebox.askyesno(
            "Confirmar",
            "¿Reiniciar todo el tracking desde el principio?"
        )
        
        if confirm:
            try:
                self.tracking_state.reset()
                self.selected_bbox_idx = None
                self._process_first_frame()
            except Exception as e:
                messagebox.showerror("Error", f"Error al reiniciar:\n{str(e)}")
                import traceback
                traceback.print_exc()
    
    def reset_from_here(self):
        """Reinicia el tracking desde el frame actual"""
        confirm = messagebox.askyesno(
            "Confirmar",
            f"¿Reiniciar el tracking desde el frame {self.tracking_state.current_frame}?"
        )
        
        if confirm:
            try:
                self.tracking_state.reset_from_current()
                self.selected_bbox_idx = None
                self.update_view()
                messagebox.showinfo(
                    "Reinicio parcial",
                    f"Tracking reiniciado desde frame {self.tracking_state.current_frame}"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Error al reiniciar:\n{str(e)}")
                import traceback
                traceback.print_exc()
