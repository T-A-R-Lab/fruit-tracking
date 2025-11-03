"""
Script principal - Punto de entrada de la aplicación
"""
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

from models.coco_reader import COCOReader
from models.tracking_state import TrackingState
from ui.main_window import TrackingEditorWindow


def select_files():
    """
    Muestra diálogos para seleccionar archivos de entrada/salida
    
    Returns:
        Tupla (coco_file, image_folder, save_file) o None si se cancela
    """
    root = tk.Tk()
    root.withdraw()
    
    # Seleccionar archivo COCO
    coco_file = filedialog.askopenfilename(
        title="Selecciona archivo de anotaciones COCO",
        filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
    )
    
    if not coco_file:
        messagebox.showerror("Error", "Debes seleccionar un archivo COCO")
        return None
    
    # Seleccionar carpeta de imágenes
    image_folder = filedialog.askdirectory(
        title="Selecciona carpeta de imágenes"
    )
    
    if not image_folder:
        messagebox.showerror("Error", "Debes seleccionar una carpeta de imágenes")
        return None
    
    # Seleccionar archivo de salida
    save_file = filedialog.asksaveasfilename(
        title="Guardar resultados MOT como...",
        initialfile="tracking_results.txt",
        defaultextension=".txt",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    
    if not save_file:
        messagebox.showerror("Error", "Debes especificar un archivo de salida")
        return None
    
    root.destroy()
    
    return coco_file, image_folder, save_file


def main():
    """Función principal"""
    print("=" * 60)
    print("TRACKING EDITOR - Simplified Version")
    print("=" * 60)
    
    # Seleccionar archivos
    result = select_files()
    if not result:
        print("Selección de archivos cancelada")
        sys.exit(1)
    
    coco_file, image_folder, save_file = result
    
    print(f"\nArchivo COCO: {coco_file}")
    print(f"Carpeta imágenes: {image_folder}")
    print(f"Archivo salida: {save_file}")
    
    try:
        # Inicializar componentes
        print("\nCargando datos COCO...")
        coco_reader = COCOReader(coco_file)
        print(f"Total de frames: {coco_reader.get_total_frames()}")
        
        print("\nInicializando tracking state...")
        tracking_state = TrackingState(coco_reader)
        
        # Crear ventana principal
        print("\nCreando interfaz gráfica...")
        root = tk.Tk()
        app = TrackingEditorWindow(root, tracking_state, image_folder, save_file)
        
        print("\n" + "=" * 60)
        print("APLICACIÓN INICIADA")
        print("=" * 60)
        print("\nInstrucciones:")
        print("  - Click en bbox para editar ID y visibilidad")
        print("  - Next >>: Procesar siguiente frame")
        print("  - << Prev: Ver frame anterior (sin reprocesar)")
        print("  - Save: Guardar resultados en formato MOT")
        print("  - Reset All: Reiniciar desde el principio")
        print("  - Reset From Here: Reiniciar desde frame actual")
        print("\nIMPORTANTE:")
        print("  - Las bounding boxes son del ground truth (no se predicen)")
        print("  - Solo se hace tracking de los IDs")
        print("  - Los IDs se validan para evitar duplicados en cada frame")
        print("=" * 60 + "\n")
        
        # Iniciar loop principal
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error Fatal", f"Error al iniciar aplicación:\n{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
