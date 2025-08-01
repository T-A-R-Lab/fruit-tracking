import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import numpy as np

def load_mot(mot_file):
    mot = {}
    with open(mot_file, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            vals = line.strip().split(",")
            if len(vals) < 10:
                continue
            frame = int(vals[0])
            tid = int(vals[1])
            x, y, w, h = map(float, vals[2:6])
            class_id = int(vals[7])
            mot.setdefault(frame, []).append({
                "id": tid, "x": x, "y": y, "w": w, "h": h, "class_id": class_id, "raw": vals[:]
            })
    return mot

def save_mot(mot, output_file):
    with open(output_file, "w") as f:
        for frame in sorted(mot.keys()):
            for det in mot[frame]:
                vals = det["raw"]
                vals[1] = str(det["id"])  # Solo track ID cambia
                f.write(",".join(str(x) for x in vals) + "\n")

def get_image_list(image_folder):
    exts = (".jpg", ".png", ".jpeg", ".bmp")
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(exts)]
    files = sorted(files)
    return [os.path.join(image_folder, f) for f in files]

def get_color(idx):
    COLORS = [
        "#ff4040", "#40ff40", "#4040ff", "#ffbf00", "#00bfbf",
        "#bf40ff", "#ff40bf", "#bfff40", "#40bfff", "#ff8040"
    ]
    return COLORS[idx % len(COLORS)]

class MOTEditor:
    def __init__(self, root, mot_file, image_folder, save_file=None):
        self.root = root
        self.mot_data = load_mot(mot_file)
        self.img_list = get_image_list(image_folder)
        self.nframes = len(self.img_list)
        self.curr_idx = 0
        self.selected_bbox = None
        self.save_file = save_file or "mot_edited.txt"
        self.image_folder = image_folder

        self.root.title("MOT Track ID Editor")
        self.canvas = tk.Canvas(root, width=1280, height=900)
        self.canvas.pack()
        
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        self.prev_btn = tk.Button(btn_frame, text="<< Prev", command=self.prev_frame)
        self.prev_btn.pack(side=tk.LEFT)
        self.next_btn = tk.Button(btn_frame, text="Next >>", command=self.next_frame)
        self.next_btn.pack(side=tk.LEFT)
        self.save_btn = tk.Button(btn_frame, text="Save", command=self.save)
        self.save_btn.pack(side=tk.LEFT)
        self.info_lbl = tk.Label(btn_frame, text="")
        self.info_lbl.pack(side=tk.LEFT)

        self.canvas.bind("<Button-1>", self.on_click)

        self.update_view()

    def update_view(self):
        if self.curr_idx < 0: self.curr_idx = 0
        if self.curr_idx >= self.nframes: self.curr_idx = self.nframes-1
        frame = self.curr_idx + 1  # Asumimos que frame 1 es imagen 0
        img_path = self.img_list[self.curr_idx]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((800, 1280, 3), np.uint8)
            cv2.putText(img, "No image", (100,100), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)
            orig_h, orig_w = img.shape[:2]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
        CANVAS_W, CANVAS_H = 1280, 900

        # Mantener aspect ratio (fit)
        scale = min(CANVAS_W / orig_w, CANVAS_H / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (CANVAS_W - new_w) // 2
        pad_y = (CANVAS_H - new_h) // 2

        img_resized = cv2.resize(img, (new_w, new_h))
        img_padded = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        img_padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(img_padded))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        bboxes = self.mot_data.get(frame, [])
        for i, det in enumerate(bboxes):
            # Ajuste de bbox con aspect ratio y padding
            x = int(det['x'] * scale) + pad_x
            y = int(det['y'] * scale) + pad_y
            w = int(det['w'] * scale)
            h = int(det['h'] * scale)
            color = get_color(det["id"])
            tag = f"bbox_{i}"
            rect = self.canvas.create_rectangle(x, y, x+w, y+h, outline=color, width=3, tags=tag)
            txt = self.canvas.create_text(x+5, y+15, anchor=tk.NW, text=f"ID:{det['id']}", fill=color, font=("Arial", 16), tags=tag)
            if self.selected_bbox == i:
                self.canvas.itemconfig(rect, width=6)
        self.info_lbl.config(text=f"Frame {self.curr_idx+1}/{self.nframes} | {img_path}")

    def prev_frame(self):
        self.selected_bbox = None
        self.curr_idx -= 1
        if self.curr_idx < 0: self.curr_idx = 0
        self.update_view()

    def next_frame(self):
        self.selected_bbox = None
        self.curr_idx += 1
        if self.curr_idx >= self.nframes: self.curr_idx = self.nframes-1
        self.update_view()

    def on_click(self, event):
        frame = self.curr_idx + 1
        bboxes = self.mot_data.get(frame, [])
        img = cv2.imread(self.img_list[self.curr_idx])
        if img is None:
            orig_h, orig_w = 800, 1280
        else:
            orig_h, orig_w = img.shape[:2]
        CANVAS_W, CANVAS_H = 1280, 900
        scale = min(CANVAS_W / orig_w, CANVAS_H / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_x = (CANVAS_W - new_w) // 2
        pad_y = (CANVAS_H - new_h) // 2
        for i, det in enumerate(bboxes):
            x = int(det['x'] * scale) + pad_x
            y = int(det['y'] * scale) + pad_y
            w = int(det['w'] * scale)
            h = int(det['h'] * scale)
            if x <= event.x <= x+w and y <= event.y <= y+h:
                self.selected_bbox = i
                self.update_view()
                # Pide nuevo ID
                result = simpledialog.askinteger("Editar ID", 
                        f"Track ID actual: {det['id']}\nNuevo track ID (entero):",
                        initialvalue=det['id'], minvalue=0)
                if result is not None:
                    det['id'] = int(result)
                self.update_view()
                return

    def save(self):
        save_mot(self.mot_data, self.save_file)
        messagebox.showinfo("Guardado", f"Archivo guardado en: {self.save_file}")

def main():
    root = tk.Tk()
    root.withdraw()  # Oculta ventana principal para selección de archivos

    mot_file = filedialog.askopenfilename(title="Selecciona archivo MOT", filetypes=[("MOT Files", "*.txt *.mot"),("All Files", "*.*")])
    if not mot_file:
        messagebox.showerror("Error", "No seleccionaste archivo MOT.")
        return

    image_folder = filedialog.askdirectory(title="Selecciona carpeta de imágenes")
    if not image_folder:
        messagebox.showerror("Error", "No seleccionaste carpeta de imágenes.")
        return

    save_file = filedialog.asksaveasfilename(title="Guardar MOT editado como...", initialfile="mot_edited.txt", defaultextension=".txt")
    if not save_file:
        messagebox.showerror("Error", "Debes elegir un archivo para guardar.")
        return

    root.deiconify()  # Muestra ventana principal
    app = MOTEditor(root, mot_file, image_folder, save_file)
    root.mainloop()

if __name__ == "__main__":
    main()