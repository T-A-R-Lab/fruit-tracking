import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import numpy as np
from collections import defaultdict
import json

def load_mot(mot_file):
    """
    Carga archivo MOT y retorna: dict {frame: [det1, det2, ...]}
    Cada det es un dict con keys: id, x, y, w, h, class_id
    """
    mot = defaultdict(list)
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
            mot[frame].append({"id": tid, "x": x, "y": y, "w": w, "h": h, "class_id": class_id})
    return mot

def get_image_list(image_folder):
    """
    Devuelve una lista ordenada de imágenes en la carpeta.
    Espera nombres tipo 00000001.jpg, ..., o frame1.jpg, etc.
    """
    exts = (".jpg", ".png", ".jpeg", ".bmp")
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(exts)]
    files = sorted(files)
    return [os.path.join(image_folder, f) for f in files]

def load_coco_categories(coco_file):
    """
    Devuelve un dict category_id -> category_name (si hay coco)
    """
    if not coco_file or not os.path.isfile(coco_file):
        return {}
    with open(coco_file,"r") as f:
        data = json.load(f)
    return {cat["id"]: cat["name"] for cat in data["categories"]}

def get_color(idx):
    COLORS = [
        "red", "green", "orange", "purple", "yellow", "blue", 
        "cyan", "magenta", "brown", "pink", "lime", "olive", "maroon", "navy"
    ]
    return COLORS[idx % len(COLORS)]

def visualize_mot(mot_file, image_folder, output_video, coco_file=None, fps=5):
    mot_data = load_mot(mot_file)
    img_list = get_image_list(image_folder)
    category_map = load_coco_categories(coco_file) if coco_file else {}
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    artists = []

    max_frame = max(mot_data.keys())
    # Chequea longitud mínima
    nframes = min(len(img_list), max_frame)
    print(f"Total imágenes: {len(img_list)}, frames MOT: {max_frame}, visualizando: {nframes}")
    for idx in range(nframes):
        frame_num = idx + 1  # MOT frames suelen empezar en 1
        img_path = img_list[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_artists = []
        img_artist = ax.imshow(img_rgb, animated=True)
        frame_artists.append(img_artist)

        # Detecciones para este frame
        dets = mot_data.get(frame_num, [])
        for det in dets:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            tid = det["id"]
            cls = det["class_id"]
            color = get_color(cls)
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            frame_artists.append(rect)
            cat_name = category_map.get(cls, f"class {cls}")
            label = f"ID:{tid} {cat_name}"
            text = ax.text(
                x, y-10, label, 
                color='black', fontsize=8, 
                bbox=dict(facecolor=color, alpha=0.7, pad=1)
            )
            frame_artists.append(text)
        # Frame info
        title = ax.text(10, 20, f"Frame: {frame_num}/{nframes}, Tracks: {len(dets)}",
                        color='white', fontsize=10, bbox=dict(facecolor='black', alpha=0.7))
        frame_artists.append(title)

        artists.append(frame_artists)

    print("Generando animación...")
    ani = animation.ArtistAnimation(fig, artists, interval=1000/fps, blit=True, repeat_delay=1000)
    print(f"Guardando video en {output_video}...")
    ani.save(output_video, writer='ffmpeg', fps=fps)
    plt.close(fig)
    print("¡Listo!")

if __name__ == "__main__":
    mot_file = "/ws/Frutillas/results/tracking/tracking_results_mot.txt"
    image_folder = "/ws/Frutillas/images/"
    coco_file = "/ws/Frutillas/annotations/annotations_bbox_1_l_week1_30_70.json"
    output_video = "/ws/Frutillas/results/videos/tracking_visualization.mp4" 
    fps = 5
    visualize_mot(mot_file, image_folder, output_video, coco_file, fps)