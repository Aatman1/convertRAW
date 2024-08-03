import tkinter as tk
from tkinter import filedialog, ttk
import rawpy
import cv2
from PIL import Image, ImageTk
import numpy as np
import os

def load_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        rgb_image = raw.postprocess()
    return rgb_image

def apply_color_grading(image, intensity=0.5):
    image = image.astype(np.float32) / 255.0
    grading_matrix = np.array([
        [1.2, 0.0, 0.0],
        [0.0, 1.1, 0.0],
        [0.0, 0.0, 1.3]
    ])
    graded_image = cv2.transform(image, grading_matrix)
    graded_image = cv2.addWeighted(image, 1 - intensity, graded_image, intensity, 0)
    graded_image = np.clip(graded_image, 0, 1)
    graded_image = (graded_image * 255).astype(np.uint8)
    return graded_image

def apply_lut(image, lut_path):
    lut = cv2.imread(lut_path)
    if lut is None:
        print(f"Failed to load LUT: {lut_path}")
        return image
    lut = cv2.cvtColor(lut, cv2.COLOR_BGR2RGB)
    lut = lut.reshape((16, 16, 16, 3))
    return cv2.LUT(image, lut)

def add_letterbox_bars(image, bar_height_ratio=0.1):
    height, width = image.shape[:2]
    bar_height = int(height * bar_height_ratio)
    bar = np.zeros((bar_height, width, 3), dtype=np.uint8)
    image_with_bars = np.vstack([bar, image, bar])
    return image_with_bars

def apply_cinematic_effect(image, style):
    if style == "Color Grading":
        return apply_color_grading(image)
    elif style.startswith("LUT:"):
        lut_path = style.split(": ", 1)[1]
        return apply_lut(image, lut_path)
    elif style == "Letterbox Bars":
        return add_letterbox_bars(image)
    return image

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("RAW files", "*.arw *.cr2 *.nef")])
    if file_path:
        image_rgb = load_raw_image(file_path)
        root.original_image = image_rgb
        display_image(image_rgb, before_image_label)
        update_image_size()

def save_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
    if file_path and root.edited_image is not None:
        cv2.imwrite(file_path, cv2.cvtColor(root.edited_image, cv2.COLOR_RGB2BGR))

def apply_style():
    style = style_combobox.get()
    if root.original_image is not None:
        edited_image = apply_cinematic_effect(root.original_image, style)
        root.edited_image = edited_image
        display_image(edited_image, after_image_label)

def display_image(image, label):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = min(root.winfo_width() // 2 - 20, width)
    new_height = int(new_width / aspect_ratio)

    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    image_tk = ImageTk.PhotoImage(image_pil)
    label.config(image=image_tk)
    label.image = image_tk

def update_image_size(event=None):
    if root.original_image is not None:
        display_image(root.original_image, before_image_label)
    if root.edited_image is not None:
        display_image(root.edited_image, after_image_label)

def load_lut_files():
    lut_options = ["Color Grading", "Letterbox Bars"]
    for folder in ["Contrast Filters", "Film Presets"]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.lower().endswith(('.cube', '.png')):
                    lut_options.append(f"LUT: {os.path.join(folder, file)}")
    return lut_options

root = tk.Tk()
root.title("Cinematic Photo Editor")
root.geometry("1024x768")

root.original_image = None
root.edited_image = None

open_button = tk.Button(root, text="Oen RAW File", command=open_file)
open_button.pack()

lut_options = load_lut_files()
style_combobox = ttk.Combobox(root, values=lut_options)
style_combobox.set("Color Grading")
style_combobox.pack()

apply_button = tk.Button(root, text="Apply Style", command=apply_style)
apply_button.pack()

image_frame = tk.Frame(root)
image_frame.pack(fill=tk.BOTH, expand=True)

before_image_label = tk.Label(image_frame)
before_image_label.pack(side="left", padx=10, pady=10, expand=True)

after_image_label = tk.Label(image_frame)
after_image_label.pack(side="right", padx=10, pady=10, expand=True)

save_button = tk.Button(root, text="Save as JPG", command=save_file)
save_button.pack()

root.bind("<Configure>", update_image_size)

root.mainloop()