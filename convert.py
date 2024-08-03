import tkinter as tk
from tkinter import ttk, filedialog
import rawpy
import cv2
from PIL import Image, ImageTk
import numpy as np
import cuda

class AdvancedCinematicPhotoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Cinematic Photo Editor")
        self.root.geometry("1200x800")
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.original_image = None
        self.edited_image = None
        self.preview_images = {}
        
        # Check GPU availability
        self.use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_gpu:
            print("GPU acceleration enabled")
        else:
            print("GPU acceleration not available. Using CPU.")
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Top frame for buttons
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=(tk.W, tk.E))
        top_frame.columnconfigure(1, weight=1)
        
        ttk.Button(top_frame, text="Open RAW File", command=self.open_file).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(top_frame, text="Save as JPG", command=self.save_file).grid(row=0, column=2, padx=(10, 0))
        
        # Left frame for original image
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=1, column=0, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        ttk.Label(left_frame, text="Original Image").grid(row=0, column=0, pady=(0, 5))
        self.original_image_label = ttk.Label(left_frame)
        self.original_image_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right frame for edited image and controls
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=1, column=1, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)
        
        ttk.Label(right_frame, text="Edited Image").grid(row=0, column=0, pady=(0, 5))
        self.edited_image_label = ttk.Label(right_frame)
        self.edited_image_label.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Controls frame
        controls_frame = ttk.Frame(right_frame)
        controls_frame.grid(row=2, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        controls_frame.columnconfigure(1, weight=1)
        
        ttk.Label(controls_frame, text="Style:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.style_combobox = ttk.Combobox(controls_frame, values=["Warm Vintage", "Cool Modern", "High Contrast", "Soft Pastel", "Cinematic Drama"])
        self.style_combobox.set("Warm Vintage")
        self.style_combobox.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.style_combobox.bind("<<ComboboxSelected>>", self.update_preview)
        
        ttk.Label(controls_frame, text="Intensity:").grid(row=1, column=0, padx=(0, 5), sticky=tk.W)
        self.intensity_slider = ttk.Scale(controls_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        self.intensity_slider.set(50)
        self.intensity_slider.grid(row=1, column=1, sticky=(tk.W, tk.E))
        self.intensity_slider.bind("<ButtonRelease-1>", self.update_preview)
        
        ttk.Button(controls_frame, text="Apply Style", command=self.apply_style).grid(row=2, column=0, columnspan=2, pady=(10, 0))
        
        # Preview frame
        preview_frame = ttk.Frame(main_frame)
        preview_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        self.preview_labels = {}
        for i, style in enumerate(["Warm Vintage", "Cool Modern", "High Contrast", "Soft Pastel", "Cinematic Drama"]):
            frame = ttk.Frame(preview_frame)
            frame.grid(row=0, column=i, padx=5)
            ttk.Label(frame, text=style).pack()
            label = ttk.Label(frame)
            label.pack()
            self.preview_labels[style] = label
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("RAW files", "*.arw *.cr2 *.nef *.dng")])
        if file_path:
            self.original_image = self.load_raw_image(file_path)
            self.display_image(self.original_image, self.original_image_label)
            self.generate_previews()
    
    def save_file(self):
        if self.edited_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                                     filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.edited_image, cv2.COLOR_RGB2BGR))
    
    def load_raw_image(self, file_path):
        with rawpy.imread(file_path) as raw:
            rgb_image = raw.postprocess()
        return rgb_image
    
    def display_image(self, image, label):
        aspect_ratio = image.shape[1] / image.shape[0]
        new_width = min(self.root.winfo_width() // 2 - 20, image.shape[1])
        new_height = int(new_width / aspect_ratio)
        
        image_pil = Image.fromarray(image)
        image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(image_pil)
        label.config(image=image_tk)
        label.image = image_tk
    
    def generate_previews(self):
        if self.original_image is not None:
            preview_size = (100, 100)
            for style in ["Warm Vintage", "Cool Modern", "High Contrast", "Soft Pastel", "Cinematic Drama"]:
                preview = self.apply_color_grading(cv2.resize(self.original_image, preview_size), style)
                preview_pil = Image.fromarray(preview)
                preview_tk = ImageTk.PhotoImage(preview_pil)
                self.preview_labels[style].config(image=preview_tk)
                self.preview_labels[style].image = preview_tk
    
    def update_preview(self, event=None):
        if self.original_image is not None:
            style = self.style_combobox.get()
            intensity = self.intensity_slider.get() / 100
            preview = self.apply_color_grading(self.original_image, style, intensity)
            self.display_image(preview, self.edited_image_label)
    
    def apply_style(self):
        if self.original_image is not None:
            style = self.style_combobox.get()
            intensity = self.intensity_slider.get() / 100
            self.edited_image = self.apply_color_grading(self.original_image, style, intensity)
            self.display_image(self.edited_image, self.edited_image_label)
    
    def apply_color_grading(self, image, style, intensity=0.5):
        analysis = self.analyze_image(image)
        
        if self.use_gpu:
            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(image)
        
        if style == "Warm Vintage":
            graded = self.apply_warm_vintage(gpu_image if self.use_gpu else image, analysis, intensity)
        elif style == "Cool Modern":
            graded = self.apply_cool_modern(gpu_image if self.use_gpu else image, analysis, intensity)
        elif style == "High Contrast":
            graded = self.apply_high_contrast(gpu_image if self.use_gpu else image, analysis, intensity)
        elif style == "Soft Pastel":
            graded = self.apply_soft_pastel(gpu_image if self.use_gpu else image, analysis, intensity)
        elif style == "Cinematic Drama":
            graded = self.apply_cinematic_drama(gpu_image if self.use_gpu else image, analysis, intensity)
        else:
            return image
        
        if self.use_gpu:
            graded = graded.download()
        
        return graded
    
    def analyze_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        height, width = image.shape[:2]
        regions = [
            ("top", hsv[:height//3]),
            ("middle", hsv[height//3:2*height//3]),
            ("bottom", hsv[2*height//3:]),
            ("left", hsv[:,:width//3]),
            ("center", hsv[:,width//3:2*width//3]),
            ("right", hsv[:,2*width//3:])
        ]
        
        region_analysis = {}
        for name, region in regions:
            region_analysis[name] = {
                "avg_hue": np.mean(region[:,:,0]),
                "avg_sat": np.mean(region[:,:,1]),
                "avg_val": np.mean(region[:,:,2])
            }
        
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        dark_threshold = 64
        bright_threshold = 192
        dark_ratio = np.sum(hist[:dark_threshold]) / np.sum(hist)
        bright_ratio = np.sum(hist[bright_threshold:]) / np.sum(hist)
        
        if dark_ratio > 0.5:
            brightness = "dark"
        elif bright_ratio > 0.5:
            brightness = "bright"
        else:
            brightness = "normal"
        
        contrast = np.std(hsv[:,:,2])
        
        pixels = np.float32(image.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant_colors = palette[np.argsort(counts)[::-1]]
        
        return {
            "region_analysis": region_analysis,
            "brightness": brightness,
            "contrast": contrast,
            "dominant_colors": dominant_colors
        }
    
    def apply_warm_vintage(self, image, analysis, intensity):
        if self.use_gpu:
            gpu_gray = cv2.cuda.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gpu_warmth_mask = cv2.cuda.normalize(gpu_gray, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            warmth_mask = gpu_warmth_mask.download()
            
            gpu_hsv = cv2.cuda.cvtColor(image, cv2.COLOR_RGB2HSV)
            hsv = gpu_hsv.download().astype(np.float32)
            
            hsv[:,:,0] = np.mod(hsv[:,:,0] + 10 * warmth_mask * intensity, 180)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + 0.2 * warmth_mask * intensity), 0, 255)
            
            gpu_hsv.upload(hsv.astype(np.uint8))
            gpu_rgb = cv2.cuda.cvtColor(gpu_hsv, cv2.COLOR_HSV2RGB)
            image = gpu_rgb.download()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            warmth_mask = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,0] = np.mod(hsv[:,:,0] + 10 * warmth_mask * intensity, 180)
            hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + 0.2 * warmth_mask * intensity), 0, 255)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        sepia = np.array([[[112, 66, 20]]], dtype=np.float32) / 255.0
        sepia_strength = (1 - warmth_mask) * intensity
        image = (image.astype(np.float32) * (1 - 0.2 * sepia_strength[:,:,np.newaxis]) + 
                 sepia * 0.2 * sepia_strength[:,:,np.newaxis]) * 255
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def apply_cool_modern(self, image, analysis, intensity):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        coolness_mask = cv2.normalize(hsv[:,:,1], None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        hsv = hsv.astype(np.float32)
        hsv[:,:,0] = np.mod(hsv[:,:,0] - 10 * coolness_mask * intensity, 180)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 - 0.1 * coolness_mask * intensity), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        contrast_strength = 1 + (min(1.2, max(1.0, 1.2 - analysis['contrast'] / 128)) - 1) * intensity
        return cv2.addWeighted(image, contrast_strength, image, 0, 0)
    
    def apply_high_contrast(self, image, analysis, intensity):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        local_contrast = cv2.Laplacian(gray, cv2.CV_32F)
        contrast_mask = cv2.normalize(local_contrast, None, 0, 1, cv2.NORM_MINMAX)
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        l = l * (1 + 0.5 * contrast_mask * intensity)
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + 0.3 * contrast_mask * intensity), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        contrast_strength = 1 + 0.2 * intensity
        return cv2.addWeighted(image, contrast_strength, image, 0, 0)
    
    def apply_soft_pastel(self, image, analysis, intensity):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_32F)
        softness_mask = 1 - cv2.normalize(texture, None, 0, 1, cv2.NORM_MINMAX)
        
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)
        l = (l - 127) * (1 - 0.2 * softness_mask * intensity) + 127
        a = (a - 127) * (1 - 0.3 * softness_mask * intensity) + 127
        b = (b - 127) * (1 - 0.3 * softness_mask * intensity) + 127
        lab = cv2.merge((l, a, b))
        image = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        soft_light = np.full_like(image, 255)
        blend_factor = softness_mask * 0.2 * intensity
        return cv2.addWeighted(image, 1 - blend_factor, soft_light, blend_factor, 0)
    
    def apply_cinematic_drama(self, image, analysis, intensity):
        if self.use_gpu:
            gpu_gray = cv2.cuda.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gpu_local_contrast = cv2.cuda.Laplacian(gpu_gray, cv2.CV_32F)
            gpu_drama_mask = cv2.cuda.normalize(gpu_local_contrast.multiply(1 - gpu_gray.divide(255)), 0, 1, cv2.NORM_MINMAX)
            drama_mask = gpu_drama_mask.download()
            
            gpu_lab = cv2.cuda.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab = gpu_lab.download().astype(np.float32)
            
            l, a, b = cv2.split(lab)
            l = l * (1 + 0.3 * drama_mask * intensity)
            lab = cv2.merge((l, a, b))
            
            gpu_lab.upload(lab.astype(np.uint8))
            gpu_rgb = cv2.cuda.cvtColor(gpu_lab, cv2.COLOR_LAB2RGB)
            image = gpu_rgb.download()
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            local_contrast = cv2.Laplacian(gray, cv2.CV_32F)
            drama_mask = cv2.normalize(local_contrast * (1 - gray / 255), None, 0, 1, cv2.NORM_MINMAX)
            
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            l, a, b = cv2.split(lab)
            l = l * (1 + 0.3 * drama_mask * intensity)
            lab = cv2.merge((l, a, b))
            image = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        dominant_color = analysis['dominant_colors'][0] / 255.0
        if np.mean(dominant_color) < 0.5:
            tint = np.array([20, 40, 80]) / 255.0  # Cool shadows
        else:
            tint = np.array([70, 50, 40]) / 255.0  # Warm highlights
        
        tint_strength = drama_mask * 0.15 * intensity
        image = (image.astype(np.float32) * (1 - tint_strength[:,:,np.newaxis]) + 
                 tint * tint_strength[:,:,np.newaxis]) * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        # Add letterbox effect
        height, width = image.shape[:2]
        letterbox_height = int(height * 0.1)
        image[:letterbox_height] = 0
        image[-letterbox_height:] = 0
        
        return image

def main():
    root = tk.Tk()
    app = AdvancedCinematicPhotoEditor(root)
    root.mainloop()

if __name__ == "__main__":
    main()