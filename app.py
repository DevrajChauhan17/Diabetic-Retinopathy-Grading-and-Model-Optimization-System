import os
import customtkinter as ctk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mysql.connector as db_connector
import numpy as np
from PIL import Image

from model import predict

# Ensure crisp rendering on Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDPIAwareness(1)
except Exception:
    pass

# UI styling setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

from dotenv import load_dotenv
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "dr_retina_db")
TABLE_NAME = os.getenv("TABLE_NAME", "retina_prediction_history")

DR_DESC = {
    "No DR": "Normal retina with no signs of diabetic retinopathy.",
    "Mild": "Initial signs such as microaneurysms are present.",
    "Moderate": "Increased lesions indicate ongoing progression.",
    "Severe": "Advanced stage with significant blood vessel damage.",
    "Proliferative DR": "Critical stage with risk of abnormal new vessel growth.",
}

PROJECT_NAME = "Diabetic Retinopathy Grading and Model Optimization System"

class RetinaApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title(PROJECT_NAME)
        self.geometry("1100x700")
        self.minsize(950, 600)
        
        self.db_conn = None
        self.db_cursor = None
        self.image_path: str = ""
        
        self._init_db()
        
        # Configure grid layout (1x2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self._build_sidebar()
        self._build_main_area()
        
    def _init_db(self):
        try:
            admin_conn = db_connector.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            admin_cursor = admin_conn.cursor()
            admin_cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`")
            admin_cursor.execute(f"USE `{DB_NAME}`")
            admin_cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS `{TABLE_NAME}` (
                    id INT NOT NULL AUTO_INCREMENT,
                    patient_name VARCHAR(120) NOT NULL,
                    image_path TEXT NOT NULL,
                    predicted_label VARCHAR(40) NOT NULL,
                    predicted_class INT NOT NULL,
                    confidence FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (id)
                )
                """
            )
            admin_conn.commit()
            admin_cursor.close()
            admin_conn.close()

            self.db_conn = db_connector.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
            )
            self.db_cursor = self.db_conn.cursor()
        except Exception as exc:
            messagebox.showwarning("Database Warning", f"MySQL backend unavailable. Predictions will not be recorded.\n\nError: {exc}")

    def _build_sidebar(self):
        # Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        
        # Logo / Title
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Retina AI", font=ctk.CTkFont(size=32, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(40, 5))
        
        self.subtitle_label = ctk.CTkLabel(self.sidebar_frame, text="DR Diagnostics System", font=ctk.CTkFont(size=14), text_color="gray")
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 30))

        # Input Area
        self.patient_name_entry = ctk.CTkEntry(self.sidebar_frame, placeholder_text="Enter Patient Name...", width=260, height=40)
        self.patient_name_entry.grid(row=2, column=0, padx=20, pady=(20, 15))
        
        self.upload_btn = ctk.CTkButton(self.sidebar_frame, text="Upload Image", command=self.upload_image, height=45, fg_color="#3b82f6", hover_color="#2563eb", text_color="#ffffff", font=ctk.CTkFont(weight="bold"))
        self.upload_btn.grid(row=3, column=0, padx=20, pady=10)
        
        self.analyze_btn = ctk.CTkButton(self.sidebar_frame, text="Analyze Image", command=self.analyze, height=45, fg_color="#10b981", hover_color="#059669", text_color="#ffffff", text_color_disabled="#ffffff", font=ctk.CTkFont(weight="bold"))
        self.analyze_btn.grid(row=4, column=0, padx=20, pady=10)
        self.analyze_btn.configure(state="disabled")

        # Result display
        self.result_textbox = ctk.CTkTextbox(self.sidebar_frame, width=260, height=200, corner_radius=10, font=ctk.CTkFont(family="Consolas", size=13))
        self.result_textbox.grid(row=5, column=0, padx=20, pady=(30, 20), sticky="nsew")
        self.result_textbox.insert("0.0", "System ready.\n\nPlease upload an image to begin.")
        self.result_textbox.configure(state="disabled")
        
        # Appearance Toggle
        self.appearance_mode_switch = ctk.CTkSwitch(self.sidebar_frame, text="Dark Mode", command=self.change_appearance_mode_event)
        self.appearance_mode_switch.grid(row=6, column=0, padx=20, pady=(10, 10), sticky="s")
        self.appearance_mode_switch.select() # Default to dark mode
        
        # Footer
        self.footer_label = ctk.CTkLabel(self.sidebar_frame, text="Supported Formats: PNG, JPG, JPEG", font=ctk.CTkFont(size=11), text_color="gray")
        self.footer_label.grid(row=7, column=0, padx=20, pady=(10, 20), sticky="s")

    def change_appearance_mode_event(self):
        if self.appearance_mode_switch.get() == 1:
            ctk.set_appearance_mode("dark")
        else:
            ctk.set_appearance_mode("light")
        # If an image is currently uploaded, refresh matplotlib so its styles update
        if self.image_path:
            if "=== DIAGNOSTIC REPORT ===" in self.result_textbox.get("0.0", "end"):
                # Meaning analysis was completed
                text = self.result_textbox.get("0.0", "end")
                # Parse out the confidence and label or just re-plot?
                # Actually, to avoid re-predicting, we shouldn't execute self.analyze()
                pass
            else:
                self.upload_image() # Re-draws the preview plot using the set path

    def _build_main_area(self):
        # Main content area
        self.main_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        
        # Top info panel
        self.info_panel = ctk.CTkFrame(self.main_frame, corner_radius=10, height=70, fg_color=("#f1f5f9", "#1e293b"), border_width=1, border_color=("#cbd5e1", "#334155"))
        self.info_panel.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        self.project_label = ctk.CTkLabel(self.info_panel, text=PROJECT_NAME, font=ctk.CTkFont(size=18, weight="bold"), text_color=("#0f172a", "#f8fafc"))
        self.project_label.pack(side="left", padx=25, pady=20)
        
        # Visualization Frame
        self.viz_frame = ctk.CTkFrame(self.main_frame, corner_radius=12, fg_color=("#ffffff", "#0f172a"), border_width=1, border_color=("#cbd5e1", "#1e293b"))
        self.viz_frame.grid(row=1, column=0, sticky="nsew")
        self.viz_frame.grid_rowconfigure(0, weight=1)
        self.viz_frame.grid_columnconfigure(0, weight=1)
        
        # Label to show when nothing is loaded
        self.placeholder_label = ctk.CTkLabel(self.viz_frame, text="No scan loaded.\n\nImage visualization will appear here.", text_color=("#64748b", "#64748b"), font=ctk.CTkFont(size=16))
        self.placeholder_label.grid(row=0, column=0)
        
        # Canvas holder for matplotlib
        self.canvas_widget = None

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not path:
            return
            
        self.image_path = path
        self.analyze_btn.configure(state="normal")
        
        self._set_result("Image loaded from disk.\nReady for clinical analysis.")
        
        # Clear previous plot
        if self.canvas_widget:
            self.canvas_widget.destroy()
            self.canvas_widget = None
            
        # Display image preview
        is_dark = self.appearance_mode_switch.get() == 1
        bg_color = '#0f172a' if is_dark else '#ffffff'
        text_color = '#94a3b8' if is_dark else '#475569'
        
        img = Image.open(path).convert("RGB")
        plt.style.use("dark_background" if is_dark else "default")
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=bg_color)
        ax.imshow(np.array(img))
        ax.axis("off")
        ax.set_title("Input Image Preview", color=text_color, pad=15, fontsize=14)
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.placeholder_label.grid_forget()
        
        # Prevent memory leaks
        plt.close(fig)

    def _set_result(self, text: str):
        self.result_textbox.configure(state="normal")
        self.result_textbox.delete("0.0", "end")
        self.result_textbox.insert("0.0", text)
        self.result_textbox.configure(state="disabled")

    def analyze(self):
        if not self.image_path:
            return
            
        patient_name = self.patient_name_entry.get().strip() or "Anonymous Patient"
        
        self._set_result("Initializing neural network...\nAnalyzing fundus scan...")
        self.update_idletasks()
        
        try:
            value, label, confidence = predict(self.image_path)
            self._save_prediction(patient_name, self.image_path, value, label, confidence)
            
            details = (
                f"=== DIAGNOSTIC REPORT ===\n\n"
                f"Patient ID:  {patient_name}\n"
                f"Diagnosis:   {label.upper()}\n"
                f"Class Logic: Level {value} DR\n"
                f"Certainty:   {confidence:.2f}%\n\n"
                f"--- Medical Assessment ---\n"
                f"{DR_DESC.get(label, 'No description available.')}"
            )
            self._set_result(details)
            self.update_plot(self.image_path, label, confidence)
            
        except Exception as exc:
            messagebox.showerror("Runtime Error", f"Model inference failed:\n{exc}")
            self._set_result("Error encountered during AI inference.")

    def _save_prediction(self, patient_name: str, image_path: str, value: int, label: str, confidence: float):
        db_conn = self.db_conn
        db_cursor = self.db_cursor

        if db_cursor is None or db_conn is None:
            return
            
        db_cursor.execute(
            f"""
            INSERT INTO `{TABLE_NAME}` (
                patient_name, image_path, predicted_label, predicted_class, confidence
            ) VALUES (%s, %s, %s, %s, %s)
            """,
            (patient_name, image_path, label, value, confidence),
        )
        db_conn.commit()

    def update_plot(self, path: str, label: str, confidence: float):
        if self.canvas_widget:
            self.canvas_widget.destroy()
            
        is_dark = self.appearance_mode_switch.get() == 1
        bg_color = '#0f172a' if is_dark else '#ffffff'
        
        img = Image.open(path).convert("RGB")
        plt.style.use("dark_background" if is_dark else "default")
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg_color)
        ax.imshow(np.array(img))
        ax.axis("off")
        
        # Color coding title based on severity
        color_map = {
            "No DR": "#10b981",           # emerald green
            "Mild": "#facc15" if is_dark else "#d97706",    # yellow/amber
            "Moderate": "#f97316",        # orange
            "Severe": "#ef4444",          # red
            "Proliferative DR": "#991b1b" if is_dark else "#7f1d1d" # dark red
        }
        title_color = color_map.get(label, "#ffffff" if is_dark else "#000000")
        
        ax.set_title(f"Prediction: {label} (Confidence: {confidence:.1f}%)", color=title_color, fontsize=17, pad=20, weight="bold")
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        
        # Prevent memory leaks
        plt.close(fig)

if __name__ == "__main__":
    app = RetinaApp()
    app.mainloop()
