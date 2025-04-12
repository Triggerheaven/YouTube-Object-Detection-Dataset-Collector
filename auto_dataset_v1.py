import os
import cv2
import yt_dlp
from ultralytics import YOLO 
from tkinter import ttk, messagebox, filedialog
import shutil
import threading
import logging
from datetime import datetime
import zipfile
import customtkinter as ctk 
from concurrent.futures import ThreadPoolExecutor, TimeoutError 
from PIL import Image 
import numpy as np
import time
import json
import subprocess 
import platform 
import onnxruntime
import tkinter as tk

ctk.set_appearance_mode("dark")  
ctk.set_default_color_theme("blue") 

APP_VERSION = "1" 
APP_TITLE = f"Image Collector v{APP_VERSION}  Discord: @xthemast"
OUTPUT_DIR = "output_images"
TEMP_DIR = "temp_videos"
LOG_FILE = "process_log.txt"
POST_DETECTION_CROP_SIZE = 416 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

logging.basicConfig(filename=LOG_FILE,level=logging.INFO,format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s", filemode='w')
logging.info(f"Anwendung gestartet: {APP_TITLE}")
logging.info(f"Output Verzeichnis: {os.path.abspath(OUTPUT_DIR)}")
logging.info(f"Temp Verzeichnis: {os.path.abspath(TEMP_DIR)}")

image_save_counter = 0
counter_lock = threading.Lock()
coco_annotations = []
coco_images = []
coco_categories = [{"id": 0, "name": "enemy", "supercategory": "person"}]
stop_processing_flag = threading.Event()
processing_thread = None
def check_ffmpeg():
    ffmpeg_executable = "ffmpeg"
    try:
        cmd = ["where", ffmpeg_executable] if platform.system() == "Windows" else ["which", ffmpeg_executable]
        process = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0)
        ffmpeg_path = process.stdout.strip().split('\n')[0]
        logging.info(f"ffmpeg executable found at: {ffmpeg_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error(f"'{ffmpeg_executable}' executable not found in system PATH.")
        messagebox.showwarning(
            "Abhängigkeit möglicherweise fehlend: FFmpeg",
            "FFmpeg wurde nicht im Systempfad (PATH) gefunden.\n\n"
            "Die Frame-Extraktion mit FFmpeg ist empfohlen für bessere Leistung und Stabilität.\n\n"
            "Bitte installieren Sie FFmpeg und fügen Sie es zu Ihrem Systempfad hinzu.\n"
            "(Die Anwendung wird versuchen fortzufahren, aber die Frame-Extraktion könnte fehlschlagen oder sehr langsam sein, falls OpenCV intern keine geeignete Methode findet.)"
        )
        return True
    except Exception as e:
        logging.error(f"Unexpected error during ffmpeg check: {e}", exc_info=True)
        messagebox.showerror("FFmpeg Check Fehler", f"Ein Fehler trat beim Prüfen von FFmpeg auf:\n{e}")
        return False
root = ctk.CTk()
root.title(APP_TITLE)
root.geometry("950x800")
root.minsize(800, 650)
root.resizable(True, True)
try:
    font_title = ctk.CTkFont(family="Segoe UI", size=26, weight="bold")
    font_heading = ctk.CTkFont(family="Segoe UI", size=18, weight="bold")
    font_label = ctk.CTkFont(family="Segoe UI", size=14)
    font_entry = ctk.CTkFont(family="Segoe UI", size=14)
    font_button = ctk.CTkFont(family="Segoe UI", size=16, weight="bold")
    font_status = ctk.CTkFont(family="Segoe UI", size=18, weight="bold")
    font_progress_label = ctk.CTkFont(family="Segoe UI", size=12)
    font_small_label = ctk.CTkFont(family="Segoe UI", size=11, slant="italic")
except Exception:
    print("Segoe UI font not found, using fallback.")
    font_title = ctk.CTkFont(size=26, weight="bold")
    font_heading = ctk.CTkFont(size=18, weight="bold")
    font_label = ctk.CTkFont(size=14)
    font_entry = ctk.CTkFont(size=14)
    font_button = ctk.CTkFont(size=16, weight="bold")
    font_status = ctk.CTkFont(size=18, weight="bold")
    font_progress_label = ctk.CTkFont(size=12)
    font_small_label = ctk.CTkFont(size=11, slant="italic")
main_frame = ctk.CTkScrollableFrame(root, fg_color="transparent")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)
main_frame.grid_columnconfigure(0, weight=1)
app_title_label = ctk.CTkLabel(main_frame, text=APP_TITLE, font=font_title, anchor="center")
app_title_label.grid(row=0, column=0, padx=10, pady=(10, 25), sticky="ew")
settings_frame = ctk.CTkFrame(main_frame, corner_radius=10)
settings_frame.grid(row=1, column=0, pady=(0, 15), padx=5, sticky="new")
settings_frame.grid_columnconfigure(1, weight=1)

settings_title = ctk.CTkLabel(settings_frame, text="Einstellungen", font=font_heading)
settings_title.grid(row=0, column=0, columnspan=3, padx=20, pady=(15, 15), sticky="w")
row_idx_settings = 1
input_pady_settings = (7, 7)
input_padx_settings = 20
search_label = ctk.CTkLabel(settings_frame, text="YouTube Suchbegriff:", font=font_label)
search_label.grid(row=row_idx_settings, column=0, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
search_entry = ctk.CTkEntry(settings_frame, placeholder_text="z.B. warzone gameplay no commentary", font=font_entry)
search_entry.grid(row=row_idx_settings, column=1, columnspan=2, padx=input_padx_settings, pady=input_pady_settings, sticky="ew")
row_idx_settings += 1
num_videos_label = ctk.CTkLabel(settings_frame, text="Max. Videos zu prüfen:", font=font_label)
num_videos_label.grid(row=row_idx_settings, column=0, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
num_videos_entry = ctk.CTkEntry(settings_frame, placeholder_text="10", font=font_entry, width=100)
num_videos_entry.grid(row=row_idx_settings, column=1, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
row_idx_settings += 1
interval_label = ctk.CTkLabel(settings_frame, text="Frame-Intervall (Sek.):", font=font_label)
interval_label.grid(row=row_idx_settings, column=0, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
interval_entry = ctk.CTkEntry(settings_frame, placeholder_text="5", font=font_entry, width=100)
interval_entry.grid(row=row_idx_settings, column=1, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
interval_desc = ctk.CTkLabel(settings_frame, text="Zeit zwischen extrahierten Bildern.", font=font_small_label, text_color="gray")
interval_desc.grid(row=row_idx_settings, column=2, padx=(0, input_padx_settings), pady=input_pady_settings, sticky="w")
row_idx_settings += 1
quality_label = ctk.CTkLabel(settings_frame, text="Videoqualität (max):", font=font_label)
quality_label.grid(row=row_idx_settings, column=0, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
quality_combobox = ctk.CTkComboBox(settings_frame, values=["best", "1080p", "720p", "480p"], state="readonly", font=font_entry, dropdown_font=font_entry, width=150)
quality_combobox.set("720p")
quality_combobox.grid(row=row_idx_settings, column=1, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
row_idx_settings += 1
model_label = ctk.CTkLabel(settings_frame, text="YOLO/ONNX-Modell:", font=font_label)
model_label.grid(row=row_idx_settings, column=0, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
model_list = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
    "yolov9c.pt", "yolov9e.pt",# Ensure ultralytics supports these
    "Custom Model (.pt/.onnx)" 
]

model_combobox = ctk.CTkComboBox(settings_frame, values=model_list, state="readonly", font=font_entry, dropdown_font=font_entry, command=lambda choice: toggle_custom_model_path())
model_combobox.set("yolov8n.pt")
model_combobox.grid(row=row_idx_settings, column=1, columnspan=2, padx=input_padx_settings, pady=input_pady_settings, sticky="ew")
row_idx_settings += 1
custom_model_frame = ctk.CTkFrame(settings_frame, fg_color="transparent")
custom_model_frame.grid(row=row_idx_settings, column=1, columnspan=2, padx=input_padx_settings, pady=0, sticky="ew")
custom_model_frame.grid_remove()
custom_model_frame.grid_columnconfigure(0, weight=1)

custom_model_path_entry = ctk.CTkEntry(custom_model_frame, placeholder_text="Pfad zur .pt oder .onnx Modelldatei", font=font_entry)

custom_model_path_entry.grid(row=0, column=0, padx=(0, 10), pady=(0,input_pady_settings[1]), sticky="ew")
custom_model_browse_button = ctk.CTkButton(custom_model_frame, text="Durchsuchen...", font=ctk.CTkFont(size=12), command=lambda: browse_custom_model(), width=100)
custom_model_browse_button.grid(row=0, column=1, padx=(0, 0), pady=(0,input_pady_settings[1]), sticky="e")
row_idx_settings += 1

def toggle_custom_model_path():
    
    if model_combobox.get() == "Custom Model (.pt/.onnx)":
    
        custom_model_frame.grid()
        settings_frame.grid_rowconfigure(row_idx_settings -1, weight=0)
    else:
        custom_model_frame.grid_remove()

def browse_custom_model():
    filepath = filedialog.askopenfilename(
        title="Custom YOLO/ONNX Model auswählen",
        filetypes=[("Model Dateien", "*.pt *.onnx"),
                   ("PyTorch Model", "*.pt"),
                   ("ONNX Model", "*.onnx"),
                   ("Alle Dateien", "*.*")]
    )
    
    if filepath:
        custom_model_path_entry.delete(0, ctk.END)
        custom_model_path_entry.insert(0, filepath)
        logging.info(f"Benutzerdefiniertes Modell ausgewählt: {filepath}")
threshold_label = ctk.CTkLabel(settings_frame, text="Erkennungsschwelle:", font=font_label)
threshold_label.grid(row=row_idx_settings, column=0, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
threshold_entry = ctk.CTkEntry(settings_frame, placeholder_text="0.4", font=font_entry, width=100)
threshold_entry.grid(row=row_idx_settings, column=1, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
threshold_desc = ctk.CTkLabel(settings_frame, text="Mindest-Konfidenz (0.01-1.0).", font=font_small_label, text_color="gray")
threshold_desc.grid(row=row_idx_settings, column=2, padx=(0, input_padx_settings), pady=input_pady_settings, sticky="w")
row_idx_settings += 1
format_label = ctk.CTkLabel(settings_frame, text="Label-Speicherformat:", font=font_label)
format_label.grid(row=row_idx_settings, column=0, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
format_combobox = ctk.CTkComboBox(settings_frame, values=["darknet", "pascal_voc", "coco"], state="readonly", font=font_entry, dropdown_font=font_entry, width=150)
format_combobox.set("darknet")
format_combobox.grid(row=row_idx_settings, column=1, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
row_idx_settings += 1
num_images_label = ctk.CTkLabel(settings_frame, text="Gewünschte Bildanzahl:", font=font_label)
num_images_label.grid(row=row_idx_settings, column=0, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
num_images_entry = ctk.CTkEntry(settings_frame, placeholder_text="100", font=font_entry, width=100)
num_images_entry.grid(row=row_idx_settings, column=1, padx=input_padx_settings, pady=input_pady_settings, sticky="w")
row_idx_settings += 1
precrop_checkbox = ctk.CTkCheckBox(settings_frame, text="Pre-Detection Crop (Bild um Mitte zuschneiden)", font=font_label,
                                    command=lambda: toggle_precrop_inputs())

precrop_checkbox.grid(row=row_idx_settings, column=0, columnspan=3, padx=input_padx_settings, pady=(input_pady_settings[0]+10, 0), sticky="w")
row_idx_settings += 1
precrop_frame = ctk.CTkFrame(settings_frame, fg_color="transparent", border_width=1, border_color=("gray70", "gray30"))
precrop_frame.grid(row=row_idx_settings, column=0, columnspan=3, padx=input_padx_settings, pady=(5, input_pady_settings[1]), sticky="ew")
precrop_frame.grid_remove()
precrop_frame.grid_columnconfigure(1, weight=1)


precrop_inner_pady = (5,5)
precrop_inner_padx = 10
entry_width_precrop = 100

precrop_desc = ctk.CTkLabel(precrop_frame, text="Schneidet das Bild quadratisch um dessen Mitte zu, bevor YOLO die Objekte sucht.", font=font_small_label, text_color="gray")
precrop_desc.grid(row=0, column=0, columnspan=2, padx=precrop_inner_padx, pady=(5, 5), sticky="w")

precrop_size_label = ctk.CTkLabel(precrop_frame, text="Zielgröße (Pixel):", font=font_label)
precrop_size_label.grid(row=1, column=0, padx=(precrop_inner_padx, 5), pady=precrop_inner_pady, sticky="w")
precrop_size_entry = ctk.CTkEntry(precrop_frame, placeholder_text="416", width=entry_width_precrop, font=font_entry)
precrop_size_entry.grid(row=1, column=1, padx=(0, precrop_inner_padx), pady=precrop_inner_pady, sticky="w")


def toggle_precrop_inputs():
    if precrop_checkbox.get() == 1:
        precrop_frame.grid()
        logging.debug("Pre-Crop Eingabefeld (Center Crop) eingeblendet.")
    else:
        precrop_frame.grid_remove()
        logging.debug("Pre-Crop Eingabefeld (Center Crop) ausgeblendet.")

row_idx_settings += 1
status_frame = ctk.CTkFrame(main_frame, corner_radius=10)
status_frame.grid(row=2, column=0, pady=(0, 15), padx=5, sticky="new")
status_frame.grid_columnconfigure(0, weight=1)

status_title = ctk.CTkLabel(status_frame, text="Status & Fortschritt", font=font_heading)
status_title.grid(row=0, column=0, padx=20, pady=(15, 10), sticky="w")

status_label = ctk.CTkLabel(status_frame, text="Status: Bereit", font=font_status, text_color="#3399FF")
status_label.grid(row=1, column=0, padx=20, pady=(5, 15), sticky="ew")
progress_labels_texts = [
    "1. Video Suche",
    "2. Video Download",
    "3. Frame Extraktion",
    "4. Pre-Crop (Mitte)",
    "5. YOLO/ONNX Erkennung",
    "6. Post-Crop / Vorbereitung",
    "7. Bild/Label Speichern",
    "8. Temp Bereinigung",
    "9. Gesamt Bilder Ziel"
]

progress_bars = []
progress_section_frame = ctk.CTkFrame(status_frame, fg_color="transparent")
progress_section_frame.grid(row=2, column=0, sticky="ew", padx=25, pady=(0, 15))
progress_section_frame.grid_columnconfigure(0, weight=1)

for i, label_text in enumerate(progress_labels_texts):
    progress_item_frame = ctk.CTkFrame(progress_section_frame, fg_color="transparent")
    progress_item_frame.grid(row=i, column=0, sticky="ew", pady=(6, 0))
    progress_item_frame.grid_columnconfigure(0, weight=1)

    lbl = ctk.CTkLabel(progress_item_frame, text=label_text + ":", font=font_progress_label, anchor="w")
    lbl.grid(row=0, column=0, sticky="ew", pady=(0, 2))

    bar = ctk.CTkProgressBar(progress_item_frame, height=16, corner_radius=8)
    bar.set(0)
    bar.grid(row=1, column=0, sticky="ew", pady=(0, 6))
    progress_bars.append(bar)
def update_gui(func, *args):
    if root and root.winfo_exists():
        try:
            root.after(0, func, *args)
        except tk.TclError as e:
            if "application has been destroyed" in str(e):
                logging.warning("GUI update attempted after window destroyed.")
            else:
                logging.error(f"GUI Update TclError: {e}")
        except Exception as e:
             logging.error(f"Unexpected error in update_gui: {e}", exc_info=True)

def set_progress(bar_index, value):
    if 0 <= bar_index < len(progress_bars):
        clamped_value = min(max(float(value), 0.0), 1.0)
        bar = progress_bars[bar_index]
        if bar and bar.winfo_exists():
             bar.set(clamped_value)
    else:
         logging.warning(f"Attempted to update non-existent progress bar index: {bar_index}")


def set_status(text):
    if status_label and status_label.winfo_exists():
         status_label.configure(text=f"Status: {text}")
def export_results():
    output_path_abs = os.path.abspath(OUTPUT_DIR)
    exportable_files_found = False
    if os.path.exists(output_path_abs) and os.listdir(output_path_abs):
         exportable_files_found = any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.txt', '.xml', '.json'))
                                      for f in os.listdir(output_path_abs))

    if not exportable_files_found:
         messagebox.showwarning("Exportieren nicht möglich",
                                f"Keine exportierbaren Ergebnisse (Bilder, Labels etc.) im Ordner gefunden:\n{output_path_abs}")
         logging.warning(f"Export fehlgeschlagen: Keine exportierbaren Dateien in {output_path_abs}")
         return

    try:
        default_zip_name = f"warzone_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = filedialog.asksaveasfilename(
            title="Ergebnisse als ZIP speichern",
            initialdir=os.getcwd(),
            initialfile=default_zip_name,
            defaultextension=".zip",
            filetypes=[("ZIP Archiv", "*.zip")]
        )

        if zip_path:
            logging.info(f"Starte Export nach: {zip_path}")
            update_gui(set_status, "Exportiere Ergebnisse als ZIP...")
            update_gui(lambda: export_button.configure(state="disabled", text="Exportiere..."))
            def do_export_thread():
                nonlocal zip_path, output_path_abs
                files_added = 0
                try:
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                        for root_dir_walk, _, files in os.walk(output_path_abs):
                            for file in files:
                                file_path = os.path.join(root_dir_walk, file)
                                arcname = os.path.relpath(file_path, output_path_abs)
                                zipf.write(file_path, arcname)
                                files_added += 1
                    logging.info(f"Export erfolgreich abgeschlossen. {files_added} Dateien zu {zip_path} hinzugefügt.")
                    update_gui(set_status, "Export abgeschlossen.")
                    messagebox.showinfo("Export Erfolgreich",
                                        f"{files_added} Ergebnis-Dateien erfolgreich exportiert nach:\n{zip_path}")

                except Exception as e_zip:
                    logging.error(f"Fehler beim Erstellen der ZIP-Datei '{zip_path}': {e_zip}", exc_info=True)
                    update_gui(set_status, "Fehler beim ZIP-Export.")
                    messagebox.showerror("Export Fehler", f"Fehler beim Erstellen der ZIP-Datei:\n{e_zip}")
                finally:
                    update_gui(lambda: export_button.configure(state="normal", text="Export als ZIP"))
            threading.Thread(target=do_export_thread, daemon=True, name="ExportThread").start()

    except Exception as e_dialog:
        logging.error(f"Fehler beim Öffnen des 'Speichern unter'-Dialogs: {e_dialog}", exc_info=True)
        update_gui(set_status, "Fehler beim Export-Dialog.")
        messagebox.showerror("Export Fehler", f"Fehler beim Vorbereiten des Exports:\n{e_dialog}")
def open_output_directory():
    path = os.path.abspath(OUTPUT_DIR)
    try:
        if not os.path.exists(path):
             messagebox.showerror("Fehler", f"Das Ausgabeverzeichnis wurde nicht gefunden:\n{path}")
             logging.error(f"Konnte Output-Verzeichnis nicht finden: {path}")
             return

        logging.info(f"Versuche, Verzeichnis zu öffnen: {path}")
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path], check=True)
        else:
            subprocess.run(["xdg-open", path], check=True)
        logging.info(f"Output-Verzeichnis erfolgreich geöffnet.")

    except FileNotFoundError:
         messagebox.showerror("Fehler", f"Das Ausgabeverzeichnis wurde nicht gefunden (erneut geprüft):\n{path}")
         logging.error(f"Konnte Output-Verzeichnis nicht finden (FileNotFoundError): {path}")
    except subprocess.CalledProcessError as e:
         messagebox.showerror("Fehler", f"Konnte das Ausgabeverzeichnis nicht öffnen (Befehl fehlgeschlagen):\n{e}")
         logging.error(f"Fehler beim Öffnen des Output-Verzeichnisses via Subprozess {path}: {e}", exc_info=True)
    except Exception as e:
        messagebox.showerror("Fehler", f"Ein unerwarteter Fehler trat beim Öffnen des Verzeichnisses auf:\n{e}")
        logging.error(f"Allgemeiner Fehler beim Öffnen des Output-Verzeichnisses {path}: {e}", exc_info=True)
actions_frame = ctk.CTkFrame(main_frame, corner_radius=10)
actions_frame.grid(row=3, column=0, pady=(0, 15), padx=5, sticky="new")
actions_frame.grid_columnconfigure((0, 1, 2), weight=1, uniform="actions")

actions_title = ctk.CTkLabel(actions_frame, text="Aktionen", font=font_heading)
actions_title.grid(row=0, column=0, columnspan=3, padx=20, pady=(15, 10), sticky="w")

button_height = 45
button_pad_actions = 15

start_button = ctk.CTkButton(actions_frame, text="Start", command=lambda: start_processing_thread_safe(), height=button_height, font=font_button)
start_button.grid(row=1, column=0, padx=button_pad_actions, pady=15, sticky="ew")
stop_fg_color = ("#D32F2F", "#E57373")
stop_hover_color = ("#B71C1C", "#D32F2F")
stop_button = ctk.CTkButton(actions_frame, text="Stop", command=lambda: request_stop_processing(), height=button_height, font=font_button,
                             state="disabled", fg_color="grey", hover=False)
stop_button.grid(row=1, column=1, padx=button_pad_actions, pady=15, sticky="ew")

export_button = ctk.CTkButton(actions_frame, text="Export als ZIP", command=export_results, height=button_height, font=font_button)
export_button.grid(row=1, column=2, padx=button_pad_actions, pady=15, sticky="ew")
output_info_frame = ctk.CTkFrame(actions_frame, fg_color="transparent")
output_info_frame.grid(row=2, column=0, columnspan=3, padx=button_pad_actions, pady=(0, 15), sticky="ew")
output_info_frame.grid_columnconfigure(0, weight=1)

output_dir_label_text = f"Ausgabe in: {os.path.abspath(OUTPUT_DIR)}"
output_dir_label = ctk.CTkLabel(output_info_frame, text=output_dir_label_text, font=font_small_label, anchor="w", wraplength=650)
output_dir_label.grid(row=0, column=0, padx=(0, 10), sticky="ew")

open_output_button = ctk.CTkButton(output_info_frame, text="Ordner öffnen", command=open_output_directory, height=30, width=120, font=ctk.CTkFont(size=12))
open_output_button.grid(row=0, column=1, padx=(0, 0), sticky="e")

def validate_inputs():
    validation_errors = []
    logging.info("Starte Eingabevalidierung...")
    search_term = search_entry.get()
    if not search_term:
        logging.warning("Suchbegriff ist leer. Standard 'warzone gameplay' wird verwendet.")
    try:
        num_videos_str = num_videos_entry.get() or "10"
        num_videos = int(num_videos_str)
        if num_videos <= 0:
            validation_errors.append("Max. Videos zu prüfen muss größer als 0 sein.")
        elif num_videos > 500:
            logging.warning(f"Angegebene max. Videoanzahl ({num_videos}) ist sehr hoch.")
    except ValueError:
        validation_errors.append("Max. Videos zu prüfen muss eine gültige Zahl sein.")
    try:
        interval_str = interval_entry.get() or "5"
        interval = float(interval_str)
        if interval <= 0.1:
            validation_errors.append("Frame-Intervall muss größer als 0.1 Sekunden sein.")
        elif interval > 60:
             logging.warning(f"Angegebenes Frame-Intervall ({interval}s) ist sehr groß.")
    except ValueError:
        validation_errors.append("Frame-Intervall muss eine gültige Zahl sein (z.B. 5 oder 2.5).")
    try:
        threshold_str = threshold_entry.get() or "0.4"
        threshold = float(threshold_str)
        if not (0.01 <= threshold <= 1.0):
            validation_errors.append("Erkennungsschwelle muss zwischen 0.01 und 1.0 liegen.")
    except ValueError:
        validation_errors.append("Erkennungsschwelle muss eine gültige Zahl sein (z.B. 0.4).")
    try:
        desired_images_str = num_images_entry.get() or "100"
        desired_images = int(desired_images_str)
        if desired_images <= 0:
            validation_errors.append("Gewünschte Bildanzahl muss größer als 0 sein.")
    except ValueError:
        validation_errors.append("Gewünschte Bildanzahl muss eine gültige Zahl sein.")
    if model_combobox.get() == "Custom Model (.pt/.onnx)":
        model_path = custom_model_path_entry.get()
        if not model_path:
             validation_errors.append("Pfad für Custom Model muss angegeben werden, wenn ausgewählt.")
        elif not os.path.exists(model_path):
             validation_errors.append(f"Custom Model Datei nicht gefunden: {model_path}")
        elif not os.path.isfile(model_path):
             validation_errors.append(f"Custom Model Pfad ist keine Datei: {model_path}")
        elif not model_path.lower().endswith((".pt", ".onnx")):
            validation_errors.append(f"Custom Model Pfad '{model_path}' muss auf .pt oder .onnx enden.")
            logging.warning(f"Custom Model Pfad '{model_path}' hat keine Standardendung (.pt, .onnx).")
    if precrop_checkbox.get() == 1:
        pc_size_str = precrop_size_entry.get()
        if not pc_size_str:
             validation_errors.append("Pre-Crop Zielgröße muss angegeben werden, wenn Pre-Crop aktiviert ist.")
        else:
            try:
                pc_size = int(pc_size_str)
                if pc_size <= 0:
                    validation_errors.append("Pre-Crop Zielgröße muss größer als 0 sein.")
                elif pc_size > 4000:
                     logging.warning(f"Angegeben Pre-Crop Größe ({pc_size}) ist sehr groß.")
            except ValueError:
                validation_errors.append("Pre-Crop Zielgröße muss eine gültige Zahl sein (z.B. 416).")
        logging.info(f"Pre-Crop Eingaben validiert: Größe={pc_size_str or 'N/A'}")
    if validation_errors:
        error_message = "Bitte korrigieren Sie die folgenden Eingaben:\n\n- " + "\n- ".join(validation_errors)
        messagebox.showerror("Ungültige Eingabe", error_message)
        logging.error(f"Input validation failed: {validation_errors}")
        return False
    else:
        logging.info("Eingabevalidierung erfolgreich.")
        return True


def request_stop_processing():
    if processing_thread and processing_thread.is_alive():
        logging.info("Stop-Anforderung erhalten.")
        update_gui(set_status, "Stopp wird angefordert...")
        stop_processing_flag.set()
        update_gui(lambda: stop_button.configure(text="Stoppt...", fg_color="grey", hover=False, state="disabled"))
    else:
        logging.info("Stop-Anforderung erhalten, aber kein Prozess läuft.")
        update_gui(lambda: stop_button.configure(state="disabled", text="Stop", fg_color="grey", hover=False))


def start_processing_thread_safe():
    global processing_thread
    if processing_thread and processing_thread.is_alive():
        logging.warning("Verarbeitung läuft bereits. Start ignoriert.")
        messagebox.showwarning("Läuft bereits", "Der Verarbeitungsprozess ist bereits aktiv.\nBitte warten Sie oder klicken Sie auf 'Stop'.")
        return
    _ = check_ffmpeg()
    if not validate_inputs():
        logging.warning("Start abgebrochen wegen ungültiger Eingaben.")
        return
    logging.info("Eingaben validiert, starte GUI-Reset und Thread...")
    try:
        update_gui(lambda: start_button.configure(state="disabled", fg_color="grey"))
        update_gui(lambda: stop_button.configure(state="normal", text="Stop", fg_color=stop_fg_color, hover_color=stop_hover_color))
        stop_processing_flag.clear()
        for i in range(len(progress_bars)):
            update_gui(set_progress, i, 0)
        update_gui(set_status, "Initialisiere...")
        if format_combobox.get() == "coco":
            global coco_annotations, coco_images
            with counter_lock:
                 coco_annotations = []
                 coco_images = []
            logging.info("COCO Daten für neuen Lauf zurückgesetzt.")

    except Exception as e_gui_reset:
         logging.error(f"Fehler beim Zurücksetzen der GUI vor Thread-Start: {e_gui_reset}", exc_info=True)
         update_gui(lambda: start_button.configure(state="normal", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]))
         update_gui(lambda: stop_button.configure(state="disabled", text="Stop", fg_color="grey", hover=False))
         return
    try:
        processing_thread = threading.Thread(target=start_process, daemon=True, name="ProcessingThread")
        processing_thread.start()
        logging.info("ProcessingThread erfolgreich gestartet.")
    except Exception as e_thread_start:
         logging.critical(f"Konnte ProcessingThread nicht starten: {e_thread_start}", exc_info=True)
         messagebox.showerror("Fehler", f"Verarbeitungsthread konnte nicht gestartet werden:\n{e_thread_start}")
         update_gui(lambda: start_button.configure(state="normal", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]))
         update_gui(lambda: stop_button.configure(state="disabled", text="Stop", fg_color="grey", hover=False))
         update_gui(set_status, "Fehler beim Thread-Start")
def search_youtube(search_term, max_results, quality, progress_bar_index):
    if stop_processing_flag.is_set():
        logging.info("search_youtube: Stop-Flag vor Beginn erkannt.")
        return []
    update_gui(set_status, f"Suche Videos für '{search_term}'...")
    update_gui(set_progress, progress_bar_index, 0.05)

    logging.info(f"Starte YouTube-Suche: Begriff='{search_term}', Max={max_results}")
    ydl_opts = {
        'extract_flat': 'discard_in_playlist',
        'playlistend': max_results,
        'default_search': f"ytsearch{max_results}",
        'ignoreerrors': True,
        'geo_bypass': True,
        'skip_download': True,
        'quiet': True,
        'noprogress': True,
        'logtostderr': False,
        'socket_timeout': 30,
    }

    video_urls = []
    try:
        logging.debug(f"Initialisiere yt_dlp.YoutubeDL mit Optionen: {ydl_opts}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = search_term if search_term.startswith("ytsearch") else f"ytsearch{max_results}:{search_term}"
            logging.info(f"Führe ydl.extract_info für Suchanfrage aus: '{search_query}'")
            start_extract = time.time()

            result = ydl.extract_info(search_query, download=False)

            end_extract = time.time()
            logging.info(f"ydl.extract_info Aufruf beendet nach {end_extract - start_extract:.2f} Sekunden.")
            if stop_processing_flag.is_set():
                logging.info("Suche abgebrochen (Stop-Flag nach extract_info).")
                return []
            if not result:
                logging.warning(f"YouTube-Suche für '{search_term}' lieferte kein Ergebnis (None oder leer).")
                update_gui(set_status, f"Keine Videos für '{search_term}' gefunden.")
                update_gui(set_progress, progress_bar_index, 1.0)
                return []
            entries = result.get('entries')

            if entries and isinstance(entries, list):
                logging.info(f"Anzahl gefundener Suchergebnisse (vor Filterung): {len(entries)}")
                count = 0
                processed_ids = set()

                for i, entry in enumerate(entries):
                    if stop_processing_flag.is_set():
                        logging.info("URL-Filterung abgebrochen (Stop-Flag).")
                        break
                    if not entry or not isinstance(entry, dict):
                        logging.warning(f"Ungültiger Eintrag #{i+1} in Suchergebnissen übersprungen: {entry}")
                        continue
                    url = entry.get('url') or entry.get('webpage_url')
                    video_id = entry.get('id')
                    title = entry.get('title', 'N/A')
                    duration = entry.get('duration')
                    is_youtube_video = url and isinstance(url, str) and ('youtube.com/watch?v=' in url or 'youtu.be/' in url)

                    if is_youtube_video and video_id and video_id not in processed_ids:
                        if duration and duration < 60:
                             logging.debug(f"  -> Video übersprungen (zu kurz: {duration}s): ID={video_id}, Title={title[:50]}...")
                             continue

                        video_urls.append(url)
                        processed_ids.add(video_id)
                        logging.debug(f"  -> Gültige & neue URL hinzugefügt: {url} (ID: {video_id}, Dauer: {duration}s)")
                        count += 1
                        update_gui(set_progress, progress_bar_index, 0.1 + 0.8 * (count / max_results))
                    elif not is_youtube_video:
                        logging.debug(f"  -> Eintrag übersprungen (keine gültige YT URL): ID={video_id}, URL={url}, Title={title[:50]}...")
                    elif video_id in processed_ids:
                         logging.debug(f"  -> Eintrag übersprungen (Duplikat ID): ID={video_id}, URL={url}, Title={title[:50]}...")
                    else:
                         logging.debug(f"  -> Eintrag übersprungen (Grund unklar): ID={video_id}, URL={url}, Title={title[:50]}...")
                    if count >= max_results:
                        logging.info(f"Maximale Anzahl ({max_results}) an gültigen URLs erreicht.")
                        break

                logging.info(f"{len(video_urls)} gültige und eindeutige YouTube Video-URLs nach Filterung gefunden.")
                if not video_urls and entries:
                    logging.warning(f"Keine gültigen URLs nach Filterung der {len(entries)} Suchergebnisse. Überprüfen Sie Filterkriterien (z.B. Dauer).")

            elif isinstance(result, dict) and ('url' in result or 'webpage_url' in result):
                 url = result.get('url') or result.get('webpage_url')
                 video_id = result.get('id')
                 if url and ('youtube.com' in url or 'youtu.be' in url):
                     logging.info(f"Einzelnes Video direkt gefunden: {url}")
                     video_urls.append(url)
                 else:
                      logging.warning(f"Einzelnes Ergebnis, aber keine gültige YT URL: {url}")
            else:
                logging.error(f"YouTube-Suche für '{search_term}' lieferte unerwartetes Ergebnisformat.")
                logging.debug(f"Raw yt-dlp result type: {type(result)}")
                logging.debug(f"Raw yt-dlp result (Ausschnitt): {str(result)[:1000]}...")
                update_gui(set_status, f"Fehler bei YouTube-Suche (Unerwartetes Format)")
        update_gui(set_progress, progress_bar_index, 1.0)
        return video_urls

    except yt_dlp.utils.DownloadError as e:
         logging.error(f"yt-dlp Fehler während der Video-Suche/Info-Extraktion: {e}", exc_info=False)
         update_gui(set_status, f"Fehler bei YouTube-Suche (yt-dlp)")
         update_gui(set_progress, progress_bar_index, 1.0)
         return []
    except Exception as e:
        logging.error(f"Unerwarteter Fehler in search_youtube: {type(e).__name__}: {e}", exc_info=True)
        update_gui(set_progress, progress_bar_index, 1.0)
        update_gui(set_status, f"Schwerer Fehler bei YouTube-Suche")
        return []
def download_video(video_url, output_path, quality, progress_bar_index, video_idx, total_videos):
    if stop_processing_flag.is_set():
        logging.info(f"download_video V:{video_idx+1}: Stop-Flag vor Beginn erkannt.")
        return None

    status_msg = f"Downloade Video {video_idx + 1}/{total_videos}..."
    update_gui(set_status, status_msg)
    update_gui(set_progress, progress_bar_index, 0)
    downloaded_file_path = None
    final_expected_path_mp4 = None
    start_time = time.time()
    def progress_hook(d):
        nonlocal downloaded_file_path, status_msg, final_expected_path_mp4
        if stop_processing_flag.is_set():
            logging.debug(f"Download-Hook V:{video_idx+1}: Stop-Flag erkannt.")
            raise yt_dlp.utils.DownloadCancelled("Download cancelled by user request via stop flag.")

        if d['status'] == 'downloading':
            total = d.get('total_bytes_estimate') or d.get('total_bytes')
            downloaded = d.get('downloaded_bytes', 0)
            if total and total > 0:
                progress = downloaded / total
                update_gui(set_progress, progress_bar_index, progress)

        elif d['status'] == 'finished':
            info_dict = d.get('info_dict', {})
            filepath_hook = info_dict.get('filepath') or d.get('filename') or info_dict.get('_filename')
            if final_expected_path_mp4 and os.path.exists(final_expected_path_mp4):
                 downloaded_file_path = final_expected_path_mp4
            elif filepath_hook:
                 downloaded_file_path = filepath_hook
            else:
                 logging.warning(f"Download-Hook V:{video_idx+1} 'finished' ohne Dateipfad.")
                 if final_expected_path_mp4: downloaded_file_path = final_expected_path_mp4

            update_gui(set_progress, progress_bar_index, 1.0)
            elapsed_time = time.time() - start_time
            logging.info(f"Download V:{video_idx+1} abgeschlossen (Status 'finished'). Datei: {os.path.basename(downloaded_file_path or 'N/A')}, Dauer: {elapsed_time:.2f}s")

        elif d['status'] == 'error':
             logging.error(f"yt-dlp Download-Hook meldet Fehler für Video {video_idx+1}. Status: {d.get('status')}")
             error_msg = d.get('error') or d.get('fragment_error') or 'Unbekannter Hook-Fehler'
             logging.error(f"  -> Hook-Fehlermeldung: {error_msg}")
             update_gui(set_status, f"Fehler Download V:{video_idx+1}")
    format_string = 'bv[ext=mp4][height<=?1080]+ba[ext=m4a]/b[ext=mp4][height<=?1080]'
    if quality != "best":
        try:
            q_val = int(quality[:-1])
            format_string = f'bv[height<={q_val}][ext=mp4]+ba[ext=m4a]/b[height<={q_val}][ext=mp4]/bv+ba/b'
            logging.info(f"Verwende benutzerdefiniertes Qualitätsformat: max {q_val}p")
        except ValueError:
            logging.warning(f"Ungültige Qualität '{quality}' angegeben, verwende Standardformat (max 1080p mp4).")
    video_id_for_file = f"vid_{video_idx+1}_{int(time.time())}"
    try:
        info_opts = {'quiet': True, 'skip_download': True, 'forceid': True, 'youtube_skip_dash_manifest': True}
        with yt_dlp.YoutubeDL(info_opts) as ydl_info:
            info_dict = ydl_info.extract_info(video_url, download=False)
            if info_dict and info_dict.get('id'):
                sanitized_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in info_dict['id'])
                video_id_for_file = sanitized_id
            else:
                 logging.warning(f"Konnte Video-ID für URL {video_url} nicht schnell extrahieren, verwende generierten Namen: {video_id_for_file}")
    except Exception as e_info:
        logging.warning(f"Fehler beim schnellen Extrahieren der Video-ID (verwende generierten Namen {video_id_for_file}): {e_info}")
    filename_template = os.path.join(output_path, f"{video_id_for_file}.%(ext)s")
    final_expected_path_mp4 = os.path.join(output_path, f"{video_id_for_file}.mp4")
    ydl_opts_dl = {
        'format': format_string,
        'outtmpl': filename_template,
        'quiet': True,
        'noprogress': True,
        'progress_hooks': [progress_hook],
        'ignoreerrors': True,
        'socket_timeout': 60,
        'retries': 5,
        'fragment_retries': 5,
        'geo_bypass': True,
        'concurrent_fragment_downloads': 4,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        },{
            'key': 'FFmpegMetadata',
            'add_metadata': True,
        },{
            'key': 'FFmpegEmbedSubtitle',
            'already_have_subtitle': False
        }],
        'ffmpeg_location': None,
        'keepvideo': False,
        'writethumbnail': False,
        'writesubtitles': False,
    }
    try:
        logging.info(f"Starte Download V:{video_idx+1} (ID: {video_id_for_file}, URL: {video_url[:50]}...)")
        logging.debug(f"Verwendete yt-dlp Optionen für Download: {ydl_opts_dl}")
        with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
            ydl.download([video_url])
        if stop_processing_flag.is_set():
            logging.info(f"Download V:{video_idx+1} abgebrochen (Stop-Flag nach ydl.download).")
            cleanup_temp_files(final_expected_path_mp4, 0, 0, 0)
            temp_part_file = final_expected_path_mp4 + ".part"
            if os.path.exists(temp_part_file): cleanup_temp_files(temp_part_file, 0, 0, 0)
            return None
        if os.path.exists(final_expected_path_mp4) and os.path.getsize(final_expected_path_mp4) > 1024:
            logging.info(f"Video V:{video_idx+1} erfolgreich heruntergeladen und als MP4 gespeichert: {final_expected_path_mp4}")
            return final_expected_path_mp4
        elif downloaded_file_path and os.path.exists(downloaded_file_path) and os.path.getsize(downloaded_file_path) > 1024:
             logging.warning(f"Finale MP4 V:{video_idx+1} nicht unter erwartetem Pfad '{final_expected_path_mp4}' gefunden/gültig, aber Hook meldete '{downloaded_file_path}'. Verwende diesen.")
             if not downloaded_file_path.lower().endswith(".mp4"):
                 try:
                     new_path = os.path.splitext(downloaded_file_path)[0] + ".mp4"
                     if os.path.exists(new_path) and downloaded_file_path != new_path:
                          logging.warning(f"Zieldatei '{new_path}' existiert bereits. Verwende ursprünglichen Hook-Pfad.")
                          return downloaded_file_path
                     shutil.move(downloaded_file_path, new_path)
                     logging.info(f"Datei umbenannt zu: {new_path}")
                     return new_path
                 except Exception as rename_err:
                     logging.error(f"Konnte heruntergeladene Datei '{downloaded_file_path}' nicht zu .mp4 umbenennen: {rename_err}")
                     return downloaded_file_path
             else:
                 return downloaded_file_path
        else:
            logging.error(f"Download V:{video_idx+1} abgeschlossen laut yt-dlp, aber gültige Datei nicht gefunden.")
            logging.error(f"  Erwartet: {final_expected_path_mp4}")
            logging.error(f"  Hook gemeldet: {downloaded_file_path}")
            found_temp_files = [f for f in os.listdir(output_path) if video_id_for_file in f]
            if found_temp_files:
                logging.error(f"  -> Möglicherweise verwandte Dateien im Temp-Ordner gefunden: {found_temp_files}.")
            else:
                 logging.error(f"  -> Keine Dateien mit ID '{video_id_for_file}' im Temp-Ordner gefunden.")
            update_gui(set_progress, progress_bar_index, 1.0)
            update_gui(set_status,f"Fehler Download V:{video_idx+1} (Datei fehlt)")
            return None

    except yt_dlp.utils.DownloadCancelled:
        logging.info(f"Download V:{video_idx+1} manuell abgebrochen durch Stop-Anforderung.")
        cleanup_temp_files(final_expected_path_mp4, 0, 0, 0)
        temp_part_file = final_expected_path_mp4 + ".part"
        if os.path.exists(temp_part_file): cleanup_temp_files(temp_part_file, 0, 0, 0)
        return None
    except yt_dlp.utils.DownloadError as e_dl:
        logging.error(f"yt-dlp Download-Fehler V:{video_idx+1}: {e_dl}", exc_info=False)
        update_gui(set_progress, progress_bar_index, 1.0)
        update_gui(set_status,f"Fehler Download V:{video_idx+1}")
        cleanup_temp_files(final_expected_path_mp4, 0, 0, 0)
        return None
    except Exception as e_generic:
        logging.error(f"Generischer Fehler während Download V:{video_idx+1}: {e_generic}", exc_info=True)
        update_gui(set_progress, progress_bar_index, 1.0)
        update_gui(set_status,f"Schwerer Fehler Download V:{video_idx+1}")
        cleanup_temp_files(final_expected_path_mp4, 0, 0, 0)
        return None
def extract_frames_ffmpeg(video_path, interval, progress_bar_index, current_video_idx, total_videos):
    if stop_processing_flag.is_set():
        logging.info(f"extract_frames V:{current_video_idx+1}: Stop-Flag vor Beginn erkannt.")
        return []

    video_basename = os.path.basename(video_path)
    update_gui(set_status, f"Extrahiere Frames V:{current_video_idx+1}/{total_videos} (ffmpeg)")
    update_gui(set_progress, progress_bar_index, 0)
    logging.info(f"Starte FFmpeg Frame-Extraktion für '{video_basename}' mit Intervall {interval}s")

    frames = []
    sanitized_basename = "".join(c if c.isalnum() else '_' for c in os.path.splitext(video_basename)[0])
    temp_frame_dir = os.path.join(TEMP_DIR, f"frames_{sanitized_basename}_{int(time.time())}")

    try:
        os.makedirs(temp_frame_dir, exist_ok=True)
        logging.debug(f"Temporäres Frame-Verzeichnis erstellt: {temp_frame_dir}")

        ffmpeg_path = "ffmpeg"
        output_pattern = os.path.join(temp_frame_dir, "frame_%07d.jpg")
        ffmpeg_cmd = [
            ffmpeg_path,
            "-hide_banner",
            "-i", video_path,
            "-vf", f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',setpts=N/FRAME_RATE/TB",
            "-an",
            "-sn",
            "-vsync", "vfr",
            "-q:v", "2",
            "-loglevel", "warning",
            output_pattern
        ]

        logging.info(f"Führe FFmpeg aus: {' '.join(ffmpeg_cmd)}")
        start_time = time.time()
        creation_flags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        process = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=False, timeout=900, creationflags=creation_flags)
        end_time = time.time()
        logging.info(f"FFmpeg-Prozess für '{video_basename}' beendet nach {end_time - start_time:.2f}s mit Return Code {process.returncode}.")
        if process.returncode != 0:
            logging.error(f"FFmpeg Fehler (Return Code {process.returncode}) bei Frame-Extraktion für '{video_basename}'.")
            stderr_output = process.stderr.strip()
            logging.error(f"FFmpeg stderr: {stderr_output[:2000]}{'...' if len(stderr_output)>2000 else ''}")
            update_gui(set_status, f"Fehler Frame-Extraktion V:{current_video_idx+1}")
            if "No such file or directory" in stderr_output and ffmpeg_path in stderr_output:
                 messagebox.showerror("FFmpeg Fehler", f"'{ffmpeg_path}' wurde nicht gefunden oder konnte nicht ausgeführt werden. Stellen Sie sicher, dass FFmpeg korrekt installiert und im PATH ist.")
            elif "Permission denied" in stderr_output:
                 messagebox.showerror("FFmpeg Fehler", f"Keine Berechtigung zum Ausführen von FFmpeg oder Lesen/Schreiben von Dateien.\nPfad: {temp_frame_dir}")
            return []
        if stop_processing_flag.is_set():
             logging.info(f"Frame-Extraktion V:{current_video_idx+1} abgebrochen (Stop-Flag nach ffmpeg-Aufruf).")
             return []
        extracted_files = sorted([f for f in os.listdir(temp_frame_dir) if f.lower().endswith(".jpg")])
        num_extracted = len(extracted_files)
        logging.info(f"FFmpeg extrahierte {num_extracted} Frames nach '{temp_frame_dir}'. Lade Bilder...")

        if num_extracted == 0:
             logging.warning(f"FFmpeg lief erfolgreich für '{video_basename}', aber keine Frames wurden im Temp-Verzeichnis gefunden. Mögliche Gründe: Video zu kurz, Intervall zu groß, Filterproblem.")
             update_gui(set_progress, progress_bar_index, 1.0)
             return []

        update_gui(set_progress, progress_bar_index, 0.1)

        frames_loaded_count = 0
        for i, filename in enumerate(extracted_files):
             if stop_processing_flag.is_set():
                 logging.info(f"Frame-Laden V:{current_video_idx+1} abgebrochen (Stop-Flag).")
                 break

             frame_path = os.path.join(temp_frame_dir, filename)
             try:
                 frame = cv2.imread(frame_path)
                 if frame is not None and frame.size > 0:
                     frames.append(frame)
                     frames_loaded_count += 1
                 else:
                     logging.warning(f"Konnte leeren oder ungültigen Frame laden: {frame_path}")
             except Exception as load_err:
                 logging.error(f"Fehler beim Laden des extrahierten Frames '{frame_path}': {load_err}", exc_info=True)
                 continue
             if num_extracted > 0:
                 update_gui(set_progress, progress_bar_index, 0.1 + 0.9 * ((i + 1) / num_extracted))
        if stop_processing_flag.is_set():
            logging.info(f"Frame-Laden V:{current_video_idx+1} wurde unterbrochen. {frames_loaded_count} Frames geladen.")
            return []

        logging.info(f"{frames_loaded_count} von {num_extracted} extrahierten Frames erfolgreich geladen aus '{video_basename}'.")
        update_gui(set_progress, progress_bar_index, 1.0)
        return frames

    except subprocess.TimeoutExpired as te:
        logging.error(f"Timeout ({te.timeout}s) bei FFmpeg-Ausführung für '{video_basename}'.", exc_info=False)
        update_gui(set_status,f"Timeout Frame-Extraktion V:{current_video_idx+1}")
        update_gui(set_progress, progress_bar_index, 1.0)
        return []
    except OSError as e:
        logging.error(f"OS-Fehler bei der Frame-Extraktion für '{video_basename}' (Temp: {temp_frame_dir}): {e}", exc_info=True)
        update_gui(set_status, f"OS-Fehler Frame-Extraktion V:{current_video_idx+1}")
        update_gui(set_progress, progress_bar_index, 1.0)
        return []
    except Exception as e:
        logging.error(f"Generischer Fehler bei der Frame-Extraktion für '{video_basename}': {e}", exc_info=True)
        update_gui(set_status,f"Schwerer Fehler Frame-Extraktion V:{current_video_idx+1}")
        update_gui(set_progress, progress_bar_index, 1.0)
        return []
    finally:
        if os.path.exists(temp_frame_dir):
            try:
                shutil.rmtree(temp_frame_dir)
                logging.debug(f"Temporäres Frame-Verzeichnis entfernt: {temp_frame_dir}")
            except Exception as e_clean:
                logging.error(f"Fehler beim Bereinigen des Temp-Frame-Verzeichnisses '{temp_frame_dir}': {e_clean}")
def detect_objects_batch(frames, model, progress_bar_index, conf_threshold, current_video_idx, total_videos, model_path_info):
    if not frames:
        logging.warning(f"detect_objects_batch V:{current_video_idx+1}: Keine Frames zum Verarbeiten empfangen.")
        return []
    if stop_processing_flag.is_set():
        logging.info(f"detect_objects_batch V:{current_video_idx+1}: Stop-Flag vor Beginn erkannt.")
        return []

    num_frames = len(frames)
    update_gui(set_status, f"Starte YOLO/ONNX Erkennung V:{current_video_idx+1}/{total_videos} ({num_frames} Frames)")
    
    update_gui(set_progress, progress_bar_index, 0)
    logging.info(f"Starte {os.path.basename(model_path_info)} Batch-Erkennung für {num_frames} Frames (Video {current_video_idx+1}) mit Schwelle {conf_threshold}")
    

    results_list = []
    batch_size = 1
    processed_count = 0

    try:
        for i in range(0, num_frames, batch_size):
            if stop_processing_flag.is_set():
                logging.info(f"YOLO/ONNX Erkennung V:{current_video_idx+1} abgebrochen (Stop-Flag während Batch-Verarbeitung).")
                
                break

            batch_start_index = i
            batch_end_index = min(i + batch_size, num_frames)
            batch_frames = frames[batch_start_index:batch_end_index]

            if not batch_frames:
                logging.warning(f"Leerer Frame-Batch bei Index {i} erhalten, überspringe.")
                continue

            logging.debug(f"Verarbeite Batch {i // batch_size + 1}, Frames {batch_start_index + 1} bis {batch_end_index}")
            try:
                batch_results = model.predict(
                    source=batch_frames,
                    conf=conf_threshold,
                    classes=[0],
                    verbose=False,
                    device=None,
                )
            except Exception as predict_err:
                 logging.error(f"Fehler während model.predict für Batch {i // batch_size + 1}: {predict_err}", exc_info=True)
                 processed_count = batch_end_index
                 update_gui(set_progress, progress_bar_index, processed_count / num_frames)
                 continue
            if isinstance(batch_results, list):
                 results_list.extend(batch_results)
            else:
                 logging.warning(f"YOLO/ONNX predict gab unerwarteten Typ zurück: {type(batch_results)}. Versuche zu konvertieren.")
                 
                 try:
                     results_list.extend(list(batch_results))
                 except TypeError:
                     logging.error(f"Konnte Batch-Ergebnis (Typ: {type(batch_results)}) nicht zur Ergebnisliste hinzufügen.")
            processed_count = batch_end_index
            update_gui(set_progress, progress_bar_index, processed_count / num_frames)
        if stop_processing_flag.is_set():
            logging.info(f"YOLO/ONNX Erkennung V:{current_video_idx+1} wurde unterbrochen.")
            
            return []
        total_detections = 0
        if results_list:
            for r in results_list:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    num_dets_in_frame = len(r.boxes)
                    total_detections += num_dets_in_frame
                elif isinstance(r, (list, np.ndarray)) and len(r) > 0:
                     logging.warning("YOLO/ONNX Ergebnis in unerwartetem Format (Liste/Array), zähle Elemente.")
                     
                     total_detections += len(r)
        logging.info(f"YOLO/ONNX Erkennung abgeschlossen V:{current_video_idx+1}. {total_detections} Objekte (Klasse 0) über Schwelle {conf_threshold} in {len(results_list)} Frames gefunden.")
        
        update_gui(set_progress, progress_bar_index, 1.0)
        return results_list

    except Exception as e:
        logging.error(f"Schwerer Fehler während YOLO/ONNX Batch-Erkennung V:{current_video_idx+1}: {e}", exc_info=True)
        update_gui(set_status, f"Schwerer Fehler YOLO/ONNX V:{current_video_idx+1}")
        
        update_gui(set_progress, progress_bar_index, 1.0)
        return [] 
    
def crop_to_final_size(image, box_coords_xywh):
    
    try:
        if image is None or image.size == 0:
             logging.error("crop_to_final_size: Ungültiges Eingabebild (None oder leer).")
             return None

        img_h, img_w = image.shape[:2]
        if img_h <= 0 or img_w <= 0:
             logging.error(f"crop_to_final_size: Ungültige Bilddimensionen {img_w}x{img_h}.")
             return None
        center_x, center_y = int(box_coords_xywh[0]), int(box_coords_xywh[1])
        if not (0 <= center_x < img_w and 0 <= center_y < img_h):
             logging.warning(f"Box-Zentrum ({center_x},{center_y}) liegt außerhalb des Bildes ({img_w}x{img_h}). Crop wird durchgeführt, aber Ergebnis könnte leer/schwarz sein.")
        half_size_target = POST_DETECTION_CROP_SIZE // 2
        crop_left = center_x - half_size_target
        crop_top = center_y - half_size_target
        crop_right = crop_left + POST_DETECTION_CROP_SIZE
        crop_bottom = crop_top + POST_DETECTION_CROP_SIZE
        read_left = max(0, crop_left)
        read_top = max(0, crop_top)
        read_right = min(img_w, crop_right)
        read_bottom = min(img_h, crop_bottom)
        if read_right <= read_left or read_bottom <= read_top:
            logging.warning(f"Crop-Bereich ({crop_left},{crop_top} to {crop_right},{crop_bottom}) liegt vollständig außerhalb des Bildes ({img_w}x{img_h}). Erzeuge schwarzes Bild ({POST_DETECTION_CROP_SIZE}x{POST_DETECTION_CROP_SIZE}).")
            return np.zeros((POST_DETECTION_CROP_SIZE, POST_DETECTION_CROP_SIZE, 3), dtype=np.uint8)
        img_cropped_part = image[read_top:read_bottom, read_left:read_right]
        final_image = np.zeros((POST_DETECTION_CROP_SIZE, POST_DETECTION_CROP_SIZE, 3), dtype=np.uint8)
        paste_x = max(0, -crop_left)
        paste_y = max(0, -crop_top)
        paste_h, paste_w = img_cropped_part.shape[:2]
        paste_end_y = paste_y + paste_h
        paste_end_x = paste_x + paste_w
        if paste_end_y > POST_DETECTION_CROP_SIZE or paste_end_x > POST_DETECTION_CROP_SIZE:
             logging.error(f"Fehler bei Pasting-Berechnung: Zielbereich ({paste_y}:{paste_end_y}, {paste_x}:{paste_end_x}) überschreitet Canvas ({POST_DETECTION_CROP_SIZE}x{POST_DETECTION_CROP_SIZE}).")
             paste_h = min(paste_h, POST_DETECTION_CROP_SIZE - paste_y)
             paste_w = min(paste_w, POST_DETECTION_CROP_SIZE - paste_x)
             paste_end_y = paste_y + paste_h
             paste_end_x = paste_x + paste_w
             img_cropped_part = img_cropped_part[:paste_h, :paste_w]
        if paste_h > 0 and paste_w > 0:
            final_image[paste_y : paste_end_y, paste_x : paste_end_x] = img_cropped_part
        else:
             logging.warning(f"Nichts zum Pasten übrig nach Berechnungen für Box {box_coords_xywh}.")
        if final_image.shape[0] != POST_DETECTION_CROP_SIZE or final_image.shape[1] != POST_DETECTION_CROP_SIZE:
            logging.error(f"Padding/Crop ergab unerwartete Endgröße: {final_image.shape[:2]}. Führe Resize als Fallback durch (sollte nicht passieren).")
            final_image = cv2.resize(final_image, (POST_DETECTION_CROP_SIZE, POST_DETECTION_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        

        return final_image

    except Exception as e:
        logging.error(f"Fehler beim Zuschneiden/Padding des Bildes: {e}", exc_info=True)
        logging.error(f"  Input image shape: {image.shape if image is not None else 'None'}, box_coords: {box_coords_xywh}")
        return None
def save_image_and_label(args):
    global image_save_counter, coco_annotations, coco_images, coco_categories
    cropped_image, label_format, original_box_w, original_box_h = args
    if cropped_image is None or cropped_image.size == 0:
         logging.error("save_image_and_label: Empfangenes Bild ist None oder leer. Überspringe Speichern.")
         return False
    if cropped_image.shape[0] != POST_DETECTION_CROP_SIZE or cropped_image.shape[1] != POST_DETECTION_CROP_SIZE:
         logging.error(f"save_image_and_label: Empfangenes Bild hat unerwartete Größe {cropped_image.shape[:2]} statt {POST_DETECTION_CROP_SIZE}x{POST_DETECTION_CROP_SIZE}. Überspringe.")
         return False
    


    try:
        with counter_lock:
            current_image_id = image_save_counter
            image_save_counter += 1
        base_filename = f"enemy_{current_image_id:07d}"
        image_filename = f"{base_filename}.jpg"
        image_path = os.path.join(OUTPUT_DIR, image_filename)
        label_path = None
        if label_format == "darknet":
            label_path = os.path.join(OUTPUT_DIR, f"{base_filename}.txt")
        elif label_format == "pascal_voc":
            label_path = os.path.join(OUTPUT_DIR, f"{base_filename}.xml")
        elif label_format == "coco":
            label_path = None
        else:
            logging.warning(f"Unbekanntes Label-Format '{label_format}' für Bild {base_filename}. Kein Label wird gespeichert.")
        try:
            if len(cropped_image.shape) == 2:
                 img_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2RGB)
            elif cropped_image.shape[2] == 4:
                 img_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGRA2RGB)
            elif cropped_image.shape[2] == 3:
                 img_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            else:
                 logging.error(f"Unerwartete Kanalanzahl ({cropped_image.shape[2]}) im Bild {base_filename}. Kann nicht speichern.")
                 return False
            img_pil = Image.fromarray(img_rgb)
            img_pil.save(image_path, format='JPEG', quality=95, subsampling=0)
            logging.debug(f"Bild erfolgreich gespeichert: {image_path}")

        except Exception as img_save_err:
            logging.error(f"Fehler beim Speichern des Bildes '{image_path}' mit Pillow: {img_save_err}", exc_info=True)
            return False
        crop_h, crop_w = cropped_image.shape[:2]
        box_center_x_crop = crop_w / 2.0
        box_center_y_crop = crop_h / 2.0
        box_w_crop = min(original_box_w, crop_w)
        box_h_crop = min(original_box_h, crop_h)
        box_w_crop = max(1.0, box_w_crop)
        box_h_crop = max(1.0, box_h_crop)
        if label_format == "darknet" and label_path:
            center_x_norm = box_center_x_crop / crop_w
            center_y_norm = box_center_y_crop / crop_h
            width_norm = box_w_crop / crop_w
            height_norm = box_h_crop / crop_h
            center_x_norm = min(max(center_x_norm, 0.0), 1.0)
            center_y_norm = min(max(center_y_norm, 0.0), 1.0)
            width_norm = min(max(width_norm, 0.0), 1.0)
            height_norm = min(max(height_norm, 0.0), 1.0)
            label_content = f"0 {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            try:
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write(label_content)
                logging.debug(f"Darknet Label gespeichert: {label_path}")
            except IOError as lbl_e:
                 logging.error(f"E/A-Fehler beim Schreiben des Darknet Labels '{label_path}': {lbl_e}", exc_info=True)
                 return False

        elif label_format == "pascal_voc" and label_path:
            xmin = int(round(box_center_x_crop - box_w_crop / 2.0))
            ymin = int(round(box_center_y_crop - box_h_crop / 2.0))
            xmax = int(round(xmin + box_w_crop))
            ymax = int(round(ymin + box_h_crop))
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(crop_w - 1, xmax)
            ymax = min(crop_h - 1, ymax)
            xmax = max(xmin, xmax)
            ymax = max(ymin, ymax)
            is_truncated = 1 if (original_box_w > crop_w or original_box_h > crop_h) else 0
            label_content = f"""<annotation>
    <folder>{os.path.basename(OUTPUT_DIR)}</folder>
    <filename>{image_filename}</filename>
    <path>{os.path.abspath(image_path)}</path>
    <source>
        <database>YouTube Warzone Collector</database>
    </source>
    <size>
        <width>{crop_w}</width>
        <height>{crop_h}</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>enemy</name>
        <pose>Unspecified</pose>
        <truncated>{is_truncated}</truncated>
        <difficult>0</difficult> <!-- Assuming not difficult -->
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>
</annotation>"""
            try:
                with open(label_path, 'w', encoding='utf-8') as f:
                    f.write(label_content)
                logging.debug(f"Pascal VOC Label gespeichert: {label_path}")
            except IOError as lbl_e:
                 logging.error(f"E/A-Fehler beim Schreiben des Pascal VOC Labels '{label_path}': {lbl_e}", exc_info=True)
                 return False

        elif label_format == "coco":
            coco_x = int(round(box_center_x_crop - box_w_crop / 2.0))
            coco_y = int(round(box_center_y_crop - box_h_crop / 2.0))
            coco_w_val = int(round(box_w_crop))
            coco_h_val = int(round(box_h_crop))
            coco_x = max(0, coco_x)
            coco_y = max(0, coco_y)
            coco_w_val = max(1, coco_w_val)
            coco_h_val = max(1, coco_h_val)
            if coco_x + coco_w_val > crop_w: coco_w_val = crop_w - coco_x
            if coco_y + coco_h_val > crop_h: coco_h_val = crop_h - coco_y
            if coco_w_val <= 0 or coco_h_val <= 0:
                logging.warning(f"COCO BBox hat ungültige Dimensionen ({coco_w_val}x{coco_h_val}) nach Clamping für Bild {base_filename}. Überspringe Annotation.")
            else:
                bbox_coco = [coco_x, coco_y, coco_w_val, coco_h_val]
                area = float(coco_w_val * coco_h_val)
                with counter_lock:
                    if not any(img['id'] == current_image_id for img in coco_images):
                        coco_images.append({
                            "id": current_image_id,
                            "width": crop_w,
                            "height": crop_h,
                            "file_name": image_filename,
                            "license": 0,
                            "flickr_url": "",
                            "coco_url": "",
                            "date_captured": datetime.now().isoformat()
                        })
                        logging.debug(f"COCO Bild-Info hinzugefügt für ID: {current_image_id}, Datei: {image_filename}")
                    annotation_id = len(coco_annotations)
                    coco_annotations.append({
                        "id": annotation_id,
                        "image_id": current_image_id,
                        "category_id": 0,
                        "segmentation": [],
                        "area": area,
                        "bbox": bbox_coco,
                        "iscrowd": 0
                    })
                    logging.debug(f"COCO Annotation hinzugefügt für Bild ID {current_image_id} (Annotation ID {annotation_id}).")
        return True

    except Exception as e:
        logging.error(f"Generischer Fehler beim Speichern von Bild/Label für Basisnamen '{base_filename}': {e}", exc_info=True)
        if label_path and os.path.exists(label_path): os.remove(label_path)
        if os.path.exists(image_path): os.remove(image_path)
        return False
def cleanup_temp_files(video_path, progress_bar_index, total_videos_processed, total_videos_found):

    action_taken = False
    file_to_remove = None

    if video_path and isinstance(video_path, str) and os.path.exists(video_path):
        file_to_remove = video_path
    elif video_path and isinstance(video_path, str):
         base_path, _ = os.path.splitext(video_path)
         part_file = base_path + ".mp4.part"
         ytdl_file = base_path + ".ytdl"
         if os.path.exists(part_file):
              file_to_remove = part_file
              logging.debug(f"Found related .part file to remove: {file_to_remove}")
         elif os.path.exists(ytdl_file):
              file_to_remove = ytdl_file
              logging.debug(f"Found related .ytdl file to remove: {file_to_remove}")


    if file_to_remove:
        file_basename = os.path.basename(file_to_remove)
        update_gui(set_status, f"Bereinige Temp-Datei: {file_basename}...")
        logging.info(f"Versuche temporäre Datei zu entfernen: {file_to_remove}")
        try:
            os.remove(file_to_remove)
            logging.info(f"Temporäre Datei erfolgreich entfernt: {file_basename}")
            action_taken = True
        except OSError as e:
            logging.error(f"OS-Fehler beim Entfernen der temporären Datei '{file_to_remove}': {e}", exc_info=False)
            update_gui(set_status, f"Fehler beim Bereinigen von {file_basename}")
        except Exception as e:
            logging.error(f"Allgemeiner Fehler beim Bereinigen von '{file_to_remove}': {e}", exc_info=True)
            update_gui(set_status, f"Fehler beim Bereinigen von {file_basename}")
    else:
         logging.debug(f"Keine temporäre Videodatei zum Bereinigen übergeben oder Pfad nicht gefunden: {video_path}")
    if total_videos_found > 0:
        progress_val = min(1.0, total_videos_processed / total_videos_found)
        update_gui(set_progress, progress_bar_index, progress_val)
    else:
         update_gui(set_progress, progress_bar_index, 1.0)

    if not action_taken:
        pass
def save_coco_annotations():
    with counter_lock:
        annotations_to_save = list(coco_annotations)
        images_to_save = list(coco_images)
        categories_to_save = list(coco_categories)
    if not annotations_to_save or not images_to_save:
        logging.warning("Keine COCO-Daten zum Speichern vorhanden (Annotationen oder Bilder fehlen). Überspringe COCO JSON-Erstellung.")
        return

    update_gui(set_status, "Speichere COCO Annotationen...")
    num_images = len(images_to_save)
    num_annos = len(annotations_to_save)
    logging.info(f"Speichere COCO Daten: {num_images} Bilder, {num_annos} Annotationen.")
    coco_data = {
        "info": {
            "description": f"Warzone Enemy Dataset - Generated by {APP_TITLE}",
            "url": "https://github.com/your-repo-if-any",
            "version": APP_VERSION,
            "year": datetime.now().year,
            "contributor": "Automated Script",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [{
             "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
             "id": 0,
             "name": "CC BY-NC-SA 4.0"
        }],
        "images": images_to_save,
        "annotations": annotations_to_save,
        "categories": categories_to_save
    }
    json_filename = f"coco_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_path = os.path.join(OUTPUT_DIR, json_filename)
    logging.info(f"Speichere COCO JSON unter: {json_path}")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)
        logging.info(f"COCO Annotationen erfolgreich gespeichert: {json_path}")
        update_gui(set_status, f"COCO JSON gespeichert ({num_images} Bilder)")
    except IOError as e_io:
         logging.error(f"E/A-Fehler beim Speichern der COCO JSON-Datei '{json_path}': {e_io}", exc_info=True)
         update_gui(set_status,"Fehler beim Speichern der COCO JSON (I/O).")
         messagebox.showerror("COCO Speicherfehler",f"Datei konnte nicht geschrieben werden:\n{e_io}\nPfad: {json_path}")
    except TypeError as e_type:
         logging.error(f"Typ-Fehler beim Serialisieren der COCO Daten zu JSON: {e_type}", exc_info=True)
         update_gui(set_status,"Fehler beim Speichern der COCO JSON (Typ).")
         messagebox.showerror("COCO Speicherfehler", f"Daten konnten nicht in JSON umgewandelt werden:\n{e_type}")
         logging.debug(f"Problematic COCO data snippet (approx): {str(coco_data)[:1000]}...")
    except Exception as e_generic:
        logging.error(f"Allgemeiner Fehler beim Speichern der COCO JSON '{json_path}': {e_generic}", exc_info=True)
        update_gui(set_status,"Schwerer Fehler beim Speichern der COCO JSON.")
        messagebox.showerror("COCO Fehler",f"COCO JSON konnte nicht gespeichert werden:\n{e_generic}")

def start_process():
    global image_save_counter
    collected_images_count_local = 0
    processed_video_count = 0
    model = None
    futures = []
    start_run_time = time.time()
    final_status_message = "Prozess gestartet..."
    with counter_lock:
        image_save_counter = 0
        if format_combobox.get() == "coco":
            global coco_annotations, coco_images
            coco_annotations = []
            coco_images = []

    try:
        search_term = search_entry.get() or "warzone gameplay no commentary"
        num_videos_to_find = int(num_videos_entry.get() or 10)
        interval = float(interval_entry.get() or 5)
        quality = quality_combobox.get()
        model_choice = model_combobox.get()
        if model_choice == "Custom Model (.pt/.onnx)":
             model_load_path = custom_model_path_entry.get()
        else:
             model_load_path = model_choice
        
        threshold = float(threshold_entry.get() or 0.4)
        label_format = format_combobox.get()
        desired_images = int(num_images_entry.get() or 100)
        precrop_enabled = precrop_checkbox.get() == 1
        precrop_target_size = None
        if precrop_enabled:
            try:
                pc_size_str = precrop_size_entry.get() or "416"
                precrop_target_size = int(pc_size_str)
                if precrop_target_size <= 0:
                     logging.error("Ungültige Pre-Crop Größe <= 0 erhalten. Deaktiviere Pre-Crop.")
                     precrop_enabled = False
                     precrop_target_size = None
            except ValueError:
                 logging.error("Konnte Pre-Crop Größe trotz Validierung nicht parsen. Deaktiviere Pre-Crop für diesen Lauf.")
                 precrop_enabled = False
                 precrop_target_size = None
        

        logging.info("----- Verarbeitungsprozess gestartet -----")
        logging.info(f"Parameter: Suche='{search_term}', Max Videos={num_videos_to_find}, Intervall={interval}s, Qualität={quality}, Modell='{model_load_path}', Schwelle={threshold}, Format={label_format}, Ziel Bilder={desired_images}")
        if precrop_enabled and precrop_target_size is not None:
            logging.info(f"Pre-Crop (Mitte) Aktiviert: Zielgröße={precrop_target_size}x{precrop_target_size}")
        else:
            logging.info("Pre-Crop (Mitte) Deaktiviert.")
        update_gui(set_status, f"Lade Modell ({os.path.basename(model_load_path)})...")
        
        try:
             model_type = "ONNX" if model_load_path.lower().endswith(".onnx") else "YOLOv8/v5/v9 PT"
             logging.info(f"Lade {model_type} Modell von: {model_load_path}")
             model = YOLO(model_load_path)
             dummy_img = np.zeros((64, 64, 3), dtype=np.uint8)
             _ = model.predict(dummy_img, verbose=False, classes=[0])
             model_device = model.device if hasattr(model, 'device') else 'N/A'
             logging.info(f"{model_type} Modell '{model_load_path}' erfolgreich geladen und initialisiert auf Gerät: {model_device}")
             update_gui(set_status, f"{model_type} Modell geladen, starte Videosuche...")
             
        except Exception as e_model:
             logging.critical(f"Fataler Fehler beim Laden/Initialisieren des Modells '{model_load_path}': {e_model}", exc_info=True)
             messagebox.showerror("Modell Ladefehler",
                                f"Das ausgewählte Modell konnte nicht geladen oder initialisiert werden:\n\n{e_model}\n\n"
                                f"Stellen Sie sicher, dass das Modell ({model_load_path}) existiert, kompatibel ist und alle Abhängigkeiten (z.B. CUDA, PyTorch, ONNXRuntime) korrekt installiert sind.\n"
                                f"Details siehe Log-Datei: '{LOG_FILE}'.")
             raise RuntimeError(f"Modell konnte nicht geladen werden") from e_model
        logging.info("Phase 1: Starte Videosuche auf YouTube")
        if stop_processing_flag.is_set(): raise InterruptedError("Stop vor Videosuche.")
        video_urls = search_youtube(search_term, num_videos_to_find, quality, 0)
        if stop_processing_flag.is_set(): raise InterruptedError("Stop nach Videosuche.")

        if not video_urls:
            logging.warning(f"Keine Videos für Suchbegriff '{search_term}' gefunden oder extrahiert. Prozess wird beendet.")
            messagebox.showwarning("Keine Videos gefunden",
                                 f"Keine gültigen Video-URLs für '{search_term}' gefunden.\n"
                                 f"Bitte prüfen Sie den Suchbegriff und die Netzwerkverbindung.\n"
                                 f"Details siehe Log-Datei: '{LOG_FILE}'.")
            final_status_message = "Bereit (Keine Videos gefunden)"
            raise RuntimeError("Keine Videos gefunden")

        num_videos_found = len(video_urls)
        logging.info(f"{num_videos_found} potenzielle Video-URLs gefunden.")
        update_gui(set_status, f"{num_videos_found} Videos gefunden, starte Verarbeitung...")
        cpu_cores = os.cpu_count() or 4
        max_save_workers = max(1, min(8, cpu_cores))
        logging.info(f"Verwende bis zu {max_save_workers} Worker-Threads für das Speichern von Bildern/Labels.")
        with ThreadPoolExecutor(max_workers=max_save_workers, thread_name_prefix='SaveWorker') as executor:
            for video_idx, url in enumerate(video_urls):
                current_video_label = f"Video {video_idx + 1}/{num_videos_found}"
                logging.info(f"----- Beginne Verarbeitung {current_video_label} ----- URL: {url[:60]}...")
                if stop_processing_flag.is_set():
                    logging.info(f"Stop-Flag erkannt vor Verarbeitung von {current_video_label}. Breche Videoschleife ab.")
                    break
                with counter_lock:
                     current_saved_count = image_save_counter
                if current_saved_count >= desired_images:
                    logging.info(f"Gewünschte Bildanzahl ({desired_images}) erreicht. Stoppe Verarbeitung weiterer Videos.")
                    break
                video_path = None
                frames = []
                frames_to_process = []

                try:
                    logging.info(f"Phase 2: Download {current_video_label}")
                    update_gui(set_progress, 1, 0)
                    if stop_processing_flag.is_set(): break
                    video_path = download_video(url, TEMP_DIR, quality, 1, video_idx, num_videos_found)
                    if stop_processing_flag.is_set(): break
                    if not video_path:
                        logging.warning(f"Überspringe {current_video_label} (Download fehlgeschlagen, abgebrochen oder Datei nicht gefunden/gültig).")
                        continue
                    logging.info(f"Phase 3: Frame-Extraktion {current_video_label}")
                    update_gui(set_progress, 2, 0)
                    if stop_processing_flag.is_set(): break
                    frames = extract_frames_ffmpeg(video_path, interval, 2, video_idx, num_videos_found)
                    if stop_processing_flag.is_set(): break
                    if not frames:
                        logging.warning(f"Keine Frames aus {os.path.basename(video_path)} extrahiert oder geladen. Überspringe Rest für dieses Video.")
                        continue
                    update_gui(set_progress, 3, 0)
                    frames_to_process = frames

                    if precrop_enabled and precrop_target_size is not None:
                        logging.info(f"Phase 4: Pre-Detection Center Cropping {current_video_label} (Größe: {precrop_target_size})")
                        update_gui(set_status, f"Pre-Cropping (Mitte) {current_video_label} ({len(frames)} Frames)...")
                        precrop_results_temp = []
                        original_frame_count_precrop = len(frames)

                        for i, frame in enumerate(frames):
                            if stop_processing_flag.is_set(): break
                            if frame is None or frame.size == 0:
                                logging.warning(f"Pre-Crop: Ungültiger Frame #{i} übersprungen.")
                                continue

                            fh, fw = frame.shape[:2]
                            center_x = fw // 2
                            center_y = fh // 2
                            half_size = precrop_target_size // 2
                            crop_x = center_x - half_size
                            crop_y = center_y - half_size
                            actual_x = max(0, crop_x)
                            actual_y = max(0, crop_y)
                            read_w = min(precrop_target_size, fw - actual_x)
                            read_h = min(precrop_target_size, fh - actual_y)
                            if read_w <= 0 or read_h <= 0:
                                logging.warning(f"Pre-Crop Region für Frame {i} von {current_video_label} ist außerhalb des Bildes ({fw}x{fh}) oder ungültig. Crop-Größe: {precrop_target_size}. Überspringe Frame.")
                                if original_frame_count_precrop > 0:
                                    update_gui(set_progress, 3, (i + 1) / original_frame_count_precrop)
                                continue
                            try:
                                cropped_part = frame[actual_y : actual_y + read_h, actual_x : actual_x + read_w]
                                target_canvas = np.zeros((precrop_target_size, precrop_target_size, 3), dtype=np.uint8)
                                paste_x = max(0, -crop_x)
                                paste_y = max(0, -crop_y)
                                paste_end_y = paste_y + read_h
                                paste_end_x = paste_x + read_w
                                if paste_end_y <= precrop_target_size and paste_end_x <= precrop_target_size:
                                    target_canvas[paste_y:paste_end_y, paste_x:paste_end_x] = cropped_part
                                    precrop_results_temp.append(target_canvas)
                                else:
                                     logging.error(f"Pre-Crop Pasting Fehler für Frame {i}: Paste Bereich ({paste_x}:{paste_end_x}, {paste_y}:{paste_end_y}) überschreitet Canvas ({precrop_target_size}x{precrop_target_size}).")

                            except Exception as crop_err:
                                logging.error(f"Fehler beim Pre-Cropping (Mitte) von Frame {i} ({current_video_label}): {crop_err}", exc_info=True)
                            if original_frame_count_precrop > 0:
                                update_gui(set_progress, 3, (i + 1) / original_frame_count_precrop)
                        if stop_processing_flag.is_set(): break

                        logging.info(f"{len(precrop_results_temp)} Frames nach Pre-Crop (Mitte) übrig für {current_video_label}.")
                        frames_to_process = precrop_results_temp
                        update_gui(set_progress, 3, 1.0)
                        if not frames_to_process:
                            logging.warning(f"Keine Frames nach Pre-Crop (Mitte) übrig für {current_video_label}. Überspringe Detection.")
                            continue

                    else:
                        logging.debug(f"Pre-Cropping für {current_video_label} übersprungen (deaktiviert oder Größe nicht gesetzt).")
                        update_gui(set_progress, 3, 1.0)
                    num_frames_for_detection = len(frames_to_process)
                    logging.info(f"Phase 5: YOLO/ONNX Erkennung {current_video_label} ({num_frames_for_detection} Frames)")
                    update_gui(set_progress, 4, 0)
                    if stop_processing_flag.is_set(): break
                    if not frames_to_process:
                         logging.warning(f"Keine Frames zur Erkennung für {current_video_label} vorhanden (möglicherweise nach Pre-Crop). Überspringe.")
                         continue
                    results = detect_objects_batch(frames_to_process, model, 4, threshold, video_idx, num_videos_found, model_load_path)
                    if stop_processing_flag.is_set(): break
                    if not results:
                        logging.info(f"Keine Objekte (Klasse 0, Schwelle {threshold}) in den Frames von {current_video_label} erkannt.")
                        continue
                    logging.info(f"Phase 6/7: Post-Detection Crop & Speichern vorbereiten {current_video_label}")
                    update_gui(set_status, f"Verarbeite Detections {current_video_label}...")
                    update_gui(set_progress, 5, 0)
                    update_gui(set_progress, 6, 0)
                    detections_to_save_args = []
                    total_detections_in_video = sum(len(r.boxes) if hasattr(r, 'boxes') and r.boxes is not None else 0 for r in results)
                    processed_detections_count = 0

                    if total_detections_in_video == 0:
                        logging.info(f"Keine Bounding Boxes in den Ergebnissen für {current_video_label} gefunden zum Verarbeiten.")
                        continue

                    for frame_idx, (frame, result) in enumerate(zip(frames_to_process, results)):
                        with counter_lock: current_saved_count_inner = image_save_counter
                        if current_saved_count_inner >= desired_images: break
                        if stop_processing_flag.is_set(): break
                        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                            for box in result.boxes:
                                with counter_lock: current_saved_count_box = image_save_counter
                                if current_saved_count_box >= desired_images: break
                                if stop_processing_flag.is_set(): break
                                try:
                                    coords_xywh = box.xywh[0].cpu().numpy()
                                    original_box_w = float(coords_xywh[2])
                                    original_box_h = float(coords_xywh[3])
                                except IndexError:
                                     logging.warning(f"Konnte xywh-Koordinaten nicht aus Box extrahieren (Frame {frame_idx}, {current_video_label}). Überspringe Box.")
                                     processed_detections_count += 1
                                     continue
                                except Exception as e_box:
                                     logging.error(f"Fehler beim Extrahieren der Box-Daten (Frame {frame_idx}, {current_video_label}): {e_box}", exc_info=True)
                                     processed_detections_count += 1
                                     continue
                                update_gui(set_progress, 5, 0.5)
                                final_cropped_image = crop_to_final_size(frame, coords_xywh)
                                
                                update_gui(set_progress, 5, 0)
                                if final_cropped_image is not None:
                                    save_args = (final_cropped_image, label_format, original_box_w, original_box_h)
                                    
                                    detections_to_save_args.append(save_args)
                                else:
                                    logging.warning(f"Überspringe Speichern: Post-Detection Crop fehlgeschlagen für Box {coords_xywh} in Frame {frame_idx} ({current_video_label}).")

                                processed_detections_count += 1
                                if total_detections_in_video > 0:
                                    update_gui(set_progress, 5, processed_detections_count / total_detections_in_video)
                            if stop_processing_flag.is_set(): break
                            with counter_lock: current_saved_count_box_check = image_save_counter
                            if current_saved_count_box_check >= desired_images: break
                    if stop_processing_flag.is_set(): break
                    update_gui(set_progress, 5, 1.0)
                    num_detections_ready_to_save = len(detections_to_save_args)
                    if num_detections_ready_to_save > 0:
                        logging.info(f"{current_video_label}: {num_detections_ready_to_save} gültige Detections gefunden. Übermittle Speicher-Tasks an ThreadPoolExecutor...")
                        update_gui(set_status, f"Speichere Bilder von {current_video_label}...")

                        submitted_save_tasks = 0
                        for save_args in detections_to_save_args:
                            with counter_lock: current_saved_count_submit = image_save_counter
                            if current_saved_count_submit >= desired_images:
                                logging.info(f"Ziel ({desired_images}) erreicht, überspringe restliche {num_detections_ready_to_save - submitted_save_tasks} Speicher-Tasks für {current_video_label}.")
                                break
                            if stop_processing_flag.is_set():
                                logging.info(f"Stop-Flag erkannt, überspringe restliche {num_detections_ready_to_save - submitted_save_tasks} Speicher-Tasks für {current_video_label}.")
                                break
                            future = executor.submit(save_image_and_label, save_args)
                            futures.append(future)
                            submitted_save_tasks += 1
                            update_gui(set_progress, 6, submitted_save_tasks / num_detections_ready_to_save)
                            current_potential_saves = current_saved_count_submit + submitted_save_tasks
                            update_gui(set_progress, 8, min(1.0, current_potential_saves / desired_images if desired_images > 0 else 1.0))


                        logging.info(f"{current_video_label}: {submitted_save_tasks} Speicher-Tasks übermittelt.")
                        if submitted_save_tasks > 0 : update_gui(set_progress, 6, 1.0)

                    else:
                        logging.info(f"Keine gültigen Detections zum Speichern für {current_video_label} vorbereitet.")
                        update_gui(set_progress, 6, 1.0)


                except InterruptedError:
                    logging.info(f"Verarbeitung für {current_video_label} durch Stop-Anforderung unterbrochen.")
                    pass
                except Exception as e_video_loop:
                    logging.error(f"Unerwarteter Fehler bei der Verarbeitung von {current_video_label}: {e_video_loop}", exc_info=True)
                    update_gui(set_status, f"Fehler bei {current_video_label} (Details im Log)")
                finally:
                    processed_video_count += 1
                    logging.info(f"Phase 8: Bereinigung für {current_video_label}")
                    cleanup_temp_files(video_path, 7, processed_video_count, num_videos_found)
                    with counter_lock: current_saved_count_final = image_save_counter
                    update_gui(set_progress, 8, min(1.0, current_saved_count_final / desired_images if desired_images > 0 else 1.0))
            logging.info("Video-Verarbeitungsschleife beendet.")
            update_gui(set_status, "Warte auf Abschluss der Speicher-Vorgänge...")
            logging.info(f"Warte auf {len(futures)} ausstehende Speicher-Threads...")

            successful_saves_count = 0
            failed_saves_count = 0
            initial_futures_count = len(futures)

            if initial_futures_count > 0:
                update_gui(set_progress, 6, 0)

                completed_count = 0
                for i, future in enumerate(futures):
                    try:
                        result_success = future.result(timeout=60)
                        if result_success:
                            successful_saves_count += 1
                        else:
                            failed_saves_count += 1
                    except TimeoutError:
                        logging.error(f"Timeout (>60s) beim Warten auf Speicher-Thread {i+1}/{initial_futures_count}. Markiere als fehlgeschlagen.")
                        failed_saves_count += 1
                    except Exception as e_future:
                        logging.error(f"Fehler beim Abrufen des Ergebnisses vom Speicher-Thread {i+1}/{initial_futures_count}: {e_future}", exc_info=True)
                        failed_saves_count += 1

                    completed_count += 1
                    update_gui(set_progress, 6, completed_count / initial_futures_count)

                logging.info(f"{initial_futures_count} Speicher-Futures abgearbeitet. Erfolgreich: {successful_saves_count}, Fehlgeschlagen/Timeout: {failed_saves_count}.")
                update_gui(set_progress, 6, 1.0)
            else:
                logging.info("Keine Speicher-Tasks zum Warten vorhanden.")
                update_gui(set_progress, 6, 1.0)
            final_collected_count = successful_saves_count
            logging.info(f"Endgültige Anzahl erfolgreich gespeicherter Bilder in diesem Lauf: {final_collected_count}")
            update_gui(set_progress, 8, min(1.0, final_collected_count / desired_images if desired_images > 0 else 1.0))
            if label_format == "coco" and not stop_processing_flag.is_set():
                with counter_lock: has_coco_data = bool(coco_annotations and coco_images)

                if has_coco_data:
                    logging.info("Phase 9: Speichere COCO Annotationen als JSON")
                    save_coco_annotations()
                else:
                     logging.info("COCO Format ausgewählt, aber keine Annotationen oder Bilddaten zum Speichern vorhanden (z.B. keine Detections).")
            elif label_format == "coco" and stop_processing_flag.is_set():
                 logging.info("COCO Speichern übersprungen, da der Prozess gestoppt wurde.")
            end_run_time = time.time()
            run_duration = end_run_time - start_run_time
            run_duration_str = time.strftime("%H:%M:%S", time.gmtime(run_duration))
            logging.info(f"----- Verarbeitung abgeschlossen (Gesamtdauer: {run_duration:.2f}s / {run_duration_str}) -----")

            if stop_processing_flag.is_set():
                final_status_message = f"Gestoppt. {final_collected_count} Bilder erfolgreich gespeichert."
                messagebox.showinfo("Prozess Gestoppt",
                                    f"Prozess manuell angehalten.\n\n"
                                    f"Erfolgreich gespeichert: {final_collected_count} Bilder\n"
                                    f"Dauer: {run_duration_str}")
            elif final_collected_count >= desired_images:
                final_status_message = f"Fertig! Ziel ({desired_images}) erreicht ({final_collected_count} gespeichert)."
                messagebox.showinfo("Ziel Erreicht",
                                    f"Ziel von {desired_images} Bildern erreicht!\n\n"
                                    f"Erfolgreich gespeichert: {final_collected_count} Bilder\n"
                                    f"Gespeichert in: {os.path.abspath(OUTPUT_DIR)}\n"
                                    f"Dauer: {run_duration_str}")
            elif final_collected_count > 0:
                 final_status_message = f"Fertig. {final_collected_count} Bilder gespeichert."
                 messagebox.showinfo("Prozess Abgeschlossen",
                                     f"Prozess abgeschlossen.\n\n"
                                     f"Erfolgreich gespeichert: {final_collected_count} Bilder\n"
                                     f"Gespeichert in: {os.path.abspath(OUTPUT_DIR)}\n"
                                     f"Dauer: {run_duration_str}")
            elif processed_video_count > 0:
                 final_status_message = "Fertig (Keine Bilder gespeichert)."
                 messagebox.showwarning("Prozess Abgeschlossen (Keine Bilder)",
                                       f"Prozess abgeschlossen, aber es wurden keine Bilder gespeichert.\n\n"
                                       f"Mögliche Gründe:\n"
                                       f"- Keine 'enemy'-Objekte gefunden (Klasse 0).\n"
                                       f"- Erkennungsschwelle zu hoch eingestellt ({threshold}).\n"
                                       f"- Pre-Crop-Bereich/Größe enthält keine Gegner.\n"
                                       f"- Fehler bei Download/Extraktion (siehe Log).\n\n"
                                       f"Überprüfen Sie die Einstellungen und die Log-Datei: '{LOG_FILE}'.\n"
                                       f"Dauer: {run_duration_str}")
            else:
                 if final_status_message == "Prozess gestartet...":
                      final_status_message = "Bereit (Keine Videos verarbeitet)"
                 logging.warning("Prozess endete ohne Videos zu verarbeiten.")

    except InterruptedError as ie:
         logging.info(f"Verarbeitungsprozess durch Benutzeranforderung gestoppt: {ie}")
         final_status_message = "Prozess gestoppt."
         with counter_lock: current_saved_count_stop = image_save_counter
         messagebox.showinfo("Prozess Gestoppt",
                             f"Prozess angehalten.\n\n"
                             f"{current_saved_count_stop} Speicherversuche wurden gestartet.\n"
                             f"(Erfolg/Abschluss dieser Speicherversuche nicht garantiert).\n"
                             f"Überprüfen Sie den Ausgabeordner.")

    except (ValueError, RuntimeError) as e_setup:
         logging.critical(f"Kritischer Setup- oder Laufzeitfehler im Verarbeitungsprozess: {e_setup}", exc_info=True)
         final_status_message = f"Fehler: {e_setup}"

    except Exception as e_main:
        logging.exception("Unerwarteter kritischer Fehler im Hauptverarbeitungsprozess.")
        final_status_message = "Kritischer Fehler!"
        messagebox.showerror("Schwerwiegender Fehler",
                             f"Ein unerwarteter Fehler ist im Verarbeitungsprozess aufgetreten:\n\n{e_main}\n\n"
                             f"Bitte überprüfen Sie die Log-Datei '{LOG_FILE}' für technische Details.")

    finally:
        logging.info(f"----- Prozess-Ende: Setze finalen Status: {final_status_message} -----")
        update_gui(set_status, final_status_message)
        if not stop_processing_flag.is_set():
             logging.debug("Setze nicht abgeschlossene Zwischen-Fortschrittsbalken auf 100% (außer Zielbalken).")
             for i in range(len(progress_bars) - 1):
                 if progress_bars[i] and progress_bars[i].winfo_exists() and progress_bars[i].get() < 1.0:
                      update_gui(set_progress, i, 1.0)
        try:
            update_gui(lambda: start_button.configure(state="normal", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"]))
            update_gui(lambda: stop_button.configure(state="disabled", text="Stop", fg_color="grey", hover=False))
            logging.info("GUI-Steuerelemente für Start/Stop zurückgesetzt.")
        except Exception as e_gui_final:
             logging.error(f"Fehler beim Zurücksetzen der GUI-Buttons am Ende: {e_gui_final}")
        global processing_thread
        processing_thread = None
        logging.info("ProcessingThread Referenz entfernt (Thread beendet oder nicht mehr gültig).")
if __name__ == "__main__":
    try:
        logging.info("==============================================")
        logging.info(f"Starte Anwendung: {APP_TITLE}")
        logging.info(f"Python Version: {platform.python_version()}")
        logging.info(f"Betriebssystem: {platform.system()} {platform.release()} ({platform.machine()})")
        logging.info(f"CustomTkinter Version: {ctk.__version__}")
        import ultralytics
        logging.info(f"Ultralytics Version: {ultralytics.__version__}")
        try:
             import onnxruntime
             logging.info(f"ONNX Runtime Version: {onnxruntime.__version__}")
        except ImportError:
             logging.warning("ONNX Runtime nicht installiert. Das Laden von .onnx Modellen schlägt fehl.")
        logging.info(f"Ausgabe Verzeichnis: {os.path.abspath(OUTPUT_DIR)}")
        logging.info(f"Log Datei: {os.path.abspath(LOG_FILE)}")
        logging.info("==============================================")
        _ = check_ffmpeg()
        toggle_custom_model_path()
        toggle_precrop_inputs()

        def on_closing():
            logging.info("Schließen-Aktion ausgelöst (WM_DELETE_WINDOW).")
            global processing_thread
            if processing_thread and processing_thread.is_alive():
                logging.warning("Schließen versucht, während Prozess aktiv ist.")
                if messagebox.askyesno("Prozess läuft",
                                       "Der Bildersammel-Prozess ist noch aktiv.\n\n"
                                       "Möchten Sie den Prozess wirklich beenden und die Anwendung schließen?\n\n"
                                       "(Aktuelle Downloads/Operationen werden abgebrochen)",
                                       icon='warning'):
                    logging.info("Benutzer bestätigt: Beende laufenden Prozess und schließe Anwendung.")
                    request_stop_processing()
                    update_gui(set_status, "Beende Prozess...")
                    root.after(500, root.destroy)

                else:
                    logging.info("Benutzer hat Schließen abgebrochen, Prozess läuft weiter.")
                    return

            else:
                logging.info("Kein aktiver Prozess gefunden. Schließe Anwendung.")
                root.destroy()
        root.protocol("WM_DELETE_WINDOW", on_closing)
        logging.info("Starte CustomTkinter Haupt-Event-Loop (root.mainloop()).")
        root.mainloop()
        logging.info("CustomTkinter Haupt-Event-Loop beendet.")

    except Exception as e_gui_main:
         logging.critical(f"Schwerwiegender Fehler im GUI Haupt-Thread: {e_gui_main}", exc_info=True)
         try:
             tk_error_root = ctk.CTk()
             tk_error_root.withdraw()
             messagebox.showerror("Fataler Anwendungsfehler",
                                f"Ein kritischer GUI-Fehler ist aufgetreten:\n\n{e_gui_main}\n\n"
                                f"Die Anwendung muss möglicherweise beendet werden.\n"
                                f"Details finden Sie in der Log-Datei:\n'{LOG_FILE}'.",
                                parent=None)
             tk_error_root.destroy()
         except Exception as fallback_err:
             print(f"FATAL ERROR: {e_gui_main}")
             print(f"Fallback error message failed: {fallback_err}")

    finally:
        logging.info("Anwendung wird beendet. Führe finale Bereinigungen durch...")
        stop_processing_flag.set()
        logging.debug("Stop-Flag wurde gesetzt (Sicherheitsmaßnahme bei Beendigung).")
        if processing_thread and processing_thread.is_alive():
             logging.info("Warte kurz (max 2s) auf Beendigung des ProcessingThreads...")
             processing_thread.join(timeout=2.0)
             if processing_thread.is_alive():
                  logging.warning("ProcessingThread hat sich nach 2 Sekunden Wartezeit nicht beendet.")
             else:
                  logging.info("ProcessingThread wurde erfolgreich beendet.")
        if os.path.exists(TEMP_DIR):
            logging.info(f"Versuche temporäres Verzeichnis '{TEMP_DIR}' zu entfernen...")
            try:
                shutil.rmtree(TEMP_DIR)
                logging.info(f"Temporäres Verzeichnis '{TEMP_DIR}' erfolgreich entfernt.")
            except PermissionError:
                 logging.error(f"Keine Berechtigung zum Löschen von '{TEMP_DIR}'. Möglicherweise sind Dateien noch in Benutzung.")
            except Exception as e_clean_temp:
                logging.error(f"Fehler beim Entfernen des temporären Verzeichnisses '{TEMP_DIR}': {e_clean_temp}", exc_info=True)
        else:
             logging.info(f"Temporäres Verzeichnis '{TEMP_DIR}' nicht gefunden, keine Bereinigung notwendig.")

        logging.info("Anwendung sauber beendet.")
        logging.info("ENDE\n\n")