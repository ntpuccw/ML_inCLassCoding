"""
GUI version of facenetPytorch_build_dataset.py.

This app captures face dataset images from webcam and stores original frames
under facenet_dataset/<person_id>/00000.png, 00001.png, ...

Shortcuts:
- k: save current original frame
- q: quit
"""

from pathlib import Path
import time
import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import Image, ImageTk


class FaceDatasetBuilderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Face Dataset Builder (GUI)")

        self.base_dir = Path(__file__).resolve().parent
        self.cascade_path = self.base_dir / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(self.cascade_path))
        if self.detector.empty():
            raise FileNotFoundError(
                f"Cannot load cascade file: {self.cascade_path}"
            )

        self.cap: cv2.VideoCapture | None = None
        self.current_orig = None
        self.current_frame = None
        self.last_face_boxes: list[tuple[int, int, int, int]] = []
        self.output_dir: Path | None = None
        self.total = 0
        self.last_auto_capture_ts = 0.0
        self.is_capturing = False

        self._build_ui()
        self.root.bind("<Key>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, padx=10, pady=10)
        top.pack(fill="x")

        tk.Label(top, text="Person ID:").grid(row=0, column=0, sticky="w")
        self.id_entry = tk.Entry(top, width=28)
        self.id_entry.grid(row=0, column=1, padx=(8, 0), sticky="w")

        tk.Label(top, text="Camera Index:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.camera_index_var = tk.IntVar(value=0)
        self.camera_index_spin = tk.Spinbox(
            top,
            from_=0,
            to=10,
            width=6,
            textvariable=self.camera_index_var,
        )
        self.camera_index_spin.grid(row=1, column=1, padx=(8, 0), sticky="w", pady=(8, 0))

        self.crop_face_var = tk.BooleanVar(value=False)
        self.crop_face_chk = tk.Checkbutton(
            top,
            text="Save Cropped Face",
            variable=self.crop_face_var,
        )
        self.crop_face_chk.grid(row=1, column=2, padx=8, pady=(8, 0), sticky="w")

        self.auto_capture_var = tk.BooleanVar(value=False)
        self.auto_capture_chk = tk.Checkbutton(
            top,
            text="Auto Capture Every (s)",
            variable=self.auto_capture_var,
        )
        self.auto_capture_chk.grid(row=1, column=3, padx=8, pady=(8, 0), sticky="w")

        self.auto_interval_var = tk.DoubleVar(value=1.0)
        self.auto_interval_entry = tk.Entry(top, width=6, textvariable=self.auto_interval_var)
        self.auto_interval_entry.grid(row=1, column=4, pady=(8, 0), sticky="w")

        self.start_btn = tk.Button(top, text="Start Capture", command=self.toggle_capture)
        self.start_btn.grid(row=0, column=2, padx=8)

        self.capture_btn = tk.Button(top, text="Capture (K)", command=self.capture_frame, state="disabled")
        self.capture_btn.grid(row=0, column=3, padx=8)

        self.quit_btn = tk.Button(top, text="Quit (Q)", command=self._on_close)
        self.quit_btn.grid(row=0, column=4)

        self.status_var = tk.StringVar(value="Enter Person ID and click Start Capture.")
        tk.Label(self.root, textvariable=self.status_var, anchor="w", padx=10).pack(fill="x")

        self.count_var = tk.StringVar(value="Saved images: 0")
        tk.Label(self.root, textvariable=self.count_var, anchor="w", padx=10).pack(fill="x")

        self.big_count_var = tk.StringVar(value="0")
        tk.Label(
            self.root,
            text="Saved Faces",
            font=("Segoe UI", 12, "bold"),
            fg="#1f4e79",
            pady=2,
        ).pack()
        tk.Label(
            self.root,
            textvariable=self.big_count_var,
            font=("Segoe UI", 44, "bold"),
            fg="#0b7a20",
            pady=4,
        ).pack()

        self.preview_label = tk.Label(self.root)
        self.preview_label.pack(padx=10, pady=10)

    def _sanitize_person_id(self, value: str) -> str:
        cleaned = "".join(ch for ch in value.strip() if ch not in '<>:"/\\|?*')
        return cleaned

    def toggle_capture(self) -> None:
        if self.is_capturing:
            self.stop_capture()
            return

        self.start_camera()

    def start_camera(self) -> None:
        person_id = self._sanitize_person_id(self.id_entry.get())
        if not person_id:
            messagebox.showerror("Invalid ID", "Please enter a valid person ID.")
            return

        self.output_dir = self.base_dir / "facenet_dataset" / person_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total = len(list(self.output_dir.glob("*.png")))
        self.count_var.set(f"Saved images: {self.total}")
        self.big_count_var.set(str(self.total))

        cam_index = self.camera_index_var.get()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(cam_index)

        if not self.cap.isOpened():
            self.cap = None
            messagebox.showerror("Camera Error", "Cannot open webcam.")
            return

        self.is_capturing = True
        self.start_btn.config(text="Stop Capture")
        self.capture_btn.config(state="normal")
        self.last_auto_capture_ts = time.time()
        self.status_var.set(f"Capturing for ID '{person_id}'. Press K to save, Q to quit.")
        self._update_frame()

    def stop_capture(self) -> None:
        self.is_capturing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.current_orig = None
        self.current_frame = None
        self.last_face_boxes = []
        self.preview_label.configure(image="")
        self.preview_label.imgtk = None

        self.start_btn.config(text="Start Capture")
        self.capture_btn.config(state="disabled")
        self.status_var.set("Capture stopped. Click Start Capture to resume.")

    def _update_frame(self) -> None:
        if not self.is_capturing or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Failed to read frame from camera.")
            return

        self.current_orig = frame.copy()
        self.last_face_boxes = []

        display = cv2.resize(frame, (400, int(frame.shape[0] * 400 / frame.shape[1])))
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        rects = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        scale_x = frame.shape[1] / display.shape[1]
        scale_y = frame.shape[0] / display.shape[0]
        for (x, y, w, h) in rects:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            ox = int(x * scale_x)
            oy = int(y * scale_y)
            ow = int(w * scale_x)
            oh = int(h * scale_y)
            self.last_face_boxes.append((ox, oy, ow, oh))

        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)
        tk_img = ImageTk.PhotoImage(image=img)
        self.preview_label.imgtk = tk_img
        self.preview_label.configure(image=tk_img)
        self.current_frame = display

        if self.auto_capture_var.get():
            interval = self.auto_interval_var.get()
            if interval > 0 and (time.time() - self.last_auto_capture_ts) >= interval:
                saved = self.capture_frame(from_auto=True)
                if saved:
                    self.last_auto_capture_ts = time.time()

        self.root.after(15, self._update_frame)

    def capture_frame(self, from_auto: bool = False) -> bool:
        if self.current_orig is None or self.output_dir is None:
            return False

        img_to_save = self.current_orig
        if self.crop_face_var.get():
            if not self.last_face_boxes:
                if not from_auto:
                    self.status_var.set("No face detected for cropped save.")
                return False

            x, y, w, h = max(self.last_face_boxes, key=lambda b: b[2] * b[3])
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(self.current_orig.shape[1], x + w)
            y1 = min(self.current_orig.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                return False
            img_to_save = self.current_orig[y0:y1, x0:x1]

        file_path = self.output_dir / f"{str(self.total).zfill(5)}.png"
        ok = cv2.imwrite(str(file_path), img_to_save)
        if ok:
            self.total += 1
            self.count_var.set(f"Saved images: {self.total}")
            self.big_count_var.set(str(self.total))
            self.status_var.set(f"Saved: {file_path.name}")
            return True
        else:
            self.status_var.set("Failed to save image.")
            return False

    def _on_key(self, event: tk.Event) -> None:
        key = event.char.lower()
        if key == "k":
            if self.is_capturing:
                self.capture_frame()
        elif key == "q":
            self._on_close()

    def _on_close(self) -> None:
        self.stop_capture()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = FaceDatasetBuilderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
