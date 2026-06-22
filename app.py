import sys
import re
import csv
from pathlib import Path
import os
import cv2
import pytesseract

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QHBoxLayout, QVBoxLayout, QSpinBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QTextEdit, QMessageBox, QDialog, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from collections import Counter


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
ID_RE = re.compile(r"\b\d{7}\b")
V_RE = re.compile(r"\d{1,2}\.\d{1,2}")

def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.abspath(relative_path)

pytesseract.pytesseract.tesseract_cmd = resource_path("tesseract.exe")

os.environ["TESSDATA_PREFIX"] = resource_path("tessdata")
class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 180);
            }
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
            }
            QProgressBar {
                background: #222;
                color: white;
                border: 1px solid #555;
                height: 30px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Analyzing Video...")
        self.label.setAlignment(Qt.AlignCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(320)

        layout.addWidget(self.label)
        layout.addWidget(self.progress)

        self.hide()

    def resizeEvent(self, event):
        if self.parent():
            self.setGeometry(self.parent().rect())


class VideoOCRApp(QWidget):
    def __init__(self):
        super().__init__()
        self.value_votes = {}
        self.setWindowTitle("Video OCR")
        self.resize(1365, 790)

        self.video_path = None
        self.frames = []
        self.results = []
        self.final_results = {}
        self.current_index = 0

        self.setStyleSheet("""
            QWidget {
                background: #151a20;
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background: #2f6fde;
                color: white;
                border: 0;
                padding: 9px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background: #3d7ff0;
            }
            QLabel {
                color: white;
            }
            QSpinBox, QTextEdit {
                background: #252a33;
                color: white;
                border: 1px solid #555;
                padding: 5px;
            }
            QTableWidget {
                background: #20252d;
                color: white;
                gridline-color: #444;
                selection-background-color: #2f6fde;
            }
            QHeaderView::section {
                background: #252a33;
                color: white;
                padding: 7px;
                border: 1px solid #333;
                font-weight: bold;
            }
            QCheckBox {
                color: white;
            }
        """)

        main = QVBoxLayout(self)

        top = QHBoxLayout()

        self.btn_select = QPushButton("Select MP4")
        self.btn_select.clicked.connect(self.select_video)
        top.addWidget(self.btn_select)

        top.addWidget(QLabel("Interval:"))

        self.interval = QSpinBox()
        self.interval.setRange(1, 3600)
        self.interval.setValue(30)
        self.interval.setSuffix(" sec")
        top.addWidget(self.interval)

        self.chk_numbers = QCheckBox("Numbers only")
        self.chk_numbers.setChecked(True)
        top.addWidget(self.chk_numbers)

        self.btn_analyze = QPushButton("Analyze Video")
        self.btn_analyze.clicked.connect(self.analyze_video)
        top.addWidget(self.btn_analyze)

        self.video_info = QLabel("No video selected")
        top.addWidget(self.video_info)

        top.addStretch()
        main.addLayout(top)

        body = QHBoxLayout()

        left = QVBoxLayout()

        self.frame_view = QLabel()
        self.frame_view.setFixedSize(725, 445)
        self.frame_view.setAlignment(Qt.AlignCenter)
        self.frame_view.setStyleSheet("""
            background:#23232c;
            border:1px solid #555;
        """)
        left.addWidget(self.frame_view)

        nav = QHBoxLayout()

        self.btn_prev = QPushButton("Previous Frame")
        self.btn_prev.clicked.connect(self.prev_frame)
        nav.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next Frame")
        self.btn_next.clicked.connect(self.next_frame)
        nav.addWidget(self.btn_next)

        left.addLayout(nav)

        left.addStretch()

        left.addWidget(QLabel("Correction:"))

        self.correction_box = QTextEdit()
        self.correction_box.setFixedHeight(150)
        left.addWidget(self.correction_box)

        self.btn_save = QPushButton("Save Correction")
        self.btn_save.clicked.connect(self.save_correction)
        left.addWidget(self.btn_save)

        body.addLayout(left, 55)

        right = QVBoxLayout()

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels([
            "Time (sec)",
            "Frame",
            "Detected Pair",
            "Corrected Text"
        ])
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 100)
        self.table.setColumnWidth(2, 230)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellClicked.connect(self.table_row_clicked)

        right.addWidget(self.table)

        bottom = QHBoxLayout()

        self.btn_final = QPushButton("Show Final Output")
        self.btn_final.clicked.connect(self.show_final_output)
        bottom.addWidget(self.btn_final)

        self.btn_csv = QPushButton("Export CSV")
        self.btn_csv.clicked.connect(self.export_csv)
        bottom.addWidget(self.btn_csv)

        self.btn_txt = QPushButton("Export TXT")
        self.btn_txt.clicked.connect(self.export_txt)
        bottom.addWidget(self.btn_txt)

        right.addLayout(bottom)

        body.addLayout(right, 45)

        main.addLayout(body)

        self.loading = LoadingOverlay(self)
        self.loading.setGeometry(self.rect())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "loading"):
            self.loading.setGeometry(self.rect())
    def find_grid_lines(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 40, 120)

        # vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 2))
        vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)

        contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        xs = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if ch > h * 0.45:
                xs.append(x)

        xs = sorted(xs)

        clean_xs = [0]
        for x in xs:
            if 20 < x < w - 20:
                if not clean_xs or abs(x - clean_xs[-1]) > 30:
                    clean_xs.append(x)
        clean_xs.append(w)

        # horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 2, 1))
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)

        contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ys = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > w * 0.45:
                ys.append(y)

        ys = sorted(ys)

        clean_ys = [0]
        for y in ys:
            if 10 < y < h - 10:
                if not clean_ys or abs(y - clean_ys[-1]) > 25:
                    clean_ys.append(y)
        clean_ys.append(h)

        # fallback
        if len(clean_xs) < 3:
            aspect = w / h
            if aspect > 1.5:
                clean_xs = [0, w // 3, (w * 2) // 3, w]
            else:
                clean_xs = [0, w // 2, w]

        if len(clean_ys) < 5:
            cols = len(clean_xs) - 1
            rows = 8 if cols == 3 else 9
            clean_ys = [int(i * h / rows) for i in range(rows + 1)]

        return clean_xs, clean_ys
    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if not path:
            return

        self.video_path = path

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps else 0
        cap.release()

        self.video_info.setText(
            f"{Path(path).name} | {duration:.1f}s | {total} frames"
        )

    def get_frame_indices(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total <= 0:
            return []

        step = int(self.interval.value() * fps)
        if step <= 0:
            step = 1

        indices = set()

        # first frame
        indices.add(0)

        # interval frames
        i = 0
        while i < total:
            indices.add(i)
            i += step

        # safer last-frame candidates
        indices.add(total - 1)

        return sorted([x for x in indices if 0 <= x < total])

    def analyze_video(self):
        if not self.video_path:
            QMessageBox.warning(
                self,
                "Error",
                "Please select video first."
            )
            return

        self.loading.show()
        self.loading.raise_()
        QApplication.processEvents()

        try:
            self.frames.clear()
            self.results.clear()
            self.final_results.clear()
            self.table.setRowCount(0)
            self.current_index = 0

            # Voting engine
            self.value_votes = {}

            cap = cv2.VideoCapture(self.video_path)

            fps = cap.get(cv2.CAP_PROP_FPS)

            indices = self.get_frame_indices(cap)

            print("Selected frames:", indices)

            if not indices:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Could not read video."
                )
                return

            for frame_no in indices:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

                ok, frame = cap.read()

                # Recover if exact frame cannot be read
                if not ok:
                    recovered = False

                    for back in range(1, 10):

                        safe_frame_no = frame_no - back

                        if safe_frame_no < 0:
                            break

                        cap.set(
                            cv2.CAP_PROP_POS_FRAMES,
                            safe_frame_no
                        )

                        ok, frame = cap.read()

                        if ok:
                            frame_no = safe_frame_no
                            recovered = True
                            break

                    if not recovered:
                        continue

                pairs, detected_text = self.ocr_frame(frame)

                # ---- Voting engine ----
                for gray_id, voltage in pairs:

                    self.value_votes.setdefault(gray_id, [])

                    self.value_votes[gray_id].append(voltage)

                # -----------------------

                item = {
                    "time": frame_no / fps,
                    "frame_no": frame_no,
                    "frame": frame,
                    "pairs": pairs,
                    "detected": detected_text,
                    "corrected": ""
                }

                self.results.append(item)
                self.frames.append(frame)

                row = self.table.rowCount()

                self.table.insertRow(row)

                self.table.setItem(
                    row,
                    0,
                    QTableWidgetItem(f"{frame_no / fps:.1f}")
                )

                self.table.setItem(
                    row,
                    1,
                    QTableWidgetItem(str(frame_no))
                )

                self.table.setItem(
                    row,
                    2,
                    QTableWidgetItem(detected_text)
                )

                self.table.setItem(
                    row,
                    3,
                    QTableWidgetItem("")
                )

                QApplication.processEvents()

            cap.release()

            # ---------- Build final results ----------
            self.final_results.clear()

            for gray_id, values in self.value_votes.items():

                counter = Counter(values)

                ranked = counter.most_common()

                final_voltage = ranked[0][0]

                # Prefer .03 over .90 when frequencies tie
                if len(ranked) > 1:

                    top_count = ranked[0][1]

                    tied = [
                        value
                        for value, count in ranked
                        if count == top_count
                    ]

                    if any(v.endswith(".03") for v in tied):
                        final_voltage = next(
                            v for v in tied
                            if v.endswith(".03")
                        )

                self.final_results[gray_id] = final_voltage

            # -----------------------------------------

            if self.frames:
                self.show_frame(0)

        finally:
            self.loading.hide()

        QMessageBox.information(
            self,
            "Done",
            "Analysis complete.\n"
            f"{len(self.results)} frames processed.\n"
            f"{len(self.final_results)} unique IDs detected."
            )
    def normalize_voltage(self, voltage):
        """
        Instant OCR correction.
        """

        corrections = {
            "1.90": "1.03",

        }

        return corrections.get(voltage, voltage)
    
    def get_video_id_range(self):
        if not self.results:
            return None, None

        first_pairs = self.results[0].get("pairs", [])
        last_pairs = self.results[-1].get("pairs", [])

        if not first_pairs or not last_pairs:
            return None, None

        # first ID = smallest visible ID in first frame
        first_frame_ids = [int(gid) for gid, _ in first_pairs if gid.isdigit()]

        # last ID = largest visible ID in last frame
        last_frame_ids = [int(gid) for gid, _ in last_pairs if gid.isdigit()]

        if not first_frame_ids or not last_frame_ids:
            return None, None

        start_id = min(first_frame_ids)
        end_id = max(last_frame_ids)

        return min(start_id, end_id), max(start_id, end_id)

    def ocr_frame(self, frame):
        xs, ys = self.find_grid_lines(frame)

        cols = len(xs) - 1
        rows = len(ys) - 1

        pairs = []

        for r in range(rows):
            for c in range(cols):
                x1 = xs[c]
                x2 = xs[c + 1]
                y1 = ys[r]
                y2 = ys[r + 1]

                cell = frame[y1:y2, x1:x2]

                if cell.size == 0:
                    continue

                cell_h, cell_w = cell.shape[:2]

                if cols == 3:
                    id_crop = cell[:, 0:int(cell_w * 0.42)]
                    v_crop = cell[:, int(cell_w * 0.58):int(cell_w * 0.98)]
                else:
                    id_crop = cell[:, 0:int(cell_w * 0.45)]
                    v_crop = cell[:, int(cell_w * 0.55):int(cell_w * 0.98)]

                gray_id = self.read_gray_id(id_crop)
                voltage = self.read_voltage(v_crop)

                if gray_id and voltage and self.is_valid_voltage(voltage):
                    voltage1 = self.normalize_voltage(voltage)
                    if cols == 3:voltage1 = self.normalize_voltage(voltage1)
                    pairs.append((gray_id, voltage))

        detected_text = " ".join([f"{gid} {volt}v" for gid, volt in pairs])
        return pairs, detected_text
    
    def read_gray_id(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

        _, th = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(th, config=config)

        match = re.search(r"\b\d{7}\b", text)
        return match.group(0) if match else None

    def read_voltage(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        configs = [
            "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.",
            "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.",
            "--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789.",
        ]

        best = None

        for config in configs:
            text = pytesseract.image_to_string(gray, config=config)
            text = text.replace(",", ".").replace(" ", "")

            matches = re.findall(r"\d{1,2}\.\d{1,2}", text)

            if matches:
                best = matches[0]
                break

        return best
    def is_valid_voltage(self, voltage):
        if not voltage:
            return False

        voltage = str(voltage).strip()

        return bool(re.fullmatch(r"\d{1,2}\.\d{2}", voltage))
    
    def show_frame(self, index):
        if not self.frames:
            return

        self.current_index = index
        frame = self.frames[index]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg).scaled(
            self.frame_view.width(),
            self.frame_view.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.frame_view.setPixmap(pix)

        self.table.selectRow(index)

        detected = self.results[index]["detected"]
        corrected = self.results[index]["corrected"]

        self.correction_box.setPlainText(corrected if corrected else detected)

    def prev_frame(self):
        if self.current_index > 0:
            self.show_frame(self.current_index - 1)

    def next_frame(self):
        if self.current_index < len(self.frames) - 1:
            self.show_frame(self.current_index + 1)

    def table_row_clicked(self, row, col):
        if 0 <= row < len(self.frames):
            self.show_frame(row)

    def save_correction(self):
        if not self.results:
            return

        text = self.correction_box.toPlainText().strip()

        self.results[self.current_index]["corrected"] = text
        self.table.setItem(
            self.current_index,
            3,
            QTableWidgetItem(text)
        )

        pairs = self.parse_pairs(text)

        for gray_id, voltage in pairs:
            voltage = self.normalize_voltage(gray_id, voltage)
            self.final_results[gray_id] = voltage

        QMessageBox.information(self, "Saved", "Correction saved.")

    def parse_pairs(self, text):
        ids = ID_RE.findall(text)
        volts = V_RE.findall(text)

        pairs = []

        for i in range(min(len(ids), len(volts))):
            pairs.append((ids[i], volts[i]))

        return pairs
    def final_text(self):
        min_id, max_id = self.get_video_id_range()

        final_by_id = {}

        for item in self.results:
            for gray_id, voltage in item.get("pairs", []):
                if not gray_id or not gray_id.isdigit():
                    continue

                voltage = self.normalize_voltage(voltage)

                if not self.is_valid_voltage(voltage):
                    continue

                gid = int(gray_id)

                # use range only if detected successfully
                if min_id is not None and max_id is not None:
                    if gid < min_id or gid > max_id:
                        continue

                final_by_id[gray_id] = voltage

        # fallback: if filtering removed everything, show all valid detections
        if not final_by_id:
            for item in self.results:
                for gray_id, voltage in item.get("pairs", []):
                    if not gray_id or not gray_id.isdigit():
                        continue

                    voltage = self.normalize_voltage(voltage)

                    if self.is_valid_voltage(voltage):
                        final_by_id[gray_id] = voltage

        values = []

        for gray_id in sorted(final_by_id.keys(), key=lambda x: int(x)):
            values.append(final_by_id[gray_id])

        return ";".join(values)

    def show_final_output(self):
        output = self.final_text()

        if not output:
            QMessageBox.warning(
                self,
                "No Final Output",
                "No final output was generated. Please check detected pairs."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Final Output")
        dialog.resize(500, 650)

        layout = QVBoxLayout(dialog)

        text = QTextEdit()
        text.setPlainText(output)
        layout.addWidget(text)

        btn_copy = QPushButton("Copy All")
        btn_copy.clicked.connect(
            lambda: QApplication.clipboard().setText(text.toPlainText())
        )
        layout.addWidget(btn_copy)

        dialog.exec_()

    def export_csv(self):
        if not self.final_results:
            QMessageBox.warning(self, "No Results", "Please analyze video first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            "results.csv",
            "CSV Files (*.csv)"
        )

        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Gray ID", "Voltage"])

            for gray_id in sorted(self.final_results.keys(), key=lambda x: int(x)):
                writer.writerow([gray_id, self.final_results[gray_id]])

        QMessageBox.information(self, "Saved", "CSV exported.")

    def export_txt(self):
        if not self.final_results:
            QMessageBox.warning(self, "No Results", "Please analyze video first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save TXT",
            "results.txt",
            "Text Files (*.txt)"
        )

        if not path:
            return

        Path(path).write_text(self.final_text(), encoding="utf-8")

        QMessageBox.information(self, "Saved", "TXT exported.")

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        win = VideoOCRApp()
        win.show()
        sys.exit(app.exec_())
    except Exception:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
