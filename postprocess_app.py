import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QLineEdit, QFormLayout, QTextEdit
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QImage
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize

class EditableEllipse:
    def __init__(self, center=QPointF(0, 0), rx=20, ry=10):
        self.center = center  # In original image coordinates
        self.rx = rx
        self.ry = ry
        self.selected = False
        self.handles = []
        self.update_handles()

    def update_handles(self):
        self.handles = [
            self.center + QPointF(self.rx, 0),
            self.center + QPointF(-self.rx, 0),
            self.center + QPointF(0, -self.ry),
            self.center + QPointF(0, self.ry),
        ]

    def contains(self, point):
        dx = (point.x() - self.center.x()) / self.rx
        dy = (point.y() - self.center.y()) / self.ry
        return dx*dx + dy*dy <= 1

    def handle_at(self, point, tol=5):
        for i, h in enumerate(self.handles):
            if (point - h).manhattanLength() < tol:
                return i
        return -1

    def move_handle(self, idx, to_point, image_size):
        if idx == 0:  # Right handle
            left_handle = self.center - QPointF(self.rx, 0)
            self.rx = max(1, abs(to_point.x() - left_handle.x())/2)
            self.center.setX(to_point.x() - self.rx)
        elif idx == 1:  # Left handle
            right_handle = self.center + QPointF(self.rx, 0)
            self.rx = max(1, abs(to_point.x() - right_handle.x())/2)
            self.center.setX(to_point.x() + self.rx)
        elif idx == 2:  # Bottom handle
            top_handle = self.center + QPointF(0, self.ry)
            self.ry = max(1, abs(to_point.y() - top_handle.y())/2)
            self.center.setY(to_point.y() + self.ry)
        elif idx == 3:  # Top handle
            bottom_handle = self.center - QPointF(0, self.ry)
            self.ry = max(1, abs(to_point.y() - bottom_handle.y())/2)
            self.center.setY(to_point.y() - self.ry)

        # Constrain size to image bounds
        self.rx = min(self.rx, image_size.width() / 2)
        self.ry = min(self.ry, image_size.height() / 2)
        self.update_handles()

    def move_center(self, delta, image_size):
        new_center = self.center + delta
        # Constrain center to stay within image bounds
        new_center.setX(max(self.rx, min(new_center.x(), image_size.width() - self.rx)))
        new_center.setY(max(self.ry, min(new_center.y(), image_size.height() - self.ry)))
        self.center = new_center
        self.update_handles()

class PolygonEditor:
    def __init__(self):
        self.points = []  # In original image coordinates
        self.drag_index = -1
        self.edit_mode = False

    def add_point(self, point):
        self.points.append(point)

    def find_point(self, pos, tol=8):
        for i, pt in enumerate(self.points):
            if (pt - pos).manhattanLength() < tol:
                return i
        return -1

    def move_point(self, index, new_pos):
        if 0 <= index < len(self.points):
            self.points[index] = new_pos

    def toggle_edit(self):
        self.edit_mode = not self.edit_mode
        self.drag_index = -1

    def clear(self):
        self.points.clear()
        self.drag_index = -1

class ImageAnnotator(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(False)  # Disable automatic scaling to use custom scaling
        self.original_image = None
        self.qimage = None
        self.scaled_image = None
        self.real_tool_diameter_mm = 5.0
        self.scale_factor = 1.0
        self.image_size = None
        self.ellipses = []
        self.selected_ellipse = -1
        self.selected_handle = -1
        self.dragging = False
        self.polygon = PolygonEditor()

    def load_image(self, path):
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            return
        height, width, ch = self.original_image.shape
        self.image_size = QSize(width, height)
        cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB, self.original_image)
        # Initialize ellipses in original image coordinates
        self.ellipses = [
            EditableEllipse(center=QPointF(width/4, height/4), rx=20, ry=10),
        ]
        self.update_scaled_image()

    def update_scaled_image(self):
        if self.original_image is None:
            return
        label_size = self.size()
        h, w, ch = self.original_image.shape
        scale_w = label_size.width() / w
        scale_h = label_size.height() / h
        self.scale_factor = min(scale_w, scale_h)

        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        resized = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        bytes_per_line = 3 * resized.shape[1]
        self.qimage = QImage(resized.data, resized.shape[1], resized.shape[0], bytes_per_line, QImage.Format_RGB888)
        self.scaled_image = QPixmap.fromImage(self.qimage)
        self.setPixmap(self.scaled_image)

    def resizeEvent(self, event):
        self.update_scaled_image()
        self.update()

    def to_image_coords(self, widget_pos):
        """Convert widget coordinates to original image coordinates."""
        return QPointF(widget_pos.x() / self.scale_factor, widget_pos.y() / self.scale_factor)

    def to_widget_coords(self, image_pos):
        """Convert original image coordinates to widget coordinates."""
        return QPointF(image_pos.x() * self.scale_factor, image_pos.y() * self.scale_factor)

    def mousePressEvent(self, event):
        pos = self.to_image_coords(event.pos())
        
        if self.polygon.edit_mode:
            idx = self.polygon.find_point(pos)
            if idx >= 0:
                self.polygon.drag_index = idx
            else:
                self.polygon.add_point(pos)
        else:
            for i, ellipse in enumerate(self.ellipses):
                h = ellipse.handle_at(pos)
                if h >= 0:
                    self.selected_ellipse = i
                    self.selected_handle = h
                    return
                elif ellipse.contains(pos):
                    self.selected_ellipse = i
                    self.selected_handle = -1
                    self.dragging = True
                    return
        self.update()

    def mouseMoveEvent(self, event):
        pos = self.to_image_coords(event.pos())
        
        if self.polygon.edit_mode and self.polygon.drag_index >= 0:
            self.polygon.move_point(self.polygon.drag_index, pos)
        elif self.selected_ellipse >= 0:
            if self.selected_handle >= 0:
                self.ellipses[self.selected_ellipse].move_handle(self.selected_handle, pos, self.image_size)
            elif self.dragging:
                delta = pos - self.ellipses[self.selected_ellipse].center
                self.ellipses[self.selected_ellipse].move_center(delta, self.image_size)
        self.update()

    def mouseReleaseEvent(self, event):
                
        self.selected_ellipse = -1
        self.selected_handle = -1
        self.dragging = False
        self.polygon.drag_index = -1
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.scaled_image is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        
        # Draw polygon in scaled coordinates
        if len(self.polygon.points) > 1:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            brush = QBrush(QColor(0, 255, 0, 100))
            painter.setBrush(brush)
            scaled_points = [self.to_widget_coords(pt) for pt in self.polygon.points]
            painter.drawPolygon(scaled_points)
            for pt in scaled_points:
                painter.drawEllipse(pt, 3, 3)

        # Draw ellipses in scaled coordinates
        for i, ellipse in enumerate(self.ellipses):
            color = QColor(0, 0, 255) if i == 0 else QColor(255, 0, 0)
            pen = QPen(color, 2 if i == 0 else 1)
            painter.setPen(pen)
            painter.setBrush(QBrush(color, Qt.NoBrush))
            center_scaled = self.to_widget_coords(ellipse.center)
            rx_scaled = ellipse.rx * self.scale_factor
            ry_scaled = ellipse.ry * self.scale_factor
            rect = QRectF(center_scaled.x() - rx_scaled, center_scaled.y() - ry_scaled,
                          2 * rx_scaled, 2 * ry_scaled)
            painter.drawEllipse(rect)
            for h in ellipse.handles:
                h_scaled = self.to_widget_coords(h)
                painter.drawEllipse(h_scaled, 4, 4)

    def set_real_tool_diameter_mm(self, diameter):
        self.real_tool_diameter_mm = float(diameter)

    def get_tool_scale(self):
        if len(self.ellipses) == 1:
            return self.real_tool_diameter_mm / self.ellipses[0].ry
        
        if len(self.ellipses) != 2:
            return None
        e1, e2 = self.ellipses
        dist_px = np.linalg.norm(np.array([e1.x(), e1.y()]) - np.array([e2.x(), e2.y()]))
        return self.real_tool_length_mm / dist_px if dist_px > 0 else None
    
    def get_pixel_to_mm_ratio(self):
        return self.get_tool_scale()

    def get_polygon_area_mm2(self):
        scale = self.get_pixel_to_mm_ratio()
        if len(self.polygon.points) < 3 or not scale:
            return None
        pts = np.array([[p.x(), p.y()] for p in self.polygon.points], dtype=np.float32)
        area_px = cv2.contourArea(pts)
        return area_px * scale ** 2

    def get_polygon_dimensions_mm(self):
        scale = self.get_pixel_to_mm_ratio()
        if len(self.polygon.points) < 3 or not scale:
            return None
        xs = [p.x() for p in self.polygon.points]
        ys = [p.y() for p in self.polygon.points]
        w = (max(xs) - min(xs)) * scale
        h = (max(ys) - min(ys)) * scale
        return w, h


class MainWindow(QMainWindow):
    def __init__(self, file_path=None):
        super().__init__()
        self.setWindowTitle("Bronchoscope Tool Annotator")
        self.resize(1200, 800)

        self.annotator = ImageAnnotator()

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        toggle_polygon_btn = QPushButton("Polygon Mode")
        toggle_polygon_btn.setCheckable(True)
        toggle_polygon_btn.clicked.connect(self.toggle_polygon_mode)

        clear_polygon_btn = QPushButton("Clear Polygon")
        clear_polygon_btn.clicked.connect(self.annotator.polygon.clear)


        form = QFormLayout()
        self.diameter_input = QLineEdit("5.0")
        self.diameter_input.textChanged.connect(self.annotator.set_real_tool_diameter_mm)
        self.length_input = QLineEdit("30.0")
        form.addRow("Tool Radius (mm):", self.diameter_input)
        # form.addRow("Tool Length (mm):", self.length_input)

        self.result_text = QTextEdit("sdjfksdfnkjs")
        self.result_text.setReadOnly(True)
        
        update_btn = QPushButton("Update")
        update_btn.clicked.connect(self.measure)
        
        left_panel = QVBoxLayout()
        left_panel.addWidget(load_btn)
        left_panel.addWidget(toggle_polygon_btn)
        left_panel.addWidget(clear_polygon_btn)
        left_panel.addLayout(form)
        left_panel.addWidget(update_btn)
        left_panel.addStretch()
        left_panel.addWidget(self.result_text)

        container = QWidget()
        layout = QHBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        layout.addWidget(left_widget)
        layout.addWidget(self.annotator, stretch=1)

        container.setLayout(layout)
        self.setCentralWidget(container)
        
        if file_path:
            self.annotator.load_image(file_path)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if file_path:
            self.annotator.load_image(file_path)

    def toggle_polygon_mode(self, checked):
        self.annotator.polygon.toggle_edit()
        self.annotator.update()
        
    def measure(self):
        scale = self.annotator.get_pixel_to_mm_ratio()
        if not scale:
            self.result_text.setPlainText("⚠️ Please mark two ellipses for the tool ends.")
            return

        dims = self.annotator.get_polygon_dimensions_mm()
        area = self.annotator.get_polygon_area_mm2()

        if not dims or not area:
            self.result_text.setPlainText("⚠️ Draw a valid polygon with at least 3 points.")
            return

        result = (
            f"📐 Tool Size: {self.annotator.ellipses[0].ry:.1f} px\n"
            f"📐 Tool Scale: {scale:.4f} mm/px\n\n"
            f"📏 Shape Width: {dims[0]:.2f} mm\n"
            f"📏 Shape Height: {dims[1]:.2f} mm\n"
            f"🧮 Area: {area:.2f} mm²"
        )
        self.result_text.setPlainText(result)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    win.show()
    sys.exit(app.exec_())