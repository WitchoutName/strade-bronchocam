import sys
import cv2
import math
import numpy as np
from abc import ABC, abstractmethod
from PyQt5.QtWidgets import (
    QApplication, QLabel, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QLineEdit, QFormLayout, QTextEdit, QGroupBox, QCheckBox, QSlider
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QImage
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize, QSizeF, QTimer

QPointF.__str__ = lambda self: f"QPointF({self.x():.2f}, {self.y():.2f})"

the_annotator = None

class Paintable(ABC):
    @abstractmethod
    def draw(self, painter, scale):
        pass

def ease_out_cubic(t):
    """Ease out cubic function."""
    return 1 - (1 - t) ** 3

def log_slider_to_value(slider_pos: int, slider_max: int = 1000) -> float:
    norm = slider_pos / slider_max  # normalize to 0–1
    return norm ** 0.2  # exponential (log-style curve)

def log_value_to_slider(val: float, slider_max: int = 1000) -> int:
    return int((val ** 5) * slider_max)  # inverse of above


def cubic_range(n, start=0, end=None):
    """Return a normalized ease-out-cubic CDF from 0 to 1, optionally sliced."""
    if n < 2:
        return [0.0]
    arr = [ease_out_cubic(i / (n - 1)) for i in range(n)]
    if end is None:
        end = n
    arr = arr[start:end]
    if len(arr) == 1:
        return [0.0]
    minv, maxv = arr[0], arr[-1]
    if maxv - minv == 0:
        return [0.0 for _ in arr]
    return [(v - minv) / (maxv - minv) for v in arr]

def ease_out_circ(t):
    """Ease out circular function."""
    return math.sqrt(1 - (t - 1) ** 2)


class ProbeData(Paintable):
    C_edge = QColor(255, 165, 0, 200)
    C_infill = QColor(150, 150, 150, 00)
    def __init__(self, start, vector, radius):        
        self.start = start  # QPointF
        self.vector = vector  # direction and scale as QPointF
        self.radius = radius  # in pixels
        
    @property
    def end(self):
        return self.start + self.vector

    
    def __str__(self):
        return f"Probe(start={self.start}, vector={self.vector}, end={self.end}, radius={self.radius})"
        

    def copy(self, probe):
        """Update the probe with new parameters."""
        self.start = probe.start
        self.vector = probe.vector
        self.radius = probe.radius


class CircleProbe(ProbeData):
    C_infill = QColor(80, 80, 80, 100)
    ANIM_HZ = 120  # Animation frames per second
    AMIN_TIME_LENGTH = 1500  # Animation time length in milliseconds
    
    def __init__(self, start, vector, radius):
        super().__init__(start, vector, radius)
        self.anim_time = 0
        self.anim_step = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.advance_animation)
        self.timer.start(CircleProbe.AMIN_TIME_LENGTH // CircleProbe.ANIM_HZ)  # Start the timer for 60 HZ animation
        
        # draw config
        self.should_draw_skeleton = False
        self.should_draw_body = True  # Flag to control body drawing
        self.should_draw_body_ruler = True  # Flag to control ruler drawing
        self.should_draw_cursor = False  # Flag to control cursor drawing
        
        self.limit_start = 0.3  # Start drawing the body at 30% of the vector length
        self.limit_end = 0.05
        
        self.cursor_label = "3mm"  # Label for the cursor, can be changed if needed
    
    def copy(self, probe):
        """Update the probe with new parameters."""
        super().copy(probe)
        self.start_animation() if probe.is_animating else self.stop_animation()
        self.anim_time = probe.anim_time
        self.anim_step = probe.anim_step
        self.limit_start = probe.limit_start
        self.limit_end = probe.limit_end

        
        
    def start_animation(self):
        self.anim_time = 0
        self.anim_step = 0
        self.timer.start(CircleProbe.AMIN_TIME_LENGTH // CircleProbe.ANIM_HZ)  # Restart the timer for animation
        
    def stop_animation(self):
        self.timer.stop()
        
    @property
    def is_animating(self):
        """Check if the animation is currently running."""
        return self.timer.isActive()
    
    @property
    def limited_start_pos(self):
        """Get the start position limited by the start percentage."""
        return self.start + self.vector * self.limit_start
    
    @property
    def limited_end_pos(self):
        """Get the end position limited by the end percentage."""
        return self.start + self.vector * (1 - self.limit_end)
    
    @property
    def limited_vector(self):
        """Get the vector limited by the start and end percentages."""
        return self.limited_end_pos - self.limited_start_pos
        
    def advance_animation(self):
        self.anim_time += CircleProbe.AMIN_TIME_LENGTH / CircleProbe.ANIM_HZ  # Increment animation time by the frame duration
        if self.anim_time >= CircleProbe.AMIN_TIME_LENGTH:
            self.anim_time = 0
            self.anim_step = 0
        else:
            self.anim_step = ease_out_cubic(self.anim_time / CircleProbe.AMIN_TIME_LENGTH)
    
    def draw_skeleton(self, painter, scale, vector_scaled, start_radius, end_radius, start_pos, end_pos):
        painter.setPen(QPen(ProbeData.C_edge, 1))
        painter.setBrush(QBrush(ProbeData.C_infill, Qt.SolidPattern))
        
        normalized_vector = vector_scaled / math.sqrt(vector_scaled.x()**2 + vector_scaled.y()**2)
        # define perpendicular vector to the vector        
        perp_vector = QPointF(-vector_scaled.y(), vector_scaled.x())
        start_perp_vector = perp_vector / math.sqrt(perp_vector.x()**2 + perp_vector.y()**2) * start_radius
        start_perp_vector_size = math.sqrt(start_perp_vector.x()**2 + start_perp_vector.y()**2)
        end_perp_vector = perp_vector / math.sqrt(perp_vector.x()**2 + perp_vector.y()**2) * end_radius
        end_perp_vector_size = math.sqrt(end_perp_vector.x()**2 + end_perp_vector.y()**2)
        
        
        # find points touching the start circle in the perpendicular direction
        start_left = start_pos + start_perp_vector
        start_right = start_pos - start_perp_vector
        start_top = start_pos + normalized_vector * start_perp_vector_size
        start_bottom = start_pos - normalized_vector * start_perp_vector_size
        
        # find points touching the end circle in the perpendicular direction
        end_left = end_pos + end_perp_vector
        end_right = end_pos - end_perp_vector
        end_top = end_pos + normalized_vector * end_perp_vector_size
        end_bottom = end_pos - normalized_vector * end_perp_vector_size
 
        # draw the cylinder as a polygon
        painter.drawPolygon(
            QPointF(start_left.x(), start_left.y()),
            QPointF(start_right.x(), start_right.y()),
            QPointF(end_right.x(), end_right.y()),
            QPointF(end_left.x(), end_left.y())
        )
        
        # painter.drawPolygon(
        #     QPointF(start_top.x(), start_top.y()),
        #     QPointF(start_bottom.x(), start_bottom.y()),
        #     QPointF(end_bottom.x(), end_bottom.y()),
        #     QPointF(end_top.x(), end_top.y())
        # )
        
        # draw the start and end circles
        painter.setPen(QPen(ProbeData.C_edge, 1))
        painter.drawEllipse(end_pos, end_radius, end_radius)
        painter.drawEllipse(start_pos, start_radius, start_radius)

    
    def draw_body(self, painter, vector_scaled, start_scaled, radius_scaled):
        if not the_annotator: return
        
        
        width = the_annotator.image_size.width()
        height = the_annotator.image_size.height()
        buffer = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        
        buffer.fill(Qt.transparent)
        
        buffer_painter = QPainter(buffer)
        buffer_painter.setRenderHint(QPainter.Antialiasing)
        
        body_start_color = 190
        body_end_color = 80
        
        pen_start_width = 2
        pen_end_width = 1
        
        def draw_slice(percentage, index, color_override=None):
            # Compute position along vector
            t_spaced = percentage
            pos = start_scaled - vector_scaled * t_spaced

            radius = radius_scaled * (1 - t_spaced)

            # Draw ellipse centered at `pos`
            color_value = int(body_start_color * (1 - t_spaced) + body_end_color * t_spaced)
            color = color_override or QColor(color_value, color_value, color_value, 255)
            pen_width = pen_start_width * (1 - t_spaced) + pen_end_width
            
            if index % 5 == 0 and self.should_draw_body_ruler:
                buffer_painter.setPen(QPen(QColor(255, 165, 0, 255), pen_width))
            else:
                buffer_painter.setPen(QPen(color, 1))
            buffer_painter.setBrush(QBrush(color, Qt.SolidPattern))
            buffer_painter.drawEllipse(pos, radius, radius)
        
        
        
        circle_count = int(vector_scaled.manhattanLength()*0.12)
        draw_slice(1 - self.limit_end, 0)
        for i, n in enumerate(reversed(range(circle_count))):
            if circle_count == 1:
                continue
            t = n / (circle_count - 1)
            t_spaced = t # linear
            t_spaced = 0.5 * (1 + math.cos(math.pi * t)) # ease in cosine
            t_spaced = 0.5 * (1 - math.cos(math.pi * t)) # ease out cosine
            t_spaced = t * t  # denser toward the end
            t_spaced = math.sqrt(t) # ease out square root

            if t_spaced < self.limit_start or t_spaced > (1 - self.limit_end):
                continue
            draw_slice(t_spaced, i)

            
        # Draw the last circle at the start position
        draw_slice(self.limit_start, 0, QColor(200, 200, 200, 255))


        buffer_painter.end()
        # Draw the buffer onto the main painter
        painter.setOpacity(0.65)  # Set opacity for the body
        painter.drawImage(0, 0, buffer)
        painter.setOpacity(1.0)  # Reset opacity for other drawings
            
        
    def draw_cursor(self, painter, start_pos, end_pos, start_radius, end_radius):
        limited_vector = start_pos - end_pos
        pos = start_pos - (limited_vector * self.anim_step)
        rad = start_radius * (1 - self.anim_step) + end_radius * self.anim_step
        
        start_width = 2.5
        end_width = 2
        width = start_width * (1 - self.anim_step) + end_width
        
        # Draw the current probe position in a different color
        painter.setPen(QPen(Qt.darkBlue, width))
        # painter.setBrush(QBrush(Qt.transparent, Qt.SolidPattern))
        painter.drawEllipse(pos, rad, rad)
        painter.drawLine(pos + QPointF(-rad, 0), pos + QPointF(rad, 0))  # Draw a line above the circle
        painter.drawText(pos + QPointF(-5, -5), self.cursor_label)  # Draw the label above the circle
        
    
    def draw(self, painter, scale):
        # Draw a cylinder-like probe
        painter.setPen(QPen(CircleProbe.C_infill, 2))
        painter.setBrush(QBrush(Qt.transparent, Qt.SolidPattern))

        # Draw the main body as a rectangle
        start_scaled = self.start * scale
        vector_scaled = self.vector * scale
        end_scaled = self.end * scale
        radius_scaled = self.radius * scale
                
        start_pos = start_scaled - vector_scaled * self.limit_start
        end_pos = start_scaled - vector_scaled * (1 - self.limit_end)
        
        start_radius = radius_scaled * (1 - self.limit_start)
        end_radius = radius_scaled * self.limit_end
        
        
        if self.should_draw_body:
            self.draw_body(painter, vector_scaled, start_scaled, radius_scaled)
            
        if self.should_draw_skeleton:
            self.draw_skeleton(painter, scale, vector_scaled, start_radius, end_radius, start_pos, end_pos)
        
        if self.should_draw_cursor:
            self.draw_cursor(painter, start_pos, end_pos, start_radius, end_radius)
                    
  

        
        # Draw the end circles
        # painter.setPen(QPen(Qt.red, 1))
        # painter.drawEllipse(end_scaled, radius_scaled, radius_scaled)


class Handle:
    def __init__(self, position, radius=4, move_func=None):
        self.position = position  # QPointF
        self.radius = radius  # in pixels
        self.is_enabled = True  # Flag to enable/disable the handle
        self.move_func = move_func  # Optional function to call on move, can be used for custom behavior

    def move(self, delta):
        if self.move_func:
            self.move_func(delta)
        else:
            self.position += delta

    def draw(self, painter, scale):
        painter.drawEllipse(self.position * scale, self.radius * scale, self.radius * scale)
        
    def contains(self, point):
        return (self.position - point).manhattanLength() <= self.radius + 2
    
    def __str__(self):
        return f"Handle(pos={self.position}, rad={self.radius})"


class TwoLineProbeDefinerBuilder(Paintable):
    def __init__(self, a, b):
        # inputs
        self.a = a
        self.b = b
        self.annotator = None  # This will be set by the annotator

        # internal state
        self.a1, self.a2, self.b1, self.b2 = None, None, None, None    
        self.intersection = None  # intersection point of the two lines   
        self.circ_center = None  # center of the circle
        self.circ_radius = None  # radius of the circle
        self.point_on_bisector = None
        self.point_on_line_a = None  # point on line a that is on the circle
        self.reference_probe_radius = None  # radius of the probe circle

    def draw(self, painter, scale):
        painter.setPen(Qt.darkGreen)
        painter.setBrush(Qt.darkGreen)

        # Draw the lines
        if self.a1 and self.a2:
            painter.drawLine(self.a1 * scale, self.a2 * scale)
        if self.b1 and self.b2:
            painter.drawLine(self.b1 * scale, self.b2 * scale)

        print(self.a[0], self.a[1], self.b[0], self.b[1])

        # Draw the intersection point
        if self.intersection:
            painter.setBrush(Qt.darkCyan)
            painter.drawEllipse(self.intersection * scale, 4 * scale, 4 * scale)
            # add a label
            painter.drawText(self.intersection * scale + QPointF(5, 5), "Intersection")
            
            

        # Draw the circle
        if self.circ_radius:
            painter.setBrush(Qt.darkGreen)
            painter.setBrush(QBrush(Qt.transparent, Qt.SolidPattern))
            painter.drawEllipse(self.intersection * scale, self.circ_radius * scale, self.circ_radius * scale)
        
        # Draw the point on the bisector
        if self.point_on_bisector:
            painter.setBrush(Qt.blue)
            painter.setPen(QPen(Qt.blue, 2))
            painter.drawEllipse(self.point_on_bisector * scale, 2 * scale, 2 * scale)
            print("Point on bisector:", self.point_on_bisector)
                   
        # Draw probe skeleton
        if self.intersection and self.point_on_bisector:
            return
            painter.setPen(QPen(Qt.blue, 2))
            probe = CircleProbe(self.point_on_bisector, self.point_on_bisector - self.intersection, self.reference_probe_radius)
            probe.draw(painter, scale)
            
    def build(self):
        # 1. Intersection A of the two lines
        self.a1, self.a2 = self.a[0].position, self.a[1].position
        self.b1, self.b2 = self.b[0].position, self.b[1].position
        angle_a = math.atan2(self.a2.y() - self.a1.y(), self.a2.x() - self.a1.x())
        angle_b = math.atan2(self.b2.y() - self.b1.y(), self.b2.x() - self.b1.x())
        angle_between = abs(angle_a - angle_b)
        # print("Angle between lines:", math.degrees(angle_between))
        
        denom = (self.a2.x() - self.a1.x()) * (self.b2.y() - self.b1.y()) - (self.a2.y() - self.a1.y()) * (self.b2.x() - self.b1.x())
        if denom == 0:
            print("Lines are parallel or coincident, cannot find intersection.")
            return None
        
        ua = ((self.b2.x() - self.b1.x()) * (self.a1.y() - self.b1.y()) - (self.b2.y() - self.b1.y()) * (self.a1.x() - self.b1.x())) / denom
        self.intersection = QPointF(
            self.a1.x() + ua * (self.a2.x() - self.a1.x()),
            self.a1.y() + ua * (self.a2.y() - self.a1.y())
        )
        # print("Intersection point:", self.intersection)
        
        def normalize(dx, dy):
            mag = math.hypot(dx, dy)
            return dx / mag, dy / mag

        # Direction unit vectors of both lines
        u1x, u1y = normalize(self.a2.x() - self.a1.x(), self.a2.y() - self.a1.y())
        u2x, u2y = normalize(self.b2.x() - self.b1.x(), self.b2.y() - self.b1.y())

        # Angle bisector direction
        bisector_dx = u1x + u2x
        bisector_dy = u1y + u2y
        bisector_dx, bisector_dy = normalize(bisector_dx, bisector_dy)

        # Point in bisector direction from A
        self.circ_radius = self.annotator.image_size.width() * 1.5 / 2 * 0.9  # 90% of the radius
        self.point_on_bisector = QPointF(
            self.intersection.x() - bisector_dx * self.circ_radius,
            self.intersection.y() - bisector_dy * self.circ_radius
        )
        
        # Point on lina a and circ
        self.point_on_line_a = QPointF(
            self.intersection.x() - u1x * self.circ_radius,
            self.intersection.y() - u1y * self.circ_radius
        )
        
        def distance(p1, p2):
            return math.hypot(p1.x() - p2.x(), p1.y() - p2.y())
        
        self.reference_probe_radius = distance(self.point_on_bisector, self.point_on_line_a)
        
        
        return CircleProbe(
            self.point_on_bisector, 
            self.point_on_bisector - self.intersection, 
            self.reference_probe_radius
        )



class TwoLineProbeDefiner(Paintable):
    def __init__(self):
        self.annotator = None  # This will be set by the annotator
        self.a = [Handle(QPointF(893, 715), 6), Handle(QPointF(700, 580))]
        self.b = [Handle(QPointF(955, 396), 6), Handle(QPointF(720, 462))]
        
        self.a = [Handle(QPointF(892/2, 716/2), 6), Handle(QPointF(700/2, 580/2))]
        self.b = [Handle(QPointF(954/2, 396/2), 6), Handle(QPointF(720/2, 462/2))]
        self._handles = self.a + self.b
        self.builder = TwoLineProbeDefinerBuilder(self.a, self.b)

        
    def set_annotator(self, annotator):
        self.annotator = annotator
        self.builder.annotator = annotator
        self.builder.a = self.a
        self.builder.b = self.b
        self.register()
    
    def register(self):
        """Register the handles with the annotator."""
        if not self.annotator:
            return
        self.annotator.manipulatable.extend(self._handles)
        self.annotator.update()
        
    def enable(self):
        """Enable the handles."""
        for handle in self._handles:
            handle.is_enabled = True
        
    def disable(self):
        """Disable the handles."""
        for handle in self._handles:
            handle.is_enabled = False
        
        
    def draw(self, painter, scale):
        painter.setPen(Qt.green)
        painter.setBrush(Qt.green)
        
        # Draw the lines
        painter.drawLine(self.a[0].position * scale, self.a[1].position * scale)
        painter.drawLine(self.b[0].position * scale, self.b[1].position * scale)
        
        # Draw the handles
        for handle in self.a + self.b:
            handle.draw(painter, scale)
        


    def build(self):
        return self.builder.build()
        
    

class ProbeTransformer(Paintable):
    def __init__(self, ref_probe, virt_probe):
        self.ref_probe = ref_probe  # CircleProbe instance
        self.virt_probe = virt_probe
        self.handle = Handle(ref_probe.limited_start_pos + ref_probe.limited_vector * 0.5, 6, move_func=self.handle_move_handle)

    def draw(self, painter, scale):
        print(f"Drawing probe transformer {self.ref_probe.limited_vector} {self.ref_probe.limited_start_pos}")
        self.handle.position = self.virt_probe.limited_start_pos/2 - self.virt_probe.limited_vector * 0.5
        # annotator_center_pos = QPointF(the_annotator.image_size.width()/2, the_annotator.image_size.height()/2)
        # self.handle.position = annotator_center_pos
        print(self.handle)
        painter.setPen(Qt.green)
        painter.setBrush(Qt.green)
        self.handle.draw(painter, scale)
        
    def handle_move_handle(self, delta):
        """Handle the movement of the probe handle."""
        if abs(delta.x()) < 20 or abs(delta.y()) < 20: return
        if self.virt_probe:
            annotator_center_pos = QPointF(the_annotator.image_size.width() / 2, the_annotator.image_size.height() / 2)
            center_to_ref_start = self.ref_probe.start - annotator_center_pos
            center_to_ref_end = self.ref_probe.end - annotator_center_pos
            center_to_virt_start = self.virt_probe.start - annotator_center_pos
            center_to_virt_end = self.virt_probe.end - annotator_center_pos
            


            new_start_x_perc = (center_to_virt_start.x() / center_to_ref_start.x()) + (delta.x() / 1000)
            new_start_y_perc = (center_to_virt_start.y() / center_to_ref_start.y()) + (delta.y() / 1000)
            new_end_x_perc = (center_to_virt_end.x() / center_to_ref_end.x()) + (delta.x() / 1000)
            new_end_y_perc = (center_to_virt_end.y() / center_to_ref_end.y()) + (delta.y() / 1000)
            
            scaled_center_to_start_dx = center_to_ref_start.x() * new_start_x_perc
            scaled_center_to_start_dy = center_to_ref_start.y() * new_start_y_perc
            scaled_center_to_end_dx = center_to_ref_end.x() * new_end_x_perc
            scaled_center_to_end_dy = center_to_ref_end.y() * new_end_y_perc 
            
            new_start = annotator_center_pos + QPointF(scaled_center_to_start_dx, scaled_center_to_start_dy)
            new_end = annotator_center_pos + QPointF(scaled_center_to_end_dx, scaled_center_to_end_dy)
            new_vector = new_end - new_start
            
            print("New start:", new_start)
            print("New end:", new_end)
            print("New vector:", new_vector)
            
            self.virt_probe.start = new_start
            self.virt_probe.vector = new_vector
            self.handle.position = new_start + new_vector * 0.5
            
    def register(self, annotator):
        """Register the handle with the annotator."""
        if not annotator:
            return
        annotator.manipulatable.append(self.handle)
        annotator.update()
        
    def enable(self):
        """Enable the handle."""
        self.handle.is_enabled = True

    def disable(self):
        """Disable the handle."""
        self.handle.is_enabled = False


class ImageAnnotator(QLabel):
    def __init__(self):
        super().__init__()
        self.setScaledContents(False)  # Disable automatic scaling to use custom scaling
        self.original_image = None
        self.qimage = None
        self.scaled_image = None
        self.real_tool_radius_mm = 0
        self.scale_factor = 1.0
        self.image_size = None
        self.manipulatable = []
        self.selected_manipulatable = -1
        self.dragging = False
        
        self.reference_probe = CircleProbe(QPointF(-10000, -10000), QPointF(-10000, -10000), 1)
        self.virtual_probe = CircleProbe(QPointF(-10000, -10000), QPointF(-10000, -10000), 1)
        
        self.virtual_probe.should_draw_skeleton = True
        self.virtual_probe.should_draw_body = True
        self.virtual_probe.should_draw_body_ruler = True
        self.virtual_probe.should_draw_cursor = True
        self.virtual_probe.stop_animation()
        self.virtual_probe.anim_step = 0.75
                
        self.is_definer_mode = True
        self.definer = TwoLineProbeDefiner()
        self.definer.set_annotator(self)
        
        self.transformer = ProbeTransformer(self.reference_probe, self.virtual_probe)
        self.transformer.register(self)
        
        self.draw_timer = QTimer()
        self.draw_timer.timeout.connect(self.update)
        self.draw_timer.start(1000 // 60)  # 60 HZ
        

    def load_image(self, path):
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            return
        height, width, ch = self.original_image.shape
        self.image_size = QSize(width, height)
        cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB, self.original_image)
        # Initialize ellipses in original image coordinates
        self.reference_probe.copy(CircleProbe(QPointF(-10000, -10000), QPointF(-10000, -10000), 1))
        self.virtual_probe.copy(CircleProbe(QPointF(-10000, -10000), QPointF(-10000, -10000), 1))
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
        
        for x in self.manipulatable:
            if isinstance(x, Handle) and x.contains(pos) and x.is_enabled:
                self.selected_manipulatable = self.manipulatable.index(x)
                self.dragging = True
                return

    def mouseMoveEvent(self, event):
        pos = self.to_image_coords(event.pos())
        
        if self.dragging and self.selected_manipulatable >= 0:
            handle = self.manipulatable[self.selected_manipulatable]
            delta = pos - handle.position
            handle.move(delta)
            self.update()

    def mouseReleaseEvent(self, event):
        if self.dragging:
            self.dragging = False
            self.selected_manipulatable = -1
            self.update()                

    def paintEvent(self, event):
        super().paintEvent(event)
        # self.definer.builder.draw(painter, self.scale_factor)
        
        if self.is_definer_mode:
            new_probe = self.definer.build()
            if new_probe:
                ProbeData.copy(self.reference_probe, new_probe)
                ProbeData.copy(self.virtual_probe, new_probe)
                # self.reference_probe.copy(new_probe)
                # self.virtual_probe.copy(new_probe)
        
        self.draw()
            
            
    def draw(self):
        if self.scaled_image is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
            
        if self.is_definer_mode: 
            self.reference_probe.draw(painter, self.scale_factor)
            print("Drawing definer probe")
            print("limit_start:", self.reference_probe.limit_start)
            print("limit_end:", self.reference_probe.limit_end)
            self.definer.draw(painter, self.scale_factor)
        else:
            if self.virtual_probe:
                self.virtual_probe.draw(painter, self.scale_factor)
                # self.transformer.draw(painter, self.scale_factor)
        
        
    def set_real_tool_radius_mm(self, diameter):
        self.real_tool_radius_mm = float(diameter)

    def get_tool_scale(self):
        return self.reference_probe.radius / self.real_tool_radius_mm
    
    def get_pixel_to_mm_ratio(self):
        """Calculate the pixel to mm ratio based on the reference probe."""
        if self.reference_probe.radius <= 0:
            return 1.0
        return self.reference_probe.radius / self.real_tool_radius_mm
        




class MainWindow(QMainWindow):
    def __init__(self, file_path=None):
        global the_annotator
        super().__init__()
        self.setWindowTitle("Bronchoscope Tool Annotator")
        self.resize(1200, 800)

        self.annotator = ImageAnnotator()
        the_annotator = self.annotator  # Global reference for easy access

        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        # --- Reference Probe Section ---
        ref_probe_group = QGroupBox("Reference Probe")
        ref_probe_layout = QVBoxLayout()

        # Toggle defining reference
        self.toggle_defining_reference = QPushButton("Define Reference")
        self.toggle_defining_reference.setCheckable(True)
        self.toggle_defining_reference.clicked.connect(self.toggle_definer_mode)
        ref_probe_layout.addWidget(self.toggle_defining_reference)

        # Real probe radius input
        ref_radius_form = QFormLayout()
        self.ref_diameter_input = QLineEdit("3")
        self.ref_diameter_input.textChanged.connect(self.handle_reference_diameter_change)
        ref_radius_form.addRow("Real Probe Diameter (mm):", self.ref_diameter_input)
        ref_probe_layout.addLayout(ref_radius_form)

        def init_probe_view_options(options, layout, probe):
            for option in ["Draw Skeleton", "Draw Body", "Draw Body Ruler", "Draw Cursor", "Animate Cursor"]:
                cb = QCheckBox(option)
                layout.addWidget(cb)
                options[option] = cb
                
            
            options["Draw Skeleton"].setChecked(probe.should_draw_skeleton)
            options["Draw Skeleton"].clicked.connect(lambda checked: setattr(probe, 'should_draw_skeleton', checked))
            options["Draw Body"].clicked.connect(lambda checked: setattr(probe, 'should_draw_body', checked))
            options["Draw Body"].setChecked(probe.should_draw_body)
            options["Draw Body Ruler"].clicked.connect(lambda checked: setattr(probe, 'should_draw_body_ruler', checked))
            options["Draw Body Ruler"].setChecked(probe.should_draw_body_ruler)
            options["Draw Cursor"].clicked.connect(lambda checked: setattr(probe, 'should_draw_cursor', checked))
            options["Draw Cursor"].setChecked(probe.should_draw_cursor)
            options["Animate Cursor"].clicked.connect(lambda checked: probe.start_animation() if checked else probe.stop_animation())
            options["Animate Cursor"].setChecked(probe.is_animating)

        
        # View options checkboxes
        ref_probe_layout.addWidget(QLabel("View Options:"))
        self.ref_view_options = {}
        # Initialize view options for reference probe
        init_probe_view_options(self.ref_view_options, ref_probe_layout, self.annotator.reference_probe)

        # Sliders limiting probe start/end %
        self.ref_start_slider = QSlider(Qt.Horizontal)
        self.ref_start_slider.setRange(0, 1000)
        self.ref_start_slider.setValue(int(self.annotator.reference_probe.limit_start * 1000))
        self.ref_start_slider_label = QLabel(f"Limit Probe Start ({log_slider_to_value(self.ref_start_slider.value(), 1000)*100:.1f}%)")
        
        def ref_start_slider_valueChanged(value):
            new_value = log_slider_to_value(value, 1000)
            self.annotator.reference_probe.limit_start = new_value
            self.ref_start_slider_label.setText(f"Limit Probe Start ({new_value*100:.1f}%)")
            
        self.ref_start_slider.valueChanged.connect(ref_start_slider_valueChanged)
        self.ref_start_slider.setSingleStep(10)  # Set single step to 10 for finer control
        self.ref_start_slider.setPageStep(100)  # Set page step to 100 for larger jumps
        self.ref_start_slider.setTickInterval(100)  # Set tick interval to 100 for better visibility
        self.ref_start_slider.setTickPosition(QSlider.TicksBelow)
        
        ref_probe_layout.addWidget(self.ref_start_slider_label)
        ref_probe_layout.addWidget(self.ref_start_slider)

        self.ref_end_slider = QSlider(Qt.Horizontal)
        self.ref_end_slider.setRange(0, 1000)
        self.ref_end_slider.setValue(int((1 - self.annotator.reference_probe.limit_end) * 1000))
        self.ref_end_slider_label = QLabel(f"Limit Probe End ({(1 - log_slider_to_value(self.ref_end_slider.value(), 1000))*100:.1f}%)")
        
        def ref_end_slider_valueChanged(value):
            new_value = 1 - log_slider_to_value(value, 1000)
            self.annotator.reference_probe.limit_end = new_value
            self.ref_end_slider_label.setText(f"Limit Probe End ({new_value*100:.1f}%)")

        self.ref_end_slider.valueChanged.connect(ref_end_slider_valueChanged)        
        self.ref_end_slider.setSingleStep(10)  # Set single step to 10 for finer control
        self.ref_end_slider.setPageStep(100)  # Set page step to 100 for larger jumps
        self.ref_end_slider.setTickInterval(100)  # Set tick interval to 100 for better visibility
        self.ref_end_slider.setTickPosition(QSlider.TicksBelow)
        self.ref_end_slider.setTracking(True)
        ref_probe_layout.addWidget(self.ref_end_slider_label)
        ref_probe_layout.addWidget(self.ref_end_slider)

        
        ref_probe_group.setLayout(ref_probe_layout)

        # --- Virtual Probe Section ---
        virt_probe_group = QGroupBox("Virtual Probe")
        virt_probe_layout = QVBoxLayout()

        # Slider multiplying real radius
        virt_probe_layout.addWidget(QLabel("Radius Multiplier"))
        self.virt_radius_multiplier_slider = QSlider(Qt.Horizontal)
        self.virt_radius_multiplier_slider.setRange(1, 1500)  # 0.01 to 5.00, scale by 100 maybe
        self.virt_radius_multiplier_slider.setValue(100)  # 1.0x default
        self.virt_radius_multiplier_slider.valueChanged.connect(self.handle_virtual_radius_multiplier_change)
        virt_probe_layout.addWidget(self.virt_radius_multiplier_slider)

        # Same view options as reference probe
        virt_probe_layout.addWidget(QLabel("View Options:"))
        self.virt_view_options = {}
        init_probe_view_options(self.virt_view_options, virt_probe_layout, self.annotator.virtual_probe)

        self.virt_start_slider = QSlider(Qt.Horizontal)
        self.virt_start_slider.setRange(0, 1000)
        self.virt_start_slider.setValue(int(self.annotator.virtual_probe.limit_start * 1000))
        self.virt_start_slider_label = QLabel(f"Limit Probe Start ({log_slider_to_value(self.virt_start_slider.value(), 1000)*100:.1f}%)")

        def virt_start_slider_valueChanged(value):
            new_value = log_slider_to_value(value, 1000)
            self.annotator.virtual_probe.limit_start = new_value
            self.virt_start_slider_label.setText(f"Limit Probe Start ({new_value*100:.1f}%)")
        
        self.virt_start_slider.valueChanged.connect(virt_start_slider_valueChanged)
        self.virt_start_slider.setSingleStep(10)  # Set single step to 10 for finer control
        self.virt_start_slider.setPageStep(100)  # Set page step to 100 for larger jumps
        self.virt_start_slider.setTickInterval(100)  # Set tick interval to 100 for better visibility
        self.virt_start_slider.setTickPosition(QSlider.TicksBelow)
        virt_probe_layout.addWidget(self.virt_start_slider_label)
        virt_probe_layout.addWidget(self.virt_start_slider)

        self.virt_end_slider = QSlider(Qt.Horizontal)
        self.virt_end_slider.setRange(0, 1000)
        self.virt_end_slider.setValue(int((1 - self.annotator.virtual_probe.limit_end) * 1000))
        self.virt_end_slider_label = QLabel(f"Limit Probe End ({(1 - log_slider_to_value(self.virt_end_slider.value(), 1000))*100:.1f}%)")
        
        def virt_end_slider_valueChanged(value):
            new_value = 1 - log_slider_to_value(value, 1000)
            self.annotator.virtual_probe.limit_end = new_value
            self.virt_end_slider_label.setText(f"Limit Probe End ({new_value*100:.1f}%)")
        
        self.virt_end_slider.valueChanged.connect(virt_end_slider_valueChanged)
        self.virt_end_slider.setSingleStep(10)  # Set single step to 10 for finer control
        self.virt_end_slider.setPageStep(100)  # Set page step to 100 for larger jumps
        self.virt_end_slider.setTickInterval(100)  # Set tick interval to 100 for better visibility
        self.virt_end_slider.setTickPosition(QSlider.TicksBelow)
        virt_probe_layout.addWidget(self.virt_end_slider_label)
        virt_probe_layout.addWidget(self.virt_end_slider)
        
        virt_probe_layout.addWidget(QLabel("Cursor Position (%)"))
        self.virt_cursor_pos_slider = QSlider(Qt.Horizontal)
        self.virt_cursor_pos_slider.setRange(0, 100)
        self.virt_cursor_pos_slider.setValue(75)  # Default to 75%
        self.virt_cursor_pos_slider.valueChanged.connect(
            lambda value: setattr(self.annotator.virtual_probe, 'anim_step', value / 100)
        )
        virt_probe_layout.addWidget(self.virt_cursor_pos_slider)

        virt_probe_group.setLayout(virt_probe_layout)

        self.handle_reference_diameter_change(self.ref_diameter_input.text())  # Initialize with default value

        # Build left sidebar
        left_panel = QVBoxLayout()
        left_panel.addWidget(load_btn)
        # left_panel.addWidget(toggle_probe_definer_btn)
        left_panel.addWidget(ref_probe_group)
        left_panel.addWidget(virt_probe_group)
        left_panel.addStretch()
        left_panel.addWidget(QLabel("Drag the handles to define the probe."))

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

    def toggle_definer_mode(self, checked):
        self.annotator.is_definer_mode = checked

        if checked:
            self.annotator.definer.enable()
            self.annotator.transformer.disable()
        else:
            self.annotator.definer.disable()
            self.annotator.transformer.enable()
            self.reset_virtual_probe()

        # Reference Probe controls enabled only if defining reference
        for cb in self.ref_view_options.values():
            cb.setEnabled(checked)
        self.ref_diameter_input.setEnabled(checked)
        self.ref_start_slider.setEnabled(checked)
        self.ref_end_slider.setEnabled(checked)

        # Virtual Probe controls enabled only if NOT defining reference
        for cb in self.virt_view_options.values():
            cb.setEnabled(not checked)
        self.virt_radius_multiplier_slider.setEnabled(not checked)
        self.virt_start_slider.setEnabled(not checked)
        self.virt_end_slider.setEnabled(not checked)
        self.virt_cursor_pos_slider.setEnabled(not checked)

        self.toggle_defining_reference.setChecked(checked)

        self.annotator.update()

    def handle_reference_diameter_change(self, value):
        """Handle changes to the reference probe diameter input."""
        try:
            radius = float(value)
            self.annotator.set_real_tool_radius_mm(radius/2)
            self.annotator.reference_probe.cursor_label = f"{radius:.1f}mm"
            self.handle_virtual_radius_multiplier_change(self.virt_radius_multiplier_slider.value())  # Update virtual probe radius based on new diameter
            
        except ValueError:
            print("Invalid radius value:", value)
            
    def handle_virtual_radius_multiplier_change(self, value):
        """Handle changes to the virtual probe radius multiplier."""
        try:
            multiplier = float(value) / 100.0  # Convert slider value to multiplier
            virt_target = self.annotator.real_tool_radius_mm * multiplier
            self.annotator.virtual_probe.radius = self.annotator.get_pixel_to_mm_ratio() * virt_target
            self.annotator.virtual_probe.cursor_label = f"{virt_target*2:.1f}mm"
            
        except ValueError:
            print("Invalid radius multiplier value:", value)

    def reset_virtual_probe(self):
        """Reset the virtual probe to a reference probe state."""
        self.annotator.virtual_probe.copy(self.annotator.reference_probe)
        self.annotator.virtual_probe.should_draw_skeleton = True
        self.annotator.virtual_probe.should_draw_body = True
        self.annotator.virtual_probe.should_draw_body_ruler = False
        self.annotator.virtual_probe.should_draw_cursor = True
        self.annotator.virtual_probe.stop_animation()
        self.annotator.virtual_probe.anim_step = 0.75
        
        # reset UI
        self.virt_start_slider.setValue(log_value_to_slider(self.annotator.virtual_probe.limit_start, 1000))
        self.virt_end_slider.setValue(log_value_to_slider(1 - self.annotator.virtual_probe.limit_end, 1000))
        self.virt_cursor_pos_slider.setValue(int(self.annotator.virtual_probe.anim_step * 100))
        self.virt_radius_multiplier_slider.setValue(100)
        
        
        self.annotator.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow(sys.argv[1] if len(sys.argv) > 1 else None)
    win.toggle_definer_mode(True)  # Start in definer mode by default
    win.show()
    sys.exit(app.exec_())