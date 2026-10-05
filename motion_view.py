"""Tkinter view for the meganan. This module never accepts or displays camera frames.

Call ``update(payload)`` from the Tk main thread. All points use the MoveNet
COCO-17 order and normalized, unmirrored (x, y, confidence) coordinates.
"""
from __future__ import annotations

import math
from collections import deque
import tkinter as tk
from tkinter import font as tkfont
from typing import Callable
from motion_topology import BODY_EDGES

from motion_topology import FACE_CONTOURS


class MotionView:
    """A lightweight skeleton-only dashboard, with no image dependencies."""

    BG = "#091117"
    PANEL = "#101e28"
    BORDER = "#223541"
    FG = "#e8f2f6"
    MUTED = "#93aab7"
    GREEN = "#6ce5b1"
    BLUE = "#6cbaff"
    AMBER = "#ffcf7b"
    JOINT_THRESHOLD = 0.30
    EDGES = BODY_EDGES

    def __init__(self, on_close: Callable[[], None],
                 on_fps_limit: Callable[[float], None] | None = None, fps_limit=20,
                 api_address="127.0.0.1:8765"):
        self.root = tk.Tk()
        self.root.title("meganan | CPU + camera")
        self.root.geometry("980x720")
        self.root.minsize(900, 700)
        self.root.configure(bg=self.BG)
        self._on_close = on_close
        self._api_address = api_address
        self._on_fps_limit = on_fps_limit
        self._fps_limit = min(30, max(1, round(float(fps_limit))))
        self._closing = False
        self._payload = {}
        self._recent_events = deque(maxlen=3)
        self._event_font = tkfont.Font(root=self.root, family="DejaVu Sans", size=9)
        self._size = (0, 0)
        self._items = {}
        self._joint_items = []
        self._edge_items = []
        self._face_items = []
        self._face_edge_items = []
        self.canvas = tk.Canvas(self.root, bg=self.BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self._fps_value = tk.DoubleVar(master=self.root, value=self._fps_limit)
        self._fps_scale = tk.Scale(
            self.canvas, from_=1, to=30, resolution=1, orient=tk.HORIZONTAL,
            variable=self._fps_value, showvalue=False, command=self._change_fps_limit,
            bg=self.BG, fg=self.FG, troughcolor=self.BORDER,
            activebackground=self.GREEN, highlightthickness=0, borderwidth=0,
            sliderlength=18, width=12, sliderrelief=tk.FLAT,
        )
        self.canvas.bind("<Configure>", self._layout)
        self.root.protocol("WM_DELETE_WINDOW", self._request_close)
        self.root.bind("<Escape>", self._request_close)
        self.root.bind("<q>", self._request_close)
        self.root.bind("<Q>", self._request_close)

    def _change_fps_limit(self, value):
        self._fps_limit = min(30, max(1, round(float(value))))
        if "fps_limit" in self._items:
            self._set("fps_limit", f"FPS LIMIT  {self._fps_limit}")
        if self._on_fps_limit is not None:
            self._on_fps_limit(float(self._fps_limit))

    def _request_close(self, event=None):
        if not self._closing:
            self._closing = True
            self._on_close()

    def close(self):
        self._closing = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _text(self, key, x, y, text="", size=12, color=None, anchor="nw", **kw):
        item = self.canvas.create_text(
            x, y, text=text, font=("DejaVu Sans", size),
            fill=color or self.FG, anchor=anchor, **kw,
        )
        self._items[key] = item
        return item

    def _set(self, key, text, color=None):
        options = {"text": text}
        if color is not None:
            options["fill"] = color
        self.canvas.itemconfigure(self._items[key], **options)

    def _layout(self, event):
        width, height = event.width, event.height
        if width < 50 or height < 50 or self._size == (width, height):
            return
        self._size = (width, height)
        c = self.canvas
        c.delete("all")
        self._items.clear()
        pad, side_width, gap = 24, 278, 18
        left_width = width - pad * 2 - gap - side_width
        side_x = pad + left_width + gap
        self._text("title", pad, 19, "meganan", 23, self.FG)
        self._text("subtitle", pad, 56, "FACE + BODY  /  CPU INFERENCE  /  ONE PERSON", 10, self.MUTED)
        self._text("uptime", width - pad, 24, "00:00", 12, self.MUTED, anchor="ne")
        self._text("privacy", width - pad, 53, "LANDMARKS ONLY", 10, self.GREEN, anchor="ne")

        top, panel_bottom = 97, height - 104
        c.create_rectangle(pad, top, pad + left_width, panel_bottom,
                           fill="#080e13", outline=self.BORDER)
        c.create_rectangle(side_x, top, width - pad, panel_bottom,
                           fill=self.PANEL, outline=self.BORDER)
        self._text("stage_label", pad + 16, top + 13, "MIRRORED MOVEMENT", 10, self.MUTED)
        self._text("confidence", pad + left_width - 15, top + 13,
                   "CONF > 0.30", 9, self.MUTED, anchor="ne")
        stage_width = left_width - 28
        stage_height = panel_bottom - top - 94
        self._sw = min(stage_width, stage_height * 4 / 3)
        self._sh = self._sw * 3 / 4
        self._sx = pad + (left_width - self._sw) / 2
        self._sy = top + 44 + (stage_height - self._sh) / 2
        # Subtle grid gives normalized motion a spatial reference, without imagery.
        for step in range(1, 4):
            x = self._sx + self._sw * step / 4
            y = self._sy + self._sh * step / 4
            c.create_line(x, self._sy, x, self._sy + self._sh, fill="#13202a")
            c.create_line(self._sx, y, self._sx + self._sw, y, fill="#13202a")
        c.create_rectangle(self._sx, self._sy, self._sx + self._sw,
                           self._sy + self._sh, outline="#1a2b36")
        self._edge_items = [
            c.create_line(0, 0, 0, 0, fill=self._joint_color(b), width=4,
                          capstyle=tk.ROUND, state="hidden")
            for a, b in self.EDGES
        ]
        self._joint_items = [
            c.create_oval(0, 0, 0, 0, fill=self._joint_color(i),
                          outline="#e8f2f6", width=1, state="hidden")
            for i in range(17)
        ]
        self._face_items = [
            c.create_oval(0, 0, 0, 0, fill="#527987", outline="", state="hidden")
            for _ in range(478)
        ]
        self._face_edge_items = [
            c.create_line(0, 0, 0, 0, fill=self.GREEN, width=1.5, state="hidden")
            for _ in FACE_CONTOURS
        ]
        self._text("empty", self._sx + self._sw / 2, self._sy + self._sh / 2,
                   "Step into view", 22, self.MUTED, anchor="center")
        self._text("guide", pad + left_width / 2, panel_bottom - 24,
                   "Show shoulders, elbows and wrists", 11, self.MUTED, anchor="center")

        x = side_x + 18
        self._text("status_label", x, top + 17, "TRACKING STATUS", 10, self.MUTED)
        self._text("status", x, top + 41, "STARTING", 21, self.AMBER)
        self._text("status_detail", x, top + 76, "Waiting for camera", 9,
                   self.MUTED, width=side_width - 36)
        c.create_line(x, top + 112, width - pad - 18, top + 112, fill=self.BORDER)
        self._text("arms_label", x, top + 124, "YOUR ARMS", 10, self.MUTED)
        self._text("left_label", x, top + 148, "LEFT", 10, self.BLUE)
        self._text("left_arm", x + 87, top + 145, "UNKNOWN", 13)
        self._text("right_label", x, top + 176, "RIGHT", 10, self.GREEN)
        self._text("right_arm", x + 87, top + 173, "UNKNOWN", 13)
        self._text("lean_label", x, top + 212, "BODY LEAN / SCREEN", 10, self.MUTED)
        self._text("lean", x, top + 235, "UNKNOWN", 17)
        self._text("angle_label", x, top + 273, "ELBOW ANGLES", 10, self.MUTED)
        self._text("angles", x, top + 296, "L  --     R  --", 14)
        self._text("speed_label", x, top + 332, "MOTION SPEED", 10, self.MUTED)
        self._text("speed", x, top + 355, "--", 14)
        self._text("event_label", x, top + 393, "RECENT EVENTS", 10, self.MUTED)
        self._event_width = side_width - 36
        self._text("events", x, top + 416, "Waiting for movement", 9, self.MUTED)

        metrics_y = height - 81
        for idx, (key, title) in enumerate((
            ("fps", "TRACKER FPS"), ("inference", "MODEL TIME"), ("age", "FRAME AGE")
        )):
            metric_x = pad + idx * 190
            self._text(key + "_label", metric_x, metrics_y, title, 9, self.MUTED)
            self._text(key, metric_x, metrics_y + 20, "--", 19)
        self._text("fps_limit", side_x + 18, height - 87,
                   f"FPS LIMIT  {self._fps_limit}", 10, self.GREEN)
        c.create_window(side_x + 18, height - 61, window=self._fps_scale,
                        anchor="nw", width=side_width - 36, height=27)
        self._text("exit", width - pad - 104, 25, "[ Esc / Q ]  Exit", 10,
                   self.MUTED, anchor="ne")
        c.itemconfigure(self._items["exit"], activefill=self.FG)
        c.tag_bind(self._items["exit"], "<Button-1>", self._request_close)
        self._text("footer", pad, height - 14,
                   "Only landmarks + movement are displayed. Camera image hidden.",
                   8, self.MUTED, anchor="sw")
        self._text("api", width - pad, height - 14, f"API  {self._api_address}",
                   8, self.GREEN, anchor="se")
        self._render()

    @classmethod
    def _joint_color(cls, index):
        if index in (5, 7, 9):
            return cls.BLUE
        if index in (6, 8, 10):
            return cls.GREEN
        if index >= 11:
            return cls.AMBER
        return cls.MUTED

    @staticmethod
    def _number(value):
        try:
            value = float(value)
            return value if math.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _format(cls, value, suffix="", digits=0):
        value = cls._number(value)
        return "--" if value is None else f"{value:.{digits}f}{suffix}"

    def update(self, payload: dict):
        """Update the existing canvas; caller must run on the Tk main thread."""
        if self._closing:
            return
        self._payload = payload
        elapsed = max(0, int(self._number(payload.get("elapsed_s")) or 0))
        if "recent_events" in payload:
            # The GUI and external clients use the same retained event list.
            self._recent_events.clear()
            now = self._number(payload.get("timestamp_unix_s")) or 0
            events = []
            for item in payload.get("recent_events") or []:
                when = self._number(item.get("timestamp_unix_s")) or now
                age = max(0, int(elapsed - max(0, now - when)))
                events.append((age, item.get("event", "")))
        else:
            events = [(elapsed, event) for event in payload.get("events") or []]
        for age, event in events:
            text = {"LEAN_LEFT": "LEAN_RIGHT", "LEAN_RIGHT": "LEAN_LEFT"}.get(str(event), str(event))
            self._recent_events.append(f"{age // 60:02d}:{age % 60:02d}  {text}")
        if self._items:
            self._render()

    def _render(self):
        p, c = self._payload, self.canvas
        points = p.get("points")
        valid = {}
        if points is not None:
            try:
                for i, point in enumerate(points):
                    if i >= 17 or len(point) < 3:
                        break
                    x, y, confidence = map(self._number, point[:3])
                    if (x is not None and y is not None and confidence is not None
                            and 0 <= x <= 1 and 0 <= y <= 1
                            and confidence > self.JOINT_THRESHOLD):
                        valid[i] = (self._sx + (1 - x) * self._sw, self._sy + y * self._sh)
            except (TypeError, ValueError):
                valid = {}
        # Visibility and readiness for action classification are distinct. A
        # cropped body remains useful visual feedback while actions stay unknown.
        body_present = (sum(i in valid for i in range(5, 13)) >= 3
                        and (5 in valid or 6 in valid))
        tracked = bool(p.get("body_tracked", p.get("tracked"))) and body_present
        face_tracked = bool(p.get("face_tracked"))
        person_present = bool(p.get("person_present", tracked or face_tracked))
        display_pose = bool(valid) and person_present and p.get("tracking_level") != "NONE"
        shoulder_low = any(i in valid and (valid[i][1] - self._sy) / self._sh > 0.85
                           for i in (5, 6))
        if face_tracked and not tracked:
            guidance = "Face motion is tracked. Move or turn your head."
        elif shoulder_low:
            guidance = "Step back or tilt camera down"
        elif not (5 in valid and 6 in valid):
            guidance = "Show shoulders, elbows and wrists"
        elif not (9 in valid and 10 in valid):
            guidance = "Bring both hands into view"
        else:
            guidance = "Move your arms or lean left and right"
        if not display_pose:
            valid = {}
        face_valid = {}
        if face_tracked:
            try:
                landmarks = p.get("face_landmarks")
                for i, point in enumerate(landmarks if landmarks is not None else []):
                    if i >= 478 or len(point) < 3:
                        break
                    x, y, z = map(self._number, point[:3])
                    if (x is not None and y is not None and z is not None
                            and 0 <= x <= 1 and 0 <= y <= 1):
                        face_valid[i] = (self._sx + (1 - x) * self._sw, self._sy + y * self._sh)
            except (TypeError, ValueError):
                face_valid = {}
        for i, item in enumerate(self._face_items):
            if i in face_valid:
                x, y = face_valid[i]
                c.coords(item, x - 1, y - 1, x + 1, y + 1)
                c.itemconfigure(item, state="normal")
            else:
                c.itemconfigure(item, state="hidden")
        for (a, b), item in zip(FACE_CONTOURS, self._face_edge_items):
            if a in face_valid and b in face_valid:
                c.coords(item, *face_valid[a], *face_valid[b])
                c.itemconfigure(item, state="normal")
            else:
                c.itemconfigure(item, state="hidden")
        # Dense facial geometry replaces the coarse pose nose/eye/ear markers.
        if face_valid:
            valid = {i: point for i, point in valid.items() if i >= 5}
        for i, item in enumerate(self._joint_items):
            if i in valid:
                x, y = valid[i]
                radius = 5 if i >= 5 else 3
                c.coords(item, x - radius, y - radius, x + radius, y + radius)
                c.itemconfigure(item, state="normal")
            else:
                c.itemconfigure(item, state="hidden")
        for (a, b), item in zip(self.EDGES, self._edge_items):
            if a in valid and b in valid:
                c.coords(item, *valid[a], *valid[b])
                c.itemconfigure(item, state="normal")
            else:
                c.itemconfigure(item, state="hidden")
        display_motion = display_pose or bool(face_valid)
        c.itemconfigure(self._items["empty"], state="hidden" if display_motion else "normal")
        self._set("status", "BODY + FACE" if tracked and face_tracked else
                  "BODY TRACKED" if tracked else "FACE TRACKED" if face_tracked else
                  "PARTIAL VIEW" if display_pose else "SEARCHING",
                  self.GREEN if tracked or face_tracked else self.AMBER)
        self._set("confidence", f"{len(face_valid)} FACE POINTS" if face_valid else "BODY CONF > 0.30")
        self._set("guide", guidance)
        self._set("status_detail", str(p.get("status", "Waiting for camera"))[:105])
        self._set("empty", "Bring your face or body into view" if points is not None else "Starting camera...")
        for key in ("left_arm", "right_arm", "lean"):
            value = str(p.get(key, "UNKNOWN")) if tracked else "UNKNOWN"
            if key == "lean":
                value = {"LEFT": "RIGHT", "RIGHT": "LEFT"}.get(value, value)
            self._set(key, value, self.GREEN if value == "UP" else self.FG if value != "UNKNOWN" else self.MUTED)
        angles = p.get("angles") or {}
        left = angles.get("left_elbow") if tracked else None
        right = angles.get("right_elbow") if tracked else None
        self._set("angles", f"L  {self._format(left, '°')}     R  {self._format(right, '°')}")
        face_motion = p.get("face_motion") or {}
        if face_tracked and not tracked:
            self._set("speed_label", "FACE MOTION")
            self._set("speed", self._format(face_motion.get("speed"), " image/s", 3))
        else:
            self._set("speed_label", "BODY MOTION")
            self._set("speed", self._format(p.get("motion_speed") if tracked else None, " widths/s", 2))
        self._set("fps", self._format(p.get("fps"), " fps", 1))
        self._set("inference", self._format(p.get("inference_ms"), " ms", 1))
        self._set("age", self._format(p.get("frame_age_ms"), " ms"))
        age = self._number(p.get("frame_age_ms"))
        c.itemconfigure(self._items["age"], fill=self.AMBER if age is not None and age > 250 else self.FG)
        elapsed = max(0, int(self._number(p.get("elapsed_s")) or 0))
        self._set("uptime", f"{elapsed // 60:02d}:{elapsed % 60:02d}")
        lines = []
        for event in self._recent_events:
            text = " ".join(event.split())
            if self._event_font.measure(text) > self._event_width:
                while text and self._event_font.measure(text + "…") > self._event_width:
                    text = text[:-1]
                text += "…"
            lines.append(text)
        self._set("events", "\n".join(lines) if lines else "Waiting for movement")
