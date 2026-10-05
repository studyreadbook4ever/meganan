# Copyright 2023 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# meganan modifications, 2026-10-05: extracted the contour, iris, and nose
# connection pairs from MediaPipe 1.0.1 FaceLandmarksConnections into a static
# tuple; combined them with the project's COCO-17 names and body connections.
# Upstream: mediapipe/tasks/python/vision/face_landmarker.py
# See THIRD_PARTY_NOTICES.txt and licenses/mediapipe/{LICENSE,NOTICE}.

"""Shared body/face line topology for rendering and the numerical API.

Only static Python data: importing this module initializes no inference SDK.
FACE_CONTOURS is the MediaPipe 1.0.1 FaceLandmarksConnections contour,
iris, and nose topology generated from its Apache-2.0-licensed definitions.
"""

KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

BODY_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
)

FACE_CONTOURS = (
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270),
    (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
    (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308), (263, 249), (249, 390),
    (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384),
    (384, 398), (398, 362), (276, 283), (283, 282), (282, 295), (295, 285),
    (300, 293), (293, 334), (334, 296), (296, 336), (33, 7), (7, 163),
    (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157),
    (157, 173), (173, 133), (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107), (10, 338), (338, 297),
    (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379),
    (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132),
    (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10), (474, 475), (475, 476),
    (476, 477), (477, 474), (469, 470), (470, 471), (471, 472), (472, 469),
    (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1),
    (1, 19), (19, 94), (94, 2), (98, 97), (97, 2), (2, 326),
    (326, 327), (327, 294), (294, 278), (278, 344), (344, 440), (440, 275),
    (275, 4), (4, 45), (45, 220), (220, 115), (115, 48), (48, 64),
    (64, 98),
)
