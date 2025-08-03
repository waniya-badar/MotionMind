# MotionMind
MotionMind - A Python-based real-time vehicle speed estimation project that uses YOLOv8 object detection, ByteTrack object tracking, and Supervision 0.19.0 for visualization and zone filtering. The project applies perspective transformation to calculate vehicle speeds accurately from a top-down view.

### Features:
- Real-time vehicle detection using Ultralytics YOLOv8
- Multi-object tracking with ByteTrack
- Polygon zone filtering to detect vehicles only in a specific region
- Perspective transformation for accurate speed estimation
- Speed calculation in km/h
- Trace lines to visualize vehicle paths
- Bounding boxes & labels for tracked vehicles

### Tech Stack:
| Category             | Technology                                                             |
| -------------------- | ---------------------------------------------------------------------- |
| **Language**         | Python 3.8+                                                            |
| **Object Detection** | [YOLOv8](https://github.com/ultralytics/ultralytics)                   |
| **Tracking**         | ByteTrack (via [Supervision](https://github.com/roboflow/supervision)) |
| **Visualization**    | OpenCV, Supervision                                                    |
| **Math/Computation** | NumPy                                                                  |
| **CLI**              | argparse                                                               |

### How To Run?
python main.py --source_video_path path/to/your/video.mp4
