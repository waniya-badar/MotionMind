import cv2
import argparse
import supervision as sv
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

SOURCE = np.array([[630,290], [1133, 286], [2600, 1014], [-790, 1014]])

TARGET_WIDTH=61
TARGET_HEIGHT=416

TARGET = np.array([[0,0], [TARGET_WIDTH -1 , 0], [TARGET_WIDTH -1, TARGET_HEIGHT -1], [0,TARGET_HEIGHT -1],]) 

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target=target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target) 

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32) 
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2) 

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle Speed Estimation using YOLOv8 and Supervision 0.19.0")
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Get video info
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    # Load YOLOv8 model
    model = YOLO("yolov8x.pt")

    # Tracker setup
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Visualization setup
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness,color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK)

    trace_annotator=sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps*2,
        position=sv.Position.BOTTOM_CENTER,
        color_lookup=sv.ColorLookup.TRACK)


    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates=defaultdict(lambda:deque(maxlen=video_info.fps))

    for frame in frame_generator:
        result = model.predict(frame)[0] 
        detections = sv.Detections.from_ultralytics(result)

       
        detections = detections[polygon_zone.trigger(detections)]

        
        detections = byte_track.update_with_detections(detections=detections)

        points=detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) 
        points = view_transformer.transform_points(points).astype(int)

        labels=[]

        for tracker_id,[_,y] in zip(detections.tracker_id, points):
            coordinates[tracker_id].append(y)
            if len(coordinates[tracker_id]) < video_info.fps/2: 
                labels.append(f"ID:{tracker_id}")
            else:
                coordinate_start=coordinates[tracker_id][-1] 
                coordinate_end=coordinates[tracker_id][0]
                distance=abs(coordinate_start-coordinate_end)
                time=len(coordinates[tracker_id])/video_info.fps
                speed=distance/time*3.6
                labels.append(f"#{tracker_id} {int(speed)}km/h")

        annotated_frame = frame.copy()
        annotated_frame=trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections)

        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break

    cv2.destroyAllWindows()
