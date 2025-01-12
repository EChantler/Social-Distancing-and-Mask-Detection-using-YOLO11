import time
import numpy as np
from scipy.spatial import distance as dist
import cv2
from ultralytics import YOLO
from norfair import Tracker, Detection
import os

tracker = Tracker(
    distance_function="euclidean",  # Distance function to use for associating detections with tracks
    distance_threshold=500  # Threshold for associating detections with tracks
)

def get_distances(centroids, heights, human_height = 170):
    
    # Compute pairwise pixel distances between centroids
    dist_pixel = dist.cdist(centroids, centroids, metric="euclidean")
    
    # Compute pairwise heights and average them
    h_pixel = np.zeros_like(dist_pixel)
    for i in range(len(heights)):
        for j in range(len(heights)):
            h_pixel[i, j] = (heights[i] + heights[j]) / 2  # Average height in pixels
    
    # Convert pixel distances to real-world distances (cm)
    dist_cm = (dist_pixel / h_pixel) * human_height

    # Add centroid pairs
    centroid_pairs = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            centroid_pairs.append((centroids[i], centroids[j]))
    
    return dist_pixel, dist_cm, centroid_pairs

# Check if violation_image directory exists, if not create it
if not os.path.exists("violation_images"):
    os.makedirs("violation_images")

# Load the YOLO model
model = YOLO("yolo11n.pt")
mask_model = YOLO("mask-detection-model\\weights\\best.pt")

# Open the video file
video_path = "5079568_Coworkers_Woman_1280x720.mp4"
# video_path = "5079566_Coworkers_Corridor_1280x720.mp4"
# video_path = "5121314_School_Classroom_1280x720.mp4"
# video_path = "sample_video.mp4"

cap = cv2.VideoCapture(video_path)
potential_violations = []
violations = []
violation_image_export_cooldown_mapping = {}
frame_count = 0

fps = 0
frame_count = 0
start_time = time.time()
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        
        results = model(frame)
        # results = model.track(frame, persist=True)

        # manually draw the bounding boxes on the current frame
        annotated_frame = frame.copy()
        boxes = results[0].boxes

        # get the person boxes (where class is 0)
        person_boxes = [box for box in boxes if box.cls == 0]

        # select xyxy for each box
        person_boxes_xyxy = [box.xyxy for box in person_boxes]
        pboxes = [box.cpu().numpy().astype(int).tolist()[0] for box in person_boxes_xyxy]
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in pboxes]
        heights = [(box[3] - box[1]) for box in pboxes]

        norfair_detections = [
            Detection(points=np.array([[c[0], c[1]]])) for c in centroids
        ]
        tracked_objects = tracker.update(norfair_detections)

        # create a mapping of bounding box index to person id based on centroid distance
        # Map tracked IDs to YOLO bounding boxes
        bb_person_id_map = {}
        for obj in tracked_objects:
            # Find the closest YOLO bounding box to this tracked centroid
            x_center, y_center = obj.estimate[0]
            closest_box = None
            min_distance = float("inf")

            closest_box_index = None
            for i, box in enumerate(pboxes):
                centroid = centroids[i]
                #distance = np.linalg.norm(np.array(centroid) - np.array([x_center, y_center]))
                # Calculate squared distance
                distance_squared = (centroid[0] - x_center) ** 2 + (centroid[1] - y_center) ** 2

                if distance_squared < min_distance:
                    min_distance = distance_squared
                    closest_box = box
                    closest_box_index = i
                    
            if closest_box_index is not None:
                # print("closest_box_index", closest_box_index)
                bb_person_id_map[closest_box_index] = obj.id

        dist_pixel, dist_cm, centroid_pairs = get_distances(centroids, heights)

        # draw line with distance between violations
        for i in range(dist_pixel.shape[0]):
            for j in range(dist_pixel.shape[1]):  # Avoid duplicates
                if(i == j):
                    continue
                real_distance = dist_cm[i, j]   # Real-world distance in cm
                if real_distance < 200:#if i in violations and j in violations:
                    # get the face area as the top 1/3 of the box
                    
                    box = pboxes[i]
                    height = box[3] - box[1]
                    face_area = (box[0], box[1], box[2], box[1] + height // 3)

                    face_start_point = (face_area[0], face_area[1])
                    face_end_point = (face_area[2], face_area[3])
                    face_frame = frame[face_area[1]:face_area[3], face_area[0]:face_area[2]]
                    # run mask detection on the face area
                    face_results = mask_model(face_frame)
                    face_boxes = face_results[0].boxes
                    face_boxes_xyxy = [face_box.xyxy for face_box in face_boxes]

                    face_color = (255,255,255)
                    # if there are no boxes, draw a red rectangle
                    if len(face_boxes_xyxy) == 0:
                        face_color = (0, 0, 0)
                    else:
                        # if there are boxes, check if the boxes are masks
                        mask_boxes = [face_box for face_box in face_boxes if face_box.cls == 0]
                        if len(mask_boxes) == 0:
                            face_color = (0, 0, 255)
                            if(bb_person_id_map.get(i) is not None and bb_person_id_map.get(j) is not None):
                                potential_violations.append(dict(frame = frame_count, dist=real_distance, centroid1=centroids[i], centroid2=centroids[j], person_id1=bb_person_id_map[i], person_id2=bb_person_id_map[j]))
                            cv2.putText(annotated_frame, "No Mask", (face_start_point[0], face_start_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        else:
                            no_mask_boxes = [face_box for face_box in face_boxes if face_box.cls == 1]
                            if len(no_mask_boxes) == 0:
                                face_color = (0, 255, 0)
                                color = (255,0,0)
                                cv2.putText(annotated_frame, "Mask", (face_start_point[0], face_start_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



                    cv2.line(annotated_frame, (int(centroids[i][0]), int(centroids[i][1])), (int(centroids[j][0]), int(centroids[j][1])), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"{real_distance:.2f} cm", (int((centroids[i][0] + centroids[j][0]) / 2), int((centroids[i][1] + centroids[j][1]) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    
        # Draw the bounding box for each person with out mapping and boxes
        for box_index, i in enumerate(bb_person_id_map):
            # get the last 30 frames from potential violations
            person_violations = [violation for violation in potential_violations if violation['person_id1'] == bb_person_id_map.get(i) and violation['frame'] >= frame_count - 30]
            export_violation = False
            if(len(person_violations) > 15):

                color = (0, 0, 255)
                violations.append(person_violations[-1])
                export_violation = True
                

            else:
                color = (0, 255, 0)  # Green for tracked objects
            box = pboxes[i]
            start_point = (box[0], box[1])
            end_point = (box[2], box[3])
            thickness = 2

            # Draw rectangle
            cv2.rectangle(annotated_frame, start_point, end_point, color, thickness)

            # Add ID label
            person_id = bb_person_id_map.get(i)
            if person_id is None:
                person_id = "Unknown"
            cv2.putText(
                annotated_frame,
                f"ID {person_id}",#{obj.id}",
                (start_point[0], start_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
            if export_violation:
                if bb_person_id_map.get(i) in violation_image_export_cooldown_mapping:
                    if frame_count - violation_image_export_cooldown_mapping[bb_person_id_map.get(i)] < 30:
                        continue

                violation_frame = frame.copy()
                cv2.rectangle(violation_frame, start_point, end_point, color, thickness)
                cv2.putText(
                    violation_frame,
                    f"ID {person_id}",
                    (start_point[0], start_point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    )
                person_violation_count = len([violation for violation in violations if violation['person_id1'] == bb_person_id_map.get(i)])
                cv2.imwrite(f"violation_images/person_{bb_person_id_map.get(i)}_{person_violation_count}_violation.png", violation_frame)
                violation_image_export_cooldown_mapping[bb_person_id_map.get(i)] = frame_count
        
        # draw the frame number on the frame
        cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        
        # Calculate the FPS
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:  # Avoid division by zero
            fps = frame_count / elapsed_time

        # Display the FPS on the frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        
        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# export potential_violations to a csv file
import csv
with open('potential_violations.csv', mode='w') as violations_file:
    violations_writer = csv.writer(violations_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    violations_writer.writerow(["frame", "person_id1", "person_id2", "distance"])
    for violation in potential_violations:
        violations_writer.writerow([violation['frame'], violation['person_id1'], violation['person_id2'], violation['dist']])

# export violations to a csv file
with open('violations.csv', mode='w') as violations_file:
    violations_writer = csv.writer(violations_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    violations_writer.writerow(["frame", "person_id1", "person_id2", "distance"])
    for violation in violations:
        violations_writer.writerow([violation['frame'], violation['person_id1'], violation['person_id2'], violation['dist']])
