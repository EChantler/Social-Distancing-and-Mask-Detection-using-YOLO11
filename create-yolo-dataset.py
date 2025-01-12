import os

# Define paths
dataset_path = "path_to_dataset/images/train"
labels_path = "path_to_dataset/labels/train"
classes = {"mask": 1, "not-mask": 2}

# Create labels directory if it doesn't exist
os.makedirs(labels_path, exist_ok=True)

# Iterate through the dataset
for class_name, class_id in classes.items():
    class_folder = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_folder):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
            img_path = os.path.join(class_folder, img_name)

            # Get image dimensions
            import cv2
            img = cv2.imread(img_path)
            height, width, _ = img.shape

            # Assume full image as bounding box
            x_center = 0.5  # Center of image
            y_center = 0.5
            bbox_width = 1.0
            bbox_height = 1.0

            # Write YOLO format label
            label_file = os.path.join(labels_path, os.path.splitext(img_name)[0] + ".txt")
            with open(label_file, "w") as f:
                f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
