import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# COCO label map (partial for demo purposes)
COCO_LABELS = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
    5: "airplane", 6: "bus", 7: "train", 8: "truck",
    9: "boat", 10: "traffic light", 11: "fire hydrant"
}

# Load pre-trained model from TensorFlow Hub
print("Loading model...")
model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

# Function to load and preprocess image
def load_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_rgb)
    input_tensor = input_tensor[tf.newaxis, ...]
    return image, input_tensor

# Function to draw bounding boxes and labels on the image
def draw_boxes(image, boxes, class_ids, scores, threshold=0.5):
    height, width, _ = image.shape
    detected_labels = []
    for i in range(len(boxes)):
        if scores[i] >= threshold:
            y1, x1, y2, x2 = boxes[i]
            start_point = (int(x1 * width), int(y1 * height))
            end_point = (int(x2 * width), int(y2 * height))
            
            class_id = class_ids[i]
            label = COCO_LABELS.get(class_id, f"ID {class_id}")
            score = scores[i]
            
            # Store detected labels
            detected_labels.append((label, start_point, end_point))
            
            # Draw bounding box
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            
            # Draw label with score
            label_text = f"{label}: {score:.2f}"
            cv2.putText(image, label_text, (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image, detected_labels

# Function to calculate grid size based on image dimensions
def calculate_grid(image_width, image_height):
    grid_size = 0
    if image_width == 300 and image_height == 300:
        grid_size = 9  # 3x3 grid
    elif image_width == 450 and image_height == 450:
        grid_size = 16  # 4x4 grid
    return grid_size


def match_labels_to_grid(detected_labels, image_width, image_height, target_label):
    grid_size = calculate_grid(image_width, image_height)
    grid_positions = []
    
    # Calculate the size of each grid cell
    grid_dim = int(grid_size ** 0.5)
    grid_width = image_width // grid_dim
    grid_height = image_height // grid_dim
    
    # Check for target label in the detected labels
    for label, start_point, end_point in detected_labels:
        if label.lower() == target_label.lower():  # Case insensitive comparison
            # Calculate the coordinates of the bounding box
            bbox_x1, bbox_y1 = start_point
            bbox_x2, bbox_y2 = end_point
            
            # Loop through all grid tiles and check for overlap
            for row in range(grid_dim):
                for col in range(grid_dim):
                    # Calculate the grid tile boundaries
                    grid_x1 = col * grid_width
                    grid_y1 = row * grid_height
                    grid_x2 = (col + 1) * grid_width
                    grid_y2 = (row + 1) * grid_height
                    
                    # Check if the bounding box overlaps with the grid tile
                    if not (bbox_x2 < grid_x1 or bbox_x1 > grid_x2 or bbox_y2 < grid_y1 or bbox_y1 > grid_y2):
                        grid_position = row * grid_dim + col
                        if grid_position not in grid_positions:
                            grid_positions.append(grid_position)
    
    return grid_positions

# Function to draw the grid on the image and highlight the target grid tiles
def draw_grid_and_highlight(image, grid_size, target_positions, image_width, image_height):
    # Calculate the grid dimensions based on the grid size
    grid_dim = int(grid_size ** 0.5)
    grid_width = image_width // grid_dim
    grid_height = image_height // grid_dim

    # Draw the grid dynamically based on the grid size (3x3, 4x4, etc.)
    for i in range(1, grid_dim):
        # Draw vertical grid lines
        cv2.line(image, (i * grid_width, 0), (i * grid_width, image_height), (255, 255, 255), 2)
        # Draw horizontal grid lines
        cv2.line(image, (0, i * grid_height), (image_width, i * grid_height), (255, 255, 255), 2)

    # Highlight the target grid positions with a different color (e.g., red)
    for pos in target_positions:
        row = pos // grid_dim
        col = pos % grid_dim

        # Calculate the starting and ending points of the tile
        start_point = (col * grid_width, row * grid_height)
        end_point = ((col + 1) * grid_width, (row + 1) * grid_height)

        # Draw the highlighted tile with a red rectangle
        cv2.rectangle(image, start_point, end_point, (0, 0, 255), 3)  # Red color

    return image

# Timing the execution
start_time = time.time()

# Load image and preprocess
image_path = "payload.jpg"
# image_path = "payload (3).jpg"  # Image downloaded from Selenium script
original_image, input_tensor = load_image(image_path)

# Run object detection
print("Running detection...")
detections = model(input_tensor)

# Extract detection results
boxes = detections["detection_boxes"].numpy()[0]
class_ids = detections["detection_classes"].numpy()[0].astype(int)
scores = detections["detection_scores"].numpy()[0]

# Draw results on the image and get detected labels
result_image, detected_labels = draw_boxes(original_image, boxes, class_ids, scores)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Target label to search for
# target_label = "fire hydrant"  # You can change this to search for other labels
target_label = "bus"
# Match detected labels to grid positions for the target label
grid_size = calculate_grid(original_image.shape[1], original_image.shape[0])
grid_positions = match_labels_to_grid(detected_labels, original_image.shape[1], original_image.shape[0], target_label)
print(f"Grid positions for '{target_label}': {grid_positions}")

# Draw grid and highlight the target positions
highlighted_image = draw_grid_and_highlight(original_image.copy(), grid_size, grid_positions, original_image.shape[1], original_image.shape[0])

# Display the final image with highlighted tiles
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Script completed in {elapsed_time:.2f} seconds")


