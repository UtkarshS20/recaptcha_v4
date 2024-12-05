import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# COCO label map (partial for demo purposes)
COCO_LABELS = {
    1: "person", 2: "bicycles", 3: "cars", 4: "motorcycles",
    5: "airplane", 6: "buses", 7: "train", 8: "truck",
    9: "boat", 10: "traffic lights", 11: "fire hydrants"
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



start_time = time.time()
driver = webdriver.Chrome()

# Open the URL
driver.get("https://www.google.com/recaptcha/api2/demo")
checkbox = WebDriverWait(driver, 10).until(
EC.element_to_be_clickable((By.CSS_SELECTOR, 'div[class="g-recaptcha"]'))
)
checkbox.click()
print("Clicked reCAPTCHA checkbox.")
time.sleep(5)

new_iframe = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, 'iframe[title="recaptcha challenge expires in two minutes"]'))
)
driver.switch_to.frame(new_iframe)

challenge_text = driver.find_element(By.CSS_SELECTOR, 'div[id="rc-imageselect"] strong')
challenge_text_check = driver.find_element(By.CSS_SELECTOR, 'div[id="rc-imageselect"]')
print("checking--------------------------->",challenge_text_check.text)
print("Extracted text:", challenge_text.text)

target_label = challenge_text.text.strip()

# Check if the extracted label matches any value in COCO_LABELS
if target_label.lower() in [label.lower() for label in COCO_LABELS.values()] and "Verify" not in challenge_text_check.text:
    image_element = driver.find_element(By.CSS_SELECTOR, 'div[class="rc-imageselect-challenge"] img')
    image_url = image_element.get_attribute("src")
    print(f"Image URL: {image_url}")

    # Download the image using requests
    response = requests.get(image_url)

    # Save the image to a file
    if response.status_code == 200:
        with open("downloaded_image.jpg", "wb") as file:
            file.write(response.content)
            print("Image downloaded successfully.")
    else:
        print("Failed to download the image.")


    # Load image and preprocess
    image_path = "downloaded_image.jpg"
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

    print(f"Target label '{target_label}' is valid. Proceeding with detection.")
    
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
    for i in range(grid_size+1):
        if i in grid_positions:
            img_btns = driver.find_elements(By.CSS_SELECTOR, f'table[class="rc-imageselect-table-44"] td[tabindex="{i+4}"]')
            print(i+4)
        # if i in grid_positions:
            # [img_btn.click() for img_btn in img_btns]
            for img_btn in img_btns:
                img_btn.click()

            time.sleep(1)

else:
    print(f"Target label '{target_label}' is not a valid COCO label. Exiting the script.")
