import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Load the pre-trained Mask R-CNN model
PATH_TO_SAVED_MODEL = 'path_to_saved_model_directory/saved_model'
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

# Load label map (COCO labels in this case)
PATH_TO_LABELS = 'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load an image
image_path = 'path_to_your_image.jpg'
image_np = np.array(cv2.imread(image_path))
image_np_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# Convert image to tensor
input_tensor = tf.convert_to_tensor(image_np_rgb)
input_tensor = input_tensor[tf.newaxis,...]

# Run the model
output_dict = model(input_tensor)

# Extract outputs
num_detections = int(output_dict['num_detections'][0])
boxes = output_dict['detection_boxes'][0].numpy()
masks = output_dict['detection_masks'][0].numpy()
class_ids = output_dict['detection_classes'][0].numpy().astype(np.int32)
scores = output_dict['detection_scores'][0].numpy()

# Visualize the results
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    boxes,
    class_ids,
    scores,
    category_index,
    instance_masks=masks,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.5  # Adjust this for detection confidence threshold
)

# Convert back to BGR for displaying with OpenCV
image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

# Display the result
cv2.imshow('Mask R-CNN Output', image_np_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the image
cv2.imwrite('output_image.jpg', image_np_bgr)
