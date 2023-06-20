from io import BytesIO
import matplotlib.pyplot as plt
import tempfile
import tacs_tools
import os
import webbrowser
import tensorflow as tf
import numpy as np
import cv2

def display_datasets_in_html(dataset_train, dataset_val):

    html_file_path = 'inspect_data.html'

    # Collect all images from the dataset
    images_with_label = []

    for image, label in dataset_train.as_numpy_iterator():
        images_with_label.append([image, label])
        
    for image, label in dataset_val.as_numpy_iterator():
        images_with_label.append([image, label])
        
    # Calculate the number of rows needed
    num_images = len(images_with_label)
    num_columns = 7
    num_rows = (num_images + num_columns - 1) // num_columns

    if os.path.exists(html_file_path):
        user_input = input("There is already a html file, do you want to generate a new one press [delete] otherwise press [display]: ")

    if user_input=="display":
        with open(html_file_path, 'r') as file:
            html = file.read()
    else:
        # Generate HTML for displaying the images
        html = ''
        for row in range(num_rows):
            html += '<div style="display:flex;">'
            start_index = row * num_columns
            end_index = min((row + 1) * num_columns, num_images)
            for i in range(start_index, end_index):
                # Encode the image as Base64
                image_data = BytesIO()
                image = tacs_tools.show_labeled_image(images_with_label[i][0], images_with_label[i][1], bool_display=False)
                plt.imshow(image)
                plt.axis('off')
                plt.savefig(image_data, format='png', bbox_inches='tight', pad_inches=0)
                plt.close()
                image_data.seek(0)
                image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')

                # Create the image HTML tag with Base64 data
                html += f'<div style="margin:5px;"><img src="data:image/png;base64,{image_base64}" style="width:200px;height:200px; object-fit: cover;"></div>'
            html += '</div>'

    with open(html_file_path, 'w') as file:
        file.write(html)

    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as temp_file:
        temp_file.write(html.encode('utf-8'))
        temp_file.flush()

        # Open the temporary file in the default web browser
        webbrowser.open('file://' + temp_file.name)


def print_image_channels_range(combined_dataset):
    # Define variables to track minimum and maximum pixel values
    min_value = float('inf')
    max_value = float('-inf')
    # Iterate over the dataset to find the minimum and maximum pixel values
    for data in combined_dataset:
        # Get the minimum and maximum pixel values in each image
        image_min = tf.reduce_min(data[0])
        image_max = tf.reduce_max(data[0])
        
        # Update the overall minimum and maximum pixel values
        min_value = tf.minimum(min_value, image_min)
        max_value = tf.maximum(max_value, image_max)

    # Display the range of pixel values
    print("Minimum pixel value:", min_value)
    print("Maximum pixel value:", max_value)

def get_images_channels_range(dataset):
    # Define variables to track minimum and maximum pixel values
    min_value = float('inf')
    max_value = float('-inf')

    #lst_dataset = tacs_tools.tensorslicedataset_2_list(dataset)
    # Iterate over the dataset to find the minimum and maximum pixel values
    for data in dataset:
        # Get the minimum and maximum pixel values in each image
        image_min = tf.reduce_min(data[0])
        image_max = tf.reduce_max(data[0])     

        # Update the overall minimum and maximum pixel values
        min_value = tf.minimum(min_value, image_min).numpy()
        max_value = tf.maximum(max_value, image_max).numpy()

    return min_value, max_value



def convert_to_tf_image(image):
    if isinstance(image, tf.Tensor):
        # If the image is already a TensorFlow tensor, return it as is
        return image
    elif isinstance(image, np.ndarray):
        # If the image is a NumPy array, convert it to a TensorFlow tensor
        return tf.convert_to_tensor(image, dtype=tf.float32)
    elif isinstance(image, Image.Image):
        # If the image is a PIL Image, convert it to a NumPy array and then to a TensorFlow tensor
        image_array = np.array(image)
        return tf.convert_to_tensor(image_array, dtype=tf.float32)
    elif isinstance(image, str):
        # If the image is a file path, load it as a PIL Image, then convert it to a TensorFlow tensor
        image_pil = Image.open(image)
        image_array = np.array(image_pil)
        return tf.convert_to_tensor(image_array, dtype=tf.float32)
    elif isinstance(image, cv2.cv2):
        # If the image is an OpenCV image, convert it to a NumPy array and then to a TensorFlow tensor
        image_array = np.array(image)
        return tf.convert_to_tensor(image_array, dtype=tf.float32)
    else:
        raise ValueError("Unsupported image type")
    
def convert_to_numpy_array(image):
    if isinstance(image, np.ndarray):
        # If the image is already a NumPy array, return it as is
        return image
    elif isinstance(image, tf.Tensor):
        # If the image is a TensorFlow tensor, convert it to a NumPy array
        return image.numpy()
    elif isinstance(image, Image.Image):
        # If the image is a PIL Image, convert it to a NumPy array
        return np.array(image)
    elif isinstance(image, str):
        # If the image is a file path, load it as a PIL Image, then convert it to a NumPy array
        image_pil = Image.open(image)
        return np.array(image_pil)
    elif isinstance(image, cv2.cv2):
        # If the image is an OpenCV image, convert it to a NumPy array
        return np.array(image)
    else:
        raise ValueError("Unsupported image type")
