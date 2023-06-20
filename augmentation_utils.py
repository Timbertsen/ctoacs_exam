import tensorflow as tf
import numpy as np
import label_utils
import image_utils
from IPython.display import display
import random
import tensorflow_addons as tfa
import math

def resize_image_tf(image, desired_size=224):
    #tf_image = image_utils.convert_to_tf_image(image)
    tf_image = tf.image.resize_with_pad(image, desired_size, desired_size)
    #numpy_array = image_utils.convert_to_numpy_array(tf_image)
    return tf_image

def resize_bbox(bbox, current_image, desired_size=224):
    width = current_image.shape[1]
    height = current_image.shape[0]
    x_factor = desired_size/width
    y_factor = desired_size/height
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max = x_min*x_factor, x_max*x_factor
    y_min, y_max = y_min*y_factor, y_max*y_factor
    bbox = ()
    return x_min, y_min, x_max, y_max

def resize_bbox_tf(bbox, image_shape, desired_size=224):
    """Resize the bounding box according to new image size.

    Parameters
    ----------
    
    bbox : tf.Tensor
        Tensor of shape `4` representing the bounding box in the 
        format `x1 y1 x2 y2`

    current_image_shape : tf.Tensor or tuple
        Tensor or tuple describing the shape of the current image. 

    desired_size : int, optional
        The desired size for the shortest dimension of the image.

    Returns
    -------

    tf.Tensor
        A Tensor of shape `4` representing the resized bounding box in the
        format `x1 y1 x2 y2`
    """

    width, height = tf.cast(image_shape[1], tf.float32), tf.cast(image_shape[0], tf.float32)
    x_factor = desired_size / width
    y_factor = desired_size / height

    x_min, y_min, x_max, y_max = tf.unstack(bbox)

    x_min, x_max = x_min * x_factor, x_max * x_factor
    y_min, y_max = y_min * y_factor, y_max * y_factor

    return tf.stack([x_min, y_min, x_max, y_max])

def get_rotated_image_shape(image, angle):
    # Obtain original image shape
    
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)
    '''# Rotate the image
    rotated_image = tfa.image.rotate(image, angle)

    # Calculate the bounding box coordinates of the rotated image
    x1 = 0
    y1 = 0
    x2 = tf.cast(width, tf.float32) * tf.math.cos(angle)
    y2 = tf.cast(width, tf.float32) * tf.math.sin(angle)
    x3 = -tf.cast(height, tf.float32) * tf.math.sin(angle)
    y3 = tf.cast(height, tf.float32) * tf.math.cos(angle)
    x4 = tf.cast(width, tf.float32) * tf.math.cos(angle) - tf.cast(height, tf.float32) * tf.math.sin(angle)
    y4 = tf.cast(width, tf.float32) * tf.math.sin(angle) + tf.cast(height, tf.float32) * tf.math.cos(angle)


    # Calculate the minimum and maximum coordinates to determine the size of the rotated image
    min_x = tf.reduce_min(tf.stack([x1, x2, x3, x4]))
    max_x = tf.reduce_max(tf.stack([x1, x2, x3, x4]))
    min_y = tf.reduce_min(tf.stack([y1, y2, y3, y4]))
    max_y = tf.reduce_max(tf.stack([y1, y2, y3, y4]))

    # Calculate the width and height of the rotated image
    rotated_width = tf.cast(tf.math.ceil(max_x - min_x), tf.int32)
    rotated_height = tf.cast(tf.math.ceil(max_y - min_y), tf.int32)'''

    return (height, width)


def resize_image_and_bbox_tf(image, bbox, angle, desired_size=224):
    resized_image = resize_image_tf(image, desired_size)
    rotated_image_shape = get_rotated_image_shape(image, angle)
    resized_bbox = resize_bbox_tf(bbox, rotated_image_shape, desired_size)
    return resized_image, resized_bbox

def rotate_label_tf(image, label, angle):
    corners = label_utils.get_corners_tf(image, label)
    rotated_corners = label_utils.rotate_box_tf(corners, image, angle, tf.shape(image)[1]/2, tf.shape(image)[0]/2, tf.shape(image)[0], tf.shape(image)[1])
    enclosed_rotated_corners = label_utils.get_enclosing_box_tf(rotated_corners)
    return enclosed_rotated_corners

def rotate_image_tf(image, angle):
    # Calculate the size of the rotated image
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    diagonal = tf.sqrt(tf.square(tf.cast(height, tf.float32)) + tf.square(tf.cast(width, tf.float32)))
    new_height = tf.cast(diagonal, tf.int32)
    new_width = tf.cast(diagonal, tf.int32)

    # Calculate the amount of padding required
    pad_height = (new_height - height) // 2
    pad_width = (new_width - width) // 2

    # Pad the image around its entire shape
    padded_image = tf.image.pad_to_bounding_box(image, pad_height, pad_width, new_height, new_width)


    # Rotate the padded image
    rotated_image = tfa.image.rotate(padded_image, tf.constant(angle))

    return rotated_image

def show_rotated_label(image, label, angle):
    rotated_image = label_utils.rotate_im(image, angle)
    rotated_bbox = rotate_label_tf(image,label,angle)
    rotated_labeled_image = tf.keras.utils.array_to_img(label_utils.draw_rect(rotated_image, rotated_bbox))
    display(rotated_labeled_image)

def rotate_randomly_image_and_label(image, label):
    angle = random.uniform(0, 2 * math.pi)
    rotated_image = rotate_image_tf(image, angle)
    rotated_bbox = rotate_label_tf(image,label,angle)
    rotated_resized_image, rotated_resized_bbox = resize_image_and_bbox_tf(rotated_image, rotated_bbox, angle, desired_size=224)
    new_label = overwrite_label_bbox_tf(rotated_resized_image, label, rotated_resized_bbox)
    return     rotated_resized_image, new_label



def flip_bbox_left_right(image, label):
    bbox = label_utils.get_bbox(image, label)
    # y coordinates remain the same
    # x coordinates mirrowed to the width/2
    width = image.shape[1]
    x_min = bbox[0]
    x_max = bbox[2]
    if x_min>width/2:
        bbox[2] = x_min-2*abs(x_min-width/2)
    else:
        bbox[2] = x_min+2*abs(x_min-width/2)

    if x_max>width/2:
        bbox[0] = x_max-2*abs(x_max-width/2)
    else:
        bbox[0] = x_max+2*abs(x_max-width/2)
    return bbox

def overwrite_label_bbox(image, label, bbox):
    bbox = np.array(bbox, dtype=np.float32) 
    x_min = bbox[0] / image.shape[1]
    y_min = bbox[1] / image.shape[0]
    x_max = bbox[2] / image.shape[1]
    y_max = bbox[3] / image.shape[0]
    bbox[0], bbox[1], bbox[2], bbox[3] = x_min, y_min, x_max, y_max
    new_label = (label[0], label[1], bbox)
    return new_label

def overwrite_label_bbox_tf(image, label, bbox):
    """Update the bounding box in label.

    Parameters
    ----------
    
    image : tf.Tensor
        Tensor representing the image.

    label : tuple
        Tuple representing the label. The third element of the tuple
        should be the bounding box.

    bbox : tf.Tensor
        Tensor of shape `4` representing the new bounding box in the 
        format `x1 y1 x2 y2`

    Returns
    -------

    tuple
        Tuple representing the updated label.
    """
    bbox = tf.cast(bbox, tf.float32)
    width, height = tf.cast(tf.shape(image)[1], tf.float32), tf.cast(tf.shape(image)[0], tf.float32)

    x_min = bbox[0] / width
    y_min = bbox[1] / height
    x_max = bbox[2] / width
    y_max = bbox[3] / height

    new_bbox = tf.stack([x_min, y_min, x_max, y_max])

    return (label[0], label[1], new_bbox)


def flip_image_label(image, label):
    flipped_image = tf.image.flip_left_right(image)
    flipped_bbox = flip_bbox_left_right(image, label)
    flipped_label = overwrite_label_bbox(image, label, flipped_bbox)
    return flipped_image, flipped_label

def rgb_to_grayscale(image, label):
    grayscaled = tf.image.rgb_to_grayscale(image)
    rgb_grayscale = tf.concat([grayscaled] * 3, axis=-1)
    return rgb_grayscale, label

def adjust_randomly_saturation(image, label):
    # Determine the range of the image
    min_val = tf.reduce_min(image)
    max_val = tf.reduce_max(image)

    # If the image is in the range [0,255], scale it to [0,1]
    if min_val >= 0 and max_val > 1:
        image = image / 255.0

    # If the image is in the range [-1,1], scale it to [0,1]
    elif min_val < 0 and max_val <= 1:
        image = (image + 1) / 2.0
    saturation_factor = tf.random.uniform([], 0.6, 3)

    saturated = tf.image.adjust_saturation(image, saturation_factor)
    # Scale the saturated image back to its original range
    if min_val >= 0 and max_val > 1:
        saturated = saturated * 255.0
    elif min_val < 0 and max_val <= 1:
        saturated = saturated * 2.0 - 1
    return saturated, label

def adjust_randomly_brightness(image, label):
    # Determine the range of the image
    min_val = tf.reduce_min(image)
    max_val = tf.reduce_max(image)

    # If the image is in the range [0,255], scale it to [0,1]
    if min_val >= 0 and max_val > 1:
        image = image / 255.0

    # If the image is in the range [-1,1], scale it to [0,1]
    elif min_val < 0 and max_val <= 1:
        image = (image + 1) / 2.0
    brightness_factor = tf.random.uniform([], -0.2, 0.2)
    brightened = tf.image.adjust_brightness(image, brightness_factor)
    # Scale the brightened image back to its original range
    if min_val >= 0 and max_val > 1:
        brightened = brightened * 255.0
    elif min_val < 0 and max_val <= 1:
        brightened = brightened * 2.0 - 1
    return brightened, label







