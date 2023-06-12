import cv2 
import numpy as np
import tacs_tools
import image_utils
import tensorflow as tf



def get_label_name(label, classes):
    if tacs_tools.is_one_hot(label[0]):
        index = np.argmax(label[0])
    else:
        index = label[0]
    name = str(index) if classes is None else classes[index]
    name = name[0].upper() + name[1:]
    return name

def get_bbox_np(image, label):
    box = label[2]

    x_min, y_min, x_max, y_max = box[0] * image.shape[1], box[1] * image.shape[0], box[2] * image.shape[1], box[3] * image.shape[0]

    return np.array([x_min, y_min, x_max, y_max])

@tf.function
def get_bbox_tf(image, label):
    box = label[2]
    x_min = box[0] * tf.cast(tf.shape(image)[1], tf.float32)
    y_min = box[1] * tf.cast(tf.shape(image)[0], tf.float32)
    x_max = box[2] * tf.cast(tf.shape(image)[1], tf.float32)
    y_max = box[3] * tf.cast(tf.shape(image)[0], tf.float32)

    return tf.stack([x_min, y_min, x_max, y_max])

def get_corners_np(image, label):
    
    x_min,y_min, x_max, y_max = get_bbox_np(image, label)
        
    x1 = x_min
    y1 = y_max

    x2 = x_max
    y2 = y_max
    
    x3 = x_max
    y3 = y_min
    
    x4 = x_min
    y4 = y_min
    
    corners = np.array([x1, y1, x2, y2, x3, y3, x4, y4])
    
    return corners

@tf.function
def get_corners_tf(image, label):
    bbox = get_bbox_tf(image, label)
    corners = tf.stack([
        bbox[0], bbox[1],   # Top-left
        bbox[2], bbox[1],   # Top-right
        bbox[2], bbox[3],   # Bottom-right
        bbox[0], bbox[3]    # Bottom-left
    ])
    return corners

def rotate_box_np(corners,angle,  cx, cy , h, w):
    """Rotate the bounding box.
    
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    
    angle : float
        angle by which the image is to be rotated
        
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
        
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
        
    h : int 
        height of the image
        
    w : int 
        width of the image
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated[0]

'''def rotate_box_tf(corners, angle, cx, cy, h, w):
    print(f"cx: {cx} cy: {cy} h: {h} w: {w}")
    corners = tf.reshape(corners, (-1, 2))
    ones = tf.ones((tf.shape(corners)[0], 1), dtype=corners.dtype)
    corners = tf.concat([corners, ones], axis=1)
    print(f"corners: {corners}")
    cos_val = tf.cos(-angle)
    sin_val = tf.sin(-angle)
    print(f"angel {angle}")
    h = tf.cast(h, dtype=tf.float32) 
    w = tf.cast(w, dtype=tf.float32)
    cx = tf.cast(cx, dtype=tf.float32) 
    cy = tf.cast(cy, dtype=tf.float32)

    nW = -cx * cos_val + cy * sin_val + cx
    nH = -cx* sin_val - cy * cos_val +cy
    
    M = tf.concat([cos_val, -sin_val, nW,
                   sin_val, cos_val, nH],
                  axis=0)
    M = tf.reshape(M, [2, 3])
    
    calculated = tf.linalg.matmul(M, corners, transpose_b=True)
    calculated = tf.transpose(calculated)
    
    calculated = tf.reshape(calculated, [-1, 8])
    print(f"calculated {calculated}")    
    return calculated[0]'''

def rotate_box_tf(corners, image, angle, cx, cy, h, w):
    corners = tf.reshape(corners, (-1, 2))
    ones = tf.ones((tf.shape(corners)[0], 1), dtype=corners.dtype)
    corners = tf.concat([corners, ones], axis=1)
    
    # Calculate the size of the rotated image
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    diagonal = tf.sqrt(tf.square(tf.cast(height, tf.float32)) + tf.square(tf.cast(width, tf.float32)))
    new_height = tf.cast(diagonal, tf.int32)
    new_width = tf.cast(diagonal, tf.int32)
    

    # Calculate the amount of padding required
    pad_height = (new_height - height) // 2
    pad_width = (new_width - width) // 2
    pad_height = tf.cast(pad_height, dtype=tf.float32)
    pad_width = tf.cast(pad_width, dtype=tf.float32)

    cos_val = tf.cos(-angle)
    sin_val = tf.sin(-angle)

    h = tf.cast(h, dtype=tf.float32) 
    w = tf.cast(w, dtype=tf.float32)
    cx = tf.cast(cx, dtype=tf.float32) 
    cy = tf.cast(cy, dtype=tf.float32)
    
    nW = -cx * cos_val + cy * sin_val + cx + pad_width
    nH = -cx * sin_val - cy * cos_val + cy + pad_height
    
    M = tf.stack([cos_val, -sin_val, nW,
                   sin_val, cos_val, nH],
                  axis=0)
    M = tf.reshape(M, [2, 3])
    
    calculated = tf.linalg.matmul(M, corners, transpose_b=True)
    calculated = tf.transpose(calculated)
    
    calculated = tf.reshape(calculated, [-1, 8])

    return calculated[0]



def draw_rect(img, cords, label = None, classes=None, color = None):
    """Draw the rectangle on the image
    
    Parameters
    ----------
    
    im : numpy.ndarray
        numpy image 
    
    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    Returns
    -------
    
    numpy.ndarray
        numpy image with bounding boxes drawn on it
        
    """
    
    img = img.copy()

    if not color:
        color = [255,0,0]
    pt1, pt2 = (cords[0], cords[1]) , (cords[2], cords[3])
            
    pt1 = int(pt1[0]), int(pt1[1])
    pt2 = int(pt2[0]), int(pt2[1])

    

    img = cv2.rectangle(img.copy(), pt1, pt2, color, int(max(img.shape[:2])/200))
    if label:
        if tacs_tools.is_one_hot(label[0]):
            class_index = np.argmax(label[0])
        else:
            class_index = label[0]
        thickness = 1
        fontScale = 0.6
        font = cv2.FONT_HERSHEY_TRIPLEX
        if classes:
            class_name = str(get_label_name(label, classes))
        else:
            class_name = str(class_index)
        anzahl = str(label[1])
        text_message = anzahl + " x " + class_name
        text_left_buttom = (pt1[0] - 3, pt1[1] - 3)
        img = cv2.putText(img, text_message , text_left_buttom, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    return img

def get_enclosing_box_np(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[[0,2,4,6]]
    y_ = corners[[1,3,5,7]]
    
    xmin = np.min(x_)
    ymin = np.min(y_)
    xmax = np.max(x_)
    ymax = np.max(y_)
    
    final = np.array([xmin, ymin, xmax, ymax])
    
    return final

def get_enclosing_box_tf(corners):
    """Get an enclosing box for rotated corners of a bounding box

    Parameters
    ----------
    
    corners : tf.Tensor
        Tensor of shape `8` containing a bounding box described by their 
        corner coordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    tf.Tensor
        Tensor containing enclosing bounding boxes of shape `4`, 
        represented in the format `x1 y1 x2 y2`
        
    """
    x_ = tf.gather(corners, [0,2,4,6])
    y_ = tf.gather(corners, [1,3,5,7])
    
    xmin = tf.reduce_min(x_)
    ymin = tf.reduce_min(y_)
    xmax = tf.reduce_max(x_)
    ymax = tf.reduce_max(y_)
    
    final = tf.stack([xmin, ymin, xmax, ymax])
    
    return final


def rotate_img_np(image, angle):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray
        Rotated Image
    
    """
    # grab the dimensions of the image and then determine the
    # centre
    image = image_utils.convert_to_numpy_array(image)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def get_label_ranges(dataset):
    min_value_class = np.iinfo(np.uint8).max
    max_value_class = np.iinfo(np.uint8).min
    min_value_count = np.iinfo(np.uint8).max
    max_value_count = np.iinfo(np.uint8).min
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')

    lst_dataset = tacs_tools.tensorslicedataset_2_list(dataset)

    for data in lst_dataset:
        # Update the overall minimum and maximum pixel values
        min_value_class = tf.minimum(min_value_class, data[1][0])
        max_value_class = tf.maximum(max_value_class, data[1][0])
        min_value_count = tf.minimum(min_value_count, data[1][1])
        max_value_count = tf.maximum(max_value_count, data[1][1])
        min_x = tf.minimum(min_x, data[1][2][0])
        min_y = tf.minimum(min_y, data[1][2][1])
        max_x = tf.maximum(max_x, data[1][2][2])
        max_y = tf.maximum(max_y, data[1][2][3])

    return (min_value_class, max_value_class, min_value_count, max_value_count, min_x, min_y, max_x, max_y)