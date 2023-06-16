from typing import Union, Any, Tuple, List, Dict
import os
import time
import io
import pickle
import json
import urllib.request

import numpy as np
import PIL.Image
import tensorflow as tf
import matplotlib.figure
import matplotlib.pyplot as plt
from IPython.display import display
import pandas as pd 
import cv2


def _load_dataset(data_type: str) -> List[Any]:

    def print_progress(index: int,
                       block: int,
                       length: int) -> None:

        progress = index * block / length

        print(f"Downloading {data_type} data: {progress:.1%}",
              end="\r")

    if data_type not in ["train", "val"]:
        raise Exception(
            f"Possible data types are ['train', 'val']. Received: '{data_type}'")

    filename = f"{data_type}.bin"

    path = os.path.abspath(f"data/{filename}")

    if os.path.exists(path):
        with open(path, "rb") as file:
            return pickle.load(file)

    start_time = time.time()

    path_temp = urllib.request.urlretrieve(f"http://www.skywhite.de/data_{data_type}.bin",
                                           reporthook=print_progress)[0]

    print()
    print(f"Finished in {time.time() - start_time:.1f} s")

    with open(path_temp, "rb") as file:
        data = file.read()

    # create directory structure otherwise no file can be opened in the next step
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    with open(path, "wb") as file:
        file.write(data)

    return pickle.load(io.BytesIO(data))


def _prepare_image(image_data: bytes,
                   image_size: Tuple[int, int] = None) -> np.ndarray:

    image = PIL.Image.open(io.BytesIO(image_data))

    if image_size is not None and image.size != image_size[:2]:
        image = image.resize(image_size[:2])

    image_data = np.array(image,
                          np.float32)

    return image_data


def load_datasets(image_size: Tuple[int, int] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:

    data_train = _load_dataset("train")
    data_val = _load_dataset("val")

    x_train = np.array([_prepare_image(entry[0], image_size)
                        for entry in data_train], np.float32)
    y_train_cls = np.array([entry[1] for entry in data_train], np.uint8)
    y_train_cnt = np.array([entry[2] for entry in data_train], np.uint8)
    y_train_loc = np.array([entry[3] for entry in data_train], np.float32)

    dataset_train = tf.data.Dataset.from_tensor_slices(
        (x_train, (y_train_cls, y_train_cnt, y_train_loc)))

    x_val = np.array([_prepare_image(entry[0], image_size)
                      for entry in data_val], np.float32)
    y_val_cls = np.array([entry[1] for entry in data_val], np.uint8)
    y_val_cnt = np.array([entry[2] for entry in data_val], np.uint8)
    y_val_loc = np.array([entry[3] for entry in data_val], np.float32)

    dataset_val = tf.data.Dataset.from_tensor_slices(
        (x_val, (y_val_cls, y_val_cnt, y_val_loc)))

    return (dataset_train, dataset_val)


def show_labeled_image(image_or_entry: Union[List[Any], bytes, np.ndarray],
                       label_or_predictions: List[Any] = None,
                       classes: List[str] = None,
                       image_size: Tuple[int, int] = None,
                       box_color: Tuple[int, int, int] = (255, 0, 0),
                       box_line: int = 2,
                       bool_display: bool = True,
                       bool_print: bool = False) -> None:

    count = None
    box = None

    if label_or_predictions is None:
        image = image_or_entry[0]

        if isinstance(image_or_entry[1], (tuple, list)):
            index = image_or_entry[1][0]
            count = image_or_entry[1][1]
            box = image_or_entry[1][2]

        else:
            index = image_or_entry[1]

            if len(image_or_entry) > 2:
                count = image_or_entry[2]
                box = image_or_entry[3]

    else:
        image = image_or_entry

        if isinstance(label_or_predictions, (tuple, list)):
            if is_one_hot(label_or_predictions[0]):
                index = np.argmax(label_or_predictions[0])
            else:
                index = label_or_predictions[0]

            if len(label_or_predictions) > 1:
                count = label_or_predictions[1]
                box = label_or_predictions[2]

        else:
            index = label_or_predictions

    if isinstance(image, bytes):
        image_data = _prepare_image(image,
                                    image_size)
    elif isinstance(image, tf.Tensor):
        image_data = image.numpy().copy()

    elif isinstance(image, np.ndarray):
        image_data = image.copy()

    else:
        print(f"Image type not supported: {type(image)}")

    index_score = None

    if isinstance(index, np.ndarray) and index.shape[0] > 1:
        index_score = np.max(index)
        index = int(np.argmax(index))

    count_score = None

    if isinstance(count, np.ndarray) and count.shape[0] > 1:
        count_score = np.max(count)
        count = int(np.argmax(count))

    name = str(index) if classes is None else classes[index]
    name = name[0].upper() + name[1:]

    if box is not None:
        x_min = int(box[0] * image_data.shape[1])
        y_min = int(box[1] * image_data.shape[0])
        x_max = int(box[2] * image_data.shape[1])
        y_max = int(box[3] * image_data.shape[0])

        box_line = max(1, box_line)

        if image_data.max() <= 3.0:
            if image_data.min() < 0.0:
                box_color = (box_color[0] / 127.5 - 1,
                             box_color[1] / 127.5 - 1,
                             box_color[2] / 127.5 - 1)

            else:
                box_color = (box_color[0] / 255,
                             box_color[1] / 255,
                             box_color[2] / 255)
        if len(image_data.shape) == 2 or (len(image_data.shape) == 3 and image_data.shape[2] == 1):
            box_color = 255
            if image_data.max() <= 3.0:
                if image_data.min() < 0.0:
                    box_color = box_color / 127.5 - 1,

                else:
                    box_color = box_color / 255


        image_data[y_min:y_min + box_line, x_min:x_max] = box_color
        image_data[y_max:y_max + box_line, x_min:x_max] = box_color
        image_data[y_min:y_max + box_line, x_min:x_min + box_line] = box_color
        image_data[y_min:y_max + box_line, x_max:x_max + box_line] = box_color

    index_str = f"{index}" if index_score is None else f"{index} ({index_score:.1%})"

    count_str = ""

    if count is not None:
        count_str = (f"{count} x " if count_score is None else
                     f"{count} ({count_score:.1%}) x ")
    if bool_print:
        print(f"{count_str}{index_str} [{name}]")

    if box is not None:
        thickness = 1
        fontScale = 0.6
        font = cv2.FONT_HERSHEY_TRIPLEX
        text_message = count_str + name
        text_left_buttom = (x_min - 3, y_min - 3)
        image_data = cv2.putText(image_data, text_message , text_left_buttom, font, fontScale, box_color, thickness, cv2.LINE_AA)

    image = tf.keras.utils.array_to_img(image_data)

    if bool_display:
        display(image)
    return image

def is_one_hot(arr):
    # check if numpy array
    if not isinstance(arr, np.ndarray):
        return False

    # check if binary
    if not np.array_equal(arr, arr.astype(bool)):
        return False

    # check if sums to 1 (use axis=None for 1-D array)
    # muss bei predictions nicht der Fall sein
    #if np.sum(arr, axis=None) != 1:
    #    return False

    return True

def plot_history(history: Dict[str, Any],
                 limit_skip_first: int = 0) -> matplotlib.figure.Figure:

    includes_mae = False

    for key in history:
        if "mae" in key:
            includes_mae = True
            break

    fig = plt.figure()
    gs = fig.add_gridspec(3 if includes_mae else 2,
                          hspace=0)

    axes: List[plt.Axes] = gs.subplots(sharex=True)
    axes[0].set(xlabel="Epoch", ylabel="Accuracy")
    axes[1].set(xlabel="Epoch", ylabel="MAE")
    axes[-1].set(xlabel="Epoch", ylabel="Loss")

    fig.suptitle("Training")
    fig.set_figwidth(12)
    fig.set_figheight(9)

    ylim = {}

    for key, entry in history.items():
        if key == "meta":
            continue

        if "accuracy" in key:
            index = 0
            key = key.replace("accuracy", "")

        if "mae" in key:
            index = 1
            key = key.replace("mae", "")

        if "loss" in key:
            index = -1
            key = key.replace("loss", "")

        label = "Validation" if "val" in key else "Train"
        key = key.replace("val", "").strip("_")

        if index not in ylim:
            ylim[index] = (np.min(entry[limit_skip_first:]),
                           np.max(entry[limit_skip_first:]))

        else:
            ylim[index] = (min(ylim[index][0], np.min(entry[limit_skip_first:])),
                           max(ylim[index][1], np.max(entry[limit_skip_first:])))

        label = f"{label} {key.upper()}"

        axes[index].plot(range(1, len(entry) + 1),
                         entry,
                         label=label)

    for index in ylim:
        axes[index].legend()
        axes[index].grid(True)

        if index != 0:
            axes[index].set_yscale("log")

        ylim_delta = 0.05 * (ylim[index][1] - ylim[index][0])

        axes[index].set_ylim(ylim[index][0] - ylim_delta,
                             ylim[index][1] + ylim_delta)

    return fig


def save_model_and_history(model: tf.keras.Model,
                           history: Dict[str, Any],
                           directory: str = None,
                           limit_skip_first: int = 0,
                           show_figure: bool = False,
                           name = False) -> None:

    if directory is None:
        directory = os.path.abspath(
            f'results/{int(time.time())}_e-{history["meta"]["epochs"]}_{name}')

    os.makedirs(directory,
                exist_ok=True)

    with open(os.path.join(directory, "history.json"), "w") as file:
        json.dump(history,
                  file)

    fig = plot_history(history,
                       limit_skip_first)

    fig.savefig(os.path.join(directory, "history.png"))

    if not show_figure:
        plt.close()

    else:
        plt.show()

    model.save(directory)

    print(f"Saved model and history to {directory}")


def tensorslicedataset_2_list(dataset):
    lst_dataset = []
    for image, label in dataset.as_numpy_iterator():
        lst_dataset.append([image, label])
    return lst_dataset

def tensorslicedataset_2_array(dataset):
    array_dataset = []
    for image, label in dataset.as_numpy_iterator():
        array_dataset.append([image, label])
    return np.array(array_dataset)


def create_class_balance_table(dataset, max_objects, classes):
    
    count_list = list(range(1, max_objects + 1, 1))
    counted_objects_list = [str(item) +' objects'   for item in count_list]

    def get_df_of_counted_dataset(dataset):
        dictionary = dict.fromkeys(classes)

        for k, klasse in enumerate(classes):
            klasse_data =[]
            for objects in range(1,max_objects+1): 
                counter=0               
                for data in dataset:
                    if is_one_hot(data[1][0]):
                        index = np.argmax(data[1][0])
                    else:
                        index = data[1][0]
                    if index==k and data[1][1]==objects:
                        counter += 1
                klasse_data.append(counter)
            dictionary[klasse]=klasse_data
        return pd.DataFrame(dictionary, index = counted_objects_list) 

    df = get_df_of_counted_dataset(dataset)

    # Add a row that sums up all rows
    def sum_up_rows(df):
        sum_row = df.sum().to_frame().T
        sum_row.index = ['Total']
        #return df.concat(sum_row, ignore_index=False)
        return pd.concat([df, sum_row])

    df = sum_up_rows(df)
    
    print(df)


