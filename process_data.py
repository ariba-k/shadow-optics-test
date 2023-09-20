import os.path
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import pandas
import random


# TODO: simplify extract_data_paths with regex
def extract_data_paths(input_path, num_runs, mask_type, led_type):
    """
    Extracts data paths from a folder of raw_data/run_num/mask_led.png into dataframe of image paths

    :param input_path: str
    :param num_runs: int
    :param mask_type: str ("fat", "skinny", "both")
    :param led_type: str ("clock", "shear", "both")
    :return: pd.DataFrame
    """

    base_path = Path(input_path)
    data_paths = defaultdict(lambda: defaultdict(lambda: [])) if mask_type == "both" else defaultdict(list)
    run_count = 0
    for run_path in base_path.iterdir():
        if run_path.is_dir() and run_count < num_runs:
            if mask_type == "both":
                for led_path in run_path.iterdir():
                    if led_type == "both":
                        if "png" in led_path.name:
                            if "F158" in led_path.name:
                                data_paths["skinny"][run_path.name].append(led_path)
                            else:
                                data_paths["fat"][run_path.name].append(led_path)
                    else:
                        led_initial = led_type[0]
                        if "png" in led_path.name and led_initial in led_path.name:
                            if "F158" in led_path.name:
                                data_paths["skinny"][run_path.name].append(led_path)
                            else:
                                data_paths["fat"][run_path.name].append(led_path)

            else:
                for led_path in run_path.iterdir():
                    if led_type == "both":
                        if "png" in led_path.name:
                            if mask_type == "skinny":
                                if "F158" in led_path.name:
                                    data_paths[run_path.name].append(led_path)  # type: ignore
                            else:
                                if "F184" in led_path.name:
                                    data_paths[run_path.name].append(led_path)  # type: ignore
                    else:
                        led_initial = led_type[0]
                        if "png" in led_path.name and led_initial in led_path.name:
                            if mask_type == "skinny":
                                if "F158" in led_path.name:
                                    data_paths[run_path.name].append(led_path)  # type: ignore
                            else:
                                if "F184" in led_path.name:
                                    data_paths[run_path.name].append(led_path)  # type: ignore

            run_count += 1

    return data_paths


# TODO: Apply pixel perturbations (1% of all pixels) - verify
def apply_perturbation(image):
    """

    Randomly "turns on" and "turns off" 1% of the pixels in an image

    :param image: np.ndarray
    :return: PIL.Image
    """
    image_arr = np.array(image)
    pixel_values = [0, 255]
    height, width = image_arr.shape[:2]
    num_pixels = height * width
    num_perts = int(num_pixels * 0.01)
    for _ in range(num_perts):
        i, j = random.randrange(0, height - 1), random.randrange(0, width - 1)
        image_arr[i][j] = random.choice(pixel_values)

    image = Image.fromarray(image_arr)

    return image


def process_images(data_paths, output_path, resize_dim, rgb, adversarial_data):
    """
    Processes image paths stored in dataframe by tiling all images in a run.
    Saves processed images in folder "processed_data" and outputs a dataframe of new image paths.

    :param data_paths: DataFrame
    :param output_path: str
    :param resize_dim: tuple (height, width)
    :param rgb: bool
    :param adversarial_data: bool
    :return: DataFrame
    """
    image_df = pd.DataFrame(columns=['image'])
    for i, run_num in enumerate(data_paths):
        stacked_images = []
        for image_path in data_paths[run_num]:
            image = Image.open(image_path)
            if rgb:
                image = image.convert("RGB")
            if resize_dim:
                image = image.resize(resize_dim)
            if adversarial_data:
                image = apply_perturbation(image)
            stacked_images.append(image)

        num_images = len(stacked_images)
        v_stack_num = int(np.sqrt(num_images))
        h_stack_num = num_images // v_stack_num
        h_stacked_images = [np.hstack(stacked_images[i::h_stack_num]) for i in range(h_stack_num)]
        v_stacked_image = np.vstack(h_stacked_images).astype(np.uint8)

        if not rgb:
            v_stacked_image = v_stacked_image * 255

        output_image = Image.fromarray(v_stacked_image)

        image_output_path = os.path.join(output_path, run_num + ".png")

        output_image.save(image_output_path)

        image_df.loc[i] = image_output_path

    return image_df


def process_labels(input_path, num_runs):
    """
    Processes labels in a dataframe with same length as above image_df

    :param input_path: str
    :param num_runs: int
    :return: pd.DataFrame
    """
    variables_df = pandas.read_csv(os.path.join(input_path, "variables.csv"))
    label_df = variables_df[['MASK_dx', 'MASK_dy', 'MASK_dclock']].head(num_runs)

    return label_df


def run(input_path, output_path,
        num_runs=1500, mask_type="both", led_type="both",
        resize_dim=None, rgb=False,
        adversarial_data=False):

    """
    Runs process_data.py and saves data.csv to output_path

    :param input_path: str
    :param output_path: str
    :param num_runs: int
    :param mask_type: str ("fat", "skinny", "both")
    :param led_type: str ("clock", "shear", "both")
    :param resize_dim: tuple (height, width)
    :param rgb: bool
    :param adversarial_data: bool
    :return: None
    """

    data_paths = extract_data_paths(input_path, num_runs, mask_type, led_type)

    output_data_path = os.path.join(output_path, "processed_data")
    os.makedirs(output_data_path, exist_ok=True)

    label = process_labels(input_path, num_runs)

    if mask_type == "both":
        mask_data = []
        for mask in data_paths:
            mask_path = os.path.join(output_data_path, mask)
            os.makedirs(mask_path, exist_ok=True)
            image = process_images(data_paths[mask], mask_path, resize_dim, rgb, adversarial_data)
            mask_data.append(pd.concat([image, label], axis=1))
        data = pd.concat(mask_data, axis=0, ignore_index=True)
    else:
        image = process_images(data_paths, output_data_path, resize_dim, rgb, adversarial_data)
        data = pd.concat([image, label], axis=1)

    data.to_csv(os.path.join(output_path, "data.csv"), index=False)


run("raw_data", "trial_results",
    mask_type="fat", led_type="clock", resize_dim=(84, 84), rgb=True)
