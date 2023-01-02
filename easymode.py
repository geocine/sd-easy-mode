import json
import os
import subprocess
import time
from ipywidgets import IntProgress, HTML, HBox
import numpy as np

class ProgressBar:
    def __init__(self, num_items, label_text='Progress'):
        self.num_items = num_items
        self.start_time = time.perf_counter()
        self.count = 0
        self.label_text = label_text

        # Create a progress bar and HTML widgets to display the labels and progress bar
        self.f = IntProgress(min=0, max=num_items)
        self.label1 = HTML(value=f'{label_text}: 0%')
        self.label2 = HTML(value='', layout=dict(margin='2px 0 0 10px'))

        # Group the widgets horizontally using the HBox layout
        display(HBox([self.label1, self.f, self.label2]))

    def update(self, label=''):
        value = 1
        self.count += value
        self.f.value += value
        percentage = f'{self.f.value / self.num_items * 100:.0f}'
        self.label1.value = f'{self.label_text}: {percentage}%'
        self.label2.value = label
        # change bar color to green if done
        if percentage == "100":
            self.f.bar_style = 'success'

    def error(self, label=''):
        self.label2.value = 'Stopped due to error'
        self.f.bar_style = 'danger'


def install_package(package, diffusers_url, xformers_url, branch="main", force_reinstall=False):
    # Check if the package is already installed using pip freeze
    installed_packages = subprocess.run(
        ["pip", "freeze"], capture_output=True).stdout.decode().split("\n")
    if not force_reinstall and any(package in s for s in installed_packages):
        return f'{package} is already installed'

    if package == 'diffusers':
        # Install the package using the URL
        result = subprocess.run(
            ["pip", "-qq", "install", diffusers_url], capture_output=True, text=True, check=True)
    elif package == 'xformers':
        # Install the package using the URL
        result = subprocess.run(
            ["pip", "install", xformers_url], capture_output=True, text=True, check=True)
    else:
        # Install the package using pip
        result = subprocess.run(
            ["pip", "install", package], capture_output=True, text=True, check=True)

    # Print the output of the command
    # print(result.stdout)
    return f'{package} is installed'

def download_regularization(sdd_class, rev="main"):
    unzip_directory = f"/content/data/{sdd_class}"

    # Check if the unzip directory exists
    try:
        # Get a list of the files in the unzip directory
        files = os.listdir(unzip_directory)
    except FileNotFoundError:
        # Create the unzip directory
        os.makedirs(unzip_directory)
        # Set the files list to an empty list
        files = []

    # Downloading the regularization images
    zip_file = f"{sdd_class}.zip"
    if not os.path.exists(zip_file):
        try:
            reg_url = f"https://huggingface.co/datasets/geocine/sd-v1-5-regularization-images/resolve/{rev}/{zip_file}"
            subprocess.run(["wget", "-q", reg_url], check=True)
        except Exception as e:
            # Print an error message and set the zip_file variable to None if the download fails or the user doesn't have access
            print(f"An error occurred while downloading the dataset: {e}")
            zip_file = None

    # Check if the unzip directory has files
    if len(files) > 0:
        # Do not run the unzip command
        print("Unzip directory has files. Skipping unzip.")
    elif zip_file is None:
        # Do not run the unzip command
        print("Skipping unzip because the zip file was not downloaded")
    else:
        command = f"unzip -l {zip_file} | wc -l"
        result = subprocess.run(command, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        file_count = int(result.stdout.decode('utf-8').strip())
        # Run the unzip command
        pb = ProgressBar(file_count, "Extracting")
        process = subprocess.Popen(
            ["unzip", "-j", zip_file, "-d", unzip_directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        while process.poll() is None:
            out = process.stdout.readline()
            if out != '':  # and (b"extracting" in out):
                current_file = out.decode("utf-8").replace("extracting: ", "")
                current_file = current_file.replace("inflating: ", "")
                pb.update(current_file)
        print("\033[92mExtracting regularization images completed\033[0m")


import json

def replace_tokens(json_file, sdd_token=None, sdd_class=None):
    # Open the JSON file and read the contents
    with open(json_file, "r") as f:
        json_data = json.load(f)

    # Check if the file still has {SDD_TOKEN} or {SDD_CLASS} to be replaced
    has_token = False
    for item in json_data:
        for key, value in item.items():
            if "{SDD_TOKEN}" in value or "{SDD_CLASS}" in value:
                has_token = True
                break
        if has_token:
            break

    # If the file has {SDD_TOKEN} or {SDD_CLASS} to be replaced, run the replacer
    if has_token:
        # Iterate over the object and replace the placeholders with the values
        for item in json_data:
            for key, value in item.items():
                if sdd_token is not None:
                    item[key] = value.replace("{SDD_TOKEN}", sdd_token)
                if sdd_class is not None:
                    item[key] = value.replace("{SDD_CLASS}", sdd_class)

        # Open the JSON file and write the updated contents
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)

def print_message(message_type, message):
    if message_type == "error":
        color_code = "\033[91m"
    elif message_type == "warning":
        color_code = "\033[93m"
    else:
        color_code = "\033[0m"
    print(f"{color_code}{message}\033[0m")
    raise SystemExit

def create_interpolation_function(points):
    def interpolate(x):
        # Extract the x and y values from the points list
        x_values, y_values = zip(*points)

        # Use the numpy polyfit function to fit a polynomial function to the data
        coefficients = np.polyfit(x_values, y_values, len(points) - 1)

        # Use the numpy polyval function to evaluate the polynomial function at x
        y = np.polyval(coefficients, x)

        # Round the result to the nearest integer
        y = int(round(y))
        return y
    return interpolate