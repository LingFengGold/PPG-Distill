#!/usr/bin/env python

import os
import sys
import time
import numpy as np
from datetime import datetime
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
import socket
from contextlib import closing
from warnings import warn
from scipy.signal import resample_poly
from fractions import Fraction
import re

import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def repeat_and_save_npy(file_path):
    # Load array and verify dimensions
    data = np.load(file_path)
    N, L = data.shape
    
    if L >= 1200:
        raise ValueError("Input array must have L < 1200")
    
    # Calculate repetition parameters
    full_repeats = 1200 // L
    remainder = 1200 % L
    
    # Create extended array
    extended = np.tile(data, (1, full_repeats))
    if remainder:
        extended = np.concatenate((extended, data[:, :remainder]), axis=1)
    
    # Generate new filename
    dir_path, filename = os.path.split(file_path)
    base_name, ext = os.path.splitext(filename)
    new_path = os.path.join(dir_path, f"{base_name}_30s{ext}")
    
    # Save and return path
    np.save(new_path, extended)
    return new_path


def is_simple_string(s: str) -> bool:
    return bool(re.fullmatch(r'^[A-Za-z0-9_.]+$', s))


def assign_key_to_dict_from_dict_safely(dict_target, key, dict_source, default_value, logger_, override=True):
    if not override:
        assert key not in dict_target, f'{key=} already exists in {dict_target=}'
    try:
        dict_target[key] = dict_source[key]
    except Exception as e:
        logger_(e)
        dict_target[key] = default_value
        logger_(f'{default_value=} assigned to dict_target')


def convert_to_path_list(path, validate=False):
    """
    Converts a path string with curly-brace expansion into a list of paths.
    
    For example, the input:
        "/path/to/{file1.npy,file2.npy}"
    becomes:
        ["/path/to/file1.npy", "/path/to/file2.npy"]
    
    If validate is True, the function checks that each generated file path exists
    and that its size is non zero. It raises a FileNotFoundError if any file does
    not exist or is empty.
    
    Parameters:
        path (str): The input path containing curly braces for expansion.
        validate (bool): If True, check each generated path for existence and non-zero size.
    
    Returns:
        list of str: A list of expanded file paths.
    """
    # Regular expression to match a pattern like {file1,file2,...}
    pattern = re.compile(r'\{([^}]+)\}')
    match = pattern.search(path)
    
    if match:
        options_str = match.group(1)
        options = options_str.split(',')
        # Replace the whole "{...}" with each option to generate the paths.
        paths = [path.replace(match.group(0), option) for option in options]
    else:
        paths = [path]
    
    if validate:
        valid_paths = []
        for p in paths:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                valid_paths.append(p)
            else:
                raise FileNotFoundError(f"File {p} does not exist or is empty.")
        return valid_paths
    
    return paths


def truncate_and_resample_npy(input_path, old_fs, new_fs):
    """
    Truncate a signal matrix in a .npy file to a whole number of seconds based on old_fs,
    resample it to new_fs, and save the result as <original_name>_resampled.npy.
    
    Parameters:
    -----------
    input_path : str
        Path to the input .npy file containing the signal matrix (shape: N x L).
    old_fs : int or float
        Original sampling frequency in Hz.
    new_fs : int or float
        Desired new sampling frequency in Hz.
    
    Returns:
    --------
    output_path : str
        Path to the saved resampled .npy file.
    
    Raises:
    -------
    FileNotFoundError:
        If the input file does not exist.
    ValueError:
        If the signal matrix is not 2D or if sampling frequencies are non-positive.
    """
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The file '{input_path}' does not exist.")
    
    # Load the signal matrix
    signal_matrix = np.load(input_path)
    
    # Validate the signal matrix
    if signal_matrix.ndim != 2:
        raise ValueError("The signal matrix must be a 2D NumPy array with shape (N, L).")
    
    if old_fs <= 0 or new_fs <= 0:
        raise ValueError("Sampling frequencies must be positive numbers.")
    
    N, L = signal_matrix.shape
    
    # Calculate whole number of seconds
    whole_seconds = L // old_fs  # Integer division to get whole seconds
    truncated_L = int(whole_seconds * old_fs)
    
    if truncated_L == 0:
        raise ValueError("The truncated length is zero. Check the input signal length and old_fs.")
    
    # Truncate the signal matrix
    truncated_matrix = signal_matrix[:, :truncated_L]
    
    # Calculate resampling factors
    ratio = Fraction(new_fs, old_fs).limit_denominator()
    up, down = ratio.numerator, ratio.denominator
    
    # Resample using resample_poly
    resampled_matrix = resample_poly(truncated_matrix, up, down, axis=1)
    
    # Generate output file path
    dir_name, base_name = os.path.split(input_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_resampled.npy"
    output_path = os.path.join(dir_name, output_filename)
    
    # Save the resampled matrix
    np.save(output_path, resampled_matrix)
    
    return output_path


def get_stats(array_or_path):
    # Check if the input is a numpy array
    if isinstance(array_or_path, np.ndarray):
        data = array_or_path
    # Check if the input is a file path (string)
    elif isinstance(array_or_path, str):
        try:
            # Load .npy files
            if array_or_path.endswith('.npy'):
                data = np.load(array_or_path)
            # Load .csv files
            elif array_or_path.endswith('.csv'):
                data = np.loadtxt(array_or_path, delimiter=',')
            else:
                raise ValueError("Unsupported file format. Please provide a .npy or .csv file.")
        except Exception as e:
            print(f"Error loading array from path: {e}")
            return
    else:
        print("Input must be a numpy array or a path to a .npy or .csv file.")
        return

    # Flatten the array to 1D
    flat_data = data.flatten()

    # Compute basic statistics
    mean = np.mean(flat_data)
    std_dev = np.std(flat_data)
    min_val = np.min(flat_data)
    max_val = np.max(flat_data)
    median = np.median(flat_data)

    # Print the statistics
    print("Basic Statistics:")
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Minimum Value: {min_val}")
    print(f"Maximum Value: {max_val}")
    print(f"Median: {median}")


def get_memory_usage():
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / (1024 ** 2)  # Convert bytes to MB
    return rss_mb

def convert_npy_to_h5_chunked(npy_file_path, h5_file_path, dataset_name='dataset', chunk_size=(100, 1200), compression='gzip'):
    # Open the .npy file in read-only mode with memory mapping
    data = np.load(npy_file_path, mmap_mode='r')
    num_instances, num_features = data.shape

    # Create an HDF5 file and dataset with chunking and compression
    with h5py.File(h5_file_path, 'w') as h5_file:
        h5_dataset = h5_file.create_dataset(dataset_name, shape=(num_instances, num_features), dtype=data.dtype,
                                            chunks=chunk_size, compression=compression)

        # Write the data in chunks
        chunk_size_instances = chunk_size[0]
        for i in range(0, num_instances, chunk_size_instances):
            end = min(i + chunk_size_instances, num_instances)
            h5_dataset[i:end] = data[i:end]
            print(f'Processed rows {i} to {end}')


class Dummy:
    pass


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def init_dummy_ddp():
    if not dist.is_initialized():
        hostname = socket.gethostname()
        os.environ['MASTER_ADDR']=hostname
        free_port = find_free_port()
        os.environ['MASTER_PORT'] = str(free_port)
        dist.init_process_group('gloo', rank=0, world_size=1)
        torch.nn.parallel.DistributedDataParallel.mixed_precision = Dummy
    else:
        warn('SKIPPING init_dummy_ddp() as dist is already initialized')


def set_dropout_rate(model, dropout_rate):
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_rate
            print(f"Set dropout rate of {name} to {dropout_rate}")


def make_weights_for_balanced_classes(labels, nclasses=2):
    n_images = labels.shape[0]
    count_per_class = [0] * nclasses
    for image_class in labels:
#        image_class = int(image_class)
        count_per_class[image_class] += 1
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = float(n_images) / float(count_per_class[i])
    weights = [0] * n_images
    for idx, image_class in enumerate(labels):
#        image_class = int(image_class)
        weights[idx] = weight_per_class[image_class]
    return weights


def find_latest_pt_file(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Filter files with the .pt extension
    pt_files = [file for file in files if file.endswith('.pth')]
    
    # Get the full path and modification time for each .pt file
    full_paths = [os.path.join(directory, file) for file in pt_files]
    
    # Find the most recently modified .pt file
    if not full_paths:
        return None  # Return None if no .pt files are found
    latest_file = max(full_paths, key=os.path.getmtime)
    
    return latest_file

def freeze_nn(model):
    print(f'EXECUTING freeze_nn()')
    for param in model.parameters():
        param.requires_grad = False     # in place

def unfreeze_nn(model):
    print(f'EXECUTING unfreeze_nn()')
    for param in model.parameters():
        param.requires_grad = True     # in place

def count_parameters(model):
    N = sum(p.numel() for p in model.parameters())
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
#    return N, n
    return f'Total:{N:,} Trainable:{n:,}'


def get_numpy_array_metadata(file_path, return_attrs=False):
    with open(file_path, 'rb') as f:
        # Determine the version of the NumPy file
        version = read_magic(f)
        
        # Read the header to get metadata based on the file version
        if version == (1, 0):
            shape, fortran_order, dtype = read_array_header_1_0(f)
        elif version >= (2, 0):
            shape, fortran_order, dtype = read_array_header_2_0(f)
        else:
            raise ValueError("Unsupported file version")

    # The metadata such as shape, data type, and array order is now available
    print("Shape:", shape)
    print("Data type:", dtype)
    print("Fortran order:", fortran_order)
    if return_attrs:
        return (shape, dtype)
