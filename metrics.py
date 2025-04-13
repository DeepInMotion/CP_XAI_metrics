import os
import json
import time
import pickle
import glob
import pandas as pd
import numpy as np
import math
import torch
from scipy.stats import truncnorm
from tqdm import tqdm

def generate_angles(n):
    """
    Generate random angles from a truncated normal distribution.
    
    Args:
        n: Number of angles to generate
        
    Returns:
        List of angles
    """
    np.random.seed(42)
    mu = np.pi  # center of the distribution
    sigma = 1.0  # standard deviation, adjust as needed
    a = (0 - mu) / sigma
    b = (2 * np.pi - mu) / sigma

    thetas = []
    for _ in range(n):
        theta = truncnorm.rvs(a, b, loc=mu, scale=sigma)
        thetas.append(theta)
    return thetas

def perturb_data(nodes, one_data, thetas, r):
    """
    Perturb coordinates by applying rotation to specified nodes.
    
    Args:
        nodes: List of node indices to perturb
        one_data: Original coordinate data
        thetas: Angles for perturbation
        r: Perturbation magnitude
        
    Returns:
        Perturbed data
    """
    body_parts_xy = [
    "head_top_x", "head_top_y", "nose_x", "nose_y", "right_ear_x", "right_ear_y",
    "left_ear_x", "left_ear_y", "upper_neck_x", "upper_neck_y", "right_shoulder_x", 
    "right_shoulder_y", "right_elbow_x", "right_elbow_y", "right_wrist_x", "right_wrist_y",
    "thorax_x", "thorax_y", "left_shoulder_x", "left_shoulder_y", "left_elbow_x",
    "left_elbow_y", "left_wrist_x", "left_wrist_y", "pelvis_x", "pelvis_y", "right_hip_x",
    "right_hip_y", "right_knee_x", "right_knee_y", "right_ankle_x", "right_ankle_y",
    "left_hip_x", "left_hip_y", "left_knee_x", "left_knee_y", "left_ankle_x", "left_ankle_y"
    ]
    data = one_data # (num_frames, node_xy)
    frames = data.shape[0]

    for i in range(frames):
        for node in nodes:
            if "_x" in body_parts_xy[node]:
                orig_x = float(data[i, node])
                dx = r * math.cos(thetas[node])
                new_x = orig_x + dx
                data[i, node] = str(new_x)
            elif "_y" in body_parts_xy[node]:
                orig_y = float(data[i, node])
                dy = r * math.sin(thetas[node])
                new_y = orig_y + dy
                data[i, node] = str(new_y)

    return data

def pgu(out_prime, orig_out):
    """
    Calculate Prediction Gap on Unmportant feature perturbation (PGU)
    
    A measure of how much the model output changes when unimportant features are perturbed.
    Lower values indicate better faithfulness.
    
    Args:
        out_prime: List of model outputs after perturbation of unimportant features
        orig_out: Original model output
        
    Returns:
        Mean absolute difference between perturbed and original outputs
    """
    pgu_loss = []
    for num_data in range(len(out_prime)):
        loss = abs(orig_out - out_prime[num_data])
        pgu_loss.append(loss)

    return sum(pgu_loss) / len(pgu_loss)

def pgi(out_prime, orig_out):
    """
    Calculate Prediction Gap on Important feature perturbation (PGI)
    
    A measure of how much the model output changes when important features are perturbed.
    Higher values indicate better faithfulness.
    
    Args:
        out_prime: List of model outputs after perturbation of important features
        orig_out: Original model output
        
    Returns:
        Mean absolute difference between perturbed and original outputs
    """
    pgi_loss = []
    for num_data in range(len(out_prime)):
        loss = abs(orig_out - out_prime[num_data])
        pgi_loss.append(loss)

    return sum(pgi_loss) / len(pgi_loss)

def min_max_normalize(array):
    """
    Normalize an array such that its values range between 0 and 1.
    
    Args:
        array: Input array to normalize
        
    Returns:
        Normalized array with values between 0 and 1
    """
    min_val = np.min(array)
    max_val = np.max(array)
    # Prevent division by zero
    if max_val == min_val:
        return np.zeros_like(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def ROS_normal(fx, fx_primes, ex, ex_primes):
    """
    Calculate Robustness to Output Stability.
    
    Measures the stability of explanations relative to changes in model output.
    Values closer to zero indicate more stable explanations.
    
    Args:
        fx: Original model output
        fx_primes: List of perturbed model outputs
        ex: Original explanation
        ex_primes: List of perturbed explanations
        
    Returns:
        Maximum stability ratio
    """
    # Lists to store differences
    explanation_diffs = []
    output_diffs = []
    max_stability_ratio = 0
    max_ratio_index = 0
    max_norms = {"fx": 0, "fx_prime": 0, "fx_diff": 0, "ex": 0, "ex_prime": 0, "ex_diff": 0}
    max_diffs = {"explanation_diff": 0, "output_diff": 0}

    p_norm = 2
    ex = min_max_normalize(ex)  # Normalize the original explanation

    # Calculate explanations differences and output differences
    for i, (fx_prime, ex_prime) in enumerate(zip(fx_primes, ex_primes)):
        ex_prime = min_max_normalize(ex_prime)  # Normalize the perturbed explanation

        explanation_diff = np.linalg.norm(ex.ravel() - ex_prime.ravel(), ord=p_norm)
        output_diff = np.abs(fx - fx_prime)

        # Check if current ratio is larger than the current max
        current_ratio = explanation_diff / output_diff if output_diff != 0 else 0
        if current_ratio > max_stability_ratio:
            max_stability_ratio = current_ratio
            max_ratio_index = i
            max_norms["fx"] = fx
            max_norms["fx_prime"] = fx_prime
            max_norms["fx_diff"] = np.abs(fx - fx_prime)
            max_norms["ex"] = np.linalg.norm(ex)
            max_norms["ex_prime"] = np.linalg.norm(ex_prime)
            max_norms["ex_diff"] = np.linalg.norm(ex - ex_prime)
            max_diffs["explanation_diff"] = explanation_diff
            max_diffs["output_diff"] = output_diff

        explanation_diffs.append(explanation_diff)
        output_diffs.append(output_diff)

    return max_stability_ratio

def RIS_normal(x, x_primes, ex, ex_primes):
    """
    Calculate Robustness to Input Stability.
    
    Measures the stability of explanations relative to changes in model input.
    Values closer to zero indicate more stable explanations.
    
    Args:
        x: Original model input
        x_primes: List of perturbed model inputs
        ex: Original explanation
        ex_primes: List of perturbed explanations
        
    Returns:
        Maximum stability ratio
    """
    epsilon_min = 1e-10
    p_norm = 2
    max_stability_ratio = 0
    max_norms = {
        "x": 0, "x_prime": 0, "x_diff": 0,
        "ex": 0, "ex_prime": 0, "ex_diff": 0
    }
    max_diffs = {"explanation_diff": 0, "input_diff": 0}
    max_ratio_index = None

    # Normalize the original inputs
    x = min_max_normalize(x)
    ex = min_max_normalize(ex)

    for i, (x_prime, ex_prime) in enumerate(zip(x_primes, ex_primes)):
        # Normalize the perturbed inputs
        x_prime = min_max_normalize(x_prime)
        ex_prime = min_max_normalize(ex_prime)

        # Compute the normalized differences
        explanation_diff = np.linalg.norm(ex.ravel() - ex_prime.ravel(), ord=p_norm)
        input_diff = np.max([np.linalg.norm(x.ravel() - x_prime.ravel(), ord=p_norm), epsilon_min])

        current_ratio = explanation_diff / input_diff
        if current_ratio > max_stability_ratio:
            max_stability_ratio = current_ratio
            max_ratio_index = i
            max_norms["x"] = np.linalg.norm(x)
            max_norms["x_prime"] = np.linalg.norm(x_prime)
            max_norms["x_diff"] = np.linalg.norm(x - x_prime)
            max_norms["ex"] = np.linalg.norm(ex)
            max_norms["ex_prime"] = np.linalg.norm(ex_prime)
            max_norms["ex_diff"] = np.linalg.norm(ex - ex_prime)
            max_diffs["explanation_diff"] = explanation_diff
            max_diffs["input_diff"] = input_diff

    return max_stability_ratio

def RRS_normal(lx, lx_primes, ex, ex_primes):
    """
    Calculate Robustness to Representation Stability.
    
    Measures the stability of explanations relative to changes in the model's internal representations.
    Values closer to zero indicate more stable explanations.
    
    Args:
        lx: Original model internal representation
        lx_primes: List of perturbed model internal representations
        ex: Original explanation
        ex_primes: List of perturbed explanations
        
    Returns:
        Maximum stability ratio
    """
    # Lists to store differences
    p_norm = 2
    max_stability_ratio = 0
    max_norms = {"ex": 0, "ex_prime": 0, "ex_diff": 0, "lx": 0, "lx_prime": 0, "lx_diff": 0}
    max_diffs = {"explanation_diff": 0, "logits_diff": 0}
    max_ratio_index = None

    # Normalize lx and ex
    lx = min_max_normalize(lx)
    ex = min_max_normalize(ex)

    # Calculate robustness and stability differences
    for i, (lx_prime, ex_prime) in enumerate(zip(lx_primes, ex_primes)):
        lx_prime = min_max_normalize(lx_prime)
        ex_prime = min_max_normalize(ex_prime)

        explanation_diff = np.linalg.norm(ex.ravel() - ex_prime.ravel(), ord=p_norm)
        logits_diff = np.linalg.norm(lx.ravel() - lx_prime.ravel(), ord=p_norm)

        current_ratio = explanation_diff / logits_diff if logits_diff != 0 else 0
        if current_ratio > max_stability_ratio:
            max_stability_ratio = current_ratio
            max_ratio_index = i
            max_norms["lx"] = np.linalg.norm(lx)
            max_norms["lx_prime"] = np.linalg.norm(lx_prime)
            max_norms["lx_diff"] = np.linalg.norm(lx - lx_prime)
            max_norms["ex"] = np.linalg.norm(ex)
            max_norms["ex_prime"] = np.linalg.norm(ex_prime)
            max_norms["ex_diff"] = np.linalg.norm(ex - ex_prime)
            max_diffs["explanation_diff"] = explanation_diff
            max_diffs["logits_diff"] = logits_diff

    return max_stability_ratio

def get_perturb_magnitude(path):
    """
    Calculate the perturbation magnitude as a percentage of infant height.
    
    Args:
        path: Path to the CSV file with tracking coordinates
        
    Returns:
        Perturbation magnitude (1% of median height)
    """
    df = pd.read_csv(path)
    head_x = df.iloc[:, 1]
    head_y = df.iloc[:, 2]
    ankle_x = df.iloc[:, 37]
    ankle_y = df.iloc[:, 38]
    heights = np.sqrt((head_x - ankle_x) ** 2 + (head_y - ankle_y) ** 2)
    median_height = np.median(heights)

    return median_height * 0.01

def calculate_metrics(prediction_function, xai_technique='cam'):
    """
    Calculate XAI metrics on a set of infant pose data (CSV files).
    
    Args:
        prediction_function: Function that takes tracking_coords, body_parts, etc. and returns
                            predictions and explanations
        xai_technique: XAI technique to evaluate ('cam' or 'gradcam')
        
    Returns:
        Dictionary with calculated metrics
    """
    json_name = 'video_fps_dict.json'

    # Load the dictionary from the JSON file
    with open(json_name, 'r') as json_file:
        video_fps_dict = json.load(json_file)

    print("JSON data loaded successfully.")

    # Add prefix 'tracked_' and suffix '.csv' to all keys in the dictionary
    video_fps_dict_modified = {'tracked_' + key + '.csv': value for key, value in video_fps_dict.items()}

    body_parts = ['head_top', 'nose', 'right_ear', 'left_ear', 'upper_neck', 'right_shoulder', 'right_elbow',
                  'right_wrist', 'thorax', 'left_shoulder', 'left_elbow', 'left_wrist', 'pelvis', 'right_hip',
                  'right_knee', 'right_ankle', 'left_hip', 'left_knee', 'left_ankle']
    num_body_parts = len(body_parts)

    # Define the path to the directory containing the CSV files
    csv_folder = './window_selections/'

    # Create an empty list to hold the paths of the CSV files
    csv_files = []

    # Iterate over all CSV files in the directory
    for csv_file_path in glob.glob(f"{csv_folder}/*.csv"):
        # Add the file path to the list
        csv_files.append(csv_file_path)

    # Specify the path to the pickle file containing the results
    results_path = f'./results_{xai_technique}.pkl'

    # Check if the pickle file exists
    if os.path.exists(results_path):
        with open(results_path, 'rb') as file:
            results = pickle.load(file)
    else:
        # File does not exist, declare an empty dictionary
        results = {}

    # Start processing each CSV file
    for csv_file_path in csv_files:
        csv_name = os.path.basename(csv_file_path)
        last_k = 1
        if csv_name in results:
            # Find the last k calculated
            if results[csv_name]:
                last_k = max(results[csv_name].keys(), key=int)
        
        if last_k == 19:
            continue  # proceed to the next csv file

        tracking_coords = read_csv_to_array(csv_file_path)
        tracking_coords = np.array(tracking_coords)
        # shape of tracking_coords (120, 38)
        frame_number = tracking_coords.shape[0]
        
        # Reconstruct the filename to match the fps dictionary
        parts = csv_name.split('_')
        # Find the index where 'tracked' begins
        tracked_index = parts.index('tracked')
        # Reconstruct the filename from the 'tracked' part onwards
        corrected_filename = '_'.join(parts[tracked_index:])
        fps = video_fps_dict_modified.get(corrected_filename)

        # Get prediction and explanation of original data
        orig_windows_cam, orig_windows_cp_risk, orig_input, orig_rep = prediction_function(
                tracking_coords=tracking_coords, 
                body_parts=body_parts, 
                frame_rate=fps, 
                total_frames=frame_number,
                pred_frame_rate=30.0,
                pred_interval_seconds=2.5,
                window_stride=2,
                num_models=10,
                num_portions=7,
                prediction_threshold=0.350307,
                xai_technique=xai_technique
            )

        if orig_windows_cam[0].all() != orig_windows_cam[1].all():
            print('Error: CAM values for windows 0 and 1 should be the same')
            continue
            
        # Get the indices that would sort the scores array
        sorted_indices = np.argsort(orig_windows_cam[0])
        # Reverse the array of indices to get the order from highest to lowest
        sorted_indices = sorted_indices[::-1]  # (19)

        # Generate random angles
        thetas = generate_angles(num_body_parts)  # (19)
        # duplicate the angles to correspond with x, y
        thetas_xy = [angle for angle in thetas for _ in range(2)]  # (38)
        
        r_magnitude = get_perturb_magnitude(csv_file_path)

        start_time = time.time()
        
        for k in range(last_k, num_body_parts + 1):
            # Extract the top k indices
            top_k = sorted_indices[:k].tolist()
            # Extract the non-top-k indices
            if k == num_body_parts:
                non_topk = None
            else:
                non_topk = sorted_indices[k:].tolist()

            # Generate perturbations for this particular set of top_k and non_topk
            if non_topk == None:
                supernodes = [top_k]
            else:
                supernodes = [top_k, non_topk]
            
            perturbed_data = []

            for index, sublist in enumerate(supernodes):  # index is 0 to 1
                if len(supernodes) == 2:
                    NUM_PERTURB = 25
                elif len(supernodes) == 1:
                    NUM_PERTURB = 50
                nodes_xy = [num for x in sublist for num in (2*x, 2*x+1)]
                # sublist is the top_k or non_topk
                # sublist tells you which nodes need to be perturbed
                # the line above converts the list of body parts to list of body parts in x&y axes
                for _ in range(NUM_PERTURB):
                    one_data = tracking_coords.copy()  # copy coords data to be perturbed
                    new_data = perturb_data(nodes_xy, one_data, thetas_xy, r_magnitude)
                    perturbed_data.append(new_data)  # (NUM_PERTURB*2,120,28)

                if index < 2:
                    print(index, end=', ')
                else:
                    print(index)
                    print()
            
            # Then take the CP prediction risk of these NUM_PERTURB*top_k and NUM_PERTURB*non_topk
            cams_perturbed = []
            cp_risks_perturbed = []
            perturbed_inputs = []
            perturbed_reps = []
            total_perturbed_data = np.asarray(perturbed_data).shape[0]
            
            for i in range(total_perturbed_data):
                perturbed_window_cam, perturbed_window_cp_risk, perturbed_input, perturbed_rep = prediction_function(
                    tracking_coords=perturbed_data[i], 
                    body_parts=body_parts, 
                    frame_rate=fps, 
                    total_frames=frame_number,
                    pred_frame_rate=30.0,
                    pred_interval_seconds=2.5,
                    window_stride=2,
                    num_models=10,
                    num_portions=7,
                    prediction_threshold=0.350307,
                    xai_technique=xai_technique
                )
                
                if perturbed_window_cam[0].all() == perturbed_window_cam[1].all():
                    cams_perturbed.append(perturbed_window_cam[0])  # (NUM_PERTURB,frames,19)
                    cp_risks_perturbed.append(perturbed_window_cp_risk[0])  # (NUM_PERTURB)
                    perturbed_inputs.append(perturbed_input)  # (num_batch, pvbxy, frames, joints) (2,6,150,19)
                    perturbed_reps.append(perturbed_rep) 
                else:
                    print('Warning: CAM values for windows 0 and 1 are not the same')
                    continue
            
            perturbed_inputs = (torch.stack(perturbed_inputs)).tolist()
            
            if k == num_body_parts:
                cp_risk_topk = cp_risks_perturbed
            else:
                cp_risk_topk = cp_risks_perturbed[:total_perturbed_data//2]
                cp_risk_nontopk = cp_risks_perturbed[total_perturbed_data//2:total_perturbed_data]

            # Get faithfulness
            PGI_k = pgi(cp_risk_topk, orig_windows_cp_risk[0])
            if k == num_body_parts:
                PGU_k = np.nan
            else:
                PGU_k = pgu(cp_risk_nontopk, orig_windows_cp_risk[0])

            # check if original data is CP or no CP
            if orig_windows_cp_risk[0] >= 0.350307:
                cp_pred_orig = True
            else:
                cp_pred_orig = False

            # get variables for stability calculation
            out_primes = []
            x_primes = []
            rep_primes = []
            explanation_primes = []
            
            for i in range(len(cp_risks_perturbed)):
                if cp_risks_perturbed[i] >= 0.350307:
                    cp_pred_perturbed = True
                else:
                    cp_pred_perturbed = False

                if cp_pred_orig == cp_pred_perturbed:  # Classification did not change
                    explanation_primes.append(cams_perturbed[i])
                    out_primes.append(cp_risks_perturbed[i])
                    x_primes.append(perturbed_inputs[i])
                    rep_primes.append(perturbed_reps[i])

            # Get stability
            if explanation_primes:  # check if empty or not
                ROS_k = (ROS_normal(orig_windows_cp_risk[0], out_primes, orig_windows_cam[0], explanation_primes))
            else:
                ROS_k = np.nan
            
            if x_primes:  # check if empty or not
                x_primes = np.array(x_primes)
                orig_input = np.array(orig_input)
                orig_inputp = orig_input[0, 0:2, :, :]
                orig_inputv = orig_input[0, 2:4, :, :]
                orig_inputb = orig_input[0, 4:6, :, :]
                xp_primes = x_primes[:, 0, 0:2, :, :]
                xv_primes = x_primes[:, 0, 2:4, :, :]
                xz_primes = x_primes[:, 0, 4:6, :, :]
                RISp_k = (RIS_normal(orig_inputp, xp_primes, orig_windows_cam[0], explanation_primes))
                RISv_k = (RIS_normal(orig_inputv, xv_primes, orig_windows_cam[0], explanation_primes))
                RISb_k = (RIS_normal(orig_inputb, xz_primes, orig_windows_cam[0], explanation_primes))
            else:
                RISp_k = np.nan
                RISv_k = np.nan
                RISb_k = np.nan

            RRS_k = RRS_normal(orig_rep, rep_primes, orig_windows_cam[0], explanation_primes)
            
            end_time = time.time()  # End time
            duration = end_time - start_time  # Duration in seconds
            print(f'k: {k}')
            print(f'duration: {duration}')

            # Store the calculated values in the data_store
            # Ensure the csv_name key exists; set it to a default empty dictionary if not
            results.setdefault(csv_name, {})

            # Ensure the key k exists within results[csv_name]; set it to a default empty dictionary if not
            results[csv_name].setdefault(k, {})

            # Now safely assign the dictionary to results[csv_name][k] without triggering a KeyError
            results[csv_name][k] = {
                'pgi': PGI_k,
                'pgu': PGU_k,
                'risp_c': RISp_k,
                'risv_c': RISv_k,
                'risb_c': RISb_k,
                'ros_c': ROS_k,
                'rrs_c': RRS_k
            }
            
            # Save the data_store to the file at each iteration
            with open(results_path, 'wb') as file:
                pickle.dump(results, file)
                
    return results

# If this file is run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate XAI metrics for CP prediction')
    parser.add_argument('--xai', type=str, default='cam', help='XAI technique to evaluate (cam or gradcam)')
    
    args = parser.parse_args()
    
    print(f"This module contains metrics for evaluating XAI techniques in cerebral palsy prediction.")
    print(f"Please import this module and call calculate_metrics with the CP prediction function.")