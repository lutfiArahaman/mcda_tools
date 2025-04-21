import arcpy
import os
import numpy as np
import pandas as pd
from scipy.stats import gmean

def calculate_fuzzy_weights(pairwise_matrix):
    """
    Calculate fuzzy weights using TFNs (Triangular Fuzzy Numbers).

    Parameters:
    - pairwise_matrix: 3D numpy array of shape (n, n, 3) for TFNs [l, m, u].

    Returns:
    - defuzzified_weights: Defuzzified crisp weights.
    """
    n = pairwise_matrix.shape[0]

    # Calculate geometric means for each row
    geometric_means = np.zeros((n, 3))
    for i in range(n):
        row = pairwise_matrix[i, :, :]
        geometric_means[i] = [
            np.prod(row[:, 0]) ** (1 / n),
            np.prod(row[:, 1]) ** (1 / n),
            np.prod(row[:, 2]) ** (1 / n),
        ]

    # Normalize fuzzy weights
    fuzzy_sums = np.sum(geometric_means, axis=0)
    normalized_weights = geometric_means / fuzzy_sums

    # Defuzzify weights using the centroid method
    defuzzified_weights = np.mean(normalized_weights, axis=1)
    return defuzzified_weights

def calculate_consistency_ratio(pairwise_matrix, weights):
    """
    Calculate the consistency ratio for a fuzzy AHP matrix.

    Parameters:
    - pairwise_matrix: 3D numpy array of TFNs [l, m, u].
    - weights: Defuzzified weights.

    Returns:
    - consistency_ratio: The consistency ratio.
    """
    n = pairwise_matrix.shape[0]
    reconstructed_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            reconstructed_matrix[i, j] = weights[i] / weights[j]

    max_eigenvalue = np.mean(np.sum(reconstructed_matrix * weights[:, None], axis=1))
    consistency_index = (max_eigenvalue - n) / (n - 1)
    
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    random_index = ri_table.get(n, 1.49)

    if random_index == 0:
        return 0.0

    consistency_ratio = consistency_index / random_index
    return consistency_ratio

def fahp(pairwise_matrix):
    weights = calculate_fuzzy_weights(pairwise_matrix)
    cr = calculate_consistency_ratio(pairwise_matrix, weights)
    return weights, cr

def normalize_weights(weights):
    total = sum(weights)
    return [w / total for w in weights]

def convert_weights_to_percentages(weights):
    total = 100
    raw_percentages = [int(w * total) for w in weights]
    current_sum = sum(raw_percentages)
    difference = total - current_sum

    if difference > 0:
        for _ in range(difference):
            max_index = raw_percentages.index(max(raw_percentages))
            raw_percentages[max_index] += 1
    elif difference < 0:
        for _ in range(abs(difference)):
            max_index = raw_percentages.index(max(raw_percentages))
            raw_percentages[max_index] -= 1

    return raw_percentages

def save_individual_percentages_to_files(percentages, output_files):
    """
    Save each percentage weight as a numerical value (without the % symbol) 
    to the corresponding output file.
    """
    if len(percentages) != len(output_files):
        raise ValueError("The number of output files must match the number of percentage weights.")

    for percentage, output_file in zip(percentages, output_files):
        with open(output_file, 'w') as file:
            file.write(f"{percentage}")  # Write only the value (integer/double)
        arcpy.AddMessage(f"Saved {percentage} to: {output_file}")

def main():
    input_files = arcpy.GetParameterAsText(0)  # MultiValue: List of .txt files
    output_files = arcpy.GetParameterAsText(1)  # MultiValue: List of output files for each percentage

    input_file_paths = input_files.split(";")
    output_file_paths = output_files.split(";")

    try:
        all_weights = []
        consistency_ratios = []

        for file_path in input_file_paths:
            file_path = file_path.strip()

            if not file_path.endswith('.txt'):
                arcpy.AddWarning(f"Skipping non-txt file: {file_path}")
                continue

            pairwise_matrix = pd.read_csv(file_path, header=None, delim_whitespace=True).to_numpy()
            pairwise_matrix = np.array([[[eval(cell) for cell in tfn.split(',')] for tfn in row] for row in pairwise_matrix])

            if pairwise_matrix.shape[0] != pairwise_matrix.shape[1]:
                raise ValueError(f"Matrix in file '{os.path.basename(file_path)}' is not square.")

            weights, cr = fahp(pairwise_matrix)
            all_weights.append(weights)
            consistency_ratios.append(cr)

        geometric_weights = gmean(np.array(all_weights), axis=0)
        normalized_weights = normalize_weights(geometric_weights)
        percentages = convert_weights_to_percentages(normalized_weights)

        save_individual_percentages_to_files(percentages, output_file_paths)

    except Exception as e:
        arcpy.AddError(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
