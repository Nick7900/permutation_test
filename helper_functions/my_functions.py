import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import pandas as pd
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def create_folder(folder_name):
    """
    Create a folder/directory if it doesn't already exist.

    Args:
    - folder_name (str): Name of the folder/directory to be created.

    Returns:
    None

    Example:
    create_folder("data_folder")
    Folder created: data_folder
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print("Folder created:", folder_name)
    else:
        print("Folder already exists:", folder_name)

def load_files(data_folder, file_name, file_type, delimiter=' '):
    import mat73
    """
    Load files from a specified data folder.

    Args:
    - data_folder (str): Path to the data folder or the file itself.
    - file_name (str): Name of the file.
    - file_type (str): Type of the file (.mat or .txt).
    - delimiter (str, optional): Delimiter used in case of a text file (default is ' ').

    Returns:
    - If file_type is '.mat':
        - data (dict): Dictionary containing the loaded data from the .mat file.
        - var_name (str): Name of the variable in the .mat file.
    - If file_type is '.txt':
        - data (DataFrame): Pandas DataFrame containing the loaded data from the text file.

    Example:
    load_files("\\data_raw", "\\headers_with_category", ".mat")
    (data, var_name)
    load_files("\\data_raw", "\\data", ".txt", delimiter=',')
    data
    """

    # Check if data_folder is a file path or a local folder
    if os.path.isdir(data_folder) == 0:
        current_directory = os.getcwd() 
        file_path = os.path.join(current_directory + data_folder)
    else:
        file_path = data_folder 

    # Load file based on the file_type
    if file_type == ".mat":
        try:
            # Load .mat file using scipy.io
            data = scipy.io.loadmat(file_path + file_name + file_type)
            var_name = list(data.keys())[-1]  # Get the name of the variable in the .mat file
            return data, var_name
        except:
            data = mat73.loadmat(file_path + file_name + file_type)
            return data
    elif file_type == ".txt":
        # Load text file using pandas
        data = pd.read_csv(file_path + file_name + file_type, delimiter=delimiter, header=None)
        return data
    
def plot_heatmap(pval, method, normalize_vals=True, figsize=(12, 7), steps=11, title_text="", annot=True, cmap='coolwarm', xlabel=None, ylabel=None):
    """
    Plot a heatmap of p-values.

    Args:
        pval (numpy.ndarray): The p-values data to be plotted.
        method (str): The method used for the permutation test. Should be one of 'regression', 'correlation', or 'correlation_com'.
        normalize_vals (bool, optional): If True, the data range will be normalized from 0 to 1. Default is True.
        figsize (tuple, optional): Figure size in inches (width, height). Default is (12, 7).
        steps (int, optional): Number of steps for x and y-axis ticks. Default is 11.
        title_text (str, optional): Title text for the heatmap. If not provided, a default title will be used.
        annot (bool, optional): If True, annotate each cell with the numeric value. Default is True.
        cmap (str, optional): Colormap to use. Default is 'coolwarm'.
        xlabel (str, optional): X-axis label. If not provided, default labels based on the method will be used.
        ylabel (str, optional): Y-axis label. If not provided, default labels based on the method will be used.

    Returns:
        None (Displays the heatmap plot).

    """

    fig, ax = plt.subplots(figsize=figsize)
    if len(pval.shape)==1:
        pval =np.expand_dims(pval,axis=0)

    if normalize_vals:
        # Normalize the data range from 0 to 1
        norm = plt.Normalize(vmin=0, vmax=1)
        heatmap = sns.heatmap(pval, ax=ax, cmap=cmap, annot=annot, fmt=".2f", cbar=False, norm=norm)
    else:
        heatmap = sns.heatmap(pval, ax=ax, cmap=cmap, annot=annot, fmt=".2f", cbar=False)

    # Add labels and title
    if method == "regression":
        if xlabel is None:
            ax.set_xlabel('Features', fontsize=12)
        else:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel is None:
            ax.set_ylabel('Time', fontsize=12)
        else:
            ax.set_ylabel(ylabel, fontsize=12)
    else:
        if xlabel is None:
            ax.set_xlabel('Predictors', fontsize=12)
        else: 
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel is None:
            ax.set_ylabel('Features', fontsize=12)
        else:
            ax.set_ylabel(ylabel, fontsize=12)

    if not title_text:
        ax.set_title('Heatmap (p-values)', fontsize=14)
    else:
        ax.set_title(title_text, fontsize=14)

    # Set the x-axis ticks
    ax.set_xticks(np.linspace(0, pval.shape[1]-1, steps).astype(int))
    ax.set_xticklabels(np.linspace(0, pval.shape[1], steps).astype(int), rotation="horizontal", fontsize=10)

    # Set the y-axis ticks
    ax.set_yticks(np.linspace(0, pval.shape[0]-1, steps).astype(int))
    ax.set_yticklabels(np.linspace(0, pval.shape[0], steps).astype(int), rotation="horizontal", fontsize=10)

    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(heatmap.get_children()[0], cax=cax)

    # Show the plot
    plt.show()

  
def plot_histograms(test_statistic, title_text=""):
    """
    Plot the histogram of the permutation mean with the observed statistic marked.

    Args:
        test_statistic (numpy.ndarray): An array containing the permutation mean values.
        title_text (str, optional): Additional text to include in the title of the plot. Default is an empty string.

    Returns:
        None: Displays the histogram plot.

    Example:
        >>> import numpy as np
        >>> test_statistic = np.random.normal(0, 1, 1000)
        >>> plot_histograms(test_statistic, "Permutation Mean Distribution")

    """
    plt.figure()
    sns.histplot(test_statistic, kde=True)
    plt.axvline(x=test_statistic[0], color='red', linestyle='--', label='Observed Statistic')
    plt.xlabel('Permutation mean')
    plt.ylabel('Density')
    
    if not title_text:
        plt.title('Distribution of Permutation Mean', fontsize=14)
    else:
        plt.title(title_text, fontsize=14)
        
    plt.legend()
    plt.show()


def plot_scatter_with_labels(p_values, alpha=0.05, title_text="", xlabel=None, ylabel=None, xlim_start =-0.1,ylim_start=0):
    """
    Create a scatter plot to visualize p-values with labels indicating significant points.

    Args:
        p_values (array-like): An array of p-values.
        alpha (float): Threshold for significance (default: 0.05)
        title_text (str, optional): The title text for the plot (default="").
        xlabel (str, optional): The label for the x-axis (default=None).
        ylabel (str, optional): The label for the y-axis (default=None).
        xlim_start (float): start position of x-axis limits (default: -5)
        ylim_start (float): start position of y-axis limits (default: -0.1)

    Returns:
        None

    Note:
        - Points with p-values less than alpha are considered significant and marked with red text.

    """

    # Create a binary mask based on condition (values below alpha)
    mask = p_values < alpha

    # Create a hue p_values based on the mask (True/False values)
    hue = mask.astype(int)

    # Set the color palette and marker style
    # sns.set_palette()
    markers = ["o", "s"]

    # Create a scatter plot with hue and error bars
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=np.arange(0, len(p_values)), y=-np.log10(p_values), hue=hue, style=hue,
                    markers=markers, s=40, edgecolor='k', linewidth=1, ax=ax)

    # Add labels and title to the plot
    if not title_text:
        ax.set_title(f'Scatter Plot of P-values, alpha={alpha}', fontsize=14, fontname='serif')
    else:
        ax.set_title(title_text, fontsize=14, fontname='serif')

    if xlabel is None:
        ax.set_xlabel('Index', fontsize=12, fontname='serif')
    else:
        ax.set_xlabel(xlabel, fontsize=12, fontname='serif')

    if ylabel is None:
        ax.set_ylabel('-log10(p-values)', fontsize=12, fontname='serif')
    else:
        ax.set_ylabel(ylabel, fontsize=12, fontname='serif')

    # Add text labels for indices where the mask is True
    for i, m in enumerate(mask):
        if m:
            ax.text(i, -np.log10(p_values[i]), str(i), ha='center', va='bottom', color='red', fontsize=10)

    # Adjust legend position and font size
    ax.legend(title="Significance", loc="upper right", fontsize=10, bbox_to_anchor=(1.25, 1))

    # Set axis limits to focus on the relevant data range
    ax.set_xlim(xlim_start, len(p_values))
    ax.set_ylim(ylim_start, np.max(-np.log10(p_values)) * 1.2)

    # Customize plot background and grid style
    sns.set_style("white")
    ax.grid(color='lightgray', linestyle='--')

    # Show the plot
    plt.tight_layout()
    plt.show()



def get_pairs(n_subjects, T=None, test_type=None):
    """
    Generate pairs of indices for paired analysis.

    Args:
    - n_subjects (int): Number of subjects.
    - T (int or None): Number of time points (for whole timeseries analysis).
    - test_type (str or None): Type of test being performed.

    Returns:
    - pairs (ndarray): NumPy array representing the pairs of indices.

    Example:
    get_pairs(4, T=10, test_type="whole_timeseries")
    array([[ 0,  1],
           [ 2,  3],
           [ 4,  5],
           [ 6,  7],
           [ 8,  9]])

    get_pairs(5, test_type="per_timepoint")
    array([[0, 1],
           [2, 3],
           [4, 5]])

    """

    if test_type == "whole_timeseries":
        # For whole timeseries analysis, generate a range of indices based on the number of subjects and time points
        array_range = np.arange(n_subjects * T)
    else:
        # For other test types, generate a range of indices based on the number of subjects
        array_range = np.arange(n_subjects)

    # Create pairs of indices by combining consecutive indices in the array_range
    pairs = np.column_stack((array_range[::2], array_range[1::2]))

    return pairs

def get_index_subjects(n_subjects, n_trials=None, test_type=None, n_timepoints=None):
    """
    Generate indices of subjects for each time point.

    Args:
    - n_subjects (int): Number of subjects.
    - n_trials (int or None): Number of trials (for whole timeseries analysis).
    - test_type (str or None): Type of test being performed.
    - n_timepoints (int or None): Number of time points.

    Returns:
    - index_subjects (ndarray): NumPy array representing the indices of subjects for each time point.

    Example:
    get_index_subjects(4, n_trials=3, test_type="whole_timeseries", n_timepoints=5)
    array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

    get_index_subjects(5, n_trials=2, test_type="per_timepoint")
    array([1, 1, 2, 2, 3])
    """

    # Set the default number of trials to be equal to the number of subjects if not provided
    n_trials = n_subjects if n_trials is None else n_trials

    if test_type == "whole_timeseries":
        # For whole timeseries analysis, repeat the range of trials based on the number of subjects and time points
        index_subjects = np.repeat(np.arange(1, n_trials + 1), n_subjects * n_timepoints // n_trials)
        
        # Adjust the length of index_subjects if necessary
        if len(index_subjects) < n_subjects * n_timepoints:
            index_subjects = np.concatenate((index_subjects, np.repeat(index_subjects[-1], n_subjects * n_timepoints - len(index_subjects))))
        else:
            index_subjects = index_subjects[:n_subjects * n_timepoints]
    else:
        # For other test types, repeat the range of trials based on the number of subjects
        index_subjects = np.repeat(np.arange(1, n_trials + 1), n_subjects // n_trials)
        
        # Adjust the length of index_subjects if necessary
        if len(index_subjects) < n_subjects:
            index_subjects = np.concatenate((index_subjects, np.repeat(index_subjects[-1], n_subjects - len(index_subjects))))
        else:
            index_subjects = index_subjects[:n_subjects]

    return index_subjects
    
def get_concatenate_data(data_tmp):
    """
    Concatenates a list of 2D arrays along the first axis.

    Args:
        data_tmp (list): List of 2D arrays to be concatenated.

    Returns:
        numpy.ndarray: Concatenated data array.

    Raises:
        ValueError: If the input data_tmp is empty or the arrays have inconsistent shapes.
    """
    if len(data_tmp) == 0:
        raise ValueError("Input data_tmp cannot be empty.")

    # Check for consistent shapes
    for i in range(1, len(data_tmp)):
        if data_tmp[i].shape != data_tmp[0].shape:
            raise ValueError("Arrays in data_tmp have inconsistent shapes.")

    # Concatenate data
    data = np.concatenate(data_tmp, axis=0)

    return data


def get_timestamp_indices(n_timestamps, n_subjects):
    """
    Generate indices of the timestamps for each subject in the data.

    Args:
    - n_timestamps (int): Number of timestamps.
    - n_subjects (int): Number of subjects.

    Returns:
    - indices (ndarray): NumPy array representing the indices of the timestamps for each subject.

    Example:
    get_timestamp_indices(5, 3)
    array([[ 0,  5],
           [ 5, 10],
           [10, 15]])
    """
    indices = np.column_stack([np.arange(0, n_timestamps * n_subjects, n_timestamps),
                               np.arange(0 + n_timestamps, n_timestamps * n_subjects + n_timestamps, n_timestamps)])

    return indices


def pval_test(pval, method='fdr_bh', alpha = 0.05):
    from statsmodels.stats import multitest as smt
    """
    This function performs multiple thresholding and correction for a 2D numpy array of p-values.

    Args:
    - pval: 2D numpy array of p-values.
    - method: method used for FDR correction. Default is 'fdr_bh'.
        bonferroni : one-step correction
        sidak : one-step correction
        holm-sidak : step down method using Sidak adjustments
        holm : step-down method using Bonferroni adjustments
        simes-hochberg : step-up method (independent)   
        hommel : closed method based on Simes tests (non-negative)
        fdr_bh : Benjamini/Hochberg (non-negative)
        fdr_by : Benjamini/Yekutieli (negative)
        fdr_tsbh : two stage fdr correction (non-negative)
        fdr_tsbky : two stage fdr correction (non-negative)
        
    - alpha: significance level. Default is 0.05.

    Returns:
    - p_values_corrected: 2D numpy array of corrected p-values.
    - rejected_corrected: 2D numpy array of boolean values indicating rejected null hypotheses.
    """
    
    # If the calibration type is not any of "regression", "univariate", "univariate_com", then it raises a ValueError
    if method not in ["fdr_bh","fdr_by","fdr_tsbh ","fdr_tsbky ","bonferroni","sidak","holm-sidak","holm","simes-hochberg","hommel"]:
        raise ValueError("Invalid method specified. Must be 'fdr_bh','fdr_by','fdr_tsbh','fdr_tsbky','bonferroni', 'sidak', 'holm-sidak','holm,'simes-hochberg or'hommel' if specified.\n Otherwise default='fdr_bh' (Benjamini/Hochberg (non-negative))")

    # perform the FDR correction using statsmodels multitest module
    rejected, p_values_corrected, _, _ = smt.multipletests(pval.flatten(), alpha=alpha, method=method, returnsorted=False)

    # reshape the corrected p-values to a 2D matrix
    p_values_corrected = p_values_corrected.reshape(pval.shape)
    rejected_corrected = rejected.reshape(pval.shape)

    # return the corrected p-values and boolean values indicating rejected null hypotheses
    return p_values_corrected,rejected_corrected



def compare_p_values(p_values, p_values_corrected, threshold=0.05):
    """
    Compare two arrays of p-values and identify the indices with similar and non-similar values.

    Args:
        p_values (ndarray): Array of p-values.
        p_values_corrected (ndarray): Array of corrected p-values.
        threshold (float): Significance threshold.

    Returns:
        similar_indices (ndarray): Indices with similar p-values below the threshold.
        not_similar_indices (ndarray): Indices with non-similar p-values below the threshold.
    """

    # Perform thresholding
    significant_indices = np.where(p_values < threshold)[0]

    # Identify significant results after correction
    significant_indices_corrected = np.where(p_values_corrected < threshold)[0]

    # Find indices where the values are similar
    similar_indices = np.intersect1d(significant_indices, significant_indices_corrected)

    # Find indices where the values are not similar
    not_similar_indices = np.setxor1d(significant_indices, significant_indices_corrected)

    return similar_indices, not_similar_indices
