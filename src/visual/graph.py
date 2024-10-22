import matplotlib.pyplot as plt
import numpy as np

def plot_sequential_data(time_axis, red_data, blue_data_list):
    """
    Plots 1D sequential data along an arbitrary time axis.
    
    Parameters:
    - time_axis: A 1D array-like structure representing the time axis.
    - red_data: A 1D array-like structure for the red data to be plotted.
    - blue_data_list: A list of 1D array-like structures for the blue data to be plotted.
    """
    plt.figure(figsize=(10, 5))
    
    # Plot the red data
    plt.scatter(time_axis, red_data, color='red', label='Red Data', linewidth=2)
    
    # Plot each blue data set
    for i, blue_data in enumerate(blue_data_list):
      plt.scatter(time_axis, blue_data, color='blue', label=f'Blue Data {i+1}', alpha=0.5)
    
    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('1D Sequential Data')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create an arbitrary time axis
    time_axis = np.arange(0, 10, 1)
    
    from src.data.file_handler import read_file
    from src.data.processing import parse_str
    
    data_array = parse_str(read_file("./dataset/data64QAM.txt"))
    data_len = len(data_array)
    
    another_data_array = parse_str(read_file("./new_BiTCN_final_predictions_471.txt"))
    another_data_len = len(another_data_array)
    # print(data_len, len(another_data_array))
    
    time_axis = np.arange(0, min(data_len, another_data_len), 1)
    data_array = data_array[:min(data_len, another_data_len)]
    # another_data_array = another_data_array[:10]
    print(len(data_array), len(another_data_array), len(time_axis))
    # # Create red data
    # red_data = np.sin(time_axis)
    
    # # Create multiple blue data sets
    # blue_data_1 = np.sin(time_axis + 0.5)  # phase shift
    # blue_data_2 = np.sin(time_axis + 1.0)  # another phase shift
    
    
    
    # Plot the data
    plot_sequential_data(time_axis, data_array, [another_data_array])
