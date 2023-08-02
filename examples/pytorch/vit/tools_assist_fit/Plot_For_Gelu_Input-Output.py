'''
    The main purpose of this code is to plot the data 
        based on the exported input-output files of the 'Gelu' during actual program execution. 
        It aims to observe the patterns in the data and the range of the truncation interval.
'''
import matplotlib.pyplot as plt
import csv

# Read data from a CSV file.
def read_data_from_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            data.extend([float(x) for x in row])
    return data

# Plot a histogram.
def plot_histogram(data):
    plt.hist(data, bins=50)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Input Data')
    plt.savefig('gelu_input.png')

# Calculate the range of the input data.
def data_range(data):
    return min(data), max(data)

# main
# For example;
filename = './ILSVRC2012_val_00000362.JPEG_gelu_output_layer1_batch_5.csv'

input_data = read_data_from_csv(filename)
plot_histogram(input_data)
data_min, data_max = data_range(input_data)
print(f"Input data range: {data_min} to {data_max}")
