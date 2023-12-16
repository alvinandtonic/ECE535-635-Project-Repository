import pandas as pd
import matplotlib.pyplot as plt
import os

# Update this file path to the location of your results.txt file
file_path = 'results/mhealth/ablation/acce_gyro/A30_B30_AB0_label_A_test_B/results.txt'  # Update this to the correct file path

# Specify the directory to save the plots
save_dir = 'results/mhealth/ablation/acce_gyro/A30_B30_AB0_label_A_test_B'  # Update this to your desired directory
os.makedirs(save_dir, exist_ok=True)

# Load the data from the text file
results_df = pd.read_csv(file_path, header=None, delimiter=',')
results_df.columns = ['Round', 'Local AE Loss', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Test F1 Score']

# Define plot titles and y-labels
plot_titles = ['Train Loss over Rounds', 'Train Accuracy over Rounds', 'Test Loss over Rounds', 
               'Test Accuracy over Rounds', 'Test F1 Score over Rounds', 'Local AE Loss over Rounds']
y_labels = ['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Test F1 Score', 'Local AE Loss']

# Create and save each plot
for i, column in enumerate(['Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Test F1 Score', 'Local AE Loss']):
    plt.figure(figsize=(8, 6))
    plt.plot(results_df['Round'], results_df[column])
    plt.title(plot_titles[i])
    plt.xlabel('Round')
    plt.ylabel(y_labels[i])
    plt.grid(True)
    plot_file_name = os.path.join(save_dir, f'{column.replace(" ", "_")}.png')
    plt.savefig(plot_file_name)
    plt.close()

print("Plots saved to", save_dir)
