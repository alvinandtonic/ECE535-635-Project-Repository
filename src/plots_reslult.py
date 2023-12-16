import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the text file
results_df = pd.read_csv('results/mhealth/ablation/acce_gyro/A30_B30_AB0_label_A_test_B/results.txt', header=None, delimiter=',')
results_df.columns = ['Round', 'Local AE Loss', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy', 'Test F1 Score']

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# Train Loss over Rounds
axs[0, 0].plot(results_df['Round'], results_df['Train Loss'], color='blue')
axs[0, 0].set_title('Train Loss over Rounds')
axs[0, 0].set_xlabel('Round')
axs[0, 0].set_ylabel('Train Loss')

# Train Accuracy over Rounds
axs[0, 1].plot(results_df['Round'], results_df['Train Accuracy'], color='green')
axs[0, 1].set_title('Train Accuracy over Rounds')
axs[0, 1].set_xlabel('Round')
axs[0, 1].set_ylabel('Train Accuracy')

# Test Loss over Rounds
axs[1, 0].plot(results_df['Round'], results_df['Test Loss'], color='red')
axs[1, 0].set_title('Test Loss over Rounds')
axs[1, 0].set_xlabel('Round')
axs[1, 0].set_ylabel('Test Loss')

# Test Accuracy over Rounds
axs[1, 1].plot(results_df['Round'], results_df['Test Accuracy'], color='purple')
axs[1, 1].set_title('Test Accuracy over Rounds')
axs[1, 1].set_xlabel('Round')
axs[1, 1].set_ylabel('Test Accuracy')

# Test F1 Score over Rounds
axs[2, 0].plot(results_df['Round'], results_df['Test F1 Score'], color='orange')
axs[2, 0].set_title('Test F1 Score over Rounds')
axs[2, 0].set_xlabel('Round')
axs[2, 0].set_ylabel('Test F1 Score')

# Local AE Loss over Rounds
axs[2, 1].plot(results_df['Round'], results_df['Local AE Loss'], color='cyan')
axs[2, 1].set_title('Local AE Loss over Rounds')
axs[2, 1].set_xlabel('Round')
axs[2, 1].set_ylabel('Local AE Loss')

plt.tight_layout()
plt.show()
