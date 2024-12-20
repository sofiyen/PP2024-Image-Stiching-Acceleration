# import matplotlib.pyplot as plt
# import numpy as np
# import re

# def extract_times(content, mode):
#     """Extract execution times from the content for a specific mode (Parallel/Sequential)"""
#     # Find the section for the specific mode
#     if mode == "Parallel":
#         section = content.split("[+]Sequential")[0]  # Get everything before Sequential
#     else:  # Sequential
#         section = content.split("[+]Sequential")[1]  # Get everything after Sequential

#     # Extract times using regex
#     times = []
#     for line in section.split('\n'):
#         if 'Total executing time:' in line:
#             time = int(re.search(r'(\d+) ms', line).group(1))
#             times.append(time)
    
#     return times

# def plot_comparison(filename):
#     # Read the profile data
#     with open(filename, 'r') as f:
#         content = f.read()

#     # Extract times for both modes
#     parallel_times = extract_times(content, "Parallel")
#     sequential_times = extract_times(content, "Sequential")

#     # Convert to seconds for better readability
#     parallel_times = np.array(parallel_times) / 1000
#     sequential_times = np.array(sequential_times) / 1000

#     # Set up the bar positions
#     image_pairs = [f'Pair {i+1}' for i in range(len(parallel_times))]
#     x = np.arange(len(image_pairs))
#     width = 0.35

#     # Create the figure and axis
#     fig, ax = plt.subplots(figsize=(12, 6))

#     # Create the bars
#     rects1 = ax.bar(x - width/2, parallel_times, width, label='Parallel', color='#2ecc71')
#     rects2 = ax.bar(x + width/2, sequential_times, width, label='Sequential', color='#e74c3c')

#     # Customize the plot
#     ax.set_ylabel('Execution Time (seconds)')
#     ax.set_title('Parallel vs Sequential Execution Time Comparison')
    
#     # Adjust x-axis to make room for speedup text
#     ax.set_xticks(x)
#     ax.set_xticklabels([])  # Remove original labels
    
#     # Calculate and display speedup
#     speedups = sequential_times / parallel_times
#     avg_speedup = np.mean(speedups)

#     # Add value labels on top of each bar
#     def autolabel(rects):
#         for rect in rects:
#             height = rect.get_height()
#             ax.annotate(f'{height:.2f}s',
#                         xy=(rect.get_x() + rect.get_width()/2, height),
#                         xytext=(0, 3),
#                         textcoords="offset points",
#                         ha='center', va='bottom')

#     autolabel(rects1)
#     autolabel(rects2)

#     # Add pair labels and speedup below the bars
#     for i in x:
#         # Add pair label
#         plt.text(i, -0.05 * ax.get_ylim()[1], f'Pair {i+1}',
#                 ha='center', va='bottom', transform=ax.transData)
#         # Add speedup below pair label
#         plt.text(i, -0.15 * ax.get_ylim()[1], f'Speedup: {speedups[i]:.1f}x',
#                 ha='center', va='bottom', transform=ax.transData)

#     # Add legend
#     ax.legend()

#     # Add average speedup text in top-left corner
#     plt.text(0.02, 0.98, f'Average Speedup: {avg_speedup:.1f}x',
#              transform=ax.transAxes, color='blue',
#              bbox=dict(facecolor='white', alpha=0.8))

#     # Adjust bottom margin to make room for labels
#     plt.subplots_adjust(bottom=0.2)

#     # Save the plot
#     plt.savefig('execution_time_comparison.png', dpi=300, bbox_inches='tight')
#     plt.close()

# if __name__ == "__main__":
#     profile_file = "./bin/match_features_profile.txt"
#     plot_comparison(profile_file)

import matplotlib.pyplot as plt
import numpy as np
import re

def extract_times(content, mode):
    """Extract execution times from the content for a specific mode (Parallel/Sequential)"""
    # Find the section for the specific mode
    if mode == "Parallel":
        section = content.split("[+]Sequential")[0]  # Get everything before Sequential
    else:  # Sequential
        section = content.split("[+]Sequential")[1]  # Get everything after Sequential

    # Extract times using regex
    times = []
    for line in section.split('\n'):
        if 'Total execution time:' in line:
            time = int(re.search(r'(\d+) ms', line).group(1))
            times.append(time)
    
    return times

def plot_comparison(filename):
    # Read the profile data
    with open(filename, 'r') as f:
        content = f.read()

    # Extract times for both modes
    parallel_times = extract_times(content, "Parallel")
    sequential_times = extract_times(content, "Sequential")

    # Convert to seconds for better readability
    parallel_times = np.array(parallel_times) / 1000
    sequential_times = np.array(sequential_times) / 1000

    # Set up the bar positions
    image_pairs = [f'Pair {i+1}' for i in range(len(parallel_times))]
    x = np.arange(len(image_pairs))
    width = 0.35

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the bars
    rects1 = ax.bar(x - width/2, parallel_times, width, label='Parallel', color='#2ecc71')
    rects2 = ax.bar(x + width/2, sequential_times, width, label='Sequential', color='#e74c3c')

    # Customize the plot
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Parallel vs Sequential Execution Time Comparison')
    
    # Adjust x-axis to make room for speedup text
    ax.set_xticks(x)
    ax.set_xticklabels([])  # Remove original labels
    
    # Calculate and display speedup
    speedups = sequential_times / parallel_times
    avg_speedup = np.mean(speedups)

    # Add value labels on top of each bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}s',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    # Add pair labels and speedup below the bars
    for i in x:
        # Add pair label
        plt.text(i, -0.05 * ax.get_ylim()[1], f'Pair {i+1}',
                ha='center', va='bottom', transform=ax.transData)
        # Add speedup below pair label
        plt.text(i, -0.15 * ax.get_ylim()[1], f'Speedup: {speedups[i]:.1f}x',
                ha='center', va='bottom', transform=ax.transData)

    # Add legend
    ax.legend()

    # Add bold average speedup text in the exact center of remaining space
    plt.text(0.5, -0.22,
             f'Average Speedup: {avg_speedup:.1f}x',
             ha='center', va='center',
             fontweight='bold', fontsize=12,
             transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Adjust bottom margin to make room for labels
    plt.subplots_adjust(bottom=0.25)

    # Save the plot
    plt.savefig('execution_time_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    profile_file = "./bin/find_keypoints_profile.txt"
    plot_comparison(profile_file)