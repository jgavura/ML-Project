import matplotlib.pyplot as plt
import numpy as np

calibration_matrix = [[[0.30, -0.20, 0.072],
                     [0.30, -0.15, 0.070],
                     [0.30, -0.10, 0.067],
                     [0.30, -0.05, 0.066],
                     [0.30, 0.00, 0.065],
                     [0.30, 0.05, 0.067],
                     [0.30, 0.10, 0.070],
                     [0.30, 0.15, 0.069],
                     [0.30, 0.20, 0.071]],
                    [[0.35, -0.20, 0.077],
                     [0.35, -0.15, 0.074],
                     [0.35, -0.10, 0.073],
                     [0.35, -0.05, 0.073],
                     [0.35, 0.00, 0.071],
                     [0.35, 0.05, 0.072],
                     [0.35, 0.10, 0.073],
                     [0.35, 0.15, 0.073],
                     [0.35, 0.20, 0.077]],
                    [[0.40, -0.20, 0.083],
                     [0.40, -0.15, 0.080],
                     [0.40, -0.10, 0.076],
                     [0.40, -0.05, 0.075],
                     [0.40, 0.00, 0.075],
                     [0.40, 0.05, 0.074],
                     [0.40, 0.10, 0.077],
                     [0.40, 0.15, 0.080],
                     [0.40, 0.20, 0.083]],
                    [[0.45, -0.20, 0.091],
                     [0.45, -0.15, 0.087],
                     [0.45, -0.10, 0.085],
                     [0.45, -0.05, 0.084],
                     [0.45, 0.00, 0.084],
                     [0.45, 0.05, 0.085],
                     [0.45, 0.10, 0.085],
                     [0.45, 0.15, 0.087],
                     [0.45, 0.20, 0.095]]]

# Get the colormap
cmap = plt.get_cmap('Oranges')

# Generate data for the grid
x = np.linspace(0, 8, 9)
y = np.linspace(0, 3, 4)
x_showticks = ["-20", "-15", "-10", "-5", "0", "5", "10", "15", "20"]
y_showticks = ["-30", "-35", "-40", "-45"]

# Flatten the calibration matrix for easy plotting
values = np.array([item[2] for sublist in calibration_matrix for item in sublist])

# Create the plot
fig, ax = plt.subplots(figsize=(9, 4))
ax.set_xticks(x)
ax.set_xticklabels(x_showticks)
ax.set_yticks(y)
ax.set_yticklabels(y_showticks)
ax.grid(True, which='both', color='grey', linestyle=':', linewidth=1, zorder=0)  # Dotted grid lines

# Add annotations and circles at each grid intersection
for i in range(len(x)):
    for j in range(len(y)):
        # print(x[i], y[j])
        value = calibration_matrix[j][i][2] * 1000
        color_ratio = (value - 64) / 31
        # print(color_ratio)
        if i in [2, 3] and j in [1, 2]:
            ax.add_artist(plt.Circle((x[i], y[j]), 0.23, color='red', clip_on=False, zorder=5))
        ax.text(x[i], y[j]+0.01, f'{int(value)}', ha='center', va='center', fontsize=15, fontweight='bold', zorder=10)
        ax.add_artist(plt.Circle((x[i], y[j]), 0.2, color=cmap(color_ratio), clip_on=False, zorder=5))
        # ax.scatter(x[i], y[j], s=500, c=value, cmap='viridis', edgecolors='black')

# Set labels and title with increased labelpad
ax.set_xlabel('X-axis Sim (*100)', labelpad=5, fontname='Times New Roman')  # Adjust the value as needed
ax.set_ylabel('Y-axis Sim (*100)', labelpad=5, fontname='Times New Roman')  # Adjust the value as needed
ax.set_title('Interpolation example', pad=20, fontname='Times New Roman')  # Adjust the value as needed

# Adjust tick labels
ax.tick_params(axis='x', pad=20)  # Adjust the value as needed
ax.tick_params(axis='y', pad=20)  # Adjust the value as needed

# Make axis lines dotted and light grey
for spine in ax.spines.values():
    spine.set_color('lightgrey')
    spine.set_linestyle(':')

# Add "NICO Arm" text outside the plot
plt.text(0.20, 1.15, "NICO Arm", transform=plt.gca().transAxes,
         fontsize=15, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5), fontname='Times New Roman')

ax.add_artist(plt.Circle((2.7, 1.6), 0.1, color='red', clip_on=False, zorder=5))
ax.text(2.7, 1.6, f'x', ha='center', va='center', fontsize=10, fontweight='bold', zorder=10)
ax.add_artist(plt.Line2D((2.7, 2.7), (1.6, 1), linestyle='--'))
ax.add_artist(plt.Line2D((2.7, 2.7), (1.6, 2), linestyle='--'))
ax.add_artist(plt.Line2D((2.7, 2), (1.6, 1.6), linestyle='--'))
ax.add_artist(plt.Line2D((2.7, 3), (1.6, 1.6), linestyle='--'))

# Improve readability
plt.gca().invert_yaxis()  # Invert y-axis if desired for better grid alignment
plt.tight_layout()  # Adjust layout to fit annotations better

# plt.show()
plt.savefig('plots/interpolation_example.png')
