import matplotlib.pyplot as plt
import numpy as np

# loading correct predictions
test_z = []

with open(f'data/testing_set.txt', 'r') as f:
    for line in f.read().split('\n'):
        if line == '':
            continue
        _, _, z = map(float, line.split(' '))
        test_z.append(z)


# loading data
data_x, data_y, data_z = [], [], []

# interpolation predictions
# filename = 'testing_set_interpolation_predictions'

# closed form reg predictions
# filename = 'testing_set_closed_form_reg_predictions'

# generalized closed form reg predictions
# filename = 'testing_set_generalized_closed_form_reg_predictions'

# neural network predictions
filename = 'testing_set_nn_predictions'

with open(f'data/{filename}.txt', 'r') as f:
    for line in f.read().split('\n'):
        if line == '':
            continue
        x, y, z = map(float, line.split(' '))
        data_x.append(y*100)
        data_y.append(-x*100)
        data_z.append(z)


# Get the colormap
cmap = plt.get_cmap('Oranges')

# Create the plot
fig, ax = plt.subplots(figsize=(9, 6))

# Add annotations and circles at each random point
for i in range(len(data_x)):
    x, y, z = data_x[i], data_y[i], data_z[i]
    prediction = z * 1000
    correct_prediction = test_z[i] * 1000
    color_ratio = (prediction - 64) / 31  # Normalize z values for the colormap
    if int(prediction) == int(correct_prediction):
        ax.add_artist(plt.Circle((x, y), 1.2, color='lightgreen', clip_on=False, zorder=5))
    ax.add_artist(plt.Circle((x, y), 1, color=cmap(color_ratio), clip_on=False, zorder=5))
    ax.text(x+0.01, y-0.1, f'{int(prediction)}', ha='center', va='center', fontsize=15, fontweight='bold', zorder=10)
    ax.text(x + 0.01, y - 1, f'{int(correct_prediction)}', ha='center', va='center', fontsize=7, fontweight='bold',
            zorder=10)

# Set labels and title
ax.set_xlabel('X-axis Sim (*100)', labelpad=10, fontname='Times New Roman')
ax.set_ylabel('Y-axis Sim (*100)', labelpad=10, fontname='Times New Roman')
ax.set_title('Z predictions from neural network model (*1000)', pad=15, fontname='Times New Roman', x=0.55)

# Set grid and axis styles
ax.grid(True, which='both', color='grey', linestyle=':', linewidth=1, zorder=0)  # Dotted grid lines
ax.set_xlim(-25, 25)
ax.set_ylim(-50, -25)
ax.tick_params(axis='both', which='both', direction='inout', length=6)

# Make axis lines dotted and light grey
for spine in ax.spines.values():
    spine.set_color('lightgrey')
    spine.set_linestyle(':')

# Add "NICO Arm" text outside the plot
plt.text(0.15, 1.05, "NICO Arm", transform=plt.gca().transAxes,
         fontsize=15, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5), fontname='Times New Roman')

# Improve readability
plt.tight_layout()

# Save or show the plot
plt.savefig('plots/nn_predictions.png')
# plt.show()
