import matplotlib.pyplot as plt
from coords_conversion import sim2tab

targets = []

# all data points
# filenames = ['grid_targets_z_confirmed', 'random_targets_z_confirmed']

# testing data points
# filenames = ['training_set']

# validation data points
# filenames = ['validation_set']

# testing data points
filenames = ['testing_set']


for name in filenames:
    file_name = f'data/{name}.txt'
    with open(file_name, 'r') as f:
        for line in f.read().split('\n'):
            if line == '':
                continue
            x, y = map(float, line.split(' ')[:2])
            x, y = sim2tab(x, y)
            targets.append((1920 - x, y))

# Unpack target positions
target_x, target_y = zip(*targets)

# Create plot
fig, ax = plt.subplots(figsize=(19.2, 7.8))

ax.set_axisbelow(True)

# Plot targets
plt.scatter(target_x, target_y, color='red', label='Targets', marker='s', zorder=5)

# Set axis limits
plt.xlim(0, 1920)
plt.ylim(300, 1080)

# Add nico hand base
plt.text(0.3, 1.08, "NICO\nArm", transform=plt.gca().transAxes,
         fontsize=20, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

# Set the font size of ticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Add labels and legend
plt.xlabel('X-axis (px)', fontsize=20)
plt.ylabel('Y-axis (px)', fontsize=20)
plt.title('Testing set data points', fontsize=25, pad=20, x=0.5)
plt.legend(fontsize=15)
plt.grid(True)

# Show plot
# plt.show()
plt.savefig('plots/testing_set.png')
