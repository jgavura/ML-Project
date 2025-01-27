import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


calibration_grid = [[[0.30, -0.20, 0.072],
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


def get_boundary_indexes(value, coordinate):
    i1, i2 = 0, 0

    if value >= calibration_grid[-1][-1][coordinate]:
        if coordinate == 0:
            i1 = i2 = len(calibration_grid) - 1
        else:
            i1 = i2 = len(calibration_grid[0]) - 1
    elif value > calibration_grid[0][0][coordinate]:
        min_value = calibration_grid[0][0][coordinate]
        step = calibration_grid[1 - coordinate][0 + coordinate][coordinate] - min_value
        i1 = int((value - min_value) / step)
        i2 = i1 + 1

    return i1, i2


def get_middle_z(value_start, value_end, value, z_start, z_end):
    z_gap = z_end - z_start
    value_gap = value_end - value_start
    value_diff_from_start = value - value_start

    if value_gap == 0:
        ratio = 0
    else:
        ratio = value_diff_from_start / value_gap

    return z_start + ratio * z_gap


def calculate_z(x, y):
    i1, i2 = get_boundary_indexes(x, 0)
    j1, j2 = get_boundary_indexes(y, 1)

    x1_z = get_middle_z(calibration_grid[i1][j1][0],
                        calibration_grid[i2][j1][0],
                        x,
                        calibration_grid[i1][j1][2],
                        calibration_grid[i2][j1][2])

    x2_z = get_middle_z(calibration_grid[i1][j2][0],
                        calibration_grid[i2][j2][0],
                        x,
                        calibration_grid[i1][j2][2],
                        calibration_grid[i2][j2][2])

    z = get_middle_z(calibration_grid[i1][j1][1],
                     calibration_grid[i1][j2][1],
                     y,
                     x1_z,
                     x2_z)

    return z


# loading testing data and interpolate for z
data_x, data_y, data_z, data_z_interpolated = [], [], [], []

with open('data/testing_set.txt', 'r') as f:
    for line in f.read().split('\n'):
        if line == '':
            continue
        x, y, z = map(float, line.split(' '))
        data_x.append(x)
        data_y.append(y)
        data_z.append(z)
        data_z_interpolated.append(calculate_z(x, y))


# save interpolated predictions
# with open('data/testing_set_interpolation_predictions.txt', 'w') as f:
#     for i in range(len(data_x)):
#         f.write(f'{data_x[i]} {data_y[i]} {data_z_interpolated[i]}\n')


# loading normalization parameters
with open('model_mean_std.txt', 'r') as f:
    data = f.read().split('\n')[0].split(' ')
    x_mean = list(map(float, data[0].split(',')))
    x_std = list(map(float, data[1].split(',')))
    y_mean = float(data[2])
    y_std = float(data[3])


data_z, data_z_interpolated = np.array(data_z), np.array(data_z_interpolated)
data_z_standardized = (data_z - y_mean) / y_std
data_z_interpolated_standardized = (data_z_interpolated - y_mean) / y_std


mse = mean_squared_error(data_z_standardized, data_z_interpolated_standardized)
mae = mean_absolute_error(data_z_standardized, data_z_interpolated_standardized)
print("\nInterpolation loss: {} | Interpolation mae: {}".format(mse, mae))


print(calculate_z(0.35 + 0.05 * 6 / 10, -0.10 + 0.05 * 7 / 10))
