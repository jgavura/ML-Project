X_UPPER_LEFT_CORNER = 0.258
X_LOWER_LEFT_CORNER = 0.473     # not corner, 20 cm from upper left corner
Y_UPPER_LEFT_CORNER = -0.25
Y_UPPER_RIGHT_CORNER = 0.23

X_tablet_20cm = 280             # lower limit of x in tablet coordinates


def tab2sim(y, x):
    if y > 1920 or y < 0 or x > 1080 or x < X_tablet_20cm:
        print("Invalid tablet coordinates")
        quit()

    y_ratio = 1 - y / 1920
    x_ratio = 1 - (x - X_tablet_20cm) / (1080 - X_tablet_20cm)
    y_sim = Y_UPPER_LEFT_CORNER + y_ratio * (Y_UPPER_RIGHT_CORNER - Y_UPPER_LEFT_CORNER)
    x_sim = X_UPPER_LEFT_CORNER + x_ratio * (X_LOWER_LEFT_CORNER - X_UPPER_LEFT_CORNER)

    return x_sim, y_sim


def sim2tab(x, y):
    if x > X_LOWER_LEFT_CORNER or x < X_UPPER_LEFT_CORNER or y > Y_UPPER_RIGHT_CORNER or y < Y_UPPER_LEFT_CORNER:
        print("Invalid sim coordinates for tablet")
        quit()

    x_ratio = 1 - (x - X_UPPER_LEFT_CORNER) / (X_LOWER_LEFT_CORNER - X_UPPER_LEFT_CORNER)
    y_ratio = 1 - (y - Y_UPPER_LEFT_CORNER) / (Y_UPPER_RIGHT_CORNER - Y_UPPER_LEFT_CORNER)
    x_tab = X_tablet_20cm + x_ratio * (1080 - X_tablet_20cm)
    y_tab = y_ratio * 1920

    return y_tab, x_tab