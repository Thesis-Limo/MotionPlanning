import matplotlib.pyplot as plt
import numpy as np
from pyclothoids import Clothoid

def plan_clothoid(start_x, start_y, start_angle, end_x, end_y, end_angle):
    clothoid = Clothoid.G1Hermite(start_x, start_y, start_angle, end_x, end_y, end_angle)
    return clothoid

def plot_clothoid(clothoid: Clothoid):
    x, y = clothoid.SampleXY(100)
    plt.plot(x, y)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    _clothoid = plan_clothoid(0, 0, np.pi, 10, 10, np.pi)
    plot_clothoid(_clothoid)

