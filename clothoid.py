import matplotlib.pyplot as plt
import numpy as np
from pyclothoids import Clothoid

def plan_clothoid(start_x, start_y, start_angle, end_x, end_y, end_angle):
    clothoid = Clothoid.G1Hermite(start_x, start_y, start_angle, end_x, end_y, end_angle)
    return clothoid

def plot_clothoid(clothoid: Clothoid, start, end):
    x, y = clothoid.SampleXY(100)
    plt.plot(x, y)
    plt.scatter([start[0], end[0]], [start[1], end[1]], color=['green', 'red'])
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Clothoid Path Planning')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    start = (0, 0, 0)
    end = (1, 0, -np.pi/2)
    _clothoid = plan_clothoid(*start, *end)
    plot_clothoid(_clothoid, start, end)

