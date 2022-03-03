# Filename: plots.py
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-np.pi, np.pi, 500)
y_cos, y_sin = np.tan(x), np.tanh(x)
# we can set the figure size (in inches)
plt.figure(figsize=(10, 6), dpi=80)
# calculate and set data range
plt.xlim(x.min() * 1.0, x.max() * 1.0)
plt.ylim(5, -5)
plt.plot(x, y_cos, color="b", linewidth=2.5, linestyle="-")
plt.plot(x, y_sin, color="r", linewidth=2.5, linestyle="--")
plt.title('Functions: tangens and hyperbolic tangens', fontsize=24)
plt.xlabel('x value', fontsize=20)
plt.ylabel('y result', fontsize=20)
# Add legend
plt.legend( ("tan(x)","tanh(x)"), loc='upper left',
fontsize=14 )
# Tell matplotlib to use LaTeX to render text
plt.rc('text', usetex=False)
# Set xticks values and description, modify yticks
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'],
fontsize=18 )
plt.yticks(ticks=np.arange(-1,1.1,0.5), fontsize=18)
plt.grid()
plt.show()
