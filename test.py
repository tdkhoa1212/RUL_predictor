import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from matplotlib.pyplot import figure

figure(figsize=(8, 6), dpi=130)


#day one, the age and speed of 13 cars:
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
x = np.arange(len(y))
plt.scatter(x, y, c='blue', label="Actual RUL")

#day two, the age and speed of 15 cars:
y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
x = np.arange(len(y))
yhat = savgol_filter(y, 6, 2)
plt.plot(x, yhat, c='orange', label='Smoothed prediction',linewidth=7.0)
plt.scatter(x, y, c='red', label="Raw prediction", s=100)
plt.xlabel("Time", fontsize=20)
plt.ylabel("Percentage", fontsize=20)
plt.legend()
plt.show()