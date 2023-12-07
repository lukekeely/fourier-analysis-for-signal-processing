import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define integral and accuracy
a = 0  # Lower limit of integration
b = 1  # Upper limit of integration
n = 12  # Number of steps (must be even)
function_label = "f(x) = e^x"
def f(x):
    return np.exp(x)

def simpsons_rule(a, b, n):
    h = (b - a) / n
    integral = f(a) + f(b)

    for i in range(1, n):
        k = a + i * h
        if i % 2 == 0:
            integral += 2 * f(k)
        else:
            integral += 4 * f(k)

    integral = integral * h / 3
    return integral

def plot(a, b, n):
    x_values = np.linspace(a, b, 1000)
    y_values = f(x_values)
    approximation_x = np.linspace(a, b, n + 1)
    approximation_y = f(approximation_x)
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_palette("magma")
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(x_values, y_values, label=function_label)
    plt.plot(approximation_x, approximation_y, linestyle='--', label="Simpson's Rule")
    
    for i in range(1, len(approximation_x) - 1, 2):
        plt.fill_between(approximation_x[i-1:i+2], approximation_y[i-1:i+2], alpha=0.4) 
    
    plt.title(f'Simpson\'s Rule Approximation of {function_label}')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc='upper right', fontsize='small')
    folder_name = "Simpsons_Rule_Integration"
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(f"{folder_name}/integration_plot.png", format='png')
    plt.savefig(f"{folder_name}/integration_plot.pdf", format='pdf')
    plt.show()

result = simpsons_rule(a, b, n)
print("The integral result is:", result)
plot(a, b, n)
