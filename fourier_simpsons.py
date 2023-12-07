import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Constants
T = 2 * np.pi  # Period
n = 100  # Steps
k_max = 5  # Number of Fourier coefficients

# Cases
functions = {
    "sin(wt)": lambda t, w=1: np.sin(w * t),
    "cos(wt)+3cos(2wt)-4cos(3wt)": lambda t, w=1: np.cos(w * t) + 3 * np.cos(2 * w * t) - 4 * np.cos(3 * w * t),
    "sin(wt)+3sin(3wt)+5sin(5wt)": lambda t, w=1: np.sin(w * t) + 3 * np.sin(3 * w * t) + 5 * np.sin(5 * w * t),
    "sin(wt)+2cos(3wt)+3sin(5wt)": lambda t, w=1: np.sin(w * t) + 2 * np.cos(3 * w * t) + 3 * np.sin(5 * w * t)
}

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_palette("magma", n_colors=2)

def simpsons_rule(func, a, b, n):
    h = (b - a) / n
    integral = func(a) + func(b)
    for i in range(1, n):
        k = a + i*h
        if i % 2 == 0:
            integral += 2 * func(k)
        else:
            integral += 4 * func(k)
    return integral * h / 3

def fourier_coefficients(f, T, n, k_max):
    omega = 2 * np.pi / T
    a0 = simpsons_rule(f, 0, T, n) / T
    ak = np.zeros(k_max)
    bk = np.zeros(k_max)
    for k in range(1, k_max + 1):
        ak_func = lambda t: f(t) * np.cos(k * omega * t)
        bk_func = lambda t: f(t) * np.sin(k * omega * t)
        ak[k-1] = 2 * simpsons_rule(ak_func, 0, T, n) / T
        bk[k-1] = 2 * simpsons_rule(bk_func, 0, T, n) / T
    return a0, ak, bk

def fourier_series(t, a0, ak, bk, T):
    omega = 2 * np.pi / T
    series = a0 / 2
    for k in range(1, len(ak) + 1):
        series += ak[k-1] * np.cos(k * omega * t) + bk[k-1] * np.sin(k * omega * t)
    return series


for function_name, function in functions.items():
    folder_name = f"fourier_simpsons_{function_name}"
    os.makedirs(folder_name, exist_ok=True)
    a0, ak, bk = fourier_coefficients(function, T, n, k_max)

    t_values = np.linspace(0, T, 400)
    f_values = [function(t) for t in t_values]
    fourier_values = [fourier_series(t, a0, ak, bk, T) for t in t_values]
    df = pd.DataFrame({'t': t_values, 'f(t)': f_values, 'Fourier Approximation': fourier_values})
    df.to_csv(f"{folder_name}/fourier_data_{function_name}.csv", index=False)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(t_values, f_values, label=f"{function_name}")
    plt.plot(t_values, fourier_values, linestyle='--', label="Fourier Series Approximation")
    plt.scatter(np.arange(1, k_max + 1), ak, label='ak coefficients')
    plt.scatter(np.arange(1, k_max + 1), bk, label='bk coefficients')
    plt.title(f"Fourier Series of {function_name}")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(f"{folder_name}/fourier_plot_{function_name}.png", format='png')
    plt.savefig(f"{folder_name}/fourier_plot_{function_name}.pdf", format='pdf')
    plt.show()
