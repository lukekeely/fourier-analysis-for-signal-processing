import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Constants
T = 2 * np.pi
n = 100
k_max = 30
alpha = 2

def rectangular_wave(t, w=1, alpha=2):
    tau = 2 * np.pi / (alpha * w)
    theta = (w * t) % (2 * np.pi)
    return 1 if 0 <= theta < tau else -1

def simpsons_rule(func, a, b, n):
    h = (b - a) / n
    integral = func(a) + func(b)
    for i in range(1, n):
        k = a + i * h
        integral += 2 * func(k) if i % 2 == 0 else 4 * func(k)
    return integral * h / 3

def fourier_coefficients(f, T, n, k_max, alpha):
    a0 = (2 / T) * simpsons_rule(lambda t: f(t, alpha=alpha), 0, T, n)
    ak = [(2 / (k * np.pi)) * np.sin(2 * k * np.pi / alpha) for k in range(1, k_max + 1)]
    bk = [(2 / (k * np.pi)) * (1 - np.cos(2 * k * np.pi / alpha)) for k in range(1, k_max + 1)]
    return a0, np.array(ak), np.array(bk)

def reconstruct_fourier_series(t, a0, ak, bk, w, k_max):
    series = a0 / 2
    for k in range(1, k_max + 1):
        series += ak[k-1] * np.cos(k * w * t) + bk[k-1] * np.sin(k * w * t)
    return series

a0, ak, bk = fourier_coefficients(rectangular_wave, T, n, k_max, alpha)

t_values = np.linspace(0, 2 * T, 800)
f_values = [rectangular_wave(t, w=1, alpha=alpha) for t in t_values]

folder_name = "fourier_rectangular_wave_analysis"
os.makedirs(folder_name, exist_ok=True)

term_numbers = [1, 2, 3, 5, 10, 20, 30]
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_palette("magma", n_colors=2)

for num_terms in term_numbers:
    reconstructed_values = [reconstruct_fourier_series(t, a0, ak, bk, 1, num_terms) for t in t_values]
    
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(t_values, f_values, label="Original Rectangular Wave")
    plt.plot(t_values, reconstructed_values, linestyle='--', label=f"Reconstructed with {num_terms} terms")
    plt.title(f"Rectangular Wave Reconstruction with {num_terms} Terms (alpha={alpha})")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(f"{folder_name}/rectangular_wave_reconstruction_{num_terms}_terms.png", format='png')
    plt.savefig(f"{folder_name}/rectangular_wave_reconstruction_{num_terms}_terms.pdf", format='pdf')
    plt.show()

coefficients_df = pd.DataFrame({
    'k': np.arange(1, k_max + 1),
    'ak': ak,
    'bk': bk
})
coefficients_df.to_csv(f"{folder_name}/fourier_coefficients.csv", index=False)

