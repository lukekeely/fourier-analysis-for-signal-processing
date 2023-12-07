import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os



def generate_signal(func, T, N):
    t_values = np.linspace(0, T, N, endpoint=False)
    return t_values, [func(t) for t in t_values]

def discrete_fourier_transform(signal):
    F = np.fft.fft(signal)
    return F.real, F.imag

def save_data_and_plot(t_values, f_values, F_real, F_imag, h, title_suffix):
    folder_name = f"DFT_analysis_{title_suffix.strip('()').replace('=', '_')}"
    os.makedirs(folder_name, exist_ok=True)

    # Save data
    df = pd.DataFrame({'t': t_values, 'f(t)': f_values, 'F_real': F_real, 'F_imag': F_imag})
    print(title_suffix, df)
    df.to_csv(f"{folder_name}/{title_suffix}_data.csv", index=False)

    # Plot
    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(t_values, f_values, label='Signal')
    plt.scatter(t_values, f_values, label='Sampled points')
    plt.title(f'Signal and Sampled Points {title_suffix}')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.savefig(f"{folder_name}/{title_suffix}signal.png")
    plt.savefig(f"{folder_name}/{title_suffix}signal.pdf")
    plt.show()
    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(F_real, label='Fn,real')
    plt.plot(F_imag, label='Fn,imaginary')
    plt.title(f'Fourier Components {title_suffix}')
    plt.xlabel('n')
    plt.ylabel('Magnitude')
    plt.legend(loc='upper right', fontsize='small')
    plt.savefig(f"{folder_name}/{title_suffix}fourier_components.png")
    plt.savefig(f"{folder_name}/{title_suffix}fourier_components.pdf")
    plt.show()

sns.set_style("darkgrid", {"axes.facecolor": ".9"})
sns.set_palette("magma", n_colors=2)

# Constants
N = 128
h = 0.1
T = N * h

# Signal generation and DFT calculation for sin(0.45πt)
sin_func = lambda t: np.sin(0.45 * np.pi * t)
t_values, f_values = generate_signal(sin_func, T, N)
F_real, F_imag = discrete_fourier_transform(f_values)
save_data_and_plot(t_values, f_values, F_real, F_imag, h, "(h=0.1)")

# Ideal sampling interval calculation and analysis for omega_1 = 0.45π
h_ideal = 2 * np.pi / (0.45 * np.pi * N)
t_values_ideal, f_values_ideal = generate_signal(sin_func, N * h_ideal, N)
F_real_ideal, F_imag_ideal = discrete_fourier_transform(f_values_ideal)
save_data_and_plot(t_values_ideal, f_values_ideal, F_real_ideal, F_imag_ideal, h_ideal, f'(h={round(h_ideal,4)})')


# Additional cases with f(t) = cos(6πt), varying h
cos_func = lambda t: np.cos(6 * np.pi * t)
N = 32
hs = [0.6, 0.01]
for h in hs:
    t_values, f_values = generate_signal(cos_func, N * h, N)
    F_real, F_imag = discrete_fourier_transform(f_values)
    save_data_and_plot(t_values, f_values, F_real, F_imag, h, f"(h={h})")
