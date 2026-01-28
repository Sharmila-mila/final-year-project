import matplotlib.pyplot as plt
from finite_size import finite_size_skr

# Asymptotic SKR from your AI + adaptive QKD system
asymptotic_skr = 0.45   # example value

# Different finite signal counts
signal_counts = [1e4, 1e5, 1e6, 1e7, 1e8]
finite_skr_values = []

for N in signal_counts:
    finite_skr = finite_size_skr(asymptotic_skr, N)
    finite_skr_values.append(finite_skr)
    print(f"N = {int(N):>8} | Asymptotic SKR = {asymptotic_skr:.3f} | Finite SKR = {finite_skr:.3f}")

# Plot comparison
plt.figure()
plt.plot(signal_counts, finite_skr_values, marker='o', label="Finite-size SKR")
plt.axhline(y=asymptotic_skr, linestyle='--', label="Asymptotic SKR")
plt.xscale("log")
plt.xlabel("Number of Signals (log scale)")
plt.ylabel("Secret Key Rate (SKR)")
plt.title("Finite-Size vs Asymptotic SKR")
plt.legend()
plt.grid(True)
plt.show()
