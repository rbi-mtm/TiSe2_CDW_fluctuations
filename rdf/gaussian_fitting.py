import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

temps = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]

r_min_fit = 3.0
r_max_fit = 4.0

use_normalized = False

def normalized_gaussian(x, mu, sigma):
    """Gaussian normalized to integrate to 1"""
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-(x - mu)**2 / (2*sigma**2))

def unnormalized_gaussian(x, A, mu, sigma):
    """Gaussian with free amplitude"""
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))

prev_guess = None

for temp in temps:
    distance_file = "./data/dists_Ti_{}K.npy".format(temp)  
    rdf_file = "./data/rdf_Ti_{}K.npy".format(temp)     

    r = np.load(distance_file)
    g_r = np.load(rdf_file)

    mask = (r >= r_min_fit) & (r <= r_max_fit)
    r_fit = r[mask]
    g_fit = g_r[mask]

    g_r_smooth = gaussian_filter1d(g_fit, sigma=3)

    if prev_guess is None:
        mu_guess = r_fit[np.argmax(g_fit)]
        sigma_guess = 0.2
        if not use_normalized:
            A_guess = g_fit.max()
            p0 = [A_guess, mu_guess, sigma_guess]
        else:
            p0 = [mu_guess, sigma_guess]
    else:
        p0 = prev_guess

    if use_normalized:
        popt, pcov = curve_fit(normalized_gaussian, r_fit, g_fit, p0=p0)
        mu_fit, sigma_fit = popt
        perr = np.sqrt(np.diag(pcov))
        print(f"Temperature {temp} K - Normalized Gaussian fit:")
        print(f"mu    = {mu_fit:.4f} ± {perr[0]:.4f}")
        print(f"sigma = {sigma_fit:.4f} ± {perr[1]:.4f}")
        g_fit_curve = normalized_gaussian(np.linspace(r_fit.min(), r_fit.max(), len(r_fit)), *popt)
        prev_guess = popt
    else:
        popt, pcov = curve_fit(unnormalized_gaussian, r_fit, g_fit, p0=p0)
        A_fit, mu_fit, sigma_fit = popt
        perr = np.sqrt(np.diag(pcov))
        print(f"Temperature {temp} K - Un-normalized Gaussian fit:")
        print(f"A     = {A_fit:.4f} ± {perr[0]:.4f}")
        print(f"mu    = {mu_fit:.4f} ± {perr[1]:.4f}")
        print(f"sigma = {sigma_fit:.4f} ± {perr[2]:.4f}")
        g_fit_curve = unnormalized_gaussian(np.linspace(r_fit.min(), r_fit.max(), len(r_fit)), *popt)
        prev_guess = popt

    mu_rdf = np.sum(r_fit * g_fit) / np.sum(g_fit)
    mu_smooth = np.sum(r_fit * g_r_smooth) / np.sum(g_r_smooth)

    r_plot = np.linspace(r_fit.min(), r_fit.max(), len(r_fit))
    plt.figure(figsize=(6,4))
    plt.plot(r_fit, g_fit, color='blue', label=f"RDF data {r_min_fit}-{r_max_fit} Å")
    plt.plot(r_fit, g_r_smooth, color='red', label=f"Smoothed RDF data {r_min_fit}-{r_max_fit} Å")
    plt.plot(r_plot, g_fit_curve, color='green', label="Gaussian fit")
    #plt.axvline(mu_rdf, ls='--', color='blue', label="RDF mean")
    #plt.axvline(mu_smooth, ls='--', lw=2, color='red', label="Smoothed RDF mean")
    #plt.axvline(mu_fit, ls='--', color='green', label="Gaussian mu")
    plt.xlabel("r [Å]")
    plt.ylabel("g(r)")
    plt.xlim(3, 4) #(3.3, 3.8) #(r_min_fit, r_max_fit)
    plt.ylim(0)
    title_type = "Normalized" if use_normalized else "Un-normalized"
    plt.title(f"{title_type} Gaussian Fit - {temp} K")
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig("./plots/gaussian_fit_Ti_{}K.png".format(temp))
    plt.close()
