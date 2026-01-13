# %%
import numpy as np
import matplotlib.pyplot as plt
from pandas import tseries

# %% simulating freqeuncy tracking with Adaptive step Bayesian frequency estimation with adaptive step size

alpha = 0.01
beta = 0.8
f = 1.e6
df = 0.0e6
T = 5000e-9

n_shots = 50


mus = []
sigmas = []
fs = []
mu = 0.0e6
for n_reps in range(500):
    mu = 0.0e6
    sigma = 0.1e6
    f = f + 0.03*np.random.randn()*1e6
    for i in range(n_shots):
        tau = (np.sqrt(16*np.pi**2*sigma**2) - 1/T)/(8 * np.pi**2 * sigma**2)
        df = 1/(4*tau) - mu
        ps = np.array([0.5 + m /2 *(alpha + beta * np.exp(-tau / T) * np.cos(2*np.pi*(f+df)*tau)) for m in [-1, 1]])
        ps = ps / np.sum(ps)
        outcome = np.random.choice([-1, 1], p=ps)
        exponential = np.exp(-tau / T - 2*np.pi**2 * sigma**2 * tau**2)
        mu = mu - (2 * np.pi * outcome * beta * sigma**2 * tau * exponential) / (1 + alpha * outcome)
        sigma = np.sqrt(sigma**2 - 1.0*(4 * np.pi**2 * beta**2 * sigma**4 * tau**2 * exponential**2) / (1 + alpha * outcome)**2)
    mus.append(mu)
    sigmas.append(sigma)
    fs.append(f)
    f = f - mu
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axs[0].plot(mus, label='mu', color='tab:blue')
axs[0].plot(fs, label='f', color='tab:red')
axs[0].set_ylabel('mu')
axs[0].legend()

axs[1].plot(sigmas, label='sigma', color='tab:orange')
axs[1].set_ylabel('sigma')
axs[1].set_xlabel('Iteration')
axs[1].legend()

plt.tight_layout()
plt.show()

# %% simulation of non-adaptive step Bayesian frequency estimation with fixed step size

alpha = 0.0
beta = 1.0

detuning = 1.5e6
df = 0
T = 5000e-9

f_min = 0.5e6
f_max = 4.0e6
f_step = 0.01e6

t_min = 10e-9
t_max = 6000e-9
t_step = 50e-9

freqs = np.arange(f_min, f_max, f_step)
ts = np.arange(t_min, t_max, t_step)
Pfs = np.ones(len(freqs)) / len(freqs)

for t in ts:
    ps = np.array([0.5 + m /2 *(alpha + beta * np.exp(-t / T) * np.cos(2*np.pi*(detuning+df)*t)) for m in [-1, 1]])
    ps = ps / np.sum(ps)
    outcome = np.random.choice([-1, 1], p=ps)
    for i, f in enumerate(freqs):
        Pfs[i] = Pfs[i] * (0.5 + outcome * (alpha + beta * np.exp(-t / T) * np.cos(2*np.pi*(f+df)*t)))
    Pfs = Pfs / np.sum(Pfs)
    
fig, axs = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
axs.plot(freqs, Pfs, label='outcome', color='tab:blue')
axs.set_ylabel('P(f)')
axs.set_xlabel('frequency (MHz)')
axs.legend()
plt.tight_layout()
plt.show()

estimated_frequency = np.sum(freqs * Pfs)
print(f'Estimated frequency: {estimated_frequency/1e6} MHz')

# %%
# %% simulation of non-adaptive step Bayesian phase estimation with fixed step size

alpha = 0.0
beta = 0.8
cz_phase = 0.5

# phses estimation axis
phase_min = 0.0
phase_max = 1
phase_step = 0.05

# phase rotation axis
phase_rotation_min = 0.0
phase_rotation_max = 1
phase_rotation_step = 0.1


phases = np.arange(phase_min, phase_max, phase_step)
phase_rotations = np.arange(phase_rotation_min, phase_rotation_max, phase_rotation_step)
Pps = np.ones(len(phases)) / len(phases)

for phase_rotation in phase_rotations:
    ps = np.array([0.5 + m /2 *(alpha + beta * np.cos(2*np.pi* (cz_phase + phase_rotation))) for m in [-1, 1]])
    ps = ps / np.sum(ps)
    outcome = np.random.choice([-1, 1], p=ps)
    for i, phase in enumerate(phases):
        Pps[i] = Pps[i] * (0.5 + outcome/2 * (alpha + beta * np.cos(2*np.pi* (phase + phase_rotation))))
    Pps = Pps / np.sum(Pps)
    
fig, axs = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
axs.plot(phases, Pps, label='outcome', color='tab:blue')
axs.set_ylabel('P(phase)')
axs.set_xlabel('phase')
axs.legend()
plt.tight_layout()
plt.show()

estimated_phase = np.sum(phases * Pps)
print(f'Estimated phase: {estimated_phase} rad')

# %% sweet spot flux estimation 
# we model a flux scan of ramsey seuqence with given duration and detuning. The aim of the estimation is to find the sweet spot flux.

# qubit parameters
quad_term = 10e9 #Hz/V**2
flux_sweet_spot = 0.003 #V
qubit_frequency = 5.0e9 #Hz
alpha = 0.0
beta = 1.0
T = 100000e-9
detuning = -1.0e6 #Hz

# flux scan parameters
duration = 200e-9

flux_scan_range = 20e-3
flux_scan_step = 1e-3
fluxes_scan_vector = np.arange(-flux_scan_range, flux_scan_range, flux_scan_step)

# estimation parmetrs
flux_estimation_min = -7e-3
flux_estimation_max = 7e-3
flux_estimation_step = 0.1e-3
fluxes_estimation_vector = np.arange(flux_estimation_min, flux_estimation_max, flux_estimation_step)

def frequency_from_flux(flux, flux_sweet_spot, quad_term, detuning):
    return quad_term * (flux - flux_sweet_spot)**2 + detuning

def model(flux, duration, detuning, quad_term, alpha, beta, T):
    frequency = frequency_from_flux(flux, flux_sweet_spot, quad_term, detuning)
    probability = [0.5 + m /2 *(alpha + beta * np.exp(-duration / T) * np.cos(2*np.pi*frequency*duration)) for m in [-1, 1]]
    probability = probability / np.sum(probability)
    return probability

def single_shot_simulation(flux, duration, detuning, quad_term, alpha, beta, T):
    probability = model(flux, duration, detuning, quad_term, alpha, beta, T)
    outcome = np.random.choice([-1, 1], p=probability)
    return outcome

def bayesian_update_probability_vector(probability, outcome, flux_estimation_vector, flux_sweet_spot, quad_term, detuning, alpha, beta, T):
    for i, p in enumerate(probability):
        probability[i] = probability[i] * (0.5 + outcome * (alpha + beta * np.exp(-duration / T) * np.cos(2*np.pi*(frequency_from_flux(flux_estimation_vector[i], flux_sweet_spot, quad_term, detuning*0))*duration)))
    probability = probability / np.sum(probability)
    return probability

def initialize_probability_vector(fluxes_estimation_vector, prev_sweet_spot):
    # Initialize the prior as a Gaussian centered at prev_sweet_spot
    sigma = 2e-3  # You can adjust this value for desired prior width
    probability = np.exp(-0.5 * ((fluxes_estimation_vector - prev_sweet_spot) / sigma) ** 2)
    probability = probability / np.sum(probability)
    return probability

def bayesian_estimation_loop(fluxes_scan_vector, fluxes_estimation_vector, duration, detuning, quad_term, alpha, beta, T):
    probability = initialize_probability_vector(fluxes_estimation_vector, flux_sweet_spot+0.2e-3)
    for flux in fluxes_scan_vector:
        outcome = single_shot_simulation(flux, duration, detuning, quad_term, alpha, beta, T)
        probability = bayesian_update_probability_vector(probability, outcome, fluxes_estimation_vector, flux_sweet_spot, quad_term, detuning, alpha, beta, T)
    return probability

probability = bayesian_estimation_loop(fluxes_scan_vector, fluxes_estimation_vector, duration, detuning, quad_term, alpha, beta, T)
print(f'Estimated flux: {fluxes_estimation_vector[np.argmax(probability)]} V')
import matplotlib.pyplot as plt

# Plot the probability histogram
plt.figure(figsize=(8, 5))
plt.bar(fluxes_estimation_vector * 1e3, probability, width=(flux_estimation_step * 1e3), alpha=0.7, label='Posterior Probability')
plt.xlabel('Flux [mV]')
plt.ylabel('Probability')
plt.title('Bayesian Estimation Posterior')

# Compute mean and standard deviation (sigma) of the flux estimate
mean_flux = np.sum(fluxes_estimation_vector * probability)
sigma_flux = np.sqrt(np.sum(((fluxes_estimation_vector - mean_flux) ** 2) * probability))

# Plot vertical lines for mean and 1 sigma
plt.axvline(mean_flux * 1e3, color='r', linestyle='--', label=f"Mean = {mean_flux*1e3:.3f} mV")
plt.axvline((mean_flux + sigma_flux) * 1e3, color='g', linestyle=':', label=f"+1σ = {sigma_flux*1e3:.3f} mV")
plt.axvline((mean_flux - sigma_flux) * 1e3, color='g', linestyle=':', label=f"-1σ")
plt.legend()
plt.tight_layout()
plt.show()

print(f"Mean flux: {mean_flux*1e3:.3f} mV")
print(f"Sigma (std dev): {sigma_flux*1e3:.3f} mV")

# %%
plt.figure(figsize=(8, 5))
frequencies = [frequency_from_flux(flux, flux_sweet_spot, quad_term, detuning) for flux in fluxes_scan_vector]
plt.plot(fluxes_scan_vector * 1e3, frequencies, '-o', label='Frequency vs. Flux')
plt.xlabel('Flux [mV]')
plt.ylabel('Frequency [arb. units]')
plt.title('Frequency vs. Flux')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot the model probability as a function of flux (using the model function)
model_probabilities = [model(flux, duration, detuning, quad_term, alpha, beta, T)
                       for flux in fluxes_scan_vector]

plt.figure(figsize=(8, 5))
plt.plot(fluxes_scan_vector * 1e3, model_probabilities, marker='o', linestyle='-', label='Model Probability')
plt.xlabel('Flux [mV]')
plt.ylabel('Model Probability')
plt.title('Model Probability vs. Flux')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()





# %%
