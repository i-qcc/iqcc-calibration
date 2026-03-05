"""
Thermal photon number at MXC through cascaded attenuation stages.
"""
import numpy as np
import matplotlib.pyplot as plt


class system:
    """Thermal photon number at MXC through cascaded attenuation stages."""
    
    HBAR = 1.05457e-34   # Planck constant over 2pi (J·s)
    KB = 1.38065e-23     # Boltzmann constant (J/K)

    STAGES = [
        ('300K', 300), ('50K', 50), ('4K', 4),
        ('ST', 0.8), ('CP', 0.1), ('MXC', 0.01),
    ]
    # T1=np.array([22.9,23.1,19,28.1,26.2])*1e-6 # Galil Arbel B
    def __init__(self, freq_hz=7e9, opx_temp=75000,
                 attenuation=None, T1=None, T2=None, chi=None, kappa=None):
        """
        freq_hz: cavity frequency (Hz)
        opx_temp: source temperature (75000 for OPX, 300 for 300K only)
        attenuation: dict of stage -> dB, e.g. {'300K':10,'50K':2,...}
        chi: dispersive shift per qubit in Hz (from state), array or list
        kappa: resonator linewidth per qubit in Hz (from state), array or list
        """
        self.w = 2 * np.pi * freq_hz
        self.opx_temp = opx_temp
        self.attenuation = attenuation or {
            '300K': 10, '50K': 2, '4K': 22, 'ST': 2, 'CP': 22, 'MXC': 22
        }
        self.T1 = np.asarray(T1) if T1 is not None else None
        self.T2 = np.asarray(T2) if T2 is not None else None
        self.chi_hz = np.asarray(chi) if chi is not None else None
        self.kappa_hz = np.asarray(kappa) if kappa is not None else None

    def _t_phi_coeff_hz(self):
        """Dephasing coefficient in Hz from thermal photons. Mean over qubits. chi, kappa in Hz."""
        if self.chi_hz is None or self.kappa_hz is None:
            raise ValueError("chi and kappa must be provided from state (e.g. machine.qubits[q].chi, .resonator.kappa)")
        chi_rad = 2 * np.pi * np.abs(self.chi_hz)
        kappa_rad = 2 * np.pi * self.kappa_hz
        coeff = 4 * chi_rad**2 * kappa_rad / (kappa_rad**2 + 4 * chi_rad**2)
        return np.mean(coeff)

    def n_BE(self, T):
        return 1 / (np.exp(self.HBAR * self.w / (self.KB * T)) - 1)

    def n_i(self, n_prev, A_dB, T):
        """Cascade: n = n_prev/A + ((A-1)/A)*n_BE(T)."""
        A = 10 ** (A_dB / 10)
        return n_prev / A + ((A - 1) / A) * self.n_BE(T)

    def n_mxc(self, **atten_dB):
        """Compute n_mxc. Override any stage with 300K=..., 50K=..., etc."""
        a = {**self.attenuation, **atten_dB}
        keys = ['300K', '50K', '4K', 'ST', 'CP', 'MXC']
        atten = [a[k] for k in keys]

        # First stage (n_i mixing at first stage)
        n_src = self.n_BE(self.opx_temp)
        n = self.n_i(n_src, atten[0], self.STAGES[0][1])

        # Remaining stages
        for i in range(1, len(self.STAGES)):
            n = self.n_i(n, atten[i], self.STAGES[i][1])
        return n

    def n_mxc_stage(self, var_range=(0, 50), n_pts=100, figsize=(4, 3),title=None):
        """Plot n_mxc vs attenuation for each stage (one stage varied).
        scatter_atten: optional list of (stage_idx, atten_dB) to add scatter points.
        title: optional custom plot title.
        """
        var = np.linspace(*var_range, n_pts)
        plt.figure(figsize=figsize)
        for i, (name, _) in enumerate(self.STAGES):
            atten = {self.STAGES[j][0]: var if j == i else self.attenuation[self.STAGES[j][0]]
                     for j in range(len(self.STAGES))}
            label = name
            plt.plot(var, self.n_mxc(**atten), label=label)
        
        plt.hlines(self.n_mxc(**self.attenuation),var_range[0],var_range[1], color='r', linestyle='--', label='default')
        plt.legend(fontsize=7)
        plt.yscale('log')
        plt.xlabel('Attenuation (dB)')
        plt.ylabel(r'$n_{\mathrm{mxc}}$')
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.title(title or r'$n_{\mathrm{mxc}}$ (Attenuation)')
        plt.tight_layout()
        plt.show()

    def n_with_extra_atten(self, extra_atten, stage_name=None, figsize=(4, 3)):
        """Bar plot of n_mxc when extra_atten dB added to each stage.
        stage_name: If None (default), shows all 6 stages. Else '300K', '50K', etc. for single stage.
        """
        if stage_name is not None:
            if stage_name not in [s[0] for s in self.STAGES]:
                raise ValueError(f"stage_name must be one of {[s[0] for s in self.STAGES]}")
            stages = [stage_name]
            atten = {s[0]: self.attenuation[s[0]] + (extra_atten if s[0] == stage_name else 0)
                     for s in self.STAGES}
            y = [self.n_mxc(**atten)]
        else:
            # All 6 stages: effect of adding extra_atten to each stage
            y = []
            for i in range(len(self.STAGES)):
                atten = {self.STAGES[j][0]: self.attenuation[self.STAGES[j][0]] + (extra_atten if j == i else 0)
                         for j in range(len(self.STAGES))}
                y.append(self.n_mxc(**atten))
            stages = [s[0] for s in self.STAGES]
        plt.figure(figsize=figsize)
        plt.bar(stages, y)
        plt.xlabel('Stage')
        plt.ylabel(r'$n_{\mathrm{mxc}}$')
        plt.ylim(1e-7, 1e-3)
        plt.yscale('log')
        plt.title(r'$n_{\mathrm{mxc}}$' + ' (extra{}dB)'.format(extra_atten))
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
  
    def T2_analysis(self, n, title=None):  # n: dimensionless (photon number); chi, kappa from state (Hz)
        T1_mean = np.mean(self.T1)
        T1_2 = 2 * T1_mean
        coeff_hz = self._t_phi_coeff_hz()
        t_phi_inv = coeff_hz * n  # 1/s
        t_phi_inv = t_phi_inv * 1e-6   # 1/s → 1/µs (1 s = 1e6 µs)
        T2e_inv = 1/T1_2 + t_phi_inv   # 1/µs (T1 in µs)
        T2e_theory = 1/T2e_inv         # µs
        # n where T2e_theory first drops below 99% of 2*T1 (1% difference)
        threshold = 0.99 * T1_2
        if T2e_theory[-1] <= threshold <= T2e_theory[0]:
            n_1pct = np.interp(threshold, T2e_theory[::-1], n[::-1])
        else:
            n_1pct = None
        # n where T2e_theory and T2 measured meet
        T2_measured = np.mean(self.T2)
        if T2e_theory[-1] <= T2_measured <= T2e_theory[0]:
            n_meet = np.interp(T2_measured, T2e_theory[::-1], n[::-1])
        else:
            n_meet = None

        plt.semilogx(n, T2e_theory)#, label='T2(T1,n)')
        plt.hlines(T1_2, xmin=n[0], xmax=n[-1],color='red', label=f'{2*T1_mean:.2f}µs:2T1 limit', linestyle='--')
        if n_1pct is not None:
            plt.axvline(n_1pct, color='red', linestyle='--', label=f'n (2T1 1% diff) = {n_1pct:.2e}')
        plt.hlines(T2_measured, xmin=n[0], color='black', xmax=n[-1], label=f'{T2_measured:.2f}µs:measured', linestyle='--')       
        if n_meet is not None:
            plt.axvline(n_meet, color='black', linestyle='--', label=f'n (T2 meas) = {n_meet:.2e}')
        plt.legend()
        plt.title(f'{title}\nT2(T1,n)' if title else 'T2(T1,n)')
        plt.xlabel('n_mxc')
        plt.ylabel('T2 (µs)')
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
        return n_1pct, n_meet