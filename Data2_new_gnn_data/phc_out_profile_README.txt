================================================================================
HDF5 FILE README
File    : phc_out_profile.h5
================================================================================

PHYSICAL CONTEXT
----------------
The structure is a Si₃N₄-on-SiO₂ PhC waveguide whose lateral width profile
varies periodically along the propagation direction (x-axis) with a period
N=10，N=20, is period number.

The FDTD sweep covers 1500–1600 nm in 50 points
(TE fundamental mode).

Port 1 (input_port)  – left end      Port 2 (output_port) – right end

S-PARAMETER PHYSICAL MEANING
-----------------------------
  S11  |S11|² – power reflected back to port 1 (back-reflection)
  S12  |S12|² – power transmitted forward from port 1 to port 2
  S21  |S21|² – power transmitted in reverse from port 2 to port 1
  S22  |S22|² – power reflected back to port 2 (back-reflection)

  By Lorentz reciprocity S12 = S21 for passive symmetric structures.
  Any difference is purely numerical (mesh, mode-overlap, finite runtime).
  simulation_error = mean|S12 − S21|  quantifies this per design.

  Energy conservation:  |S11|² + |S12|² ≤ 1   (= 1 if lossless)

SCATTERING LOSS FORMULA  (cutback method)
-----------------------------------------
  For Si₃N₄ at telecom wavelengths absorption ≈ 0, so all missing power is
  scattered out of the guided mode (radiation, substrate leakage, mode mismatch).

  Step 1 – insertion loss per excitation direction:
    IL_fwd[w] = -10 × log₁₀( S12[w] + S11[w] )   port-1 excitation
    IL_bwd[w] = -10 × log₁₀( S21[w] + S22[w] )   port-2 excitation

  Step 2 – cutback: subtract n10 from n20 to cancel facet coupling losses,
    divide by Δperiods = 10 to get per-period contribution:
    fwd_loss_per_period_dB[w] = ( IL_fwd_n20[w] − IL_fwd_n10[w] ) / 10
    bwd_loss_per_period_dB[w] = ( IL_bwd_n20[w] − IL_bwd_n10[w] ) / 10

  Step 3 – forward direction only; mask interference fringes:
    scattering_loss_per_period_dB[w] = fwd[w]   if fwd[w] > 0  (real scattering loss)
                                     = NaN       if fwd[w] ≤ 0  (Fabry-Pérot fringe, excluded)
    bwd_loss_per_period_dB is stored as a diagnostic; negative fwd values at isolated
    wavelengths are characteristic of interference, not gain.

  Step 4 – average over positive (reliable) wavelengths only:
    average_scattering_loss_dB = nanmean_λ( scattering_loss_per_period_dB )

  Positive values = loss (dB/period).  NaN where value ≤ 0 or data missing.

GLOBAL FILE ATTRIBUTES
----------------------
  wavelength_start_nm   float    Start of sweep (nm)
  wavelength_stop_nm    float    Stop  of sweep (nm)
  sample_points         int      Number of wavelength points = 50
  n_samples             int      Number of design groups = 665
  n10_*                          Attributes from outer_shape_10_period_te_rt.h5
  n20_*                          Attributes from outer_shape_20_period_te_rt.h5
  (reciprocity error stats are prefixed n10_ / n20_)

GLOBAL DATASETS
---------------
  wavelengths_nm        float32  (50,)  Wavelength axis (nm, ascending)

DESIGN GROUP STRUCTURE  /design_XXXX/
--------------------------------------
Datasets at group level:
  y_width_arrays        float32  (100,)       Waveguide width profile (µm)
  inner_hole_arrays     float32  (32, 2)      Hole polygon vertices (µm); zeros = no hole
  index_images          float32  (40, 20)     Refractive-index map at ~1550 nm (from n10)
  image_matrices        float32  (2000, 490)  Binary geometry mask (1=waveguide, from n10)
  fwd_loss_per_period_dB  float32  (50,)  Raw loss/period, port-1 excitation (may be negative)
  bwd_loss_per_period_dB  float32  (50,)  Raw loss/period, port-2 excitation (diagnostic)
  scattering_loss_per_period_dB float32 (50,)  fwd only; NaN where ≤ 0 (interference excluded)
  average_scattering_loss_dB    float32  scalar            nanmean over positive wavelengths only (dB/period)

Subgroup  S_n_10/   and   S_n_20/   (identical layout):
  s11_power             float32  (50,)  |S11|²
  s12_power             float32  (50,)  |S12|²  (forward transmission)
  s21_power             float32  (50,)  |S21|²  (reverse transmission)
  s22_power             float32  (50,)  |S22|²
  sim_time_s            float64  scalar        FDTD wall-clock time (s)
  simulation_error      float32  scalar        mean|S12 − S21|  per design

HOW TO LOAD IN PYTHON
---------------------
  import h5py, numpy as np

  with h5py.File('phc_combined.h5', 'r') as f:

      wl = f['wavelengths_nm'][:]          # (W,) nm

      # Access one design
      grp = f['design_0042']

      # Geometry
      y   = grp['y_width_arrays'][:]       # (100,) µm
      img = grp['index_images'][:]         # (40, 20)

      # S-parameters for n=10
      s12_10 = grp['S_n_10/s12_power'][:]  # (W,)  forward transmission
      s11_10 = grp['S_n_10/s11_power'][:]  # (W,)  back-reflection
      err_10 = grp['S_n_10/simulation_error'][()]  # scalar

      # S-parameters for n=20
      s12_20 = grp['S_n_20/s12_power'][:]
      s11_20 = grp['S_n_20/s11_power'][:]

      # Scattering loss per period
      loss   = grp['scattering_loss_per_period_dB'][:]  # (W,)  dB/period
      avg_L  = grp['average_scattering_loss_dB'][()]    # scalar

  # Iterate all completed designs using a generator
  with h5py.File('phc_combined.h5', 'r') as f:
      wl = f['wavelengths_nm'][:]
      for name in sorted(f.keys()):
          if not name.startswith('design_'):
              continue
          s12  = f[name]['S_n_10/s12_power'][:]
          loss = f[name]['average_scattering_loss_dB'][()]
          print(name, f'T_peak={s12.max():.3f}  scatter_loss={loss:.2f} dB/period')
