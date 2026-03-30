# GNN_PhC
The Data2_new_gnn_data can be download from this link: https://heibox.uni-heidelberg.de/f/fbf2d44f0b2443a2b8e8/?dl=1 
This is final project of Generative Neural Networks for the Sciences
Traditionally, discovering high-performing photonic devices relies on computationally
expensive physics simulations. Furthermore, the “inverse problem” of mapping a target
optical spectrum back to a physical geometry is notoriously ill-posed; because multiple
geometries can produce the same spectrum, standard deterministic models often fail by
outputting blurred, physically unbuildable averages. We addressed this through a multi
part data-driven pipeline using fabrication-constrained datasets. First, we solved the for
ward problem bytraining robust Convolutional Neural Networks (CNNs) and Multi-Layer
Perceptrons (MLPs) to near-instantaneously predict bandgaps, reflection, transmission,
and scattering losses, effectively replacing slow physics simulators. In the waveguide
focused part of the project, we trained image-based models for predicting reflection,
transmission, and loss-related optical responses from photonic-crystal geometries, and
also explored inverse models that generate candidate structures from target spectral be
havior. Second, we tackled the inverse design challenge by comparing three distinct
probabilistic and constrained approaches: a conditional Variational Autoencoder (cVAE)
for stochastic candidate generation, a Geometry Autoencoder combined with an MLP sur
rogate for gradient-based Latent-Space Optimization, and physics-constrained Denoising
Diffusion Probabilistic Models (DDPMs). These methods successfully generated valid,
high-performing structural blueprints. Crucially, we identified a “Metric Misalignment
Paradox”: optimizing models purely for mathematical accuracy (e.g., Mean Squared Er
ror) forces them to reproduce jagged, unmanufacturable simulation artifacts, whereas our
constrained latent and generative models naturally yield smooth, cleanroom-ready de
signs. Currently, our pipeline successfully proposes candidate geometries evaluated via
neural network surrogates. Validating these AI-generated structures through full-wave
electromagnetic simulations and experimental cleanroom fabrication remains the primary
focus for future work
