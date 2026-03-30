Overview
-Autoencoder learns a latent space for valid geometries
-ForwardNet predicts S12 spectra from geometries
Inverse design is done via latent optimization (+ small geometry correction)


Setup
-pip install torch numpy h5py matplotlib
-Place phc_out_profile.h5 in project folder and update path in main.py

Run
-python main.py


Pipeline
1. Load dataset (.h5)
2. Train Autoencoder
3. Train ForwardNet
4. Run inverse design (multiple targets, ensemble)
5. Plot + save results


Project Structure
-main.py – full pipeline
-phc_data.py – dataset loading
-geometry_ae.py – autoencoder
-forward_net.py – forward model
-latent_inverse_design.py – optimization
-plotting_utils.py – plots
-utils.py – helpers


Output
-inverse_design_results/ – normalized outputs
-inverse_design_results_physical/ – physical values


Notes
-All parameters are defined in Args (in main.py)
-Ensemble (n_starts) improves robustness
-Works well for smooth targets, struggles with sharp spectral features