# Black Hole Dynamics Simulator

A physics-informed neural network (PINN) implementation for simulating black hole orbital dynamics and gravitational waveform generation. This project trains a neural network to learn and predict gravitational waveforms from relativistic orbits around black holes.

## Overview

This simulator models the orbital dynamics of objects around black holes using both Newtonian and relativistic frameworks, then employs a neural network to learn the complex gravitational waveforms produced by these orbits. The approach combines classical physics differential equations with modern machine learning techniques.

## Features

- **Newtonian Orbit Simulation**: Classical orbital mechanics using Newton's laws
- **Relativistic Orbit Simulation**: General relativistic orbital dynamics accounting for spacetime curvature
- **Neural Network Model**: Physics-informed neural network that learns orbital corrections
- **Gravitational Waveform Computation**: Calculates gravitational wave signals (h+ and h× polarizations) from orbital motion
- **Training Visualization**: Real-time plotting of predicted vs. true gravitational waveforms

## Physics Background

The simulator computes orbits using parameterized equations where:
- `p`: Semi-latus rectum (orbital parameter)
- `M`: Black hole mass 
- `e`: Orbital eccentricity
- `χ`: Radial phase angle
- `ϕ`: Azimuthal angle

Gravitational waveforms are derived from the second time derivative of the quadrupole moment tensor of the orbiting mass distribution.

## Installation

### Requirements

- Python 3.7+
- PyTorch
- NumPy
- SciPy
- Matplotlib

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Blackholedynamics

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch numpy scipy matplotlib
```

Alternatively, install from requirements:

```bash
pip install -r requirements.txt
```

## Usage

Run the main simulation script:

```bash
python "Black Hole Dynamics.py"
```

The script will:
1. Generate reference data using relativistic orbit equations
2. Initialize and train a neural network to predict gravitational waveforms
3. Display training progress with loss metrics every 10 epochs
4. Plot the comparison between true relativistic waveforms and NN predictions

### Example Output

```
Epoch 0 | Loss = 0.234567
Epoch 10 | Loss = 0.123456
Epoch 20 | Loss = 0.067890
...
Epoch 190 | Loss = 0.001234
```

A matplotlib window will display the gravitational waveform comparison.

## Code Structure

### Main Components

- **`newtonian_orbit()`**: Implements Newtonian orbital dynamics differential equations
- **`relativistic_orbit()`**: Implements post-Newtonian relativistic corrections
- **`NNOrbit`**: Neural network architecture (3-layer feedforward network with 32 hidden units)
- **`abstract_nn_orbit()`**: Neural network-augmented orbit integrator
- **`compute_waveform()`**: Gravitational waveform calculator from quadrupole moment
- **`soln_to_orbit()`**: Converts orbital parameters to Cartesian coordinates

### Neural Network Architecture

```
Input: χ (radial phase angle)
    ↓
Linear(1 → 32) + Tanh
    ↓
Linear(32 → 32) + Tanh
    ↓
Linear(32 → 2)
    ↓
Output: [χ_correction, ϕ_correction]
```

The network outputs multiplicative corrections to the Newtonian orbit equations to approximate relativistic effects.

## Parameters

Default simulation parameters (can be modified in the code):

```python
params = [100.0, 1.0, 0.5]  # [p, M, e]
u0 = [np.pi, 0.0]            # Initial conditions [χ₀, ϕ₀]
t_span = (0, 6e4)            # Time range
t_eval = np.linspace(*t_span, 250)  # 250 evaluation points
```

Training hyperparameters:
- Optimizer: Adam
- Learning rate: 1e-3
- Epochs: 200
- Loss function: Mean Squared Error (MSE)

## Scientific Applications

This code demonstrates:
- **Gravitational Wave Astronomy**: Simulating waveforms for LIGO/Virgo-like detectors
- **Physics-Informed ML**: Using neural networks constrained by physical laws
- **Orbital Mechanics**: Comparing Newtonian vs. relativistic dynamics
- **Differential Equation Solving**: Hybrid numerical-ML approach to ODEs

## Customization

To modify the simulation:

1. **Change orbital parameters**: Edit `params = [p, M, e]`
2. **Adjust network architecture**: Modify the `NNOrbit` class
3. **Training duration**: Change the epoch count in the training loop
4. **Learning rate**: Adjust the `lr` parameter in the optimizer

## Future Enhancements

- [ ] Support for spinning black holes (Kerr metric)
- [ ] Multi-orbit scenarios (binary systems)
- [ ] GPU acceleration for larger networks
- [ ] Export waveforms in standard formats (HDF5)
- [ ] Interactive parameter tuning interface
- [ ] Comparison with numerical relativity simulations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- Post-Newtonian orbital mechanics
- Gravitational wave theory (Einstein's quadrupole formula)
- Physics-informed neural networks (PINNs)

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Note**: This is a research/educational tool. For production gravitational wave analysis, use validated codes like LALSuite or specialized numerical relativity packages.
