import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Newtonian and Relativistic Orbit Models


def newtonian_orbit(t, u, params):
    χ, ϕ = u
    p, M, e = params
    numer = (1 + e * np.cos(χ)) ** 2
    denom = M * (p ** (3/2))
    χ_dot = numer / denom
    ϕ_dot = numer / denom
    return [χ_dot, ϕ_dot]


def relativistic_orbit(t, u, params):
    χ, ϕ = u
    p, M, e = params
    numer = (p - 2 - 2 * e * np.cos(χ)) * (1 + e * np.cos(χ))**2
    denom = np.sqrt((p - 2)**2 - 4 * e**2)
    χ_dot = numer * np.sqrt(p - 6 - 2 * e * np.cos(χ)) / (M * (p**2) * denom)
    ϕ_dot = numer / (M * (p**(3/2)) * denom)
    return [χ_dot, ϕ_dot]

class NNOrbit(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )

    def forward(self, u):
        return 1 + self.net(u)  # multiplicative correction

def abstract_nn_orbit(t, u, params, model):
    χ, ϕ = u
    p, M, e = params
    u_t = torch.tensor([[χ]], dtype=torch.float32)
    nn_out = model(u_t).detach().numpy()[0]
    numer = (1 + e * np.cos(χ)) ** 2
    denom = M * (p ** (3/2))
    χ_dot = (numer / denom) * nn_out[0]
    ϕ_dot = (numer / denom) * nn_out[1]
    return [χ_dot, ϕ_dot]

def soln_to_orbit(soln, params):
    χ, ϕ = soln
    p, M, e = params
    r = p / (1 + e * np.cos(χ))
    x = r * np.cos(ϕ)
    y = r * np.sin(ϕ)
    return x, y


def second_derivative(v, dt):
    return np.gradient(np.gradient(v, dt), dt)


def compute_waveform(dt, soln, params):
    x, y = soln_to_orbit(soln, params)
    Ixx = x**2 - (x**2 + y**2) / 3
    Iyy = y**2 - (x**2 + y**2) / 3
    Ixy = x * y

    d2Ixx = second_derivative(Ixx, dt)
    d2Iyy = second_derivative(Iyy, dt)
    d2Ixy = second_derivative(Ixy, dt)

    h_plus = 2 * (d2Ixx - d2Iyy)
    h_cross = 4 * d2Ixy
    return h_plus, h_cross

# Create data
params = [100.0, 1.0, 0.5]
u0 = [np.pi, 0.0]
t_span = (0, 6e4)
t_eval = np.linspace(*t_span, 250)
dt = t_eval[1] - t_eval[0]

sol_ref = solve_ivp(relativistic_orbit, t_span, u0, t_eval=t_eval, args=(params,))
waveform_true, _ = compute_waveform(dt, sol_ref.y, params)

# Initialize NN
model = NNOrbit()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
for epoch in range(200):
    sol_nn = solve_ivp(lambda t, u: abstract_nn_orbit(t, u, params, model),
                       t_span, u0, t_eval=t_eval)
    w_pred, _ = compute_waveform(dt, sol_nn.y, params)

    # convert to torch
    y_true = torch.tensor(waveform_true, dtype=torch.float32)
    y_pred = torch.tensor(w_pred, dtype=torch.float32)

    loss = criterion(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss = {loss.item():.6f}")

plt.plot(t_eval, waveform_true, label="Relativistic (True)")
plt.plot(t_eval, w_pred, label="NN Prediction", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Gravitational Waveform")
plt.legend()
plt.show()