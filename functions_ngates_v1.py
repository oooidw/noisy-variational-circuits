import numpy as np
import torch as th
from icecream import ic

# --- Gate and Operator Definitions ---

def U3_rotation(theta, phi, lambda_, device='cpu'):
    """Rotation matrix for a single qubit (direct tensor construction)."""
    cos_t = th.cos(theta / 2)
    sin_t = th.sin(theta / 2)
    exp_il = th.exp(1j * lambda_)
    exp_ip = th.exp(1j * phi)
    
    # Initialize an empty tensor and fill its elements
    U = th.empty((2, 2), dtype=th.complex128, device=device)
    U[0, 0] = cos_t
    U[0, 1] = -exp_il * sin_t
    U[1, 0] = exp_ip * sin_t
    U[1, 1] = exp_ip * exp_il * cos_t
    return U

def global_rotation(angles, N, device='cpu'):
    """Apply a global rotation to the state vector."""
    U = U3_rotation(angles[0, 0], angles[0, 1], angles[0, 2], device=device)
    for i in range(1, N):
        U_i = U3_rotation(angles[i, 0], angles[i, 1], angles[i, 2], device=device)
        U = th.kron(U, U_i)
    return U

def crx_layer(N, theta, device='cpu'):
    """Return the operator for a layer of controlled RX gates."""
    Id = th.eye(2, dtype=th.complex128, device=device)
    cos_t = th.cos(th.tensor(theta / 2, device=device))
    sin_t = th.sin(th.tensor(theta / 2, device=device))
    Rx = th.tensor(
        [[cos_t, -1j * sin_t],
         [-1j * sin_t, cos_t]],
        dtype=th.complex128, device=device) 

    crx = th.block_diag(Id, Rx) 
    
    identities = [th.eye(2**i, dtype=th.complex128, device=device) for i in range(N)]
    crx_layer_op = th.kron(identities[0], th.kron(crx, identities[N-2]))

    for i in range(1, N-1):
        Id1 = identities[i]
        Id2 = identities[N-i-2]
        crx_layer_op = crx_layer_op @ th.kron(Id1, th.kron(crx, Id2))
    return crx_layer_op

def generate_H(N, device='cpu'):
    """Generate the Hamiltonian for the system."""
    h = 9 / N
    scale = 2**(N / 2)
    Z = th.tensor([[1, 0], [0, -1]], dtype=th.complex128, device=device)
    ZZ = th.kron(Z, Z)
    
    H = th.zeros((2**N, 2**N), dtype=th.complex128, device=device)
    identities = [th.eye(2**i, dtype=th.complex128, device=device) for i in range(N)]

    for i in range(N - 1):
        Id1 = identities[i]
        Id2 = identities[N - i - 2]
        H += th.kron(Id1, th.kron(ZZ, Id2))
    return h * scale * H

# --- Noise Model Definitions ---

def get_jump_operators(N, noise_type='dephasing', device='cpu'):
    """Pre-calculates and returns jump operators."""
    if noise_type == 'dephasing':
        jumpOP_local = th.tensor([[1, 0], [0, -1]], dtype=th.complex128, device=device)
    elif noise_type == 'bitflip':
        jumpOP_local = th.tensor([[0, 1], [1, 0]], dtype=th.complex128, device=device)
    elif noise_type == 'amplitude_damping':
        jumpOP_local = th.tensor([[0, 1], [0, 0]], dtype=th.complex128, device=device)
    
    jumpOPs = th.empty(N, 2**N, 2**N, dtype=th.complex128, device=device)
    identities = [th.eye(2**i, dtype=th.complex128, device=device) for i in range(N + 1)]

    for i in range(N):
        Id1 = identities[i]
        Id2 = identities[N - i - 1]
        jumpOPs[i] = th.kron(Id1, th.kron(jumpOP_local, Id2))
    return jumpOPs

def noisy_gate_batch(phi_batch, jumpOPs, adj):
    """Calculates a batch of noisy gates. Device is inferred from input tensors."""
    jumpOPs = jumpOPs.unsqueeze(0)
    
    if adj:
        exponent_arg = th.sum(-1j * phi_batch * jumpOPs, axis=1)
    else:
        JdagJ = jumpOPs.mH @ jumpOPs
        JJ = jumpOPs @ jumpOPs
        exponent_arg = th.sum(-1j * phi_batch * jumpOPs + (phi_batch**2 / 2) * (JJ - JdagJ), axis=1)
        
    return th.matrix_exp(exponent_arg)

# --- Main Simulation Functions ---

def calc_variance_ng(N, L, noise, theta=1, device='cpu', n_sim=100, n_sim_noise=100):
    H = generate_H(N, device=device)
    n_noise = noise[0].shape[0]
    obs_hist = th.empty([n_sim, n_sim_noise, 3 * L * N + n_noise + 1], dtype=th.float64, device=device)

    for i in range(n_sim):
        params1 = th.acos(1.0 - 2 * th.rand((L, N), device=device))
        params2 = 2 * th.pi * th.rand((L, N), device=device)
        params3 = 2 * th.pi * th.rand((L, N), device=device)
        params = th.stack([params1, params2, params3], dim=2)
        flat_params = th.cat([params1.flatten(), params2.flatten(), params3.flatten()])
        
        phi_batch = th.normal(theta/4, np.log(N)/(n_noise * L), size=(n_sim_noise, n_noise, 1, 1), device=device)
        
        # Create a batch of initial states |0...0> as COLUMN vectors
        state_batch = th.zeros(n_sim_noise, 2**N, 1, dtype=th.complex128, device=device)
        state_batch[:, 0, 0] = 1.0

        for l in range(L):
            U = global_rotation(params[l], N, device=device)
            state_batch = U @ state_batch
            
            noise_batch = noisy_gate_batch(phi_batch, noise[0], noise[1])
            state_batch = th.bmm(noise_batch, state_batch)

        # Calculate expectation <psi|H|psi> for the batch
        obs_batch = (state_batch.mH @ (H @ state_batch)).squeeze().real
        
        obs_hist[i, :, :3*L*N] = flat_params
        obs_hist[i, :, 3*L*N:-1] = phi_batch.squeeze()
        obs_hist[i, :, -1] = obs_batch

    return obs_hist

def calc_variance_pure(N, L, theta=np.pi/20, device='cpu', n_sim=100):
    H = generate_H(N, device=device)
    ent_gate = crx_layer(N, theta, device=device)
    obs_hist = th.empty([n_sim, 3 * L * N + 1], dtype=th.float64, device=device)

    for i in range(n_sim):
        params1 = th.acos(1.0 - 2 * th.rand((L, N), device=device))
        params2 = 2 * th.pi * th.rand((L, N), device=device)
        params3 = 2 * th.pi * th.rand((L, N), device=device)
        params = th.stack([params1, params2, params3], dim=2)
        flat_params = th.cat([params1.flatten(), params2.flatten(), params3.flatten()])
        
        # State is a COLUMN vector
        state = th.zeros(2**N, 1, dtype=th.complex128, device=device)
        state[0] = 1.0

        for l in range(L):
            U = global_rotation(params[l], N, device=device)
            state = U @ state
            
            state = ent_gate @ state

        # Expectation <psi|H|psi>
        obs = (state.mH @ H @ state).squeeze().real

        obs_hist[i, :3*L*N] = flat_params
        obs_hist[i, -1] = obs

    return obs_hist
