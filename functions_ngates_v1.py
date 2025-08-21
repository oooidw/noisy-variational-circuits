import numpy as np
import torch as th

from icecream import ic


device = th.device('cuda' if th.cuda.is_available() else 'cpu')
device = 'cpu'
print(f"Using device: {device}")


def U3_rotation(theta, phi, lambda_):
    """Rotation matrix for a single qubit."""

    cos_t = th.cos(theta / 2)
    sin_t = th.sin(theta / 2)

    exp_il = th.exp(1j * lambda_)
    exp_ip = th.exp(1j * phi)
    
    U = th.tensor(
        [[cos_t, -exp_il * sin_t],
         [exp_ip * sin_t, exp_ip * exp_il * cos_t]],
        dtype=th.complex128, device=device) 
    return U


def X_rotation(theta):
    """RX rotation matrix."""
    ic(theta)

    cos_t = th.cos(theta / 2)
    sin_t = th.sin(theta / 2)
    U = th.tensor(
        [[cos_t, -1j * sin_t],
         [-1j * sin_t, cos_t]],
        dtype=th.complex128, device=device) 
    return U


def global_rotation(angles, N):
    """Apply a global rotation to the state vector."""

    U = U3_rotation(angles[0, 0], angles[0, 1], angles[0, 2])
    for i in range(1, N):
        U_i = U3_rotation(angles[i, 0], angles[i, 1], angles[i, 2])
        U = th.kron(U, U_i)
    return U


def cnot_gates(N):
    """Return a n-1 cascading CNOT list."""

    cnot = th.tensor([
        [1, 0, 0, 0], [0, 1, 0, 0],
        [0, 0, 0, 1], [0, 0, 1, 0]], dtype=th.complex128, device=device)
    
    identities = [th.eye(2**i, dtype=th.complex128, device=device) for i in range(N)]
    
    cnot_gate_list = []
    for i in range(N - 1):
        Id1 = identities[i]
        Id2 = identities[N - i - 2]
        cnot_gate_list.append(th.kron(Id1, th.kron(cnot, Id2)))
    return cnot_gate_list


def crx_layer(N, theta):
    """Return a list of controlled RX gates."""

    Id = th.eye(2, dtype=th.complex128, device=device)
    cos_t = np.cos(theta / 2)
    sin_t = np.sin(theta / 2)
    Rx = th.tensor(
        [[cos_t, -1j * sin_t],
         [-1j * sin_t, cos_t]],
        dtype=th.complex128, device=device) 

    crx = th.block_diag(Id, Rx) 
    
    identities = [th.eye(2**i, dtype=th.complex128, device=device) for i in range(N)]

    crx_layer = th.kron(identities[0], th.kron(crx, identities[N-2]))

    for i in range(1,N-1):
        Id1 = identities[i]
        Id2 = identities[N-i-2]
        crx_layer = crx_layer @ th.kron(Id1, th.kron(crx, Id2))

    return crx_layer


def generate_H(N):
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


def get_jump_operators(N, noise_type='dephasing'):
    """Pre-calculates and returns jump operators."""

    if noise_type == 'dephasing':
        jumpOP_local = th.tensor([[1, 0], [0, -1]], dtype=th.complex128, device=device) # sigma_z
    elif noise_type == 'bitflip':
        jumpOP_local = th.tensor([[0, 1], [1, 0]], dtype=th.complex128, device=device) # sigma_x
    elif noise_type == 'amplitude_damping':
        jumpOP_local = th.tensor([[0, 1], [0, 0]], dtype=th.complex128, device=device) # sigma_-
    
    jumpOPs = th.empty(N, 2**N, 2**N, dtype=th.complex128, device=device)
    identities = [th.eye(2**i, dtype=th.complex128, device=device) for i in range(N + 1)]

    for i in range(N):
        Id1 = identities[i]
        Id2 = identities[N - i - 1]
        jumpOPs[i] = th.kron(Id1, th.kron(jumpOP_local, Id2))
    return jumpOPs


def noisy_gate_batch(phi_batch, jumpOPs, adj):
    """Calculates a batch of noisy gates."""

    # phi_batch shape: (n_sim_noise, N, 1, 1)
    # jumpOPs shape: (N, 2**N, 2**N) -> add batch dim -> (1, N, 2**N, 2**N)
    jumpOPs = jumpOPs.unsqueeze(0)
    
    if adj:
        exponent_arg = th.sum(-1j * phi_batch * jumpOPs, axis=1)
    else:
        JdagJ = jumpOPs.mH @ jumpOPs
        JJ = jumpOPs @ jumpOPs
        exponent_arg = th.sum(-1j * phi_batch * jumpOPs + (phi_batch**2 / 2) * (JJ - JdagJ), axis=1)
        
    return th.matrix_exp(exponent_arg)


def calc_variance_ng(N, L, noise, n_sim=100, n_sim_noise=100):
    
    H = generate_H(N)

    obs_hist = th.empty([n_sim, n_sim_noise, 3 * L * N + N + 1], dtype=th.float64, device=device)

    n_noise = noise[0].shape()[0]


    for i in range(n_sim):
        params1 = th.acos(1.0 - 2 * th.rand((L, N), device=device))
        params2 = 2 * th.pi * th.rand((L, N), device=device)
        params3 = 2 * th.pi * th.rand((L, N), device=device)
        params = th.stack([params1, params2, params3], dim=2)
        flat_params = th.cat([params1.flatten(), params2.flatten(), params3.flatten()])
        
        # Sample all noise values for the batch at once
        phi_batch = th.normal(0, np.log(N)/(n_noise * L), size=(n_sim_noise, N, 1, 1), device=device)
        
        # Create a batch of initial states |00...0>
        state_batch = th.zeros(n_sim_noise, 2**N, dtype=th.complex128, device=device)
        state_batch[:, 0] = 1.0

        # Evolve all states in the batch through the layers
        for l in range(L):
            # Apply global rotation (same for all states in the batch)
            U = global_rotation(params[l], N)
            state_batch = state_batch @ U.T
            
            # Apply noisy gate (DIFFERENT for each state in the batch)
            noise_batch = noisy_gate_batch(phi_batch, noise[0], noise[1])
            # Use batched matrix-vector multiplication
            state_batch = th.bmm(state_batch.unsqueeze(1), noise_batch).squeeze(1)

        # Calculate expectation value for the entire batch
        obs_batch = th.sum(state_batch.conj() * (state_batch @ H.T), dim=1).real
        ic(obs_batch.shape)
        
        # Store results
        obs_hist[i, :, :3*L*N] = flat_params
        obs_hist[i, :, 3*L*N:-1] = phi_batch.squeeze()
        obs_hist[i, :, -1] = obs_batch

    return obs_hist.reshape(n_sim * n_sim_noise, -1)


def calc_variance_pure(N, L, theta=np.pi/20, n_sim=100):
    
    H = generate_H(N)

    ent_gate = crx_layer(N, theta)

    obs_hist = th.empty([n_sim, 3 * L * N + 1], dtype=th.float64, device=device)


    for i in range(n_sim):
        params1 = th.acos(1.0 - 2 * th.rand((L, N), device=device))
        params2 = 2 * th.pi * th.rand((L, N), device=device)
        params3 = 2 * th.pi * th.rand((L, N), device=device)
        params = th.stack([params1, params2, params3], dim=2)
        flat_params = th.cat([params1.flatten(), params2.flatten(), params3.flatten()])
        
        # Create a batch of initial states |00...0>
        state = th.zeros(2**N, dtype=th.complex128, device=device)
        state[0] = 1.0

        # Evolve all states in the batch through the layers
        for l in range(L):
            # Apply global rotation (same for all states in the batch)
            U = global_rotation(params[l], N)
            state = U @ state
            
            # Apply entangling gates (CRX) to the batched states
            state = ent_gate @ state

        # Calculate expectation value for the entire batch
        obs = th.vdot(state, H @ state).real

        # Store results
        obs_hist[i, :3*L*N] = flat_params
        obs_hist[i, -1] = obs

    return obs_hist
