import numpy as np
import torch as th
from icecream import ic

# It's good practice to define the device at the top
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Gate Definitions (Unchanged, but we'll move tensors to the correct device) ---

def U3_rotation(theta, phi, lambda_):
    """Rotation matrix for a single qubit."""
    cos_t = th.cos(theta / 2)
    sin_t = th.sin(theta / 2)
    # Use torch.exp for consistency
    exp_il = th.exp(1j * lambda_)
    exp_ip = th.exp(1j * phi)
    
    U = th.tensor(
        [[cos_t, -exp_il * sin_t],
         [exp_ip * sin_t, exp_ip * exp_il * cos_t]],
        dtype=th.complex128, device=device) # Move to device
    return U

def X_rotation(theta):
    """RX rotation matrix."""
    cos_t = th.cos(theta / 2)
    sin_t = th.sin(theta / 2)
    U = th.tensor(
        [[cos_t, -1j * sin_t],
         [-1j * sin_t, cos_t]],
        dtype=th.complex128, device=device) # Move to device
    return U

# --- Operator Generation (Mostly unchanged, but now accepts device) ---

def global_rotation(angles, N):
    """Apply a global rotation to the state vector."""
    # Note: angles tensor should be on the correct device
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
    
    # Pre-calculate identity matrices of powers of 2
    identities = [th.eye(2**i, dtype=th.complex128, device=device) for i in range(N)]
    
    cnot_gate_list = []
    for i in range(N - 1):
        Id1 = identities[i]
        Id2 = identities[N - i - 2]
        cnot_gate_list.append(th.kron(Id1, th.kron(cnot, Id2)))
    return cnot_gate_list

def crx_gates(N, theta=th.pi/20):
    """Return a list of controlled RX gates."""
    Id = th.eye(2, dtype=th.complex128, device=device)
    Rx = X_rotation(theta)
    # Note: block_diag needs tensors on the same device.
    crx = th.block_diag(Id, Rx) 
    
    identities = [th.eye(2**i, dtype=th.complex128, device=device) for i in range(N)]
    
    crx_gate_list = []
    for i in range(N-1):
        Id1 = identities[i]
        Id2 = identities[N-i-2]
        crx_gate_list.append(th.kron(Id1, th.kron(crx, Id2)))
    return crx_gate_list

def generate_H(N):
    """Generate the Hamiltonian for the system."""
    h = 9 / N
    # This scaling factor is unusual, but keeping it per the original code
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

# --- Batched Noisy Gate Calculation ---

def noisy_gate_batch(phi_batch, jumpOPs, adj):
    """Calculates a batch of noisy gates."""
    # phi_batch shape: (n_sim_noise, N, 1, 1)
    # jumpOPs shape: (N, 2**N, 2**N) -> add batch dim -> (1, N, 2**N, 2**N)
    jumpOPs = jumpOPs.unsqueeze(0)
    
    # Argument for matrix_exp, shape: (n_sim_noise, 2**N, 2**N)
    if adj:
        # Sum over the N dimension
        exponent_arg = th.sum(-1j * phi_batch * jumpOPs, axis=1)
    else:
        # Using .mH for Hermitian conjugate (adjoint)
        JdagJ = jumpOPs.adjoint() @ jumpOPs
        JJ = jumpOPs @ jumpOPs
        exponent_arg = th.sum(-1j * phi_batch * jumpOPs + (phi_batch**2 / 2) * (JJ - JdagJ), axis=1)
        
    # torch.matrix_exp is batched automatically!
    return th.matrix_exp(exponent_arg)

# --- OPTIMIZED Main Simulation Function ---

def calc_variance_optimized(N, L, n_sim=100, n_sim_noise=100, fast_ent=True, noise='dephasing'):
    
    # 1. PRE-CALCULATION: Generate constant operators once
    H = generate_H(N)
    ent_gates = cnot_gates(N) if fast_ent else crx_gates(N)
    jumpOPs = get_jump_operators(N, noise)
    
    adj = (noise != 'amplitude_damping')

    # Storage for results
    obs_hist = th.empty([n_sim, n_sim_noise, 3 * L * N + N + 1], dtype=th.float64, device=device)

    # Outer loop for parameter sets
    for i in range(n_sim):
        # Generate parameters for all layers at once on the correct device
        params1 = th.acos(1.0 - 2 * th.rand((L, N), device=device))
        params2 = 2 * th.pi * th.rand((L, N), device=device)
        params3 = 2 * th.pi * th.rand((L, N), device=device)
        params = th.stack([params1, params2, params3], dim=2)
        flat_params = th.cat([params1.flatten(), params2.flatten(), params3.flatten()])

        # 2. VECTORIZATION: Prepare for batch processing
        
        # Sample all noise values for the batch at once
        phi_batch = th.normal(0, np.log(N)/(N * L), size=(n_sim_noise, N, 1, 1), device=device)
        
        # Create a batch of initial states |00...0>
        # state_batch shape: [n_sim_noise, 2**N]
        state_batch = th.zeros(n_sim_noise, 2**N, dtype=th.complex128, device=device)
        state_batch[:, 0] = 1.0

        # Evolve all states in the batch through the layers
        for l in range(L):
            # Apply global rotation (same for all states in the batch)
            U = global_rotation(params[l], N)
            state_batch = state_batch @ U.T # Apply to each row vector
            
            # Apply entangling gates (same for all states in the batch)
            for gate in ent_gates:
                state_batch = state_batch @ gate.T

            # Apply noisy gate (DIFFERENT for each state in the batch)
            noise_batch = noisy_gate_batch(phi_batch, jumpOPs, adj)
            # Use batched matrix-vector multiplication
            # state_batch shape: (B, D), noise_batch shape: (B, D, D)
            # We need (B, 1, D) @ (B, D, D) -> (B, 1, D) -> squeeze to (B, D)
            state_batch = th.bmm(state_batch.unsqueeze(1), noise_batch).squeeze(1)

        # Calculate expectation value for the entire batch
        # (state_batch @ H.T) -> elementwise mul -> sum
        obs_batch = th.sum(state_batch.conj() * (state_batch @ H.T), dim=1).real
        
        # Store results
        # Broadcasting flat_params and phi_batch to fit into the obs_hist tensor
        obs_hist[i, :, :3*L*N] = flat_params
        obs_hist[i, :, 3*L*N:-1] = phi_batch.squeeze()
        obs_hist[i, :, -1] = obs_batch

    # Reshape to match original output format
    return obs_hist.reshape(n_sim * n_sim_noise, -1)