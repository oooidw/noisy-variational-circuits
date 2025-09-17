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
    
    L = theta.shape[0]

    # Initialize an empty tensor and fill its elements
    U = th.empty((L,2, 2), dtype=th.complex128, device=device)
    U[:, 0, 0] = cos_t
    U[:, 0, 1] = -exp_il * sin_t
    U[:, 1, 0] = exp_ip * sin_t
    U[:, 1, 1] = exp_ip * exp_il * cos_t
    return U

def global_rotation(angles, N, device='cpu'):
    """Apply a global rotation to the state vector."""
    U = U3_rotation(angles[:, 0, 0], angles[:, 0, 1], angles[:, 0, 2], device=device)
    for i in range(1, N):
        U_i = U3_rotation(angles[:, i, 0], angles[:, i, 1], angles[:, i, 2], device=device)
        U = _kron_batched(U, U_i)
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
    
    Ids = [th.eye(2**i, dtype=th.complex128, device=device) for i in range(N)]

    crx_layer_op = th.kron(crx, Ids[N-2])
    
    for i in range(1, N-1):
        crx_layer_op = th.kron(Ids[i], th.kron(crx, Ids[N-i-2])) @ crx_layer_op
    return crx_layer_op

def _kron_batched(A, B):
    """Calculate the tensor product in a batched way"""
    size1 = A.shape[-2:]
    size2 = B.shape[-2:]

    # 1. Use einsum with ellipsis to compute the outer product for each batch element
    intermediate = th.einsum('...ij,...kl->...ikjl', A, B)

    # 2. Reshape to get the final Kronecker product form
    batch_dims = A.shape[:-2]
    res = intermediate.reshape(*batch_dims, size1[0]*size2[0], size1[1]*size2[1])

    return res

def build_crx_layer_operator_batch(N, L, phi_batch, device='cpu'):
    """
    Constructs a batch of CRX layer operators.
    phi_batch has shape (batch_size, N-1).
    Returns a tensor of shape (batch_size, 2**N, 2**N).
    """
    batch_size = phi_batch.shape[0]
    
    # Pre-calculate identities to reuse
    identities = [th.eye(2**i, dtype=th.complex128, device=device).repeat(batch_size, L, 1, 1) for i in range(N + 1)]

    # Build the full layer operator for this single simulation
    # This logic is identical to the build_crx_layer_operator function
    cos_t = th.cos(phi_batch / 2)
    sin_t = th.sin(phi_batch / 2)
    
    crx_gates = th.zeros(batch_size, N-1, L, 4, 4, dtype=th.complex128, device=device)
    crx_gates[:, :, :, 0, 0] = 1
    crx_gates[:, :, :, 1, 1] = 1
    crx_gates[:, :, :, 2, 2] = cos_t
    crx_gates[:, :, :, 3, 3] = cos_t
    crx_gates[:, :, :, 2, 3] = -1j * sin_t
    crx_gates[:, :, :, 3, 2] = -1j * sin_t

    crx_gates = -1 * crx_gates

    full_ops = _kron_batched(identities[0], _kron_batched(crx_gates[:, 0], identities[N-2]))
    
    for j in range(1, N-1):
        Id1 = identities[j]
        Id2 = identities[N-j-2]
        full_ops = _kron_batched(Id1, _kron_batched(crx_gates[:, j], Id2)) @ full_ops

    return full_ops

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

# --- Main Simulation Functions ---

def calc_variance_pure(N, L, theta=np.pi/20, device='cpu', n_sim=100):
    H = generate_H(N, device=device)
    ent_gate = crx_layer(N, theta, device=device)
    obs_hist = th.empty([n_sim], dtype=th.float64, device=device)

    for i in range(n_sim):
        params1 = th.acos(1.0 - 2 * th.rand((L, N), device=device))
        params2 = 2 * th.pi * th.rand((L, N), device=device)
        params3 = 2 * th.pi * th.rand((L, N), device=device)
        params = th.stack([params1, params2, params3], dim=2)
        U_gates = global_rotation(params, N, device=device)
        
        # State is a COLUMN vector
        state = th.zeros(2**N, 1, dtype=th.complex128, device=device)
        state[0] = 1.0

        for l in range(L):
            state = U_gates[l] @ state

            state = ent_gate @ state

        # Expectation <psi|H|psi>
        obs = (state.mH @ H @ state).squeeze().real

        obs_hist[i] = obs

    return obs_hist

def calc_variance_ng_crx_batched(N, L, theta=np.pi/20, device='cpu', n_sim=100, n_sim_noise=100):
    """
    Fully batched version of calc_variance_ng_crx.
    The inner loop over n_sim_noise is replaced with batched tensor operations.
    """
    H = generate_H(N, device=device)
    obs_hist = th.empty([n_sim, n_sim_noise], dtype=th.float64, device=device)

    for i in range(n_sim):
        # --- 1. Parameter and State Initialization ---
        params1 = th.acos(1.0 - 2 * th.rand((L, N), device=device))
        params2 = 2 * th.pi * th.rand((L, N), device=device)
        params3 = 2 * th.pi * th.rand((L, N), device=device)
        params = th.stack([params1, params2, params3], dim=2)
        U_gates = global_rotation(params, N, device=device)

        # Sample a batch of noise parameters, one set for each simulation in the batch
        phi_batch = th.normal(0, np.sqrt(np.log(N)/((N-1)*L))*theta/2, size=(n_sim_noise, N-1, L), device=device)  
        noise_gates = build_crx_layer_operator_batch(N, L, phi_batch, device=device)

        # Create a batch of initial states |0...0> as COLUMN vectors
        state_batch = th.zeros(n_sim_noise, 2**N, 1, dtype=th.complex128, device=device)
        state_batch[:, 0, 0] = 1.0

        for l in range(L):            
            state_batch = U_gates[l] @ state_batch

            state_batch = th.bmm(noise_gates[:, l], state_batch)

        obs_batch = (state_batch.mH @ (H @ state_batch)).squeeze().real        

        obs_hist[i, :] = obs_batch.real

    return obs_hist