import numpy as np
import torch as th

from icecream import ic


def U3_rotation(theta,phi,lambda_):
    """Rotation matrix for a single qubit."""
    U = th.tensor(
            [[np.cos(theta / 2)                     , -np.exp(1j * lambda_) * np.sin(theta / 2)         ],
            [np.exp(1j * phi) * np.sin(theta / 2)   , np.exp(1j * (phi+lambda_)) * np.cos(theta / 2)    ]],
            dtype=th.complex128)
    return U


def X_rotation(theta):
    U = th.tensor(
            [[np.cos(theta / 2)                     , -1j * np.sin(theta / 2)         ],
            [-1j * np.sin(theta / 2)               , np.cos(theta / 2)              ]],
            dtype=th.complex128)
    return U


def global_rotation(angles,N):
    """Apply a global rotation to the state vector."""
    U = U3_rotation(angles[0,0], angles[0,1], angles[0,2])

    for i in range(1,N):
        U_i = U3_rotation(angles[i,0], angles[i,1], angles[i,2])
        U = th.kron(U, U_i)
    return U


def cnot_gates(N):
    """Return a n-1 cascading CNOT."""

    cnot = th.tensor([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
        ], dtype=th.complex128)
    
    cnot_gates = []
    
    for i in range(N-1):    
        p1 = i
        p2 = N-i-2

        Id1 = th.eye(2**p1, dtype=th.complex128)
        Id2 = th.eye(2**p2, dtype=th.complex128)

        cnot_gates.append(th.kron(Id1, th.kron(cnot, Id2)))
    
    return cnot_gates


def crx_gates(N, theta=np.pi/20):
    """Return a controlled RX gate with rotation angle theta."""
    Id = th.eye(2, dtype=th.complex128)
    Rx = X_rotation(theta)
    crx = th.block_diag(Id, Rx)

    crx_gates = []
    
    for i in range(N-1):    
        p1 = i
        p2 = N-i-2

        Id1 = th.eye(2**p1, dtype=th.complex128)
        Id2 = th.eye(2**p2, dtype=th.complex128)

        crx_gates.append(th.kron(Id1, th.kron(crx, Id2)))
    return crx_gates


def generate_H(N):
    """Generate the Hamiltonian for the system."""
    h = 9/N
    scale = 2**(N/2)

    Z = th.tensor([[1, 0], [0, -1]], dtype=th.complex128)

    ZZ = th.kron(Z, Z)

    H = th.zeros(2**N, 2**N, dtype=th.complex128)

    for i in range(N-1):
        H += th.kron( th.eye(2**i), th.kron(ZZ, th.eye(2**(N-i-2))) )

    return h*scale*H


def noisy_gate_adj(phi, jumpOPs, gamma=0.01):
    res = th.matrix_exp(th.sum(-1j * phi * jumpOPs, axis=0))

    return res


def noisy_gate_nadj(phi, jumpOPs, gamma=0.01):
    res = th.matrix_exp(th.sum(-1j * phi * jumpOPs + (phi**2 / 2) * (jumpOPs @ jumpOPs - jumpOPs.transpose(2, 1).conj() @ jumpOPs), axis=0))

    return res


def noisy_gate(phi, jumpOPs, adj, gamma=0.01):
    if adj==True:
        return noisy_gate_adj(phi, jumpOPs, gamma)
    else:
        return noisy_gate_nadj(phi, jumpOPs, gamma)


def layer(state, angles, N, gates, phi, jumpOPs, adj):
    """Apply a layer of gates to the state vector."""
    U = global_rotation(angles,N)
    
    # Apply the rotation
    state = U @ state
    
    # Apply CNOT gates
    for gate in gates:
        state = gate @ state

    # Apply noisy gate
    noise = noisy_gate(phi, jumpOPs, adj)
    state = noise @ state

    return state


def calc_variance(N, L, n_sim=100, n_sim_noise=100, fast_ent=True, noise='dephasing'):

    H = generate_H(N)

    if fast_ent:
        ent_gates = cnot_gates(N)
    else:
        ent_gates = crx_gates(N)

    if noise == 'dephasing':
        jumpOP = th.tensor([[1, 0], [0, -1]], dtype=th.complex128) # sigma_z
        adj = True
    elif noise == 'bitflip':
        jumpOP = th.tensor([[0, 1], [1, 0]], dtype=th.complex128) # sigma_x
        adj = True
    elif noise == 'amplitude_damping':
        jumpOP = th.tensor([[0, 1], [0, 0]], dtype=th.complex128) # sigma_-
        adj = False

    jumpOPs = th.empty(N, 2**N, 2**N, dtype=th.complex128)
    for i in range(N):
        jumpOPs[i] = th.kron(th.eye(2**i, dtype=th.complex128), th.kron(jumpOP, th.eye(2**(N-i-1), dtype=th.complex128))) # Id x jumpOP x Id


    obs_hist = th.empty([n_sim*n_sim_noise, 3*L*N+N+1], dtype=th.float64)

    for i in range(n_sim):
        params1 = th.acos(th.ones((L, N)) - 2 * th.rand((L, N)))
        params2 = 2 * np.pi * th.rand((L, N))
        params3 = 2 * np.pi * th.rand((L, N))
        params = th.stack([params1, params2, params3], dim=2)
        
        flat_params = th.cat([params1.flatten(), params2.flatten(), params3.flatten()])

        for j in range(n_sim_noise):
            phi = th.normal(0, np.log(N)/(N * L), (N,1,1))

            state = th.zeros(2**N, dtype=th.complex128)
            state[0] = 1.0  # |00...0>

            for l in range(L):
                state = layer(state, params[l], N, ent_gates, phi, jumpOPs, adj)

            obs = th.vdot(state, H @ state).real # Expectation value should be real

            obs_hist[i * n_sim_noise + j, :] = th.cat([flat_params, phi.flatten(), obs.unsqueeze(0)])

    return obs_hist


# --- Batched Noise Model Definitions (doesn't work in these functions) ---

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
