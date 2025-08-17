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


def generate_ghz(N):
    state = th.zeros(2**N, dtype=th.complex128)
    state[0] = 1/np.sqrt(2)
    state[-1] = 1/np.sqrt(2)
    return state


def generate_H(N):
    h = 9/N
    scale = 2**(N/2)

    Z = th.tensor([[1, 0], [0, -1]], dtype=th.complex128)

    ZZ = th.kron(Z, Z)

    H = th.zeros(2**N, 2**N, dtype=th.complex128)

    for i in range(N-1):
        H += th.kron( th.eye(2**i), th.kron(ZZ, th.eye(2**(N-i-2))) )

    return h*scale*H


def generate_ghz_dm(N):
    """Generate a GHZ state for N qubits."""
    ghz_state = generate_ghz(N)
    return th.outer(ghz_state, ghz_state)


def noisy_gate(n, L, gamma=0.01):
    sigma_z = th.tensor([[1, 0], [0, -1]], dtype=th.complex128)
    sigma_z_loc = []

    for i in range(n):
        sigma_z_loc.append(th.kron(th.eye(2**i, dtype=th.complex128), th.kron(sigma_z, th.eye(2**(n-i-1), dtype=th.complex128)))) # Id x Z_i x Id

    phi = np.random.normal(0, np.log(n)/(n * L), n)

    gate = th.matrix_exp(-1j * phi[0] * sigma_z_loc[0])
    for i in range(1,n):
        tmp = th.matrix_exp(-1j * phi[i] * sigma_z_loc[i])
        gate = gate @ tmp

    return gate


def layer(state, angles, N, L, gates):
    """Apply a layer of gates to the state vector."""
    U = global_rotation(angles,N)
    
    # Apply the rotation
    state = U @ state
    
    # Apply CNOT gates
    for gate in gates:
        state = gate @ state

    # Apply noisy gate
    noise = noisy_gate(N, L)
    state = noise @ state

    return state


def calc_variance(N, L, n_sim=100, n_sim_noise=100, fast_ent=True):

    H = generate_H(N)

    if fast_ent:
        ent_gates = cnot_gates(N)
    else:
        ent_gates = crx_gates(N)

    obs = []

    for _ in range(n_sim):
        params1 = np.arccos(np.ones((L, N)) - 2 * np.random.uniform(size=(L,N)))
        params2 = np.random.uniform(0, 2*np.pi, size=(L, N))
        params3 = np.random.uniform(0, 2*np.pi, size=(L, N))

        params = np.stack((params1, params2, params3), axis=2)

        rho_state = th.zeros(2**N, 2**N, dtype=th.complex128)
        for _ in range(n_sim_noise):
            state = th.zeros(2**N, dtype=th.complex128)
            state[0] = 1.0  # |00...0>

            for l in range(L):
                state = layer(state, params[l], N, l+1, ent_gates)

            rho_state += th.outer(state, state.conj())
        rho_state /= n_sim_noise
        

        obs.append(th.trace(rho_state @ H).real)
    
    return np.mean(obs), np.std(obs)


if __name__ == "__main__":
    N = 4
    L = 10

    state = th.zeros(2**N, dtype=th.complex128)
    state[0] = 1.0  # |00...0>

    params1 = np.arccos(np.ones((L, N)) - 2 * np.random.uniform(size=(L,N)))
    params2 = np.random.uniform(0, 2*np.pi, size=(L, N))
    params3 = np.random.uniform(0, 2*np.pi, size=(L, N))

    params = np.stack((params1, params2, params3), axis=2)

    ic(calc_variance(N, L, n_sim=10, n_sim_noise=10, fast_ent=True))
