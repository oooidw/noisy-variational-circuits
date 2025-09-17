import torch as th
import pennylane as qml
import numpy as np
import torch.nn as nn
# It's better to import the specific tqdm instance you need
from tqdm.notebook import tqdm

def U3_rotation(theta, phi, lambda_, device='cpu'):
    """
    Rotation matrix for a single qubit (direct tensor construction).
    Accepts batched inputs for the angles.
    """
    cos_t = th.cos(theta / 2)
    sin_t = th.sin(theta / 2)
    exp_il = th.exp(1j * lambda_)
    exp_ip = th.exp(1j * phi)
    
    # The batch size L is derived from the input tensor shape
    L = theta.shape[0]

    # Initialize an empty tensor and fill its elements
    U = th.empty((L, 2, 2), dtype=th.complex128, device=device)
    U[:, 0, 0] = cos_t
    U[:, 0, 1] = -exp_il * sin_t
    U[:, 1, 0] = exp_ip * sin_t
    U[:, 1, 1] = exp_ip * exp_il * cos_t
    return U

def _kron_batched(A, B):
    """
    Calculates the Kronecker product in a batched way.
    NOTE: Modern PyTorch has `th.kron` which handles this, but this
    custom implementation is also fine.
    """
    size1 = A.shape[-2:]
    size2 = B.shape[-2:]

    intermediate = th.einsum('...ij,...kl->...ikjl', A, B)

    batch_dims = A.shape[:-2]
    res = intermediate.reshape(*batch_dims, size1[0] * size2[0], size1[1] * size2[1])

    return res

def global_rotation(angles, N, device='cpu'):
    """
    Apply a global rotation to the state vector by building the
    full unitary operator for one layer.
    `angles` should have shape (L, N, 3) for L layers.
    """
    # Build the first qubit's rotation
    U = U3_rotation(angles[:, 0, 0], angles[:, 0, 1], angles[:, 0, 2], device=device)
    # Iteratively tensor with the rest of the qubits' rotations
    for i in range(1, N):
        U_i = U3_rotation(angles[:, i, 0], angles[:, i, 1], angles[:, i, 2], device=device)
        U = _kron_batched(U, U_i)
    return U

def build_entangling_layer_operator(N, L, phi, device='cpu'):
    """
    Constructs a batch of CNOT layer operators.
    phi_batch has shape (batch_size, N-1).
    Returns a tensor of shape (batch_size, 2**N, 2**N).
    """
    
    # Pre-calculate identities to reuse
    identities = [th.eye(2**i, dtype=th.complex128, device=device).repeat(L, 1, 1) for i in range(N + 1)]

    cnot = th.zeros((L, N-1, 4, 4), dtype=th.complex128, device=device)
    cnot[...,0,0] = 1
    cnot[...,1,1] = 1
    cnot[...,2,2] = (1 + th.exp(1j * 4 * phi))/2
    cnot[...,3,3] = (1 + th.exp(1j * 4 * phi))/2
    cnot[...,2,3] = (1 - th.exp(1j * 4 * phi))/2
    cnot[...,3,2] = (1 - th.exp(1j * 4 * phi))/2
    cnot = -1 * cnot

    full_ops = _kron_batched(identities[0], _kron_batched(cnot[:, 0], identities[N-2]))
    
    for j in range(1, N-1):
        Id1 = identities[j]
        Id2 = identities[N-j-2]
        full_ops = full_ops @ _kron_batched(Id1, _kron_batched(cnot[:, j], Id2))

    return full_ops

class VQEModule(nn.Module):
    def __init__(self, H, L, device):
        super(VQEModule, self).__init__()
        self.device = device

        self.N = 8  # Number of qubits
        self.L = L  # Number of layers in the ansatz

        # Load Hamiltonian for He2 molecule
        # Ensure you have the dataset downloaded, e.g., via `qml.data.load` once outside the script
        self.H = H

        # Initialize rotation parameters
        params1 = th.acos(1.0 - 2 * th.rand((self.L, self.N), device=device))
        params2 = 2 * th.pi * th.rand((self.L, self.N), device=device)
        params3 = 2 * th.pi * th.rand((self.L, self.N), device=device)
        self.params = nn.Parameter(th.stack([params1, params2, params3], dim=2))

        # Initialize entangling parameters.
        theta = np.pi / 4
        layers_tensor = th.arange(1, self.L + 1, device=device, dtype=th.float32)
        stds = th.sqrt(th.log(th.tensor(self.N, device=device)) / ((self.N - 1) * layers_tensor) * 2) * theta / 2
        self.phi = nn.Parameter(th.normal(mean=0.0, std=stds.unsqueeze(1).repeat(1, self.N-1)))

    def forward(self,):
        # Build the operators for all layers at once
        rotation_ops = global_rotation(self.params, self.N, device=self.device)
        entangling_ops = build_entangling_layer_operator(self.N, self.L, self.phi, device=self.device)

        # Initialize state in |00...0>
        state = th.zeros(2**self.N, 1, dtype=th.complex128, device=self.device)
        state[0] = 1.0

        # Apply each layer of the ansatz
        for l in range(self.L):
            state = rotation_ops[l] @ state
            state = entangling_ops[l] @ state

        # Calculate expectation value <psi|H|psi>
        obs = (state.mH @ self.H @ state).squeeze().real
        return obs

def optimize_vqe(module, opt, steps=100):
    """Optimize the VQEModule parameters using Adam."""

    energy_history = []
    grad_norm_history = []
    grad_var_history = []

    # Use a tqdm instance for the progress bar
    pbar = tqdm(range(steps))
    for step in pbar:
        opt.zero_grad()
        loss = module()
        loss.backward()
        opt.step()

        energy_history.append(loss.item())

        # Collect gradients as a single flattened tensor for analysis
        with th.no_grad():
            grad_list = []
            for p in module.parameters():
                if p.grad is not None:
                    grad_list.append(p.grad.flatten())
            
            if grad_list:
                grad_tensor = th.cat(grad_list)
                grad_norm = th.norm(grad_tensor).item()
                grad_var = grad_tensor.var().item()
                grad_norm_history.append(grad_norm)
                grad_var_history.append(grad_var)

                # Update the progress bar description
                pbar.set_description(
                    f"E={energy_history[-1]:.8f} Ha, |∇E|={grad_norm:.4f}, Var(∇E)={grad_var:.4f}"
                )

    return energy_history, grad_norm_history, grad_var_history