from pennylane import numpy as np
import pennylane as qml
from pennylane.operation import Operation
import torch
import torch.optim as optim
from tqdm.notebook import tqdm



class NG_CNOT(Operation):
    """A custom gate built from a specific decomposition."""
    num_params = 1
    num_wires = 2
    par_domain = 'R'

    def decomposition(self):
        """Returns the gate's decomposition into native PennyLane gates."""
        phi = self.parameters[0]
        control_wire = self.wires[0]
        target_wire = self.wires[1]
        
        return [
            qml.PhaseShift(2 * phi, wires=control_wire),
            qml.CRX(4 * phi, wires=[control_wire, target_wire]),
            qml.GlobalPhase(np.pi)
        ]


def vqe_uccsd(H, qubits, hf_state, singles, doubles, opt, max_iterations=100, conv_tol=1e-06):
    dev = qml.device("lightning.qubit", wires=qubits)

    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights):
        qml.UCCSD(weights, wires=range(qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state)
        return qml.expval(H)

    # We can just use the circuit as the cost function
    cost_fn = circuit
    grad_fn = qml.grad(cost_fn)

    num_weights = len(singles) + len(doubles)
    weights = np.zeros(num_weights, requires_grad=True)

    energy = [cost_fn(weights)]
    grad_norms = []
    grad_variances = []

    for n in tqdm(range(max_iterations)):
        # --- EFFICIENT OPTIMIZATION STEP ---
        
        # 1. Calculate the gradient ONLY ONCE
        gradient = grad_fn(weights)
        
        # 2. Use the gradient to calculate metrics
        flat_gradient = np.hstack([g.flatten() for g in gradient])
        norm = np.linalg.norm(flat_gradient)
        variance = np.var(flat_gradient)
        grad_norms.append(norm)
        grad_variances.append(variance)
        
        # 3. Use the SAME gradient to update the weights.
        # This avoids the second, redundant gradient calculation.
        weights = opt.apply_grad(gradient, weights)
        
        # 4. Store the previous energy and compute the new one
        prev_energy = energy[-1]
        current_energy = cost_fn(weights)
        energy.append(current_energy)
        
        # ------------------------------------

        conv = np.abs(current_energy - prev_energy)

        if n % 10 == 0:
            tqdm.write(
                f"It={n}, E={energy[-1]:.8f} Ha, "
                f"|∇E|={norm:.6f}, Var(∇E)={variance:.6f}"
            )

        if conv <= conv_tol:
            print("\nConvergence achieved!")
            break

    return energy, weights, grad_norms, grad_variances


def hardware_efficient_ansatz_1(params, wires):
    num_layers = params.shape[0]
    num_qubits = len(wires)

    for layer in range(num_layers):
        # Layer of rotation gates
        for i in range(num_qubits):
            qml.U3(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)

        # Layer of entangling gates
        for i in range(num_qubits):
            qml.CNOT(wires=[i, (i + 1) % num_qubits])


def vqe_hee(H, qubits, L, lr=0.1, max_iterations=100, conv_tol=1e-06, verbose=True):
    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(params1, params2, params3):  # Accept separate parameters
        # Stack parameters inside the circuit
        weights = torch.stack([params1, params2, params3], dim=2)
        hardware_efficient_ansatz_1(weights, wires=range(qubits))
        return qml.expval(H)

    def cost_fn(params1, params2, params3):
        return circuit(params1, params2, params3)

    # Initialize individual parameter tensors as leaf tensors
    params1 = torch.tensor(
        np.arccos(1.0 - 2 * np.random.rand(L, qubits)), 
        requires_grad=True, 
        dtype=torch.float64
    )
    params2 = torch.tensor(
        2 * np.pi * np.random.rand(L, qubits), 
        requires_grad=True, 
        dtype=torch.float64
    )
    params3 = torch.tensor(
        2 * np.pi * np.random.rand(L, qubits), 
        requires_grad=True, 
        dtype=torch.float64
    )

    # Create PyTorch optimizer with different learning rates for parameter groups
    optimizer = optim.Adam([
        {'params': [params1, params2, params3], 'lr': lr}
    ])

    # Store the values of the cost function
    energy = [cost_fn(params1, params2, params3).item()]
    grad_norms = []
    grad_variances = []
    convergence = False

    for n in tqdm(range(max_iterations), disable=not verbose):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        current_energy = cost_fn(params1, params2, params3)
        
        # Backward pass
        current_energy.backward()
        
        # --- METRIC CALCULATION ---
        # Collect gradients from all parameter tensors
        all_grads = []
        for param in [params1, params2, params3]:
            if param.grad is not None:
                all_grads.append(param.grad.flatten())
        
        # Concatenate all gradients
        if all_grads:
            flat_gradient = torch.cat(all_grads).detach().numpy()
            norm = np.linalg.norm(flat_gradient)
            variance = np.var(flat_gradient)
            grad_norms.append(norm)
            grad_variances.append(variance)
        # ---------------------------

        # Store previous energy for convergence check
        prev_energy = energy[-1]
        
        # Optimizer step
        optimizer.step()
        
        # Store current energy
        energy.append(current_energy.item())
        
        # Check convergence
        conv = np.abs(energy[-1] - prev_energy)
        
        if n % 10 == 0 and verbose:
            tqdm.write(
                f"It={n}, E={energy[-1]:.8f} Ha, "
                f"|∇E|={norm:.6f}, Var(∇E)={variance:.6f}"
            )

        if conv <= conv_tol:
            convergence = True

    return energy, grad_norms, grad_variances, convergence


def hardware_efficient_ansatz_2(params, phi, wires):
    num_layers = params.shape[0]
    num_qubits = len(wires)

    for layer in range(num_layers):
        # Layer of rotation gates
        for i in range(num_qubits):
            qml.U3(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)

        # Layer of entangling gates
        for i in range(num_qubits):
            NG_CNOT(phi[layer, i], wires=[i, (i + 1) % num_qubits])


def vqe_ng(H, qubits, L, weights_lr=0.01, phi_lr=0.1, max_iterations=100, conv_tol=1e-06, verbose=True):
    dev = qml.device("lightning.qubit", wires=qubits)
    N = qubits

    @qml.qnode(dev, interface="torch")
    def circuit(params1, params2, params3, phi):  # Accept separate parameters
        # Stack parameters inside the circuit
        weights = torch.stack([params1, params2, params3], dim=2)
        hardware_efficient_ansatz_2(weights, phi, wires=range(qubits))
        return qml.expval(H)

    def cost_fn(params1, params2, params3, phi):
        return circuit(params1, params2, params3, phi)

    # Initialize individual parameter tensors as leaf tensors
    params1 = torch.tensor(
        np.arccos(1.0 - 2 * np.random.rand(L, qubits)), 
        requires_grad=True, 
        dtype=torch.float64
    )
    params2 = torch.tensor(
        2 * np.pi * np.random.rand(L, qubits), 
        requires_grad=True, 
        dtype=torch.float64
    )
    params3 = torch.tensor(
        2 * np.pi * np.random.rand(L, qubits), 
        requires_grad=True, 
        dtype=torch.float64
    )
    
    phi = torch.tensor(
        np.random.normal(0, np.sqrt(np.log(N)/((N-1)*L))*(np.pi/4)/2, size=(L, qubits)), 
        requires_grad=True, 
        dtype=torch.float64
    )

    # Create PyTorch optimizer with different learning rates for parameter groups
    optimizer = optim.Adam([
        {'params': [params1, params2, params3], 'lr': weights_lr},  # Weight parameters
        {'params': [phi], 'lr': phi_lr}                           # Phi parameters
    ])

    # Store the values of the cost function
    energy = [cost_fn(params1, params2, params3, phi).item()]
    grad_norms = []
    grad_variances = []
    convergence = False

    for n in tqdm(range(max_iterations), disable=not verbose):
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        current_energy = cost_fn(params1, params2, params3, phi)
        
        # Backward pass
        current_energy.backward()
        
        # --- METRIC CALCULATION ---
        # Collect gradients from all parameter tensors
        all_grads = []
        for param in [params1, params2, params3, phi]:
            if param.grad is not None:
                all_grads.append(param.grad.flatten())
        
        # Concatenate all gradients
        if all_grads:
            flat_gradient = torch.cat(all_grads).detach().numpy()
            norm = np.linalg.norm(flat_gradient)
            variance = np.var(flat_gradient)
            grad_norms.append(norm)
            grad_variances.append(variance)
        # ---------------------------

        # Store previous energy for convergence check
        prev_energy = energy[-1]
        
        # Optimizer step
        optimizer.step()
        
        # Store current energy
        energy.append(current_energy.item())
        
        # Check convergence
        conv = np.abs(energy[-1] - prev_energy)
        
        if n % 10 == 0 and verbose:
            tqdm.write(
                f"It={n}, E={energy[-1]:.8f} Ha, "
                f"|∇E|={norm:.6f}, Var(∇E)={variance:.6f}"
            )

        if conv <= conv_tol:
            convergence = True

    return energy, grad_norms, grad_variances, convergence

