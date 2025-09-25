from pennylane import numpy as np
import pennylane as qml
from pennylane.operation import Operation
import torch
import torch.optim as optim
from tqdm.notebook import tqdm



def hardware_efficient_ansatz_1(params, theta, wires):
    num_layers = params.shape[0]
    num_qubits = len(wires)

    for layer in range(num_layers):
        # Layer of rotation gates
        for i in range(num_qubits):
            qml.U3(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)

        # Layer of entangling gates
        for i in range(0, num_qubits - 1, 2):
            qml.CRX(theta, wires=[i, i + 1])

        for i in range(1, num_qubits, 2):
            qml.CRX(theta, wires=[i, (i + 1) % num_qubits])


def vqe_hee(H, qubits, L, lr, theta=np.pi, max_iterations=100, conv_tol=1e-06, verbose=True):
    dev = qml.device("lightning.qubit", wires=qubits)
    opt = qml.AdamOptimizer(stepsize=lr)

    @qml.qnode(dev, interface="autograd")
    def circuit(weights):
        hardware_efficient_ansatz_1(weights, theta, wires=range(qubits))
        return qml.expval(H)

    def cost_fn(param):
        return circuit(param)

    grad_fn = qml.grad(cost_fn)

    params1 = np.arccos(1.0 - 2 * np.random.rand(L, qubits, requires_grad=True))
    params2 = 2 * np.pi * np.random.rand(L, qubits, requires_grad=True)
    params3 = 2 * np.pi * np.random.rand(L, qubits, requires_grad=True)
    weights = np.stack([params1, params2, params3], axis=2)

    # store the values of the cost function
    energy = [cost_fn(weights)]
    grad_norms = []
    grad_variances = []
    convergence = False

    for n in tqdm(range(max_iterations), disable=not verbose):
        # --- METRIC CALCULATION ---
        # 1. Calculate the gradient for the current weights
        gradient = grad_fn(weights)
        
        # 2. Flatten the gradient into a 1D vector to compute stats
        # The gradient has the same nested shape as 'weights', so we unravel it.
        flat_gradient = np.hstack([g.flatten() for g in gradient])
        
        # 3. Compute and store the norm and variance
        norm = np.linalg.norm(flat_gradient)
        variance = np.var(flat_gradient)
        grad_norms.append(norm)
        grad_variances.append(variance)
        # ---------------------------

        # Optimizer step
        weights, prev_energy = opt.step_and_cost(cost_fn, weights)
        energy.append(cost_fn(weights))

        conv = np.abs(energy[-1] - prev_energy)

        if n % 10 == 0 and verbose:
            # Updated print statement to include the new metrics
            tqdm.write(
                f"It={n}, E={energy[-1]:.8f} Ha, "
                f"|∇E|={norm:.6f}, Var(∇E)={variance:.6f}"
            )

        if conv <= conv_tol:
            convergence = True

    return energy, grad_norms, grad_variances, convergence


def vqe_hee_th(H, qubits, L, lr, theta=np.pi/5, max_iterations=100, conv_tol=1e-06, verbose=True):
    dev = qml.device("lightning.qubit", wires=qubits)
    N = qubits

    @qml.qnode(dev, interface="torch")
    def circuit(params1, params2, params3):  # Accept separate parameters
        # Stack parameters inside the circuit
        weights = torch.stack([params1, params2, params3], dim=2)
        hardware_efficient_ansatz_1(weights, theta, wires=range(qubits))
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
        for i in range(0, num_qubits - 1, 2):
            qml.CRX(phi[layer, i], wires=[i, i + 1])

        for i in range(1, num_qubits, 2):
            qml.CRX(phi[layer, i], wires=[i, (i + 1) % num_qubits])


def vqe_ng(H, qubits, L, weights_lr=0.1, phi_lr=0.01, max_iterations=100, conv_tol=1e-06, verbose=True):
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
    params1 = torch.tensor(np.arccos(1.0 - 2 * np.random.rand(L, qubits)), dtype=torch.float64)
    params2 = torch.tensor(2 * np.pi * np.random.rand(L, qubits), requires_grad=True, dtype=torch.float64)
    params3 = torch.tensor(2 * np.pi * np.random.rand(L, qubits), requires_grad=True, dtype=torch.float64)

    phi = torch.tensor(np.random.normal(0, np.sqrt(np.log(N)/((N)*L))*(np.pi/4)/2, size=(L, qubits)), requires_grad=True, dtype=torch.float64)

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

