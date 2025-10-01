from pennylane import numpy as np
import pennylane as qml
from pennylane.operation import Operation
import torch
import torch.optim as optim
from tqdm.notebook import tqdm


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
