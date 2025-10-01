import torch as th
import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict

# --- Import your custom calculation functions ---
from functions_ngates_v1 import calc_variance_pure as calc_variance_ng_crx_pure
from functions_ngates_v1 import calc_variance_ng_crx_batched
from functions_ngates_v2 import calc_variance_pure as calc_variance_ng_cnot_pure
from functions_ngates_v2 import calc_variance_ng_cnot_batched

# --- Global Parameters ---
N = 8
WORKERS = 20

def run_single_task(gate_type, method, layers):
    """
    Worker function to run one specific variance calculation on the CPU.
    This function is called in parallel by joblib.
    """
    result = None
    # --- Select the correct function based on parameters ---
    if gate_type == 'crx':
        if method == 'hea':
            result = calc_variance_ng_crx_pure(N, layers, theta=np.pi/2, device='cpu', n_sim=4000)
        elif method == 'qresnet':
            # Forcing the 'batched' function to run on the CPU
            result = calc_variance_ng_crx_batched(N, layers, theta=np.pi/2, device='cpu', n_sim=3000, n_sim_noise=1000)
    
    elif gate_type == 'cnot':
        if method == 'hea':
            result = calc_variance_ng_cnot_pure(N, layers, device='cpu', n_sim=4000)
        elif method == 'qresnet':
            # Forcing the 'batched' function to run on the CPU
            result = calc_variance_ng_cnot_batched(N, layers, device='cpu', n_sim=3000, n_sim_noise=1000)

    # Return a dictionary with the result and its identifying parameters
    return {'gate': gate_type, 'method': method, 'layers': layers, 'result': result}


if __name__ == "__main__":
    all_tasks = []
    layers_crx = np.arange(2, 33, 4)
    layers_cnot = np.arange(2, 17, 2)

    for l in layers_crx:
        all_tasks.append({'gate_type': 'crx', 'method': 'hea', 'layers': l})
        all_tasks.append({'gate_type': 'crx', 'method': 'qresnet', 'layers': l})
    
    for l in layers_cnot:
        all_tasks.append({'gate_type': 'cnot', 'method': 'hea', 'layers': l})
        all_tasks.append({'gate_type': 'cnot', 'method': 'qresnet', 'layers': l})

    print(f"Found {len(all_tasks)} total tasks to run.")
    print(f"Distributing tasks across {WORKERS} available CPU cores.")

    # --- Run all tasks in parallel using joblib ---
    # n_jobs=-1 automatically uses all available CPU cores requested from Slurm.
    # The 'verbose' flag provides progress updates.
    list_of_results = Parallel(n_jobs=WORKERS, verbose=10)(
        delayed(run_single_task)(**task) for task in all_tasks
    )

    # --- Aggregate all results into a single dictionary ---
    print("\nAggregation phase: Combining all results...")
    final_results = defaultdict(lambda: defaultdict(dict))
    for res in list_of_results:
        if res and res['result'] is not None:
            final_results[res['gate']][res['method']][res['layers']] = res['result']

    # --- Save the final, single file ---
    output_filename = 'results_' + str(N) + 'q.pt'
    th.save(dict(final_results), output_filename)

    print(f"\nWorkflow complete. All results saved to: {output_filename}")
