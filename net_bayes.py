import numpy as np
from itertools import product

def prior_true_probabilities(nodes, edges, priors):
    # Helper to get parents of a node
    def get_parents(node):
        return [edge[0] for edge in edges if edge[1] == node]

    # Store marginal probabilities
    marginals = {}

    # Compute in topological order (assuming nodes are ordered)
    for node in nodes:
        parents = get_parents(node)
        if not parents:
            # Root node: use marginal
            marginals[node] = priors[(node, True)]
        else:
            # Marginalize over all parent combinations
            combos = list(product([True, False], repeat=len(parents)))
            prob_true = 0.0
            for combo in combos:
                parent_prob = 1.0
                for p, val in zip(parents, combo):
                    parent_prob *= marginals[p] if val else (1 - marginals[p])
                # For nodes with >1 parent, priors key is (node, *combo)
                key = (node,) + combo
                prob_true += parent_prob * priors[key]
            marginals[node] = prob_true
    return marginals

def netBayes(nodes, edges, priors, evidence):
    """
    Compute the posterior probability of each node being True given evidence.
    """

    def get_parents(node):
        return [edge[0] for edge in edges if edge[1] == node]

    # All variables not in evidence
    hidden_nodes = [n for n in nodes if n not in evidence]

    # Generate all possible assignments for hidden nodes
    assignments = list(product([True, False], repeat=len(hidden_nodes)))

    # For each node, sum probabilities where node is True
    posteriors = {}
    for node in nodes:
        # For evidence nodes, posterior is 1.0 if True, 0.0 if False
        if node in evidence:
            posteriors[node] = 1.0 if evidence[node] else 0.0
        else:
            num = 0.0 # Numerator of posterior calculus
            denom = 0.0 # Denominator of posterior calculus
            for assign in assignments:
                full_state = evidence.copy()
                for n, v in zip(hidden_nodes, assign):
                    full_state[n] = v
                # Calculate joint probability for this assignment
                joint = 1.0
                for n in nodes:
                    parents = get_parents(n)
                    if not parents:
                        prob = priors[(n, full_state[n])]
                    else:
                        parent_vals = tuple(full_state[p] for p in parents) # Get state values of parents
                        prob = priors[(n,) + parent_vals] if full_state[n] else 1 - priors[(n,) + parent_vals]
                    joint *= prob
                if full_state[node]:
                    num += joint
                denom += joint
            
            posteriors[node] = num / denom if denom > 0 else 0.0 # Posterior probability of node being True is equal to the ratio of all True assignments to all assignments
    return posteriors

def joint_probability(nodes, edges, priors, assignment):
    """
    Calculate the joint probability of a full assignment of all nodes.
    assignment: dict {node: True/False, ...}
    """
    def get_parents(node):
        return [edge[0] for edge in edges if edge[1] == node]

    joint = 1.0
    for node in nodes:
        parents = get_parents(node)
        if not parents:
            prob = priors[(node, assignment[node])]
        else:
            parent_vals = tuple(assignment[p] for p in parents)
            prob = priors[(node,) + parent_vals] if assignment[node] else 1 - priors[(node,) + parent_vals]
        joint *= prob
    return joint

nodes = ['D11', 'D12', 'D21', 'D22', 'D1', 'D2', 
         'M1', 'M2', 'M3', 'M13', 'M23', 
         'N', 'P1', 'P2', 'S1', 'S2', 'S12', 'TE'] #nodes = ['A', 'B', 'C', 'D', 'E']
edges = [('D11', 'D1'), ('D12', 'D1'), ('D21', 'D2'), ('D22', 'D2'),
         ('M1', 'M13'), ('M3', 'M13'), ('M2', 'M23'), ('M3', 'M23'),
         ('P1', 'S1'), ('P2','S2'), ('D1','S1'), ('D2','S2'), 
         ('M13','S1'), ('M23','S2'), ('S1','S12'), ('S2','S12'), 
         ('S12','TE'), ('N','TE')] 

t = 5000 # time
lambda_D = 8e-5 # Disk failure rate
lambda_M = 3e-8 # Memory failure rate
lambda_N = 2e-9 # Bus failure rate
lambda_P = 5e-7 # Processor failure rate

# Prior and conditional probabilities
priors = {
    # Marginal probabilities for root nodes
    ('D11', True): 1 - np.exp(-lambda_D*t),
    ('D11', False): np.exp(-lambda_D*t),
    ('D12', True): 1 - np.exp(-lambda_D*t),
    ('D12', False): np.exp(-lambda_D*t),
    ('D21', True): 1 - np.exp(-lambda_D*t),
    ('D21', False): np.exp(-lambda_D*t),
    ('D22', True): 1 - np.exp(-lambda_D*t),
    ('D22', False): np.exp(-lambda_D*t),
    ('M1', True): 1- np.exp(-lambda_M*t),
    ('M1', False): np.exp(-lambda_M*t),
    ('M2', True): 1 - np.exp(-lambda_M*t),
    ('M2', False):np.exp(-lambda_M*t),
    ('M3', True): 1 - np.exp(-lambda_M*t),
    ('M3', False):np.exp(-lambda_M*t),
    ('N', True): 1 - np.exp(-lambda_N*t),
    ('N', False): np.exp(-lambda_N*t),
    ('P1', True): 1 - np.exp(-lambda_P*t),
    ('P1', False): np.exp(-lambda_P*t),
    ('P2', True): 1 - np.exp(-lambda_P*t),
    ('P2', False): np.exp(-lambda_P*t),

    # Conditional probabilities for D1 given D11 and D12 (AND gate)
    ('D1', True, True): 1,   
    ('D1', True, False): 0,  
    ('D1', False, True): 0,  
    ('D1', False, False): 0, 
    # Conditional probabilities for D2 given D21 and D22 (AND gate)
    ('D2', True, True): 1,   
    ('D2', True, False): 0,  
    ('D2', False, True): 0,  
    ('D2', False, False): 0, 
    # Conditional probabilities for M13 given M1 and M3 (AND gate)
    ('M13', True, True): 1,   
    ('M13', True, False): 0,  
    ('M13', False, True): 0,  
    ('M13', False, False): 0, 
    # Conditional probabilities for M23 given M2 and M3 (AND gate)
    ('M23', True, True): 1,   
    ('M23', True, False): 0,  
    ('M23', False, True): 0,  
    ('M23', False, False): 0, 
    # Conditional probabilities for S1 given D1, M13, and P1 (OR gate)
    ('S1', True, True, True): 1,   
    ('S1', True, True, False): 1,  
    ('S1', True, False, True): 1,  
    ('S1', True, False, False): 1,
    ('S1', False, True, True): 1,   
    ('S1', False, True, False): 1,  
    ('S1', False, False, True): 1,  
    ('S1', False, False, False): 0, 
    # Conditional probabilities for S2 given D2, M23, and P2 (OR gate)
    ('S2', True, True, True): 1,   
    ('S2', True, True, False): 1,  
    ('S2', True, False, True): 1,  
    ('S2', True, False, False): 1,
    ('S2', False, True, True): 1,   
    ('S2', False, True, False): 1,  
    ('S2', False, False, True): 1,  
    ('S2', False, False, False): 0,
    # Conditional probabilities for S12 given S1 and S3 (AND gate)
    ('S12', True, True): 1,   
    ('S12', True, False): 0,  
    ('S12', False, True): 0,  
    ('S12', False, False): 0,  
    # Conditional probabilities for TE given N and S12 (OR gate)
    ('TE', True, True): 1,   
    ('TE', True, False): 1,  
    ('TE', False, True): 1,  
    ('TE', False, False): 0,
}


# assignment = {'A': True, 'B': False, 'C': True, 'D': True, 'E': False}
evidence = {'TE': True}

prior_probs = prior_true_probabilities(nodes, edges, priors)
print(f"Prior probabilities:\n")
for key, value in prior_probs.items():
    print(f"{key}: {value:.12f}")

posteriors = netBayes(nodes, edges, priors, evidence)
print(f"Posterior probabilities given the evidence TE:\n")
for key, value in posteriors.items():
    print(f"{key}: {value:.12f}")


# overall_prob = joint_probability(nodes, edges, priors, assignment)
# print("Joint probability of evidence:", overall_prob)

