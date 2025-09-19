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

nodes = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 
         'E7', 'E8', 'E9', 'CAT', 'VSF', 
         'SRF', 'SGF', 'TE'] #nodes = ['A', 'B', 'C', 'D', 'E']
edges = [('CAT', 'E1'), ('CAT', 'VSF'), ('VSF', 'E2'), ('VSF', 'E3'),
         ('VSF', 'E4'), ('VSF', 'E5'), ('SRF', 'E6'), ('SRF', 'E7'),
         ('SRF', 'E8'), ('SGF','SRF'), ('SGF','E9'), ('TE','CAT'), 
         ('TE','SGF')] 

#t = 5000 # time
#lambda_D = 8e-5 # Disk failure rate
#lambda_M = 3e-8 # Memory failure rate
#lambda_N = 2e-9 # Bus failure rate
#lambda_P = 5e-7 # Processor failure rate

def beta_mean(a, b):
    return a / (a + b)

# Prior and conditional probabilities
priors = {
    # Marginal probabilities for root nodes
    ('E1', True): 1 - beta_mean(1.99, 72.22),
    ('E1', False): beta_mean(1.99, 72.22),
    ('E2', True): 1 - beta_mean(1.11, 58.23),
    ('E2', False): beta_mean(1.11, 58.23),
    ('E3', True): 1 - beta_mean(1.1,58.09),
    ('E3', False): beta_mean(1.1,58.09),
    ('E4', True): 1 - beta_mean(1.34,69.45),
    ('E4', False): beta_mean(1.34,69.45),
    ('E5', True): 1- beta_mean(1.1,96.95),
    ('E5', False): beta_mean(1.1,96.95),
    ('E6', True): 1 - beta_mean(1.89,159.93),
    ('E6', False):beta_mean(1.89,159.93),
    ('E7', True): 1 - beta_mean(1.15,123.01),
    ('E7', False):beta_mean(1.15,123.01),
    ('E8', True): 1 - beta_mean(2.41,149.22),
    ('E8', False): beta_mean(2.41,149.22),
    ('E9', True): 1 - beta_mean(1.86,27.23),
    ('E9', False): beta_mean(1.86,27.23),


    # Conditional probabilities for D1 given D11 and D12 (AND gate)
    ('CAT', True, True): 0,   
    ('CAT', True, False): 1,  
    ('CAT', False, True): 1,  
    ('CAT', False, False): 1, 

    # Conditional probabilities for S1 given D1, M13, and P1 (OR gate)
    ('SRF', True, True, True): 0,   
    ('SRF', True, True, False): 1,  
    ('SRF', True, False, True): 1,  
    ('SRF', True, False, False): 1,
    ('SRF', False, True, True): 1,   
    ('SRF', False, True, False): 1,  
    ('SRF', False, False, True): 1,  
    ('SRF', False, False, False): 1, 
    # Conditional probabilities for M13 given M1 and M3 (AND gate)
    ('VSF', True, True, True, True): 0,   
    ('VSF', True, True, True, False): 1,  
    ('VSF', True, True, False, True): 1,  
    ('VSF', True, True, False, False): 1, 
    ('VSF', True, False, True, True): 1,  
    ('VSF', True, False, True, False): 1, 
    ('VSF', True, False, False, True): 1, 
    ('VSF', True, False, False, False): 1,
    ('VSF', False, True, True, True): 1,   
    ('VSF', False, True, True, False): 1,  
    ('VSF', False, True, False, True): 1,  
    ('VSF', False, True, False, False): 1, 
    ('VSF', False, False, True, True): 1,  
    ('VSF', False, False, True, False): 1, 
    ('VSF', False, False, False, True): 1, 
    ('VSF', False, False, False, False): 1,
    # Conditional probabilities for M23 given M2 and M3 (AND gate)
    ('SGF', True, True): 0,   
    ('SGF', True, False): 1,  
    ('SGF', False, True): 1,  
    ('SGF', False, False): 1, 

    # Conditional probabilities for TE given N and S12 (OR gate)
    ('TE', True, True): 0,   
    ('TE', True, False): 1,  
    ('TE', False, True): 1,  
    ('TE', False, False): 1,
}


# assignment = {'A': True, 'B': False, 'C': True, 'D': True, 'E': False}
evidence = {'D1': True}

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

