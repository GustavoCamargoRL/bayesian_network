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
        print("Node being processed:", node)
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

nodes = ['E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 
         'E17', 'E18', 'E19', 'E20', 'E21' , 'E22', 'E23' , 'E24', 'E25', 'DVB' ,'VSF', 
         'DS', 'IE', 'SS', 'CAT', 'SF', 'TE'] #nodes = ['A', 'B', 'C', 'D', 'E']
edges = [('E10', 'DVB'), ('E11', 'DVB'), ('E12', 'DVB'), ('E13', 'VSF'), ('E14', 'VSF'), ('E15', 'VSF'),
         ('E16', 'DS'), ('E17', 'DS'), ('E18', 'DS'), ('E19', 'IE'), ('E20', 'IE'), ('E21', 'IE'), ('E22', 'IE'),
         ('E22', 'SS'), ('E23', 'SS'), ('E24', 'SS'), ('DVB', 'CAT'), ('VSF', 'CAT'), ('DS', 'CAT'), ('IE', 'SF'),
         ('SS', 'SF'), ('E25','SF'), ('CAT','TE'), ('SF','TE')] 

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
    ('E10', True): beta_mean(1.12, 184.78),
    ('E10', False):1 -  beta_mean(1.12, 184.78),
    ('E11', True): beta_mean(1.15, 186.97),
    ('E11', False):1 -  beta_mean(1.15, 186.97),
    ('E12', True): beta_mean(1.15, 187.21),
    ('E12', False):1 -  beta_mean(1.15, 187.21),
    ('E13', True): beta_mean(1.15, 186.92),
    ('E13', False):1 -  beta_mean(1.15, 186.92),
    ('E14', True): beta_mean(1.14, 186.19),
    ('E14', False):1 -  beta_mean(1.14, 186.19),
    ('E15', True): beta_mean(1.15, 187.21),
    ('E15', False):1 - beta_mean(1.15, 187.21),
    ('E16', True): beta_mean(1.14, 186.22),
    ('E16', False):1 - beta_mean(1.14, 186.22),
    ('E17', True): beta_mean(1.15, 187.28),
    ('E17', False):1 -  beta_mean(1.15, 187.28),
    ('E18', True): beta_mean(1.14, 185.96),
    ('E18', False):1 -  beta_mean(1.14, 185.96),
    ('E19', True): beta_mean(1.15, 187.12),
    ('E19', False):1 -  beta_mean(1.15, 187.12),
    ('E20', True): beta_mean(1.16, 187.37),
    ('E20', False):1 -  beta_mean(1.16, 187.37),
    ('E21', True): beta_mean(1.11, 184.04),
    ('E21', False):1 -  beta_mean(1.11, 184.04),
    ('E22', True): beta_mean(1.13, 185.16),
    ('E22', False):1 -  beta_mean(1.13, 185.16),
    ('E23', True): beta_mean(1.13, 185.53),
    ('E23', False):1 -  beta_mean(1.13, 185.53),
    ('E24', True): beta_mean(1.15, 187.04),
    ('E24', False):1 -  beta_mean(1.15, 187.04),
    ('E25', True): beta_mean(1.15, 187.13),
    ('E25', False):1 -  beta_mean(1.15, 187.13),


    ('DVB', True, True, True): 1,   
    ('DVB', True, True, False): 1,  
    ('DVB', True, False, True): 1,  
    ('DVB', True, False, False): 1,
    ('DVB', False, True, True): 1,   
    ('DVB', False, True, False): 1,  
    ('DVB', False, False, True): 1,  
    ('DVB', False, False, False): 0, 

    ('VSF', True, True, True): 1,   
    ('VSF', True, True, False): 1,  
    ('VSF', True, False, True): 1,  
    ('VSF', True, False, False): 1,
    ('VSF', False, True, True): 1,   
    ('VSF', False, True, False): 1,  
    ('VSF', False, False, True): 1,  
    ('VSF', False, False, False): 0, 
    
    ('DS', True, True, True): 1,   
    ('DS', True, True, False): 1,  
    ('DS', True, False, True): 1,  
    ('DS', True, False, False): 1,
    ('DS', False, True, True): 1,   
    ('DS', False, True, False): 1,  
    ('DS', False, False, True): 1,  
    ('DS', False, False, False): 0, 

    ('IE', True, True, True, True): 1,   
    ('IE', True, True, True, False): 1,  
    ('IE', True, True, False, True): 1,  
    ('IE', True, True, False, False): 1, 
    ('IE', True, False, True, True): 1,  
    ('IE', True, False, True, False): 1, 
    ('IE', True, False, False, True): 1, 
    ('IE', True, False, False, False): 1,
    ('IE', False, True, True, True): 1,   
    ('IE', False, True, True, False): 1,  
    ('IE', False, True, False, True): 1,  
    ('IE', False, True, False, False): 1, 
    ('IE', False, False, True, True): 1,  
    ('IE', False, False, True, False): 1, 
    ('IE', False, False, False, True): 1, 
    ('IE', False, False, False, False): 0,

    ('SS', True, True, True): 1,   
    ('SS', True, True, False): 1,  
    ('SS', True, False, True): 1,  
    ('SS', True, False, False): 1,
    ('SS', False, True, True): 1,   
    ('SS', False, True, False): 1,  
    ('SS', False, False, True): 1,  
    ('SS', False, False, False): 0, 

    ('CAT', True, True, True): 1,   
    ('CAT', True, True, False): 1,  
    ('CAT', True, False, True): 1,  
    ('CAT', True, False, False): 1,
    ('CAT', False, True, True): 1,   
    ('CAT', False, True, False): 1,  
    ('CAT', False, False, True): 1,  
    ('CAT', False, False, False): 0, 

    ('SF', True, True, True): 1,   
    ('SF', True, True, False): 1,  
    ('SF', True, False, True): 1,  
    ('SF', True, False, False): 1,
    ('SF', False, True, True): 1,   
    ('SF', False, True, False): 1,  
    ('SF', False, False, True): 1,  
    ('SF', False, False, False): 0, 


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

