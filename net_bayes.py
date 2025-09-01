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

def noisy_or(probabilities):
    """
    Calculate the probability of at least one event occurring given individual probabilities.
    probabilities: list of probabilities [p1, p2, ..., pn]
    """
    prod = 1.0
    for p in probabilities:
        prod *= (1 - p)
    return 1 - prod

def noisy_and(probabilities):
    """
    Calculate the probability of all events occurring given individual probabilities.
    probabilities: list of probabilities [p1, p2, ..., pn]
    """
    prod = 1.0
    for p in probabilities:
        prod *= p
    return prod

def prior_with_noisy_nodes(nodes, edges, priors, noisy_nodes):
    """
    Compute prior probabilities considering noisy nodes.
    noisy_nodes: dict {node: 'AND'/'OR', ...}
    """
    def get_parents(node):
        return [edge[0] for edge in edges if edge[1] == node]

    marginals = {}
    for node in nodes:
        parents = get_parents(node)
        if not parents:
            marginals[node] = priors[(node, True)]
        else:
            combos = list(product([True, False], repeat=len(parents)))
            prob_true = 0.0
            for combo in combos:
                parent_prob = 1.0
                for p, val in zip(parents, combo):
                    parent_prob *= marginals[p] if val else (1 - marginals[p])
                if node in noisy_nodes:
                    probs = [priors[(p, True)] if val else 0.0 for p, val in zip(parents, combo)]
                    if noisy_nodes[node] == 'AND':
                        prob_node_true = noisy_and(probs)
                    elif noisy_nodes[node] == 'OR':
                        prob_node_true = noisy_or(probs)
                    else:
                        raise ValueError("Noisy node must be 'AND' or 'OR'")
                else:
                    key = (node,) + combo
                    prob_node_true = priors[key]
                prob_true += parent_prob * prob_node_true
            marginals[node] = prob_true
    return marginals

def Beta(a, b):
    return (a/(a+b))

def noisy_priors(nodes_with_no_parents, nodes_distributions):
    priors = {}
    for node in nodes_with_no_parents:
        if (node, 'Beta') in nodes_distributions:
            a, b = nodes_distributions[(node, 'Beta')]
            priors[(node, True)] = Beta(a, b)
            priors[(node, False)] = 1 - priors[(node, True)]
        else:
            raise ValueError(f"No distribution found for node {node}")
    return priors


root_nodes = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 
              'E7', 'E8', 'E9'] #root_nodes = ['A', 'B']

nodes_distributions = {
    ('E1', 'Beta') : [1.1,75.69] , ('E2', 'Beta'): [1.1,85.40], ('E3', 'Beta'): [1.1,85.40], ('E4', 'Beta'): [1.1,86.95], ('E5', 'Beta'): [1.1,86.95],
    ('E6', 'Beta'): [1.89,159.93], ('E7', 'Beta'): [1.15,123.01], ('E8', 'Beta'): [2.41,149.22], ('E9', 'Beta'): [5.24,192.90]}

nodes = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 
         'E7', 'E8', 'E9', 'SF', 'VSF', 
         'SFP', 'CAT', 'VF'] #nodes = ['A', 'B', 'C', 'D', 'E']
edges = [('CAT', 'E1'), ('VSF', 'E2'), ('VSF', 'E3'), ('VSF', 'E4'),
         ('VSF', 'E5'), ('SF', 'E6'), ('SF', 'E7'), ('SF', 'E8'),
         ('SFP', 'E9'), ('CAT','VSF'), ('SFP','SF'), ('VF','CAT'), 
         ('VF','SFP')] #parents = [('son1', 'father'), ('son2', 'father')] 




# Prior and conditional probabilities
priors = noisy_priors(root_nodes, nodes_distributions)
    # Marginal probabilities for root nodes
#    ('D11', True): 1 - np.exp(-lambda_D*t),
#    ('D11', False): np.exp(-lambda_D*t),
print(priors)

# assignment = {'A': True, 'B': False, 'C': True, 'D': True, 'E': False}
#evidence = {'TE': True}

#prior_probs = prior_true_probabilities(nodes, edges, priors)
#print(f"Prior probabilities:\n")
#for key, value in prior_probs.items():
#    print(f"{key}: {value:.12f}")

#posteriors = netBayes(nodes, edges, priors, evidence)
#print(f"Posterior probabilities given the evidence TE:\n")
#for key, value in posteriors.items():
#    print(f"{key}: {value:.12f}")


# overall_prob = joint_probability(nodes, edges, priors, assignment)
# print("Joint probability of evidence:", overall_prob)

