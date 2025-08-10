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

nodes = ['A', 'B', 'C', 'D', 'E']
edges = [('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')]
# Prior and conditional probabilities
priors = {
    # Marginal probabilities for root nodes
    ('A', True): 0.8,
    ('A', False): 0.2,
    ('B', True): 0.1,
    ('B', False): 0.9,
    # Conditional probabilities for C given A and B
    ('C', True, True): 0.9,    # C | A=True, B=True
    ('C', True, False): 0.75,  # C | A=True, B=False
    ('C', False, True): 0.6,   # C | A=False, B=True
    ('C', False, False): 0.05, # C | A=False, B=False
    # Conditional probabilities for D given B
    ('D', True): 0.2,   # D | B=True
    ('D', False): 0.75, # D | B=False
    # Conditional probabilities for E given C
    ('E', True): 0.15,  # E | C=True
    ('E', False): 0.95  # E | C=False
}

assignment = {'A': True, 'B': False, 'C': True, 'D': True, 'E': False}
evidence = {'A': True, 'C': True}

prior_probs = prior_true_probabilities(nodes, edges, priors)
print(prior_probs)

posteriors = netBayes(nodes, edges, priors, evidence)
print(posteriors)

overall_prob = joint_probability(nodes, edges, priors, assignment)
print("Joint probability of evidence:", overall_prob)

