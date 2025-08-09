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

prior_probs = prior_true_probabilities(nodes, edges, priors)
print(prior_probs)