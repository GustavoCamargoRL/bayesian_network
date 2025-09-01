import numpy as np
from itertools import product, combinations

# -------------------------------------------
# Noisy gates (Excel-inspired formulas)
# -------------------------------------------

def noisy_or_sum_over_combos(p_list):
    """Compute noisy-OR probability using sum over parent combinations."""
    if not p_list:
        return 0.0
    total = 0.0
    n = len(p_list)
    idxs = range(n)
    for r in range(1, n+1):
        for S in combinations(idxs, r):
            # Effect term: 1 - ∏(1 - p_i), i ∈ S
            effect = 1.0
            for i in S:
                effect *= (1 - p_list[i])
            effect = 1 - effect
            # Evidence weight: ∏_{i∈S}(1 - p_i) * ∏_{j∉S} p_j
            evid = 1.0
            for i in S:
                evid *= (1 - p_list[i])
            for j in idxs:
                if j not in S:
                    evid *= p_list[j]
            total += effect * evid
    return total

def noisy_and_sum_over_combos(p_list):
    """Compute noisy-AND probability by swapping p <-> (1-p) in noisy-OR."""
    if not p_list:
        return 0.0
    q_list = [1 - p for p in p_list]
    q_total = noisy_or_sum_over_combos(q_list)
    return 1 - q_total

# -------------------------------------------
# Prior computation for marginals
# -------------------------------------------

def prior_with_noisy_nodes(nodes, edges, priors, noisy_nodes):
    """
    Compute marginal probabilities of all nodes in topological order.
    - priors: {(node, True/False): prob} for root nodes
    - noisy_nodes: dict {node: 'OR'/'AND'} for non-root nodes
    """
    def get_parents(node):
        return [edge[0] for edge in edges if edge[1] == node]

    marginals = {}
    for node in nodes:
        parents = get_parents(node)
        if not parents:
            # Root node
            marginals[node] = priors[(node, True)]
        else:
            parent_probs = [marginals[p] for p in parents]
            if noisy_nodes[node] == 'OR':
                marginals[node] = noisy_or_sum_over_combos(parent_probs)
            elif noisy_nodes[node] == 'AND':
                marginals[node] = noisy_and_sum_over_combos(parent_probs)
            else:
                raise ValueError(f"Invalid gate type for node {node}")
    return marginals

# -------------------------------------------
# Posterior inference
# -------------------------------------------

def net_bayes(nodes, edges, priors, noisy_nodes, evidence):
    """
    Compute posterior probability of each node being True given evidence.
    """

    def get_parents(node):
        return [edge[0] for edge in edges if edge[1] == node]

    hidden_nodes = [n for n in nodes if n not in evidence]
    assignments = list(product([True, False], repeat=len(hidden_nodes)))

    posteriors = {}
    for node in nodes:
        if node in evidence:
            posteriors[node] = 1.0 if evidence[node] else 0.0
        else:
            num = 0.0
            denom = 0.0
            for assign in assignments:
                full_state = evidence.copy()
                for n, v in zip(hidden_nodes, assign):
                    full_state[n] = v

                # Joint probability for this assignment
                joint = 1.0
                for n in nodes:
                    parents = get_parents(n)
                    if not parents:
                        # Root node
                        prob = priors[(n, True)] if full_state[n] else priors[(n, False)]
                    else:
                        # Collect parent states (True/False)
                        parent_states = [full_state[p] for p in parents]
                        # Convert them into probabilities: True->1, False->0
                        parent_probs = [1.0 if st else 0.0 for st in parent_states]

                        if noisy_nodes[n] == 'OR':
                            prob_true = noisy_or_sum_over_combos(parent_probs)
                        else:
                            prob_true = noisy_and_sum_over_combos(parent_probs)

                        prob = prob_true if full_state[n] else 1 - prob_true

                    joint *= prob

                if full_state[node]:
                    num += joint
                denom += joint

            posteriors[node] = num / denom if denom > 0 else 0.0
    return posteriors

# -------------------------------------------
# Helper for Beta priors
# -------------------------------------------

def Beta(a, b):
    return a/(a+b)

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

# -------------------------------------------
# Example usage
# -------------------------------------------

root_nodes = ['E1','E2','E3','E4','E5','E6','E7','E8','E9']

nodes_distributions = {
    ('E1','Beta'):[1.1,75.69], ('E2','Beta'):[1.1,85.40], ('E3','Beta'):[1.1,85.40],
    ('E4','Beta'):[1.1,86.95], ('E5','Beta'):[1.1,86.95], ('E6','Beta'):[1.89,159.93],
    ('E7','Beta'):[1.15,123.01], ('E8','Beta'):[2.41,149.22], ('E9','Beta'):[5.24,192.90]
}

nodes = ['E1','E2','E3','E4','E5','E6','E7','E8','E9','VSF','SF','CAT','SFP','VF']

edges = [
    ('E2','VSF'), ('E3','VSF'), ('E4','VSF'), ('E5','VSF'),
    ('E6','SF'), ('E7','SF'), ('E8','SF'),
    ('E1','CAT'), ('VSF','CAT'),
    ('E9','SFP'), ('SF','SFP'),
    ('CAT','VF'), ('SFP','VF')
]

noisy_nodes = {
    'VSF': 'OR',
    'SF' : 'OR',
    'CAT': 'OR',
    'SFP': 'OR',
    'VF' : 'OR'
}

priors = noisy_priors(root_nodes, nodes_distributions)
marginals = prior_with_noisy_nodes(nodes, edges, priors, noisy_nodes)
print("Marginals:", marginals)

evidence = {'E1': True}
posteriors = net_bayes(nodes, edges, priors, noisy_nodes, evidence)
print("Posteriors given evidence:", posteriors)
