import numpy as np
from itertools import product


def get_parents(node):
    return [edge[0] for edge in edges if edge[1] == node]


def noisy_or(node, non_root_nodes, combo, parents, leak=0.0):
    """Noisy-OR: chance of Y=1 given at least one Xi=1"""
    prodTrue = 1.0
    for n, val in enumerate(combo):  # prob = pi
        if val is True:
            prodTrue = prodTrue * (1 - non_root_nodes[(node, parents[n])])
    return (1 - (1 - leak) * prodTrue)


def noisy_and(node, non_root_nodes, combo, parents, leak=0.0):
    """Noisy-AND: chance of Y=1 given all Xi=1"""
    prodTrue = 1.0
    for n, prob in enumerate(combo):  # prob = pi
        if combo[n] is False:
            prodTrue *= prob
    return ((1 - leak) * prodTrue)


def Beta(a, b):
    return (a / (a + b))


def prob_priors(nodes_with_no_parents, nodes_distributions):
    """Compute priors for root nodes using Beta distributions"""
    priors = {}
    for node in nodes_with_no_parents:
        if (node, 'Beta') in nodes_distributions:
            a, b = nodes_distributions[(node, 'Beta')]
            priors[(node, True)] = Beta(a, b)
            priors[(node, False)] = 1 - Beta(a, b)
        else:
            raise ValueError(f"No distribution found for node {node}")
    return priors


def noisy_priors(noisy_nodes, nodes, priors, intermed_priors):
    """Initialize priors for noisy nodes"""
    for node in nodes:
        if node in noisy_nodes:
            parents = get_parents(node)
            combos = list(product([True, False], repeat=len(parents)))
    return priors


def noisy_propagation(intermed_priors, non_root_nodes, nodes, noisy_nodes):
    """Compute intermediate probabilities for noisy-OR and noisy-AND nodes"""
    for node in nodes:
        if node in noisy_nodes:
            parents = get_parents(node)
            combos = list(product([True, False], repeat=len(parents)))
            for combo in combos:
                if noisy_nodes[node] == 'OR':
                    prob_true = noisy_or(node, non_root_nodes, combo, parents)
                elif noisy_nodes[node] == 'AND':
                    prob_true = noisy_and(node, non_root_nodes, combo, parents)
                else:
                    raise ValueError(f"Unknown noisy function {noisy_nodes[node]} for node {node}")
                intermed_priors[(node,) + combo] = prob_true
    return intermed_priors


def update_child_priors(priors, intermed_priors, nodes, noisy_nodes):
    """
    Update child priors using intermediate probabilities.
    P(child=True) = sum_over_combos( P(child|combo) * product(P(parent=state)) )
    P(child=False) = 1 - P(child=True)
    """
    for node in nodes:
        if node in noisy_nodes:  # only update noisy nodes
            parents = get_parents(node)
            combos = list(product([True, False], repeat=len(parents)))

            prob_true = 0.0
            for combo in combos:
                # conditional probability P(child|combo)
                p_child_given_combo = intermed_priors[(node,) + combo]

                # probability of the parents being in this combo
                p_parents_combo = 1.0
                for p, state in zip(parents, combo):
                    p_parents_combo *= priors[(p, state)]

                prob_true += p_child_given_combo * p_parents_combo

            priors[(node, True)] = prob_true
            priors[(node, False)] = 1 - prob_true

    return priors


def apply_evidence(priors, evidence):
    """
    Apply evidence directly into priors.
    evidence: dict {node: True/False}
    """
    for node, val in evidence.items():
        priors[(node, True)] = 1.0 if val else 0.0
        priors[(node, False)] = 0.0 if val else 1.0
    return priors


def update_network(priors, nodes, noisy_nodes, non_root_nodes, evidence=None):
    """
    Recompute the Bayesian network after applying evidence.
    1. Apply evidence
    2. Recompute intermediate probabilities
    3. Update child priors
    """
    # Step 1: Apply evidence
    if evidence:
        priors = apply_evidence(priors, evidence)

    # Step 2: Recompute intermediate probabilities
    intermed_priors = {}
    for node in nodes:
        if node in noisy_nodes:
            parents = get_parents(node)
            combos = list(product([True, False], repeat=len(parents)))
            for combo in combos:
                intermed_priors[(node,) + combo] = 1.0  # initialize

    intermed_priors = noisy_propagation(intermed_priors, non_root_nodes, nodes, noisy_nodes)

    # Step 3: Update child priors
    priors = update_child_priors(priors, intermed_priors, nodes, noisy_nodes)

    return priors, intermed_priors


# -----------------------------
# Example usage
# -----------------------------

#root_nodes = ['E6','E7','E8','E9']

#nodes_distributions = {
#    ('E6','Beta'):[1.89,159.93], ('E7','Beta'):[1.15,123.01], ('E8','Beta'):[2.41,149.22], ('E9','Beta'):[5.24,192.90] }


#nodes = ['E6','E7','E8','E9', 'SF','SFP']

#edges = [
#    ('E6','SF'), ('E7','SF'), ('E8','SF'),('E9','SFP'), ('SF','SFP')]

#noisy_nodes = {
#    'SF':'OR',
#    'SFP': 'OR'
#}


root_nodes = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9']
nodes_distributions = {
    ('E1', 'Beta'): [1.1, 75.69], ('E2', 'Beta'): [1.1, 85.40], ('E3', 'Beta'): [1.1, 85.40],
    ('E4', 'Beta'): [1.1, 86.95], ('E5', 'Beta'): [1.1, 86.95], ('E6', 'Beta'): [1.89, 159.93],
    ('E7', 'Beta'): [1.15, 123.01], ('E8', 'Beta'): [2.41, 149.22], ('E9', 'Beta'): [5.24, 192.90]
}

non_root_nodes = {
    ('VSF', 'E2'): 0.9, ('VSF', 'E3'): 0.8, ('VSF', 'E4'): 0.7, ('VSF', 'E5'): 0.6,
    ('SF', 'E6'): 0.5, ('SF', 'E7'): 0.6, ('SF', 'E8'): 0.7,
    ('CAT', 'E1'): 0.8, ('CAT', 'VSF'): 0.9,
    ('SFP', 'E9'): 0.8, ('SFP', 'SF'): 0.7,
    ('VF', 'CAT'): 0.6, ('VF', 'SFP'): 0.9
}

nodes = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'VSF', 'SF', 'CAT', 'SFP', 'VF']
edges = [
    ('E2', 'VSF'), ('E3', 'VSF'), ('E4', 'VSF'), ('E5', 'VSF'),
    ('E6', 'SF'), ('E7', 'SF'), ('E8', 'SF'),
    ('E1', 'CAT'), ('VSF', 'CAT'),
    ('E9', 'SFP'), ('SF', 'SFP'),
    ('CAT', 'VF'), ('SFP', 'VF')
]
noisy_nodes = {
    'VSF': 'OR', 'SF': 'OR', 'CAT': 'OR', 'SFP': 'OR', 'VF': 'OR'
}

intermed_priors = {}
priors = prob_priors(root_nodes, nodes_distributions)

# Initialize priors for noisy nodes
for node in nodes:
    if (node, True) not in priors:
        priors[node, True] = 0
        priors[node, False] = 0
        parents = get_parents(node)
        combos = list(product([True, False], repeat=len(parents)))
        for combo in combos:
            intermed_priors[node, *combo] = 1.0

# Initial propagation
intermed_priors = noisy_propagation(intermed_priors, non_root_nodes, nodes, noisy_nodes)
priors = update_child_priors(priors, intermed_priors, nodes, noisy_nodes)

print("Initial priors:", priors, "\n")

# Example: set evidence
evidence = {'E2': False, 'E7': False}
priors, intermed_priors = update_network(priors, nodes, noisy_nodes, non_root_nodes, evidence)

print("Priors after evidence:", priors, "\n")
