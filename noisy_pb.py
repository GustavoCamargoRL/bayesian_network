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


root_nodes = nodes = ['E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 
         'E17', 'E18', 'E19', 'E20', 'E21' , 'E22', 'E23' , 'E24', 'E25']
nodes_distributions = {
    ('E10', 'Beta'): [1.12, 184.78], ('E11', 'Beta'): [1.15, 186.97], ('E12', 'Beta'): [1.15, 187.21],
    ('E13', 'Beta'): [1.15, 186.92], ('E14', 'Beta'): [1.14, 186.19], ('E15', 'Beta'): [1.15, 197.21],
    ('E16', 'Beta'): [1.14, 186.22], ('E17', 'Beta'): [1.15, 187.28], ('E18', 'Beta'): [1.14, 185.96], 
    ('E19', 'Beta'): [1.15, 187.12], ('E20', 'Beta'): [1.16, 187.37], ('E21', 'Beta'): [1.11, 184.04],
    ('E22', 'Beta'): [1.13, 185.16], ('E23', 'Beta'): [1.13, 185.53], ('E24', 'Beta'): [1.15, 187.04],
    ('E25', 'Beta'): [1.15, 187.13]
}

# Filho,Pai
non_root_nodes = {
    ('DVB', 'E10'): 0.99, ('DVB', 'E11'): 0.99, ('DVB', 'E12'): 0.99,
    ('VSF', 'E13'): 0.99, ('VSF', 'E14'): 0.99, ('VSF', 'E15'): 0.99,
    ('DS', 'E16'): 0.99,  ('DS', 'E17'): 0.99, ('DS', 'E18'): 0.99,
    ('IE', 'E19'): 0.99, ('IE', 'E20'): 0.99, ('IE', 'E21'): 0.99, ('IE', 'E22'): 0.99,
    ('SS', 'E22'): 0.99, ('SS', 'E23'): 0.99, ('SS', 'E24'): 0.99,
    ('CAT', 'DVB'): 0.99, ('CAT', 'VSF'): 0.99, ('CAT', 'DS'): 0.99,
    ('SF', 'IE'): 0.99, ('SF', 'SS'): 0.99, ('SF', 'E25'): 0.99,
    ('TE', 'CAT'): 0.99, ('TE', 'SF'): 0.99
}

nodes = ['E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 
         'E17', 'E18', 'E19', 'E20', 'E21' , 'E22', 'E23' , 'E24', 'E25', 'DVB' ,'VSF', 
         'DS', 'IE', 'SS', 'CAT', 'SF', 'TE'] #nodes = ['A', 'B', 'C', 'D', 'E']
# Pai, Filho
edges = [('E10', 'DVB'), ('E11', 'DVB'), ('E12', 'DVB'), ('E13', 'VSF'), ('E14', 'VSF'), ('E15', 'VSF'),
         ('E16', 'DS'), ('E17', 'DS'), ('E18', 'DS'), ('E19', 'IE'), ('E20', 'IE'), ('E21', 'IE'), ('E22', 'IE'),
         ('E22', 'SS'), ('E23', 'SS'), ('E24', 'SS'), ('DVB', 'CAT'), ('VSF', 'CAT'), ('DS', 'CAT'), ('IE', 'SF'),
         ('SS', 'SF'), ('E25','SF'), ('CAT','TE'), ('SF','TE')] 

noisy_nodes = {
    'DVB': 'OR', 'VSF': 'OR', 'DS': 'OR', 'IE': 'OR', 'SS': 'OR', 'CAT': 'OR', 'SF': 'OR', 'TE': 'OR'
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
evidence = {'TE': True}
priors, intermed_priors = update_network(priors, nodes, noisy_nodes, non_root_nodes, evidence)

print("Priors after evidence:", priors, "\n")
