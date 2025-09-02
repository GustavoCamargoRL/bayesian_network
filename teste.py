import numpy as np
from itertools import product

def prior_true_probabilities(nodes, edges, priors):
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
                key = (node,) + combo
                prob_true += parent_prob * priors[key]
            marginals[node] = prob_true
    return marginals

def get_parents(node):
        return [edge[0] for edge in edges if edge[1] == node]

def noisy_or(probabilities, combo, leak=0.0):
    """Noisy-OR: chance de Y=1 dado que pelo menos um Xi=1"""
    prodTrue = 1.0
    prodAll = 1.0
    for n, prob in enumerate(probabilities):
        if combo[n] == True:
            prodTrue = prodTrue * (1 - prob)
            prodAll = prodAll * (1 - prob)
        else:
            prodAll = prodAll * (1 - prob)
        
        
    return (1 - (1 - leak) * prodTrue) * prodAll


def noisy_and(probabilities, combo, leak=0.0):
    """Noisy-AND: chance de Y=1 dado que todos Xi=1"""
    prodTrue = 1.0
    prodAll = 1.0
    for n, prob in enumerate(probabilities):
        if combo[n] == False:
            prodTrue *= (prob)
            prodAll *= (prob)
        else:
            prodAll *= (1 - prob)
        
        
    return ((1 - leak) * prodTrue) * prodAll


def Beta(a, b):
    return (a / (a + b))


def prob_priors(nodes_with_no_parents, nodes_distributions):
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
    for node in nodes:
        if node in noisy_nodes:
            parents = get_parents(node)
            combos = list(product([True, False], repeat=len(parents)))
            for combo in combos:
                parent_probs = []
                for p, val in zip(parents, combo):
                        parent_probs.append(priors[(p, val)])
                if noisy_nodes[node] == 'OR':
                    prob_true = noisy_or(parent_probs, combo)
                elif noisy_nodes[node] == 'AND':
                    prob_true = noisy_and(parent_probs, combo)
                else:
                    raise ValueError(f"Unknown noisy function {noisy_nodes[node]} for node {node}")
                intermed_priors[(node,) + combo] = prob_true
                priors[node, True] = priors[node, True] + prob_true
                priors[node, False] = priors[node, False] + (1-prob_true)
                #priors[(node,) + tuple(not v for v in combo)] = 1 - prob_true

    return priors



# -----------------------------
# Example usage
# -----------------------------

root_nodes = ['E6','E7','E8','E9']

nodes_distributions = {
    ('E6','Beta'):[1.89,159.93], ('E7','Beta'):[1.15,123.01], ('E8','Beta'):[2.41,149.22], ('E9','Beta'):[5.24,192.90] }

nodes = ['E6','E7','E8','E9', 'SF','SFP']

edges = [
    ('E6','SF'), ('E7','SF'), ('E8','SF'),('E9','SFP'), ('SF','SFP')]

noisy_nodes = {
    'SF':'OR',
    'SFP': 'OR'
}

#root_nodes = ['E1','E2','E3','E4','E5','E6','E7','E8','E9']
#nodes_distributions = {
#    ('E1','Beta'):[1.1,75.69], ('E2','Beta'):[1.1,85.40], ('E3','Beta'):[1.1,85.40],
#    ('E4','Beta'):[1.1,86.95], ('E5','Beta'):[1.1,86.95], ('E6','Beta'):[1.89,159.93],
#    ('E7','Beta'):[1.15,123.01], ('E8','Beta'):[2.41,149.22], ('E9','Beta'):[5.24,192.90]
#}

#nodes = ['E1','E2','E3','E4','E5','E6','E7','E8','E9','VSF','SF','CAT','SFP','VF']
#edges = [
#    ('E2','VSF'), ('E3','VSF'), ('E4','VSF'), ('E5','VSF'),
#    ('E6','SF'), ('E7','SF'), ('E8','SF'),
#    ('E1','CAT'), ('VSF','CAT'),
#    ('E9','SFP'), ('SF','SFP'),
#    ('CAT','VF'), ('SFP','VF')
#]
#noisy_nodes = {
#    'VSF':'OR','SF':'OR','CAT':'OR','SFP':'OR','VF':'OR'
#}

intermed_priors = {}

# Build priors with evidence
#evidence = {'E2': True}

priors = prob_priors(root_nodes, nodes_distributions)

## fill in the rest of the priors with uniform distributions before computing marginals
for node in nodes:
    check = (node, True)
    if check in priors.keys():
        continue
    else:
        priors[node, True] = 0
        priors[node, False] = 0
        parents = get_parents(node)
        combos = list(product([True, False], repeat=len(parents)))
        for combo in combos:
            intermed_priors[node, *combo] = 1.0

priors = noisy_priors(noisy_nodes, nodes, priors, intermed_priors)
print(priors,"\n\n")
print(intermed_priors, "\n\n")
# Compute posterior
#posteriors = net_bayes(nodes, edges, priors, evidence)
#print("Posteriors:", posteriors)
