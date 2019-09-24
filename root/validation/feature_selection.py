import itertools
from k_fold import kfold_validate

test_tuples = [(1,2), (1,4), (5,6), (7,8)]

# Generate every possible combination of tuples
def combine_tuples(tuple_list, count):
    combinations = list(itertools.combinations(tuple_list, count))
    for i in range(0, len(combinations)):
        # flatten tuples then remove duplicates
        combinations[i] = sum(combinations[i], ())
        combinations[i] = tuple(set(combinations[i])) # TODO: Do we need something less hacky to remove duplicates?
    return combinations


# Gradually grow feature set size by eliminating smaller feature combinations that have weaker predictive power.
def select_features(dataframe, num_features, gradient):

