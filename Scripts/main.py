import math
from blackjack import Blackjack
from rl_algorithms import RLAlgorithms, RLResult
from data_vis import DataVis

# Epsilon configurations
def ε_fixed(_):
    return 0.1

def ε_inv(n):
    return 1 / n

def ε_exp_fast(n):
    return pow(math.e, -n / 10000)

def ε_exp_slow(n):
    return pow(math.e, -n / 100000)

env = Blackjack()
agent = RLAlgorithms(env, gamma=1.0)

algorithms = [
    {
        "name": "MC",
        "mode": "montecarlo",
        "is_exploring_start": [True, False],
        "epsilons": {
            True: [ε_inv],
            False: [ε_inv, ε_exp_fast, ε_exp_slow]
        }
    },
    {
        "name": "SARSA",
        "mode": "sarsa",
        "is_exploring_start": [False],
        "epsilons": {
            False: [ε_fixed, ε_inv, ε_exp_fast, ε_exp_slow]
        }
    },
    {
        "name": "QL",
        "mode": "qlearning",
        "is_exploring_start": [False],
        "epsilons": {
            False: [ε_fixed, ε_inv, ε_exp_fast, ε_exp_slow]
        }
    },
    {
        "name": "DQL",
        "mode": "doubleq",
        "is_exploring_start": [False],
        "epsilons": {
            False: [ε_fixed, ε_inv, ε_exp_fast, ε_exp_slow]
        }
    }
]

all_labels = []
all_stats = []

for algo in algorithms:
    counts = []
    labels = []
    winrate_results = []
    sa_pairs_list = []
    Qs_list = []

    for explore_flag in algo["is_exploring_start"]: 
        for epsilon in algo["epsilons"][explore_flag]:
            if algo['mode'] == 'montecarlo':
                explore_label = "Es" if explore_flag else "NoES"
                label = f"{algo['name']}, {explore_label}, {epsilon.__name__}"
            else:
                label = f"{algo['name']}, {epsilon.__name__}"

            rlResult = agent.run_rl_agent(
                mode=algo["mode"],
                epsfunc=epsilon,
                is_exploring_start=explore_flag
            )

            # Collecting data for plots
            counts.append(rlResult.num_unique_sa)
            labels.append(label)
            all_labels.append(label)
            all_stats.append(rlResult.results)

            winrate_results.append(rlResult.results)
            sa_pairs_list.append(rlResult.sa_pairs)
            Qs_list.append(rlResult.Q1)

    # After all configs per algo:
    DataVis.plot_win_rate_per100_batch(winrate_results, labels, algo['name'])
    DataVis.plot_sa_counts_batch(sa_pairs_list, labels, algo['name'])
    DataVis.plot_strat_table_batch(Qs_list, labels, algo['name'])
    DataVis.plot_total_unique_sa_count_each_config(counts, labels, algo['name'], algo['name'])

# After all training:
DataVis.plot_dealer_advantage(all_stats, all_labels)

print("Done")