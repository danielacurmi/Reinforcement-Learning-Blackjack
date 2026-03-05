import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import os

pd.set_option('future.no_silent_downcasting', True)

class DataVis:
    @staticmethod
    def plot_win_rate_per100_batch(results_list, labels_list, algo_name):
        fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True)
        fig.suptitle(f"Win/Loss/Draw Rates - {algo_name}", fontsize=24) 

        for idx, (results, label) in enumerate(zip(results_list, labels_list)):
            ax = axs[idx // 2, idx % 2]
            wins = [entry['win'] for entry in results]
            losses = [entry['loss'] for entry in results]
            draws = [entry['draw'] for entry in results]
            episodes = [i * 1000 for i in range(1, len(wins) + 1)]

            ax.plot(episodes, wins, label='Wins', color='green', linewidth=2)  
            ax.plot(episodes, losses, label='Losses', color='red', linewidth=2)
            ax.plot(episodes, draws, label='Draws', color='blue', linewidth=2) 
            ax.set_title(label, fontsize=18) 
            ax.set_xlabel("Episodes", fontsize=14)
            ax.set_ylabel("Count", fontsize=14)  
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(True)

        fig.legend(['Wins', 'Losses', 'Draws'], 
            loc='upper right', ncol=3, fontsize=14)
            
        plt.tight_layout()
        output_dir = "Plots/WinRate"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{algo_name}.png")
        plt.close()

    @staticmethod
    def plot_sa_counts_batch(sa_pairs_list, labels_list, algo_name, top_n=20):
        fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharex=False, sharey=True)
        fig.suptitle(f"State-Action Counts Top {top_n} - {algo_name}", fontsize=18)

        for idx, (sa_pairs_counter, label) in enumerate(zip(sa_pairs_list, labels_list)):
            ax = axs[idx // 2, idx % 2]
            sa_pairs_sorted = sa_pairs_counter.most_common(top_n)
            sa_pairs, count = zip(*sa_pairs_sorted)

            transformed_sa_pairs = []
            for sa_pair in sa_pairs:
                state, action = sa_pair
                player_total, dealer_card, usable_ace = state
                transformed_sa_pairs.append(f"({player_total}, {dealer_card}, {usable_ace}), {action}")

            ax.bar(transformed_sa_pairs, count)
            ax.set_title(label, fontsize = 18)
            ax.set_xlabel("State-Action")
            ax.set_ylabel("Count")
            ax.tick_params(axis='x', rotation=90, labelsize=14)
            ax.tick_params(axis='y', labelsize=14)


        # Super annotation
        fig.text(0.5, 0.02,
                 "<State> = (player_total, dealer_card, usable_ace)    |    <Action> = Hit or Stand",
                 ha='center', va='center', fontsize=12, color='gray')

        plt.tight_layout()
        output_dir = "Plots/StateActionCounts"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{algo_name}.png")
        plt.close()

    @staticmethod
    def plot_total_unique_sa_count_each_config(counts: list[int], configs: list[str], algo_name: str, label: str) -> None:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(configs, counts)
        plt.title(f"Total Unique State Action Count Per Config - {label}")
        plt.xlabel("Config")
        plt.xticks(rotation=45, fontsize=10)
        plt.ylabel("Unique (State, Action) Count")
        plt.tight_layout()

        # Annotate each bar with its value
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height, 
                f"{int(height)}", 
                ha='center', va='bottom', fontsize=10  
            )

        output_dir = "Plots/TotalUniqueSACounts"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{label}.png")
        plt.close()

    @staticmethod
    def _bj_strat_table(Q: defaultdict):
        dealer_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        player_totals = list(range(20, 11, -1))

        rows = []
        values = []

        for usable_ace in [True, False]:
            for player_total in player_totals:
                row = (usable_ace, player_total)
                row_values = []
                for dealer_card in dealer_cards:
                    state = (player_total, dealer_card, usable_ace)
                    if state in Q:
                        best_action = max(Q[state], key=Q[state].get)
                        row_values.append('H' if best_action == 'hit' else 'S')
                    else:
                        row_values.append('')
                rows.append(row)
                values.append(row_values)

        index = pd.MultiIndex.from_tuples(rows, names=["Usable Ace", "Player Total"])
        df = pd.DataFrame(values, index=index, columns=dealer_cards)

        return df

    @staticmethod
    def plot_strat_table_batch(Qs_list, labels_list, algo_name):
        fig, axs = plt.subplots(4, 2, figsize=(20, 24), sharex=True, sharey=True)
        fig.suptitle(f"Strategy Tables - {algo_name}", fontsize=24) 

        for row_idx, (Q, label) in enumerate(zip(Qs_list, labels_list)):
            df = DataVis._bj_strat_table(Q)
            df_useable_ace = df.xs(True)
            df_no_useable_ace = df.xs(False)

            mapper = {'H': 1, 'S': 0, '': np.nan}
            df_useable_ace_vals = df_useable_ace.replace(mapper).infer_objects()
            df_no_useable_vals = df_no_useable_ace.replace(mapper).infer_objects()

            sns.heatmap(df_useable_ace_vals, cmap="coolwarm", cbar=False,
                        linewidths=0.5, linecolor='gray', annot=df_useable_ace, fmt='', 
                        annot_kws={"fontsize": 14}, ax=axs[row_idx, 0])   
            axs[row_idx, 0].set_title(f"{label} - Usable Ace", fontsize=18) 
            axs[row_idx, 0].set_xlabel("Dealer's Card", fontsize=14)  
            axs[row_idx, 0].set_ylabel("Player's Total", fontsize=14) 
            axs[row_idx, 0].tick_params(axis='x', labelsize=12)  
            axs[row_idx, 0].tick_params(axis='y', labelsize=12)  

            sns.heatmap(df_no_useable_vals, cmap="coolwarm", cbar=False,
                        linewidths=0.5, linecolor='gray', annot=df_no_useable_ace, fmt='',
                        annot_kws={"fontsize": 12}, ax=axs[row_idx, 1]) 
            axs[row_idx, 1].set_title(f"{label} - No Usable Ace", fontsize=18)  
            axs[row_idx, 1].set_xlabel("Dealer's Card", fontsize=14) 
            axs[row_idx, 1].set_ylabel("", fontsize=14)  
            axs[row_idx, 1].tick_params(axis='x', labelsize=12)  
            axs[row_idx, 1].tick_params(axis='y', labelsize=12)  

        plt.tight_layout()
        output_dir = "Plots/StrategyTable"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{algo_name}.png")
        plt.close()

    @staticmethod
    def plot_dealer_advantage(all_stats: list, algos_names: list[str]):
        dealer_advs = []
        for stats in all_stats:
            sorted_stat = sorted(stats, key=lambda entry: entry["episode"], reverse=True)
            sorted_stat = sorted_stat[:11]

            dealer_advs.append(DataVis._calc_dealer_adv(stats))

        combined = list(zip(dealer_advs, algos_names))
        combined.sort(key=lambda x: x[0])
        dealer_advs_sorted, algos_names_sorted = zip(*combined)

        plt.figure(figsize=(20, 10))
        plt.bar(algos_names_sorted, dealer_advs_sorted)
        plt.title(f"Dealer Advantage vs Algorithms", fontsize=20) 
        plt.xlabel("Algorithm Configurations", fontsize=14)  
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.ylabel("Dealer Advantage", fontsize=14) 
        plt.yticks(fontsize=14)  
        plt.tight_layout()

        output_dir = "Plots/DealerAdvantage"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/DealerAdvantage.png")
        plt.close()

    @staticmethod
    def _calc_dealer_adv(stats):
        total_wins = sum(d['win'] for d in stats)
        total_losses = sum(d['loss'] for d in stats)
        dealer_advantage = (total_losses - total_wins) / (total_losses + total_wins)
        return dealer_advantage