import sys
import bisect
import numpy as np
import pandas as pd
import re
import math

import matplotlib.pyplot as plt
import matplotlib.lines
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from spinup.utils.plot import get_datasets, get_all_datasets

animation_control = 'stop'
animation_offset = 0
animation_direction = 'right'

def observation_and_action_features(df):
    observation_features = []
    action_features = []
    for column in df.columns:
        if re.search("Observations\d+", column):
            observation_features.append(column)
        if re.search("Actions\d+", column):
            action_features.append(column)
    return observation_features, action_features

def principal_components(df, features):
    components = df.loc[:, features].values
    scaled_components = StandardScaler().fit_transform(components)
    pca = PCA(n_components=1)
    principal_components = pca.fit_transform(scaled_components)
    return principal_components.flatten()


def make_animation(all_logdirs, colormap_name='Spectral', num_visible_episodes=5, values=["Observations0", "Reward", "GoalError"]):

    colormap = colormap=cm.get_cmap(colormap_name)
    data = get_all_datasets(all_logdirs, filename="traces.txt")
    df = data[0]

    observation_features, action_features = observation_and_action_features(df)

    observations = principal_components(df, observation_features) if len(observation_features) > 1 else df['Observations0'].values
    actions = principal_components(df, action_features) if len(action_features) > 1 else df['Actions0'].values

    # Useful values
    num_goals = df['Goal'].max() + 1
    num_epochs = df['Epoch'].max()
    num_episodes = df['Episode'].max()
    unique_episodes = set(df['Episode'])

    goals = df.loc[:,['Goal']].values.flatten()
    rewards_scaled = MinMaxScaler((5, 40)).fit_transform(df.loc[:,['Reward']].values).flatten()

    goals_series = df['Goal']
    goals_runs_starts = goals_series.loc[goals_series.shift() != goals_series].index

    goals_predicted_series = df['GoalPredicted']

    episodes_series = df['Episode']
    episode_starts = episodes_series.loc[episodes_series.shift() != episodes_series]

    #  Create subplots
    fig, (ax_charts_root, ax_traces) = plt.subplots(2, 1, figsize = (20,10))

    ax_charts = []
    for i in range(len(values)):
        ax = ax_charts_root if i == 0 else ax_charts_root.twinx()
        value_series = df[values[i]]
        ax.set_ylim(value_series.min(), value_series.max())
        ax_charts.append(ax)

    # Traces
    ax_traces.set_title('Goals mapped over Action/Observation space', fontsize = 10)
    ax_traces.xlim=(observations.min(), observations.max())
    ax_traces.ylim=(actions.min(), actions.max())
    ax_traces.grid()

    plots = []

    def onKey(event):
        global animation_offset
        global animation_control
        if animation_control == 'stop' and event.key in ['left', 'right']:
            animation_direction = event.key
            animation_offset += num_visible_episodes if animation_direction == 'left' else -num_visible_episodes
            remove_old_plots()
        if event.key == ' ':
            animation_control = 'stop' if animation_control == 'go' else 'go'

    def init():
        ax_traces.set_title('Goals mapped over Action/Observation space')

    def animate(animation_step):
        episode = episode_from_step(animation_step)

        if animation_control == 'stop' and len(plots) > 0:
            return

        ax_traces.set_title('Goals mapped over Action/Observation space (episode {} of {})'.format(episode, num_episodes))

        remove_old_plots()

        episode_start, episode_end, episode_len = episode_start_end(episode, num_visible_episodes)

        plots.extend(plot_sccatter_traces(episode_start, episode_end, goals_series[episode_start:episode_end].values))
        plots.extend(plot_sccatter_traces(episode_start, episode_end, goals_predicted_series[episode_start:episode_end].values, 0.5))

        ax_charts[0].set(xlim=(episode_start, episode_end))
        colors = ['r', 'b', 'g', 'y', 'm', 'c']
        lines = []
        for i in range(len(values)):
            value = values[i]
            color = colors[i % len(colors)]
            scatter, line = plot_value_over_time(ax_charts[i], episode_start, episode_end, value, color, i==0)
            if i != 0:
                ax_charts[i].spines['right'].set_color(color)
                ax_charts[i].spines['right'].set_position(('axes', 1 + (i - 1) * 0.03))
            lines.append(line)
            plots.append(line)
            if scatter != None:
                plots.append(scatter)
        ax_charts[0].legend(lines, values, loc=0)

    def episode_from_step(animation_step):
        global animation_offset
        idx = int((animation_step - animation_offset) % len(episode_starts.values))
        if animation_control == 'stop':
            animation_offset += 1
        return episode_starts.values[idx]

    def episode_start_end(episode, num_episodes=1):
        episode_idxs = episode_starts.loc[episode_starts >= episode]
        start = episode_idxs.index[0]
        last_episode_len_idx = num_episodes if num_episodes < len(episode_idxs) else len(episode_idxs) - 1
        end = episode_idxs.index[last_episode_len_idx] - 1

        return start, end, end - start

    def remove_old_plots():
        for plot in plots:
            plot.remove()
        plots.clear()

    def plot_sccatter_traces(start, end, goal_values, scale_factor=1.0):
        scatter = ax_traces.scatter(observations[start:end], actions[start:end], c=goal_values,
                                    s=rewards_scaled[start:end]*scale_factor, cmap=colormap, marker='o',
                                    vmin=0, vmax=(num_goals - 1))

        goal_markers = []
        for goal in range(num_goals):
            goal_markers.append(matplotlib.lines.Line2D([], [], color=goal_color(goal), marker='o', linestyle='None',
                                                        markersize=5, label=str(goal)))
        legend = plt.legend(handles=goal_markers, fontsize='small', bbox_to_anchor=(-0.05, 1.0))
        return [scatter]

    def plot_value_over_time(ax, start, end, column_name, color, show_scatter=True):
        if not column_name in df.columns:
            raise

        ep_df = df.iloc[start:end]

        x = np.arange(start, end)
        y = ep_df[column_name].values
        line = ax.plot(x, y, lw=0.5, c=color, label=column_name)
        if show_scatter:
            scatter =  ax.scatter(x, y, c=ep_df['Goal'].values, cmap=colormap, marker='o', s=5, vmin=0, vmax=(num_goals - 1))
        else:
            scatter = None

        return scatter, line[0]

    def goal_color(goal):
        return colormap(float(goal / (num_goals - 1)))


    fig.canvas.mpl_connect('key_press_event', onKey)
    anim = FuncAnimation(fig, animate, init_func=init, interval=300, frames=sys.maxsize)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--colormap', '-c', default='Spectral')
    parser.add_argument('--visible_episodes', '-v', type=int, default=5)
    parser.add_argument('--value', '-y', default='Reward', nargs='*')
    args = parser.parse_args()
    """

    Args:
        logdir (strings): As many log directories (or prefixes to log
            directories, which the plotter will autocomplete internally) as
            you'd like to plot from.

    """

    make_animation(args.logdir, args.colormap, args.visible_episodes)

    #TODO use args.value
    # make_animation(args.logdir, args.colormap, args.visible_episodes, args.value)

if __name__ == "__main__":
    main()
