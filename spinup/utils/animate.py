import sys
import bisect
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from spinup.utils.plot import get_datasets, get_all_datasets

animation_control = 'stop'
animation_offset = 0
animation_direction = 'right'

def make_animation(all_logdirs, legend=None, xaxis=None, values=None, count=False,
                   font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', pca=False,
                   colormap_name='Spectral', num_visible_episodes=5, line_traces=False):

    colormap = colormap=cm.get_cmap(colormap_name)
    data = get_all_datasets(all_logdirs, legend, select, exclude, filename="pca.txt")
    df = data[0]


    observation_features = ['Observations0', 'Observations1']

    observations_components = df.loc[:, observation_features].values
    num_epochs = df['Epoch'].max()
    num_episodes = df['Episode'].max()
    unique_episodes = set(df['Episode'])

    observations_components = StandardScaler().fit_transform(observations_components)
    pca = PCA(n_components=1)
    observations_principal_components = pca.fit_transform(observations_components)
    observations_principal_df = pd.DataFrame(data = observations_principal_components, columns = ['c1'])
    final_df = pd.concat([observations_principal_df, df[['Actions0']], df[['Goal']], df[['Epoch']], df[['Episode']]], axis = 1)

    observations = final_df['c1']
    actions = final_df['Actions0']

    goals = df.loc[:,['Goal']].values.flatten()
    rewards = MinMaxScaler((1, 40)).fit_transform(df.loc[:,['Reward']].values).flatten()

    goals_series = df['Goal']
    goals_runs_starts = goals_series.loc[goals_series.shift() != goals_series].index

    episodes_series = df['Episode']
    episode_starts = episodes_series.loc[episodes_series.shift() != episodes_series]

    #  Create subplots
    fig, (ax_charts, ax_traces) = plt.subplots(2, 1, figsize = (20,10))

    # Traces
    ax_traces.set_title('Goals mapped over Action/Observation space', fontsize = 10)
    ax_traces.xlim=(observations.min(), observations.max())
    ax_traces.ylim=(actions.min(), actions.max())
    ax_traces.grid()

    # Useful values
    num_goals = final_df['Goal'].max() + 1

    plots = []

    def onKey(event):
        global animation_offset
        global animation_control
        if animation_control == 'stop' and event.key in ['left', 'right']:
            animation_direction = event.key
            animation_offset += num_visible_episodes if animation_direction == 'left' else -num_visible_episodes
        if event.key == ' ':
            animation_control = 'stop' if animation_control == 'go' else 'go'

    def init():
        ax_traces.set_title('Goals mapped over Action/Observation space')

    def animate(animation_step):
        episode = episode_from_step(animation_step)

        ax_traces.set_title('Goals mapped over Action/Observation space (episode {} of {})'.format(episode, num_episodes))

        remove_old_plots(episode)

        episode_start, episode_end, episode_len = episode_start_end(episode, num_visible_episodes)

        if line_traces:
            plots.extend(plot_line_traces(episode_start, episode_end))
        else:
            plots.extend(plot_sccatter_traces(episode_start, episode_end))

        plots.extend(plot_value_over_time(episode_start, episode_end, 'Observations0'))
        plots.extend(plot_value_over_time(episode_start, episode_end, 'Reward'))

        ax_charts.set(xlim=(episode_start, episode_end))

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

    def remove_old_plots(episode):
        for plot in plots:
            plot.remove()
        plots.clear()

    def plot_line_traces(start, end):
        plots = []
        goal_start_i = bisect.bisect_left(goals_runs_starts, start)

        while goals_runs_starts[goal_start_i] < end:
            start_i = goals_runs_starts[goal_start_i]
            end_i = goals_runs_starts[goal_start_i+1] + 1
            color = goal_color(df['Goal'][start_i])
            plot = ax_chart.plot(observations[start_i:end_i], actions[start_i:end_i], lw=0.5, c=color)
            plots.append(plot)
            goal_start_i += 1

        return plots

    def plot_sccatter_traces(start, end):
        scatter = ax_traces.scatter(observations[start:end], actions[start:end], c=goals_series[start:end], s=rewards[start:end],
                                    cmap=colormap, marker='o', vmin=0, vmax=(num_goals - 1))
        return [scatter]

    def plot_value_over_time(start, end, column_name, offset = 0):
        plots = []
        ep_df = df.iloc[start:end]
        x = np.arange(start, end)
        y = ep_df[column_name].values

        plots.extend( ax_charts.plot(x, y, c='gray', lw=0.5) )
        plots.append( ax_charts.scatter(x, y, c=ep_df['Goal'].values, cmap=colormap, marker='o', s=2, vmin=0, vmax=(num_goals - 1)) )

        return plots

    def goal_color(goal):
        return colormap(float(goal / (num_goals - 1)))

    fig.canvas.mpl_connect('key_press_event', onKey)
    anim = FuncAnimation(fig, animate, init_func=init, interval=100, frames=sys.maxsize)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--legend', '-l', nargs='*')
    parser.add_argument('--xaxis', '-x', default='TotalEnvInteracts')
    parser.add_argument('--value', '-y', default='Performance', nargs='*')
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--select', nargs='*')
    parser.add_argument('--exclude', nargs='*')
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()
    """

    Args:
        logdir (strings): As many log directories (or prefixes to log
            directories, which the plotter will autocomplete internally) as
            you'd like to plot from.

        legend (strings): Optional way to specify legend for the plot. The
            plotter legend will automatically use the ``exp_name`` from the
            config.json file, unless you tell it otherwise through this flag.
            This only works if you provide a name for each directory that
            will get plotted. (Note: this may not be the same as the number
            of logdir args you provide! Recall that the plotter looks for
            autocompletes of the logdir args: there may be more than one
            match for a given logdir prefix, and you will need to provide a
            legend string for each one of those matches---unless you have
            removed some of them as candidates via selection or exclusion
            rules (below).)

        xaxis (string): Pick what column from data is used for the x-axis.
             Defaults to ``TotalEnvInteracts``.

        value (strings): Pick what columns from data to graph on the y-axis.
            Submitting multiple values will produce multiple graphs. Defaults
            to ``Performance``, which is not an actual output of any algorithm.
            Instead, ``Performance`` refers to either ``AverageEpRet``, the
            correct performance measure for the on-policy algorithms, or
            ``AverageTestEpRet``, the correct performance measure for the
            off-policy algorithms. The plotter will automatically figure out
            which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for
            each separate logdir.

        count: Optional flag. By default, the plotter shows y-values which
            are averaged across all results that share an ``exp_name``,
            which is typically a set of identical experiments that only vary
            in random seed. But if you'd like to see all of those curves
            separately, use the ``--count`` flag.

        smooth (int): Smooth data by averaging it over a fixed window. This
            parameter says how wide the averaging window will be.

        select (strings): Optional selection rule: the plotter will only show
            curves from logdirs that contain all of these substrings.

        exclude (strings): Optional exclusion rule: plotter will only show
            curves from logdirs that do not contain these substrings.

    """

    make_animation(args.logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est)

if __name__ == "__main__":
    main()
