import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

# pca
import sys
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import bisect

DIV_LINE_WIDTH = 50

# Global vars for tracking and labeling data at load time.
exp_idx = 0
units = dict()

def plot_data(data, xaxis='Epoch', value="AverageEpRet", condition="Condition1", smooth=1, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x,y,'same') / np.convolve(z,y,'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best').set_draggable(True)

    """
    For the version of the legend used in the Spinning Up benchmarking page,
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.tight_layout(pad=0.5)

animation_control = 'go'
animation_offset = 0
animation_direction = 'right'

def plot_pca(df, colormap=cm.get_cmap('Spectral'), num_visible_episodes=5, line_traces=False):

    observation_features = ['Observations0', 'Observations1']

    #
    observations_components = df.loc[:, observation_features].values
    num_epochs = df['Epoch'].max()
    num_episodes = df['Episode'].max()

    observations_components = StandardScaler().fit_transform(observations_components)
    pca = PCA(n_components=1)
    observations_principal_components = pca.fit_transform(observations_components)
    observations_principal_df = pd.DataFrame(data = observations_principal_components, columns = ['c1'])
    final_df = pd.concat([observations_principal_df, df[['Actions0']], df[['Goal']], df[['Epoch']], df[['Episode']]], axis = 1)
    # final_df = df

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

    # scatter = ax.scatter([], [], c = [], cmap=plt.cm.bwr, s = 3)

    plots = {}
    # bars = {}

    def onKey(event):
        global animation_offset
        global animation_control
        if animation_control == 'stop' and event.key in ['left', 'right']:
            animation_direction = event.key
            animation_offset += 1 if animation_direction == 'left' else -1
        if event.key == ' ':
            animation_control = 'stop' if animation_control == 'go' else 'go'

    def init():
        ax_traces.set_title('Goals mapped over Action/Observation space')

    def animate(animation_step):
        episode = episode_from_step(animation_step)

        ax_traces.set_title('Goals mapped over Action/Observation space (episode {} of {})'.format(episode, num_episodes))

        remove_old_plots(episode)

        episode_start, episode_end, episode_len = episode_start_end(episode)

        if line_traces:
            plots[episode] = plot_line_traces(episode_start, episode_end)
        else:
            plots[episode] = plot_sccatter_traces(episode_start, episode_end)

        plots[episode].extend(plot_value_over_time(episode_start, episode_end, 'Observations0'))
        plots[episode].extend(plot_value_over_time(episode_start, episode_end, 'Reward'))

        ax_charts.set(xlim=(episode_start - (num_visible_episodes - 1) * episode_len, episode_end))

    def episode_from_step(animation_step):
        global animation_offset
        if animation_control == 'stop':
            animation_offset += 1
        return int((animation_step - animation_offset) % num_episodes)

    def episode_start_end(episode):
        start = episode_starts.loc[episode_starts == episode].index[0]
        end_row = episode_starts.loc[episode_starts == (episode + 1)]
        end = end_row.index[0] if len(end_row) > 0 else len(df)
        return start, end, end - start

    def remove_old_plots(episode):
        if animation_direction == 'right':
            to_remove = episode - num_visible_episodes if episode >= num_visible_episodes else (episode - num_visible_episodes + num_episodes)
        else:
            to_remove = episode + 1

        if to_remove in plots:
            [plot.remove() for plot in plots[to_remove]]
            del plots[to_remove]
        if episode in plots:
            [plot.remove() for plot in plots[episode]]
            del plots[episode]

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
        scatter = ax_traces.scatter(observations[start:end], actions[start:end], c=goals[start:end], s=rewards[start:end],
                                    cmap=colormap, marker='o', vmin=0, vmax=(num_goals - 1))
        return [scatter]

    def plot_value_over_time(start, end, column_name, offset = 0):
        plots = []
        ep_df = df.iloc[start:end]
        x = np.arange(start, end)
        y = ep_df[column_name].values

        plots.extend( ax_charts.plot(x, y, c='gray', lw=0.5) )
        plots.append( ax_charts.scatter(x, y, c=ep_df['Goal'].values, cmap=colormap, marker='o', s=2) )

        return plots

    def goal_color(goal):
        return colormap(float(goal / (num_goals - 1)))

    fig.canvas.mpl_connect('key_press_event', onKey)
    anim = FuncAnimation(fig, animate, init_func=init, interval=100, frames=sys.maxsize)
    plt.show()

def get_datasets(logdir, condition=None, filename='progress.txt'):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.

    Assumes that any file "progress.txt" is a valid hit.
    """
    global exp_idx
    global units
    datasets = []
    for root, _, files in os.walk(logdir):
        if filename in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root,'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')
            condition1 = condition or exp_name or 'exp'
            condition2 = condition1 + '-' + str(exp_idx)
            exp_idx += 1
            if condition1 not in units:
                units[condition1] = 0
            unit = units[condition1]
            units[condition1] += 1

            try:
                exp_data = pd.read_table(os.path.join(root, filename))
            except:
                print('Could not read from %s'%os.path.join(root, filename))
                continue
            performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            exp_data.insert(len(exp_data.columns),'Unit',unit)
            exp_data.insert(len(exp_data.columns),'Condition1',condition1)
            exp_data.insert(len(exp_data.columns),'Condition2',condition2)
            if performance in exp_data:
                exp_data.insert(len(exp_data.columns),'Performance',exp_data[performance])
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs, legend=None, select=None, exclude=None, pca=False):

    """
    For every entry in all_logdirs,
        1) check if the entry is a real directory and if it is,
           pull data from it;

        2) if not, check to see if the entry is a prefix for a
           real directory, and pull data from that.
    """
    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir) and logdir[-1]==os.sep:
            logdirs += [logdir]
        else:
            basedir = osp.dirname(logdir)
            fulldir = lambda x : osp.join(basedir, x)
            prefix = logdir.split(os.sep)[-1]
            listdir= os.listdir(basedir)
            logdirs += sorted([fulldir(x) for x in listdir if prefix in x])

    """
    Enforce selection rules, which check logdirs for certain substrings.
    Makes it easier to look at graphs from particular ablations, if you
    launch many jobs at once with similar names.
    """
    if select is not None:
        logdirs = [log for log in logdirs if all(x in log for x in select)]
    if exclude is not None:
        logdirs = [log for log in logdirs if all(not(x in log) for x in exclude)]

    # Verify logdirs
    print('Plotting from...\n' + '='*DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '='*DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    assert not(legend) or (len(legend) == len(logdirs)), \
        "Must give a legend title for each set of experiments."

    # Load data from logdirs
    filename = "progress.txt" if not pca else "pca.txt"

    data = []
    if legend:
        for log, leg in zip(logdirs, legend):
            data += get_datasets(log, leg, filename=filename)
    else:
        for log in logdirs:
            data += get_datasets(log, filename=filename)
    return data

def make_plots(all_logdirs, legend=None, xaxis=None, values=None, count=False,
               font_scale=1.5, smooth=1, select=None, exclude=None, estimator='mean', pca=False):

    data = get_all_datasets(all_logdirs, legend, select, exclude, pca)

    if pca:
        plot_pca(data[0])
    else:
        values = values if isinstance(values, list) else [values]
        condition = 'Condition2' if count else 'Condition1'
        estimator = getattr(np, estimator)      # choose what to show on main curve: mean? max? min?
        for value in values:
            plt.figure()
            plot_data(data, xaxis=xaxis, value=value, condition=condition, smooth=smooth, estimator=estimator)
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
    parser.add_argument('--pca', action='store_true')
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

    make_plots(args.logdir, args.legend, args.xaxis, args.value, args.count,
               smooth=args.smooth, select=args.select, exclude=args.exclude,
               estimator=args.est, pca=args.pca)

if __name__ == "__main__":
    main()
