import logging

import numpy as np
import torch
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.collections import PolyCollection
import matplotlib.ticker as ticker


def show_args(args):
    logging.info("----------Parameters-----------")
    for arg in vars(args):
        logging.info(str(arg) + ' = ' + str(getattr(args, arg)))
    logging.info("------------------------------\n")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)


def set_logging(args, description: str):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.test_result_dir):
        os.makedirs(args.test_result_dir)

    log_filename = args.log_dir + '/' + description.lower() + '_' + datetime.now().strftime("%m%d-%H%M%S") + '.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='w',
                        format='%(asctime)-15s %(levelname)-8s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("\x1b[38;20m" + ' %(message)s' + "\x1b[0m"))
    logging.getLogger().addHandler(console)


def set_device(args):
    """
    Set device to gpu if available
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        logging.info("\n No GPU found, program is running on CPU! \n")
        return torch.device("cpu")
    return torch.device("cuda")


def plot_loss_acc(args, tr_loss, val_loss, good_pred_perc, prag_distance_list,
                  program_start_time, tr_acc=None, val_acc=None):
    """
    plots the training metrics

    Arguments
    ---------
    tr_loss: (list of tuple), the average epoch loss on the training set for each epoch
    val_loss: (list of tuple), the average epoch loss on the validation set for each epoch
    tr_acc: (list, optional), the epoch categorization accuracy on the training set for each epoch
    val_acc: (list, optional), the epoch categorization accuracy on the validation set for each epoch

    """
    if not os.path.exists(args.fig_dir):
        os.makedirs(args.fig_dir)

    figure, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    train_n = [int(item[0]) + 1 for item in tr_loss]
    train_loss = [item[1] for item in tr_loss]
    valid_n = [int(item[0]) + 1 for item in val_loss]
    val_loss = [item[1] for item in val_loss]

    ax1.set_xticks(train_n)
    ax1.plot(train_n, train_loss, 'bs-', markersize=4, label="training loss")
    ax1.plot(valid_n, val_loss, 'rs-', markersize=4, label="validation loss")
    ax1.plot(valid_n, prag_distance_list, 'ys-', markersize=4, label="pragmatic distance gap")
    ax1.legend(loc="upper center", fontsize=12, ncol=2)
    ax1.set_title("Training and Validation Losses")
    ax1.set_ylabel("Epoch Loss", fontsize=15)
    ax1.set_xlabel("Epoch Number", fontsize=15)
    # rotate x-axis labels
    plt.xticks(rotation=45)
    # set the maximum y-axis value to 4
    ax1.set_ylim(0, min(4, max(max(train_loss), max(val_loss))) * 1.1)

    # Create a second y-axis for the bar chart
    ax2 = ax1.twinx()

    # draw bars for good_pred_perc
    for i, good_pred in enumerate(good_pred_perc):
        ax2.bar(valid_n[i], good_pred, color='green', alpha=0.3, edgecolor='black', linewidth=0.5, width=0.6)
        # ax1.text(valid_n[i], good_pred+0.1, f"{good_pred*100:.1f}%", fontsize=10, color='black', ha='center')

    ax2.tick_params(axis='y', labelcolor='black')

    ax2.axhline(y=max(good_pred_perc), color='grey', linestyle='--')
    ax2.text(x=plt.xlim()[1] + 1, y=max(good_pred_perc), s=f'{max(good_pred_perc):.2f}',
             verticalalignment='center',
             horizontalalignment='right',
             color='red')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.set_ylabel('Valid Accuracy', color='black', fontsize=20)

    # save the plot
    plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_loss_plot.png'))
    plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_loss_plot.pdf'))


def plot_scores(args, score1_list, score2_list, program_start_time, point_size=100, show_text=False):
    if not os.path.exists(args.fig_dir):
        os.makedirs(args.fig_dir)

    # just choose a subset of the scores to plot
    part_ids = np.random.choice(range(len(score1_list)), point_size, replace=False)
    score1_list = [score1_list[i] for i in part_ids]
    score2_list = [score2_list[i] for i in part_ids]

    score_pairs = zip(score1_list, score2_list)
    pair_index = range(1, len(score1_list) + 1)
    figure, ax = plt.subplots(1, 1, figsize=(6, 8))
    ax.scatter(score1_list, pair_index, color='blue', label='statement 1 (implicit)')
    ax.scatter(score2_list, pair_index, color='red', label='statement 2 (explicit)')
    # draw a dotted line between the two points
    for i, (score1, score2) in enumerate(score_pairs):
        ax.plot([score1, score2], [pair_index[i], pair_index[i]], color='black', linewidth=1, linestyle='dotted')
        if show_text:
            # put scores label on the points
            ax.text(score1, pair_index[i], f"{score1:.2f}", fontsize=8, color='black', ha='left')
            ax.text(score2, pair_index[i], f"{score2:.2f}", fontsize=8, color='black', ha='right')
    ax.set_title(f"Scores of statements ({point_size} sample pairs)")
    ax.set_ylabel("Pair Index", fontsize=15)
    ax.set_xlabel("Implicitness Score", fontsize=15)
    # draw legend
    legend = ax.legend(loc="upper right")
    legend.get_frame().set_alpha(0.5)
    # make the x-axis from 0 to 1
    # ax.set_xlim(0, 1)

    # save the plot
    plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_score_point.png'))
    plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_score_point.pdf'))


def plot_violin_distribution(args, program_start_time, imp_scores1_list, imp_scores2_list, prag_scores_list, prag_labels_list):
    df = pd.DataFrame({'imp_score1': imp_scores1_list,
                       'imp_score2': imp_scores2_list,
                       'prag_score': prag_scores_list,
                       'prag_label': prag_labels_list})

    # Assuming 'df' is your original dataframe
    data = {
        'imp_score1': df['imp_score1'][df['prag_label'] == '1'],
        'imp_score2_at_1': df['imp_score2'][df['prag_label'] == '1'],
        'imp_score2_at_0': df['imp_score2'][df['prag_label'] == '0'],
        'prag_score_at_1': df['prag_score'][df['prag_label'] == '1'],
        'prag_score_at_0': df['prag_score'][df['prag_label'] == '0']
    }

    # Convert dictionary into DataFrame
    plot_data = pd.DataFrame(data)

    # Melt the data to long format for easier plotting with Seaborn
    plot_data_long = plot_data.melt(var_name='Groups', value_name='Scores')

    # Define specific colors for each violin
    color_list = ['#a6caec', '#b4e5a2', '#f6c6ad', '#d9f2d0', '#fbe3d6']
    transparency = 1  # Set transparency level: 0 (transparent) to 1 (opaque)

    # Plotting
    plt.figure(figsize=(8, 7))
    ax = sns.violinplot(x='Scores', y='Groups',
                        hue='Groups',
                        data=plot_data_long,
                        orient='h',
                        palette=color_list,
                        saturation=1.0,
                        gap=0,
                        inner='box',
                        linewidth=1,
                        linecolor='black',
                        width=0.6)

    # set x-axis label size
    ax.set_xlabel('Scores', fontsize=15)
    # set x-axis number size
    ax.tick_params(axis='x', labelsize=15)

    # make the x-axis valus to float
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    # remove y-axis tick values
    ax.set_yticklabels([])
    # remove y-axis label
    ax.set_ylabel('')

    for violin, alpha in zip(ax.collections[::2], [transparency] * 5):
        violin.set_alpha(alpha)
        violin.set_edgecolor('black')

    plt.title('Distribution of Implicitness and Pragmatic Scores', fontsize=20)

    # Calculate and draw vertical lines for average values
    for i, group in enumerate(plot_data_long['Groups'].unique()):
        if group == 'imp_score1':
            line_color = 'black'
            y_position = 0.9
        elif group == 'imp_score2_at_1':
            line_color = 'black'
            y_position = 0.7
        elif group == 'imp_score2_at_0':
            line_color = 'black'
            y_position = 0.5
        elif group == 'prag_score_at_1':
            line_color = 'black'
            y_position = 0.3
        else:
            line_color = 'black'
            y_position = 0.1
        mean_val = plot_data_long[plot_data_long['Groups'] == group]['Scores'].mean()
        plt.axvline(x=mean_val, color=line_color, linestyle='--', linewidth=1,
                    ymin=0, ymax=y_position)
        # add text to the line
        # plt.text(mean_val, 4.5, f"{mean_val:.2f}", fontsize=11, color='black', ha='right', rotation=90)

    # Assuming the y-ticks correspond to the order of the groups:
    n_highlight = 5  # Number of violins to highlight from the top
    height = 0.99  # Height of the rectangle to cover the violin plot adequately
    y_ticks = ax.get_yticks()
    for i in range(n_highlight):
        if i < 3:
            rect = Rectangle((plt.xlim()[0], y_ticks[i] - height / 2), plt.xlim()[1] - plt.xlim()[0], height,
                             color='#d0e6f6', zorder=0, alpha=0.2)
        else:
            rect = Rectangle((plt.xlim()[0], y_ticks[i] - height / 2), plt.xlim()[1] - plt.xlim()[0], height,
                             color='#ffffe0', zorder=0, alpha=0.3)
        ax.add_patch(rect)

    # Adding horizontal lines to separate violin plots
    for y in y_ticks:
        ax.axhline(y + 0.5, color='grey', linestyle='--', linewidth=0.6)  # Adjust the y position as needed

    # # Create legend handles manually with the same transparency
    # handles = [mpatches.Patch(color=color_dict[name], label=name) for name in color_dict]
    # ax.legend(handles=handles, title='')

    # Set hatch pattern on the first violin
    count = 0
    for art in ax.findobj(match=lambda x: isinstance(x, PolyCollection)):
        count += 1
        if count <= 3:
            pass
            art.set_hatch(".")  # Apply hatch pattern
        else:
            art.set_hatch("//")
        art.set_linewidth(0.8)

    plt.tight_layout()
    # save the plot
    plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_score_distribution.png'))
    # plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_score_distribution.pdf'))

    # --------------------------------------------------------------

    # plot the implicitness scores only
    imp_data = {
        'imp_score1': df['imp_score1'][df['prag_label'] == '1'],
        'imp_score2_at_1': df['imp_score2'][df['prag_label'] == '1'],
        'imp_score2_at_0': df['imp_score2'][df['prag_label'] == '0']
    }

    # Convert dictionary into DataFrame
    plot_data = pd.DataFrame(imp_data)
    # Melt the data to long format for easier plotting with Seaborn
    plot_data_long = plot_data.melt(var_name='Groups', value_name='Scores')
    # Define specific colors for each violin
    color_list = ['#a6caec', '#b4e5a2', '#f6c6ad']
    transparency = 1

    # Plotting
    plt.figure(figsize=(7, 6))
    ax = sns.violinplot(x='Scores', y='Groups',
                        hue='Groups',
                        data=plot_data_long,
                        orient='h',
                        palette=color_list,
                        saturation=1.0,
                        gap=0,
                        inner='box',
                        linewidth=1,
                        linecolor='black',
                        width=0.6)

    # set x-axis label size
    ax.set_xlabel('Implicitness Score', fontsize=25)
    # set x-axis number size
    ax.tick_params(axis='x', labelsize=25)

    # make the x-axis valus to float
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    # remove y-axis tick values
    ax.set_yticklabels([])
    # remove y-axis label
    ax.set_ylabel('')

    for violin, alpha in zip(ax.collections[::2], [transparency] * 3):
        violin.set_alpha(alpha)
        violin.set_edgecolor('black')

    plt.title('Distribution of Implicitness Scores', fontsize=25)

    # Calculate and draw vertical lines for average values
    for i, group in enumerate(plot_data_long['Groups'].unique()):
        if group == 'imp_score1':
            line_color = 'black'
            y_position = 5*1/6.0
        elif group == 'imp_score2_at_1':
            line_color = 'black'
            y_position = 3*1/6.0
        else:
            line_color = 'black'
            y_position = 1/6.0
        mean_val = plot_data_long[plot_data_long['Groups'] == group]['Scores'].mean()
        plt.axvline(x=mean_val, color=line_color, linestyle='--', linewidth=1,
                    ymin=0, ymax=y_position)
        # add text to the line
        # plt.text(mean_val, 4.5, f"{mean_val:.2f}", fontsize=11, color='black', ha='right', rotation=90)

    # Adding horizontal lines to separate violin plots
    y_ticks = ax.get_yticks()
    for y in y_ticks:
        ax.axhline(y + 0.5, color='grey', linestyle='--', linewidth=0.6)  # Adjust the y position as needed

    plt.tight_layout()
    # save the plot
    plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_imp_score_distribution.png'))
    # plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_imp_score_distribution.pdf'))

    # --------------------------------------------------------------

    # plot the pragmatic scores only
    prag_data = {
        'prag_score_at_1': df['prag_score'][df['prag_label'] == '1'],
        'prag_score_at_0': df['prag_score'][df['prag_label'] == '0']
    }

    # Convert dictionary into DataFrame
    plot_data = pd.DataFrame(prag_data)
    # Melt the data to long format for easier plotting with Seaborn
    plot_data_long = plot_data.melt(var_name='Groups', value_name='Scores')
    # Define specific colors for each violin
    # color_list = ['#b4e5a2', '#f6c6ad']
    color_list = ['#d9f2d0', '#fbe3d6']
    transparency = 1

    # Plotting
    plt.figure(figsize=(7, 6))
    ax = sns.violinplot(x='Scores', y='Groups',
                        hue='Groups',
                        data=plot_data_long,
                        orient='h',
                        palette=color_list,
                        saturation=1.0,
                        gap=0,
                        inner='box',
                        linewidth=1,
                        linecolor='black',
                        width=0.6)

    # set x-axis label size
    ax.set_xlabel('Pragmatic Distance', fontsize=25)
    # set x-axis number size
    ax.tick_params(axis='x', labelsize=25)

    # make the x-axis valus to float
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    # remove y-axis tick values
    ax.set_yticklabels([])
    # remove y-axis label
    ax.set_ylabel('')

    for violin, alpha in zip(ax.collections[::2], [transparency] * 3):
        violin.set_alpha(alpha)
        violin.set_edgecolor('black')

    plt.title('Distribution of Pragmatic Distance', fontsize=25)

    # Calculate and draw vertical lines for average values
    for i, group in enumerate(plot_data_long['Groups'].unique()):
        if group == 'prag_score_at_1':
            line_color = 'black'
            y_position = 3/4.0
        else:
            line_color = 'black'
            y_position = 1/4.0
        mean_val = plot_data_long[plot_data_long['Groups'] == group]['Scores'].mean()
        plt.axvline(x=mean_val, color=line_color, linestyle='--', linewidth=1,
                    ymin=0, ymax=y_position)

    # Adding horizontal lines to separate violin plots
    y_ticks = ax.get_yticks()
    for y in y_ticks:
        ax.axhline(y + 0.5, color='grey', linestyle='--', linewidth=0.6)  # Adjust the y position as needed

    plt.tight_layout()
    # save the plot
    plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_prag_score_distribution.png'))
    # plt.savefig(os.path.join(args.fig_dir, f'{program_start_time}_prag_score_distribution.pdf'))
