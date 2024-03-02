from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dython.nominal import associations


plt.style.use("bmh")


def plot_distribution(
    df_list: List[pd.DataFrame], column: str, df_names_list: Optional[List[str]] = None
):
    """Plot the distribution of a specific column from a list of dataframes."""

    if pd.api.types.is_numeric_dtype(df_list[0][column]):
        # If the column is numeric, plot a histogram
        # plt.figure(figsize=(12, 6))

        for i, temp_df in enumerate(df_list):
            sns.histplot(
                temp_df[column],
                label=f"{df_names_list[i]} {column}",
                kde=True,
                stat="density",
            )

        plt.legend()
        plt.title(f"Distribution of {column}")
        plt.show()

    elif pd.api.types.is_categorical_dtype(
        df_list[0][column]
    ) or pd.api.types.is_object_dtype(df_list[0][column]):
        # If the column is categorical, plot a bar chart
        # plt.figure(figsize=(12, 6))

        pivot_df = pd.DataFrame(
            data={
                column: sum([temp_df[column].to_list() for temp_df in df_list], []),
                "dataset": sum(
                    [
                        [df_names_list[i]] * len(temp_df)
                        for i, temp_df in enumerate(df_list)
                    ],
                    [],
                ),
            }
        )

        count_df = pivot_df.filter([column, "dataset"]).pivot_table(
            index=[column], aggfunc="size", columns="dataset"
        )
        prop_df = count_df.div(count_df.sum(axis=0), axis=1)
        prop_df.sort_values(by=df_names_list[0]).plot.bar()

        plt.legend()
        plt.title(f"Distribution of {column}")
        plt.show()


def compare_distributions(
    df_list: List[pd.DataFrame], df_names_list: Optional[List[str]] = None
):
    """
    Compare distributions of all columns between a list of dataframes with the same columns. One by one plot.
    """
    if df_names_list is None:
        df_names_list = [f"DataFrame{i}" for i in range(len(df_list))]

    for column in df_list[0].columns:
        plot_distribution(df_list, column, df_names_list)


def compare_distributions_grid(
    df_list: List[pd.DataFrame],
    df_names_list: Optional[List[str]] = None,
    nrows: int = 1,
    ncols: int = 1,
):
    """
    Compare distributions of all columns between a list of dataframes with the same columns. Plot in a grid.
    """
    if df_names_list is None:
        df_names_list = [f"DataFrame{i}" for i in range(len(df_list))]

    fig, axs = plt.subplots(nrows, ncols, figsize=(8 * ncols, 5 * nrows))
    axs = axs.ravel()

    df1 = df_list[0]
    for i, column in enumerate(df1.columns):
        ax = axs[i]

        temp_df = pd.DataFrame(
            data={
                column: sum([temp_df[column].to_list() for temp_df in df_list], []),
                "dataset": sum(
                    [
                        [df_names_list[i]] * len(temp_df)
                        for i, temp_df in enumerate(df_list)
                    ],
                    [],
                ),
            }
        )

        if pd.api.types.is_numeric_dtype(df1[column]):
            # sns.kdeplot(
            #     data=temp_df,
            #     x=column,
            #     hue="dataset",
            #     ax=ax,
            # )
            # if i == 0:
            #     ax.set_xlim(9.5, 13)
            # else:
            #     ax.set_xlim(0, 1)

            sns.histplot(
                data=temp_df,
                x=column,
                hue="dataset",
                ax=ax,
                stat="density",
                kde=True,
                common_norm=False,
            )

            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel("")

        elif pd.api.types.is_categorical_dtype(
            df1[column]
        ) or pd.api.types.is_object_dtype(df1[column]):
            # If the column is categorical, plot a bar chart
            # plt.figure(figsize=(12, 6))

            count_df = temp_df.filter([column, "dataset"]).pivot_table(
                index=[column], aggfunc="size", columns="dataset"
            )
            count_df = count_df[df_names_list]

            prop_df = count_df.div(count_df.sum(axis=0), axis=1)
            prop_df.sort_values(by=df_names_list[0]).plot(ax=ax, kind="bar")

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

            category_size = len(temp_df[column].value_counts())
            if category_size <= 10:
                labelsize = 10
            elif category_size <= 20:
                labelsize = 8
            else:
                labelsize = 6
            ax.xaxis.set_tick_params(labelsize=labelsize)

            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel("")

    i += 1
    while i < len(axs):
        axs[i].set_visible(False)
        i += 1

    # set the spacing between subplots
    plt.subplots_adjust(
        # left  = 0.125,  # the left side of the subplots of the figure
        # right = 0.9,    # the right side of the subplots of the figure
        # bottom = 0.1,   # the bottom of the subplots of the figure
        # top = 0.9,      # the top of the subplots of the figure
        # wspace = 0.2,   # the amount of width reserved for blank space between subplots
        hspace=0.45,  # the amount of height reserved for white space between subplots
    )


def heatmap_correlation(
    df1: pd.DataFrame, df2: pd.DataFrame, df1_name="df1", df2_name="df2"
):
    """Correlation heatmap and absolute difference between two dataframes of mixed-type columns."""
    true_corr = associations(
        df1,
        theil_u=True,
        annot=False,
        cmap="Blues",
        plot=False,
        nan_strategy="drop_samples",
    )["corr"]
    fake_corr = associations(
        df2,
        theil_u=True,
        annot=False,
        cmap="Blues",
        plot=False,
        nan_strategy="drop_samples",
    )["corr"]
    diff_corr = np.abs(true_corr - fake_corr)
    # heatmaps
    sns.set(rc={"axes.facecolor": "white", "figure.facecolor": "white"})
    cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True, n_colors=100)

    kwargs = {
        "cmap": cmap,
        "annot": False,
        "xticklabels": False,
        "yticklabels": False,
        "cbar": False,
        "square": True,
        "vmin": 0,
        "vmax": 1,
    }

    fig, axs = plt.subplots(figsize=(40, 20))

    ax1 = plt.subplot(1, 3, 1)
    img = sns.heatmap(true_corr, **kwargs, ax=ax1)
    ax1.set_title(df1_name, weight="bold", fontsize=30)

    ax2 = plt.subplot(1, 3, 2)
    sns.heatmap(fake_corr, **kwargs, ax=ax2)
    ax2.set_title(df2_name, weight="bold", fontsize=30)

    ax3 = plt.subplot(1, 3, 3)
    sns.heatmap(diff_corr, **kwargs, ax=ax3)
    ax3.set_title("Absolute Difference", weight="bold", fontsize=30)

    mappable = img.get_children()[0]
    cbar = plt.colorbar(
        mappable, ax=[ax1, ax2, ax3], orientation="horizontal", pad=0.05
    )
    cbar.ax.tick_params(labelsize=30)

    axs.axis("off")
