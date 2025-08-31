"""Module to define plotting utilities to visualize the network and experiment results."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backend_bases import PickEvent

matplotlib.use("Qt5Agg")

custom_palette = {
    "benchmark1": "#CB3335",
    "benchmark2": "#477CA8",
    "modelname": "#905998",
}

if TYPE_CHECKING:
    from matplotlib.backend_bases import MouseEvent, Event


class Graphplot:
    """Class to plot and interact with a NetworkX graph."""

    def __init__(self, graph: nx.Graph, show_on_click: list[str] | None = None) -> None:
        self.graph = graph
        if show_on_click is None:
            show_on_click = ["name", "industry_group", "industry_sector"]
        self.show_on_click = show_on_click
        self.picker_tolerance = 0.03
        self.annot_axes = None
        self.annotation = None

    def plot(self) -> None:
        """Plot the graph and set up the click event."""
        self.figure, self.axes = plt.subplots(1, 1, figsize=(9, 9))
        self.node_pos = nx.drawing.spring_layout(self.graph)
        self.node_pos_df = pd.DataFrame(self.node_pos, index=["x_coord", "y_coord"]).T

        nx.draw_networkx(
            self.graph,
            pos=self.node_pos,
            ax=self.axes,
            hide_ticks=False,
            node_size=15,
            font_size=10,
            width=0.5,
            labels=nx.get_node_attributes(self.graph, "ticker_ids"),
            with_labels=False,
        )
        self.axes.set_picker(True)
        self.axes.patch.set_alpha(0.0)
        self.figure.canvas.mpl_connect("pick_event", self.on_left_click)
        plt.show()

    def add_buttons(self) -> None:
        """"""

    @staticmethod
    def euclidean_distance(
        x1: float | pd.Series,
        y1: float | pd.Series,
        x2: float | pd.Series,
        y2: float | pd.Series,
    ) -> pd.Series:
        """Calculate the Euclidean distance between two points."""
        return pd.Series(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    def on_left_click(self, event: Event | PickEvent) -> Any:
        """Show information about the closest node to the click."""
        if not isinstance(event, PickEvent):
            return
        click_x, click_y = event.mouseevent.xdata, event.mouseevent.ydata
        if click_x is None or click_y is None:
            return

        distances = self.euclidean_distance(
            self.node_pos_df["x_coord"],
            self.node_pos_df["y_coord"],
            click_x,
            click_y,
        )
        self.node_pos_df["distance_to_click"] = distances
        if distances.min() > self.picker_tolerance:
            print("No node close enough to the clicked point.")
            return
        closest_idx = self.node_pos_df["distance_to_click"].argmin()
        found_ticker = self.node_pos_df.iloc[closest_idx].name

        print("---------------------")
        print(found_ticker)
        node_attributes = [
            nx.get_node_attributes(self.graph, col)[found_ticker]
            for col in self.show_on_click
        ]

        annotations = "\n".join(
            [f"{col}: {attr}" for col, attr in zip(self.show_on_click, node_attributes)]
        )
        print(annotations)
        print("---------------------")

        self.remove_annotated_points()
        self.annot_axes = self.axes.twiny()
        self.annot_axes.set_axis_off()
        self.annot_axes.sharex(self.axes)
        self.annot_axes.set_zorder(0)
        self.axes.set_zorder(1)

        self.annot_axes.set_title(
            annotations, fontdict={"fontsize": 12, "color": "red", "weight": "bold"}
        )
        self.annot_axes.plot(
            self.node_pos_df.iloc[closest_idx]["x_coord"],
            self.node_pos_df.iloc[closest_idx]["y_coord"],
            marker="o",
            markersize=10,
            markeredgecolor="red",
            markerfacecolor="red",
            markeredgewidth=2,
        )
        # if self.annotation is None:
        #     self.annotation = self.axes.annotate(
        #         annotations,
        #         (-1.0, -1.0),
        #         textcoords="offset points",
        #         xytext=(5, 5),
        #         ha="left",
        #         fontsize=10,
        #         color="red",
        #         weight="bold",
        #     )
        # self.annotation.set_text(annotations)

        self.figure.canvas.draw()

    def remove_annotated_points(self) -> None:
        """Remove the connected points and axes object from the plot."""
        if self.annot_axes is not None:
            print("Removing connected points.")
            self.annot_axes.remove()
            self.annot_axes = None
            self.figure.canvas.draw()

    def on_hover(self, event: MouseEvent):
        """"""
        # Logically similar to boxplot hover.


class Boxplot:
    """Class to create an interactive boxplot."""

    def __init__(self, df: pd.DataFrame, show_on_click: list[str] | None = None):
        self.df = df
        self.connect_axes = None
        self.strip_axes = None
        if show_on_click is None:
            show_on_click = ["name", "industry_group", "industry_sector"]
        self.show_on_click = show_on_click
        sns.set(context="paper", style="darkgrid")

    def plot(
        self, metric: str, group_by: str, split_by: str, reference_by: str
    ) -> None:
        """Plot the boxplot with the given parameters."""
        self.metric = metric
        self.group_by = group_by
        self.split_by = split_by
        self.reference_by = reference_by
        self.figure, self.axes = plt.subplots(figsize=(10, 6))
        self.axes = sns.boxplot(
            data=self.df,
            x=split_by,
            y=metric,
            hue=group_by,
            dodge=True,
            ax=self.axes,
            showfliers=False,
            saturation=1,
        )
        self.strip_axes = self.axes.twiny()
        self.strip_axes = sns.stripplot(
            data=self.df,
            x=split_by,
            y=metric,
            hue=group_by,
            dodge=True,
            ax=self.strip_axes,
            marker="o",
            linewidth=1,
            size=6,
            picker=5,
            legend=False,
        )
        self.strip_axes.set_axis_off()
        self.assign_references_to_axes(
            group_order=self.df[group_by].unique().tolist(),
            split_order=self.df[split_by].unique().tolist(),
        )
        self.axes.set_xticks(
            self.axes.get_xticks(),
            [txt.get_text() for txt in self.axes.get_xticklabels()],
            rotation=45,
            ha="right",
        )
        self.axes.set_xticklabels(self.axes.get_xticklabels(), size=12)
        self.axes.set_yticklabels(self.axes.get_yticklabels(), size=12)
        self.axes.set_xlabel(self.axes.get_xlabel(), size=18)
        self.axes.set_ylabel(self.axes.get_ylabel(), size=18)
        self.figure.canvas.mpl_connect("pick_event", self.on_left_click)

        plt.tight_layout()
        plt.show()

    def assign_references_to_axes(
        self, group_order: list[str], split_order: list[str]
    ) -> None:
        """Assign reference IDs to each artist in the boxplot."""
        if self.strip_axes is None:
            return
        group_len = len(group_order)
        # print("Group order:", group_order)
        # print("Split order:", split_order)

        for idx, artist in enumerate(self.strip_axes.collections):
            artist.group_name = group_order[idx % group_len]  # type: ignore[attr-defined]
            artist.split_name = split_order[idx // group_len]  # type: ignore[attr-defined]
            artist.ref_ids = self.df[  # type: ignore[attr-defined]
                (self.df[self.group_by] == artist.group_name)  # type: ignore[attr-defined]
                & (self.df[self.split_by] == artist.split_name)  # type: ignore[attr-defined]
            ][self.reference_by].values

        return

    def connect_points(self, reference_id: str) -> None:
        """Connect points with the same reference ID across different groups."""
        if self.strip_axes is None:
            return
        print("Connecting points for ID:", reference_id)
        connect_locations: list[int] = []
        for artist in self.strip_axes.collections:
            if reference_id in artist.ref_ids:  # type: ignore[attr-defined]
                loc_idx = np.where(artist.ref_ids == reference_id)  # type: ignore[attr-defined]
                offset_pos = np.array(artist.get_offsets())[loc_idx]
                connect_locations.extend(offset_pos)
        highlight_data = np.array(connect_locations)

        self.connect_axes = self.strip_axes.twiny()
        self.connect_axes = sns.lineplot(
            x=highlight_data[:, 0],
            y=highlight_data[:, 1],
            ax=self.connect_axes,
            color="red",
            linewidth=2,
            marker="o",
            markersize=8,
            markerfacecolor="red",
            markeredgecolor="black",
            markeredgewidth=1,
        )
        annotations = "\n".join(
            [
                f"{col}: {self.df[self.df[self.reference_by] == reference_id][col].values[0]}"
                for col in self.show_on_click
            ]
        )

        self.connect_axes.annotate(
            annotations,
            (highlight_data[-1, 0], highlight_data[-1, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=10,
            color="red",
            weight="bold",
        )
        self.connect_axes.set_axis_off()
        self.connect_axes.sharex(self.strip_axes)
        self.connect_axes.set_zorder(1)
        self.strip_axes.set_zorder(2)
        self.figure.canvas.draw()

    def remove_connect_points(self) -> None:
        """Remove the connected points and axes object from the plot."""
        if self.connect_axes is not None:
            print("Removing connected points.")
            self.connect_axes.remove()
            self.connect_axes = None
            self.figure.canvas.draw()

    def on_left_click(self, event: Event | PickEvent) -> str:
        """Display information about the clicked point.

        Parameters
        ----------
        event : Event | PickEvent
            The pick event containing information about the clicked point.

        Returns
        -------
        str
            The reference ID of the clicked point.
        """
        self.remove_connect_points()
        selected_points = event.artist.ref_ids[event.ind]  # type: ignore[attr-defined]
        selected_id = selected_points[0]

        print("---------------------")
        print("Clicked on point:", selected_id)
        for col in self.show_on_click:
            print(self.df[self.df[self.reference_by] == selected_id][col].values[0])
        print("---------------------")

        self.connect_points(selected_id)
        return selected_id


class ConfusionMatrixPlot:
    """"""

    def __init__(self):
        pass

    def plot(self):
        """"""

    def on_left_click(self, event: PickEvent):
        """"""


def plot_results_boxplot(
    out_path: str, metric: str = "mse", size_multiplier: int = 3
) -> None:
    """Plot the given benchmarks in a boxplot."""
    dataset_names = ["dataset1", "dataset2"]
    benchmarks = ["benchmark1", "benchmark2", "modelname"]
    custom_palette = {
        "benchmark1": "#CB3335",
        "benchmark2": "#477CA8",
        "modelname": "#905998",
    }

    # Collect data
    all_dfs = []
    for benchmark in benchmarks:
        for dataset_name in dataset_names:
            path = os.path.join(
                out_path,
                benchmark,
                f"{dataset_name}",
                f"{dataset_name}_{metric}.csv",
            )
            df = pd.DataFrame()
            result_arr = pd.read_csv(path, header=None).values

            df["Metric"] = np.squeeze(result_arr)
            df["Dataset"] = dataset_name
            df["Benchmark"] = benchmark
            if benchmark == "modelname":
                df["Benchmark"] += df["Benchmark"] + " (ours)"
            all_dfs.append(df)

    all_combined = pd.concat(all_dfs).reset_index(drop=True)

    sns.set(
        context="paper",
        style="darkgrid",
        rc={
            "figure.dpi": 100 * size_multiplier,
            "figure.figsize": (10, 5),
            # "axes.titlesize": 20 / size_multiplier,
            # "axes.labelsize": 20 / size_multiplier,
            # "axes.linewidth": 1.0 / size_multiplier,
            "xtick.labelsize": 25.0,
            # "xtick.major.pad": 3.5 / size_multiplier,
            # "xtick.minor.pad": 3.4 / size_multiplier,
            "ytick.labelsize": 15.0,
            # "ytick.major.pad": 3.5 / size_multiplier,
            # "ytick.minor.pad": 3.4 / size_multiplier,
            # "legend.fontsize": 20 / size_multiplier,
            # "legend.title_fontsize": 20 / size_multiplier,
            # "figure.titlesize": 20 / size_multiplier,
            # "lines.linewidth": 1.0 / size_multiplier,
            # "lines.markersize": 6.0 / size_multiplier,
            # "boxplot.flierprops.markersize": 6.0 / size_multiplier,
            # "boxplot.flierprops.markeredgewidth": 1.0 / size_multiplier,
            # "boxplot.flierprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.boxprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.whiskerprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.capprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.medianprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.meanprops.linewidth": 1.0 / size_multiplier,
            # "boxplot.meanprops.markersize": 6.0 / size_multiplier,
            # "boxplot.showmeans": True,
        },
    )

    # Create and display the plot
    ax = sns.boxplot(
        x="Dataset",
        y="Metric",
        hue="Benchmark",
        data=all_combined,
        # palette="Set1",
        palette=custom_palette,
        width=0.8,
        dodge=True,
        legend=False,
    )
    # ax = sns.swarmplot(
    #     x="Dataset", y="Metric", hue="Benchmark", data=all_combined, dodge=True
    # )
    if metric == "mse":
        ax.set_ylim((0.0, 0.02))
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{metric}.png")
    plt.close()
    all_combined.to_csv(f"./boxplot_table_{metric}.csv", index=False)
