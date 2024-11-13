import copy
from string import ascii_uppercase
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from .utils import result2df


PALLETE = np.asarray(
    [
        [0.65098039, 0.80784314, 0.89019608],
        [0.12156863, 0.47058824, 0.70588235],
        [0.69803922, 0.8745098, 0.54117647],
        [0.2, 0.62745098, 0.17254902],
        [0.98431373, 0.60392157, 0.6],
        [0.89019608, 0.10196078, 0.10980392],
        [0.99215686, 0.74901961, 0.43529412],
        [1.0, 0.49803922, 0.0],
        [0.79215686, 0.69803922, 0.83921569],
        [0.41568627, 0.23921569, 0.60392157],
        [1.0, 1.0, 0.6],
        [253,191,111],
        [0.89762718, 0.67752062, 0.16665834],
        [0.08745955, 0.34371572, 0.77180043],
        [0.81152818, 0.53247345, 0.9027888],
        [0.4687154, 0.36006836, 0.17939007],
        [0.89106359, 0.38987048, 0.0492416],
        [0.21113298, 0.14047327, 0.66375923],
        [0.60034632, 0.23381699, 0.4618079],
        [0.17950182, 0.57224155, 0.44925753],
        [0.51774669, 0.54758115, 0.43924649],
        [0.4214879, 0.40567483, 0.13917379],
        [0.10930892, 0.4085702, 0.92445868],
        [0.87177099, 0.70851517, 0.1635471],
        [0.25377735, 0.22651516, 0.81022119],
        [0.58438536, 0.95091206, 0.89078385],
        [0.11884829, 0.7531675, 0.07229407],
        [0.05173027, 0.25899175, 0.42809596],
        [0.58487207, 0.58348158, 0.74867259],
        [0.46886089, 0.36065668, 0.58227664],
        [0.31023186, 0.95488994, 0.56283996],
        [0.79888512, 0.33990301, 0.45320525],
    ]
)
COLORS = ["#66c2a5", "#fc8d62", "#ffd92f", "#8da0cb", "#e78ac3"]
SCENARIOS = ["Base", "REPowerEU", "Year 2022", "Coal", "Clean"]
SCENARIOS = ["Base", "REPowerEU", "Aftermath", "Coal", "Clean"]

top_bar_font_size = 6
fig_length = {
    1: 3.50394,  # 1 column
    1.5: 5.35433,  # 1.5 columns
    2: 7.20472,
}  # 2 columns
fig_height = 9.72441  # maxium height
fontsize_title = 9
fontsize_label = 8
fontsize_legend = 8
fontsize_axs = 8

spineline_width = 0.3

sns.set_style("ticks")  # darkgrid, white grid, dark, white and ticks
plt.style.use("default")

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = spineline_width

plt.rcParams["font.family"] = "calibri"  # "times new roman"
plt.rcParams["legend.facecolor"] = "white"
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.facecolor"] = "white"
plt.rcParams["legend.edgecolor"] = "black"
plt.rcParams["legend.shadow"] = False
plt.rcParams["font.size"] = fontsize_axs
plt.rcParams["legend.fontsize"] = fontsize_legend
plt.rcParams["axes.labelsize"] = fontsize_axs
plt.rcParams["ytick.labelsize"] = fontsize_axs
plt.rcParams["xtick.labelsize"] = fontsize_axs
plt.rcParams["axes.labelpad"] = 0.0
plt.rcParams["axes.linewidth"] = spineline_width
plt.rcParams["axes.spines.bottom"] = True
plt.rcParams["axes.spines.left"] = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.titlesize"] = fontsize_label
plt.rcParams["xtick.labelsize"] = fontsize_label
plt.rcParams["xtick.major.width"] = spineline_width
plt.rcParams["ytick.major.width"] = spineline_width
plt.rcParams["xtick.minor.width"] = spineline_width
plt.rcParams["ytick.minor.width"] = spineline_width
plt.rcParams["figure.titlesize"] = fontsize_title
plt.rcParams["grid.linewidth"] = spineline_width
plt.rcParams["axes.grid.axis"] = "y"


def plot_proc_contrib(
    scenario,
    save_fig=False,
    path2save_fig="./figs/prod_contrib_NG_scenarios.png",
    verbose=False,
    dict_stats_mc=None,
    quality_level=None,
    stat_relevant_index=None,
):
    PALLETE = [
        "#9ACD32",
        "#1E90FF",
        "#C0C0C0",
        "#8A2BE2",
        "#FFD700",
        "#FF4500",
        "#008080",
        "#D2B48C",
        "#DDA0DD",
        "#FDBF6F",
        "#808000",
    ]
    PALLETE += PALLETE

    n_scenarios = len(scenario)
    label_txt = SCENARIOS[0 : len(scenario)]
    y = [label_txt[s] for s in range(n_scenarios)]
    y_pos = np.arange(len(y))
    keys = list(scenario[0].process_contribution.keys())
    internal_keys = list(scenario[0].process_contribution[keys[0]].keys())
    categories = [" " for _ in scenario[0].categories]
    for ite, cate in enumerate(scenario[0].categories):
        try:
            categories[ite] = (
                cate.split(" no LT ")[1]
                .replace(" human health", "")
                .replace(":", "")
                .replace(" ", "_")
                .replace("-", "")
                .replace("/", "_")
            )
        except IndexError:
            categories[ite] = "climate change"

    regularizer = np.array(
        [
            1e9,
            1e12,
            1e12,
            1e13,
            1e7,
            1e8,
            1e9,
            1e2,
            1e3,
            1e10,
            1e11,
            1e6,
            1e5,
            1e3,
            1e9,
            1e10,
        ]
    )
    categories = [
        f"{nm.split(': h')[0].replace('_',' ').capitalize()}\n["
        # + f"{rg:0.0e}"
        + r"$10^{{{0:0}}}$".format(int(np.floor(np.log10(abs(rg)))))
        + f" {u.replace('-Eq','-eq').replace(' eq.','-eq').replace('CO2','CO$_2$').replace('H+','H$^+$').replace('m3','m$^3$')}]"
        for nm, rg, u in zip(categories, regularizer, scenario[0].categories_unit)
    ]

    if len(quality_level) > 0:
        categories = [
            cat.replace(k, f"{k} ({v})")
            for cat, (k, v) in zip(categories, quality_level.items())
        ]

    if stat_relevant_index is not None:
        categories = [categories[i] for i in stat_relevant_index]

    dpi = 300 if save_fig else 100
    n_rows = 4
    fig, ax = plt.subplots(
        n_rows,
        int(len(categories) / 4),
        sharex=True,
        figsize=(7.20472, 7.20472),
        facecolor="white",
        dpi=dpi,
    )
    ax = ax.flatten()

    legend_txt = [",".join(kk.split(",")[:3]) for kk in keys]
    legend_txt = [kk.split("-")[0] for kk in keys]

    not_index = []  # [0,1,4]
    for i, kk in enumerate(keys):
        s = sum(
            [
                sce.process_contribution[kk][cat]
                for sce in scenario
                for cat in internal_keys
            ]
        )
        if s < 1e-6:
            not_index.append(i)
    not_index = set(not_index)
    indexes = [ite for ite in range(len(keys)) if ite not in not_index]
    if stat_relevant_index is not None:
        ax_ind = stat_relevant_index
    else:
        ax_ind = range(len(ax))
    for ax_i, (f, axx) in enumerate(zip(ax_ind, ax)):
        f_new = f + 1 if "IPCC" in internal_keys[0] else f
        for n, i in enumerate(indexes):
            k = keys[i]
            label = legend_txt[i]

            if i == indexes[0]:
                b = np.zeros(n_scenarios)
                val_ite = np.array(
                    [
                        scenario[s].process_contribution[k][internal_keys[f_new]]
                        for s in range(n_scenarios)
                    ]
                )
                val_ite = val_ite / regularizer[f]
                label = "electricity, NG"
            elif i == indexes[1]:
                b = b + val_ite
                val_ite = np.array(
                    [
                        scenario[s].process_contribution[k][internal_keys[f_new]]
                        for s in range(n_scenarios)
                    ]
                )
                val_ite = val_ite / regularizer[f]
            else:
                b = b + val_ite
                val_ite = (
                    np.array(
                        [
                            scenario[s].process_contribution[k][internal_keys[f_new]]
                            for s in range(n_scenarios)
                        ]
                    )
                    / regularizer[f]
                )
                # label = k.split(",")[:2]
                label = legend_txt[i]
                if type(label) == list:
                    label = ",".join(label[0:2])
                # print(label)
                if "heat, NG" in label:
                    label = "heat, CHP, NG"
                if "nuclear" in label:
                    label = "electricity, nuclear"
                if "heat production from NG, central" in label:
                    label = "heat in households/services, NG"
                    label = "heat, households, NG"
                if "heat production from electricity" in label:
                    label = "heat in households/services, heat pumps"
                    label = "heat, households, electricity"
                if "heat production" in label and "industrial furnace" in label:
                    if "natural gas" in label:
                        label = "heat in industry, NG"
                    elif "hard coal" in label:
                        label = "heat in industry, coal"
                    elif "fuel oil" in label:
                        label = "heat in industry, oil"
            label = label.replace(" production", "").replace(" from", ",")

            error = None
            internal_keys_mc = [
                key.replace("('", "")
                .replace("')", "")
                .replace("'", "")
                .replace(",", " |")
                for key in internal_keys
            ]
            if dict_stats_mc and len(dict_stats_mc) > 0 and n == len(indexes) - 1:
                impact = internal_keys_mc[f_new]
                error = [
                    (val_ite + b)[mc_i]
                    * dict_stats_mc[key_stat].loc[impact, "QCD [%]"]
                    / 100
                    for mc_i, (key_stat) in enumerate(SCENARIOS[0 : len(scenario)])
                    # for mc_i, (key_stat) in enumerate(dict_stats_mc.items())
                ]

            axx.bar(
                y,
                val_ite,
                0.75,
                yerr=error,
                error_kw=(
                    {}
                    if error == None
                    else {
                        "capsize": 2.0,
                        "linewidth": 0.5,
                    }
                ),
                bottom=b,
                label=label.capitalize().replace(" ng", " NG").replace(" chp", " CHP"),
                color=PALLETE[n],
                edgecolor="white",
                linewidth=0.35,
            )

        axx.annotate(
            ascii_uppercase[ax_i] + ")",
            xy=(0, 1),
            xycoords="axes fraction",
            xytext=(+0.5, -0.5),
            textcoords="offset fontsize",
            fontsize="medium",
            verticalalignment="top",
            fontfamily="calibri",
            # bbox=dict(facecolor="0.7", edgecolor="none", pad=3.0),
        )
        ylim = max(b + val_ite) + (max(error) if error else 0)
        axx.set_ylim([0, 1.2 * ylim])

        # annotations on top of the bars
        rects = axx.patches
        labels_top_rects = 100 * (b + val_ite) / (b[0] + val_ite[0])
        for ite, (rect, label) in enumerate(zip(rects, labels_top_rects)):
            height = b[ite] * 1.025 + val_ite[ite] * 1.025  # + 1.01 * error[ite]
            axx.text(
                rect.get_x() + (1.2 if error else 1.0) * rect.get_width() / 2,
                height,
                f"{label:0.0f}%",
                ha="left" if error else "center",
                va="bottom",
                fontsize=8,
            )

        if verbose:
            print(f, categories[ax_i], sep="------------>")
        axx.set_title(categories[ax_i])
        axx.yaxis.set_major_formatter(FormatStrFormatter("%0.1f"))
        # axx.yaxis.set_major_locator(AutoMinorLocator(5))
        axx.yaxis.set_minor_locator(AutoMinorLocator(3))
        axx.tick_params(axis="y", which="minor", length=2)
        axx.tick_params(axis="y", which="major", length=4)

    n_xticks = -math.floor(len(categories) / n_rows) if len(ax) > 4 else -1
    for axs in ax[n_xticks::]:
        axs.xaxis.set_tick_params(labelbottom=True, rotation=30)
        axs.xaxis.set_tick_params(pad=1.0, bottom="on")
        plt.setp(axs.get_xticklabels(), ha="right", rotation_mode="anchor")

    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    handles, labels = ax[-1].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        ncol=4,
        loc="lower center",
        frameon=True,
        bbox_to_anchor=(0.5, -0.07),
    )
    leg.get_frame().set_linewidth(0.3)

    plt.tight_layout()

    if save_fig:
        plt.savefig(path2save_fig.replace(".png", ".svg"), bbox_inches="tight")

    return fig, ax


def plot_pb_flat(
    scenario,
    save_fig=False,
    path2save_fig=r".\figs\pb_scenario.png",
    verbose=False,
    log_scale=False,
):
    n_scenarios = len(scenario)
    df_results = result2df(scenario, verbose=False)

    # Planetary boundaries without cut
    pb_percapta_dict = {
        "climate_change": 501,  # 985,
        "ozone_depletion": 0.078,
        "eutrophication_marine": 290,
        "eutrophication_freshwater": 0.84,
        "eutrophication_terrestrial": 887,
        "acidification": 145,
        "land_use": 1840,
        "water_use": 26300,
        "particulate_matter": 7.47e-5,
        "photochemical_ozone_formation": 58.8,
        "human_toxicity_cancer": 1.39e-4,
        "human_toxicity_noncancer": 5.93e-4,
        "ecotoxicity_freshwater": 1.9e4,
        "ionising_radiation": 7.62e4,
        "energy_resources_nonrenewable": 3.24e4,
        "material_resources_metals_minerals": 3.18e-2,
    }
    EU_POPULATION = 447.0
    pb_europe_dict = {}
    for k, v in pb_percapta_dict.items():
        pb_europe_dict[k] = v * EU_POPULATION * 1e6

    pd.options.display.float_format = "{:0.2e}".format
    df_results_lca_only = [df.iloc[:, 0] for df in df_results]
    df_results_lca_only = pd.concat(df_results_lca_only, axis=1)
    df_results_lca_only.sort_index(inplace=True)
    col = [col for col in df_results_lca_only.columns]
    for ite, colu in enumerate(df_results_lca_only.columns):
        try:
            col[ite] = (
                colu.split(" no LT ")[1]
                .replace(" human health", "")
                .replace(":", "")
                .replace(" ", "_")
                .replace("-", "")
                .replace("/", "_")
            )
        except IndexError:
            if "land use" in colu:
                col[ite] = "Land use"
            else:
                col[ite] = "climate change"

    df_results_lca_only.columns = col
    df_pb_eu = pd.DataFrame.from_dict(pb_europe_dict, orient="index", columns=["PB_EU"])
    df_pb_eu.sort_index(inplace=True)
    df_pb_eu.index = col
    df_pb_eu = df_pb_eu.T
    df_LCA_PB = pd.concat([df_results_lca_only, df_pb_eu])

    X = df_LCA_PB.columns

    for sce in scenario:
        if type(sce) == list or type(sce) == np.ndarray:
            pass
        else:
            sce.categories = X

    YY = []
    # print(df_LCA_PB.head(10))
    for i in range(n_scenarios):
        if log_scale == "lol":
            YY.append(
                np.log10(
                    100 * df_LCA_PB.iloc[i, :].values / df_LCA_PB.iloc[-1, :].values
                )
            )
        else:
            YY.append(100 * df_LCA_PB.iloc[i, :].values / df_LCA_PB.iloc[-1, :].values)

    N = len(YY[0])
    ind = np.arange(N)
    width = 0.25

    X_axis = np.arange(len(X))

    if save_fig:
        fig, ax1 = plt.subplots(
            figsize=(190 / 25.4, 150 / 25.4),
            facecolor="white",
            tight_layout=True,
            dpi=600,
        )
    else:
        fig, ax1 = plt.subplots(
            figsize=(10, 8), facecolor="white", tight_layout=True, dpi=120
        )

    width = 0.8 / n_scenarios
    label_txt = [
        "Base case",
        "REPower scenario",
        "50 BCMeq. coal",
        "50 BCMeq. wind+solar+savings",
        "Test scenario",
    ]
    COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#bcbddc"]
    for i in range(n_scenarios):
        ax1.bar(
            X_axis - 0.4 + float(2 * i + 1) * width / 2,
            YY[i],
            width,
            # label=f'scenario_{i+1}',
            label=label_txt[i],
            # color=pallete[2*i+1],
            color=COLORS[i],
            edgecolor="k",
        )

    X_axis_mod = copy.copy(X_axis)
    X_axis_mod[0] -= 10
    X_axis_mod[-1] += 10
    if log_scale == "lol":
        ax1.plot(
            X_axis_mod, 0 * X_axis_mod + 2, "k--", lw=1, label="Safe planetary boundary"
        )
    else:
        ax1.plot(
            X_axis_mod,
            0 * X_axis_mod + 100,
            "k--",
            lw=1,
            label="Safe planetary boundary",
        )

    plt.yscale("log") if log_scale else plt.yscale("linear")

    plt.xticks(
        X_axis + width,
        [" ".join(txt.split("_")).capitalize() for txt in X],
        rotation=30,
        fontsize=10,
        ha="right",
    )
    ax1.set_xlim(X_axis[0] - 3 * width, X_axis[-1] + 3 * width)
    plt.yticks()
    plt.xlabel("Impact category", fontsize=10)
    plt.ylabel("Comparison with PB safe limit [%]", fontsize=10)
    plt.suptitle("PB of NG scenarios", fontsize=12)
    ax1.legend(
        fontsize=8,
        #    title='Legend',
        loc="upper right",
    )
    if save_fig:
        # plt.savefig('./lca_figs/PB_NG_scenarios.png', bbox_inches='tight')
        plt.savefig(path2save_fig, bbox_inches="tight")

    return fig, ax1


def plot_ef_pb_hb(
    df_LCA_PB,
    save_fig=False,
    path2save_fig=r".\figs\pb_ef_hb.png",
    verbose=False,
    log_scale=False,
):
    X = df_LCA_PB.columns
    n_scenarios = len(df_LCA_PB.index)

    YY = []
    # print(df_LCA_PB.head(10))
    for i in range(n_scenarios):
        if log_scale == "lol":
            YY.append(np.log10(df_LCA_PB.iloc[i, :].values))
        else:
            YY.append(df_LCA_PB.iloc[i, :].values)

    N = len(YY[0])
    ind = np.arange(N)
    width = 0.25

    X_axis = np.arange(len(X))

    fig, ax1 = plt.subplots(figsize=(9.72441, 7.20472), facecolor="white", dpi=300 if save_fig else 120)

    width = 0.8 / n_scenarios
    label_txt = df_LCA_PB.index
    for i in range(n_scenarios):
        ax1.barh(
            X_axis - 2 * width + float(2 * i + 1) * width / 2,
            YY[i],
            width,
            # label=f'scenario_{i+1}',
            label=label_txt[i],
            # color=pallete[2*i+1],
            color=COLORS[i],
            edgecolor="k",
            linewidth=0.3,
        )

    X_axis_mod = copy.copy(X_axis)
    X_axis_mod[0] -= 10
    X_axis_mod[-1] += 10

    plt.xscale("log") if log_scale else plt.xscale("linear")

    plt.yticks(
        X_axis + 0 * width,
        [
            " ".join(txt.split("_"))
            .capitalize()
            .replace("nonrenewable", "non-renewable")
            for txt in X
        ],
        rotation=0,
        fontsize=8,
        ha="right",
    )
    # ax1.set_xlim(0, df_LCA_PB.max().max()*(1.0 + 0.01 if log_scale else 0))
    plt.xlim([0, 4.8 * 10**2])
    ax1.set_ylim(X_axis[0] - 3 * width, X_axis[-1] + 3 * width)
    ax1.invert_yaxis()  # labels read top-to-bottom
    # plt.xticks()
    plt.ylabel("Impact category", fontsize=10)
    plt.xlabel("Planetary boundary allocation [%]", fontsize=10)
    # plt.suptitle("PB of NG scenarios", fontsize=12)
    ax1.legend(
        fontsize=8,
        #    title='Legend',
        loc="lower right",
    )
    if save_fig:
        # plt.savefig('./lca_figs/PB_NG_scenarios.png', bbox_inches='tight')
        new_path2save_fig = path2save_fig.split(".svg")
        if log_scale:
            new_path2save_fig[0] = new_path2save_fig[0] + "_log"
        new_path2save_fig = ".svg".join(new_path2save_fig)
        plt.savefig(new_path2save_fig, bbox_inches="tight")

    return fig, ax1


def plot_ef_pb_hb_err(
    df_LCA_PB,
    df_LCA_PB_EU,
    df_mc_lca,
    categories,
    save_fig=False,
    path2save_fig="./figs/pb_ef_hb_err.png",
    verbose=False,
    log_scale=False,
    plot_mean=False,
):
    X = df_LCA_PB.columns
    n_scenarios = len(df_LCA_PB.index)

    YY = []
    # print(df_LCA_PB.head(10))
    for i in range(n_scenarios):
        if log_scale == "lol":
            YY.append(np.log10(df_LCA_PB.iloc[i, :].values))
        else:
            YY.append(df_LCA_PB.iloc[i, :].values)

    N = len(YY[0])
    ind = np.arange(N)
    width = 0.25

    X_axis = np.arange(len(X))

    fig, ax1 = plt.subplots(figsize=(9.72441, 7.20472), facecolor="white", dpi=300 if save_fig else 120)

    width = 0.8 / n_scenarios
    label_txt = SCENARIOS + [
        "Baseline" if plot_mean else "Mean value",
    ]
    COLORS_local = COLORS + ["#bcbddc"]
    for i, df in enumerate(df_mc_lca.values()):
        ax1.barh(
            X_axis - 2 * width + float(2 * i + 1) * width / 2,
            (
                (
                    (100 * df.mean() / df_LCA_PB_EU).apply(lambda x: np.log10(x))
                    if log_scale == "lol"
                    else 100 * df.mean() / df_LCA_PB_EU
                )
                if plot_mean
                else YY[i]
            ),
            width,
            xerr=100 * df.std() / df_LCA_PB_EU,
            error_kw=dict(lw=0.3, capsize=0.8, capthick=0.3),
            label=label_txt[i],
            color=COLORS_local[i],
            alpha=0.8,
            edgecolor="k",
            linewidth=0.3,
        )
        ax1.scatter(
            (
                YY[i]
                if plot_mean
                else (
                    (100 * df.mean() / df_LCA_PB_EU).apply(lambda x: np.log10(x))
                    if log_scale == "lol"
                    else 100 * df.mean() / df_LCA_PB_EU
                )
            ),
            X_axis - 2 * width + float(2 * i + 1) * width / 2,
            marker=".",
            color="k",
            label=label_txt[-1] if i == 0 else "",
            s=6,
            alpha=0.8,
        )
    ax1.axvline(x=100, color="red", lw=0.75, ls=":")

    X_axis_mod = copy.copy(X_axis)
    X_axis_mod[0] -= 10
    X_axis_mod[-1] += 10

    plt.xscale("log") if log_scale else plt.xscale("linear")

    plt.yticks(
        X_axis + 0 * width,
        [
            " ".join(txt.split("_")).replace("nonrenewable", "non-renewable")
            for txt in categories
        ],
        rotation=0,
        fontsize=8,
        ha="right",
    )
    plt.xlim([0, 10**3])
    ax1.set_ylim(X_axis[0] - 3 * width, X_axis[-1] + 3 * width)
    ax1.invert_yaxis()  # labels read top-to-bottom
    plt.ylabel("Impact category", fontsize=10)
    plt.xlabel(
        "Share of Earth's carrying capacity downscaled to EU's energy consumption [%]",
        fontsize=10,
    )
    ax1.legend(
        fontsize=8,
        loc="lower right",
        ncols=2,
    )
    if save_fig:
        new_path2save_fig = path2save_fig.split(".svg")
        if plot_mean:
            new_path2save_fig[0] = new_path2save_fig[0] + "_mean"
        if log_scale:
            new_path2save_fig[0] = new_path2save_fig[0] + "_log"
        new_path2save_fig = ".svg".join(new_path2save_fig)
        plt.savefig(new_path2save_fig, bbox_inches="tight")

    return fig, ax1


def plot_ipcc_breakdown(
    scenarios,
    label_list,
    color_scenario="#fc8d62",
    axs=None,
    dict_stats_mc=None,
    save_fig=False,
    path2save="./figs/tmp",
):
    ipcc_res_0 = [sce.lca_results[0].score / 1e12 for sce in scenarios]
    ipcc_res_1 = [sce.lca_results[1].score / 1e12 for sce in scenarios]
    ylim_low = round(min([ipcc_res_0[0], ipcc_res_0[-1]]) / 4 * 3, ndigits=1)

    COLORS = ["#DEDEDE" for _ in range(len(scenarios) + 1)]
    COLORS[0] = "#66c2a5"  # base
    COLORS[-1] = color_scenario  # repower color

    height = 0
    col_width = 0.5
    if axs is None:
        fig, axs = plt.subplots(
            1,
            2,
            figsize=(fig_length[2] * 2 / 3, 0.3 * fig_height),
            dpi=300 if save_fig else 120,
            sharex=True,
            sharey=True,
        )
    ax = axs[0]
    ax2 = axs[1]
    max_height = 0
    lwd = 0.3
    for ite, ax in enumerate(axs):
        ipcc_res = [sce.lca_results[ite].score / 1e12 for sce in scenarios]
        # ite_method = 0
        for i, r in enumerate(ipcc_res):
            if i == 0:
                b, top = 0, r
            else:
                if r > top:
                    b, top = height, r - height
                else:
                    b, top = r, top - r

            error = None
            if dict_stats_mc and len(dict_stats_mc) > 0 and i in {0, len(ipcc_res)}:
                impact = dict_stats_mc[list(dict_stats_mc.keys())[0]].index.tolist()[
                    ite
                ]
                error = (
                    top
                    * dict_stats_mc[list(dict_stats_mc.keys())[0]].loc[
                        impact, "QCD [%]"
                    ]
                    / 100
                )

            ax.bar(
                i,
                top,
                bottom=b,
                width=col_width,
                label=label_list[i],
                edgecolor="k",
                yerr=error,
                error_kw=(
                    {}
                    if error == None
                    else {
                        "capsize": 1.0,
                        "linewidth": 0.4,
                        "capthick": 0.4,
                        "ecolor": "magenta",
                    }
                ),
                linewidth=lwd,
                color=COLORS[i],
                alpha=1.0,
                zorder=2.0,
            )
            # link columns
            if i == 0:
                ax.plot(
                    [i + col_width / 2, i + col_width / 2 + (1 - col_width)],
                    [top, top],
                    ls="dotted",
                    color="k",
                    lw=lwd,
                )
            # elif i != len(label_list)-1:
            else:
                if b < height:
                    ax.plot(
                        [i + col_width / 2, i + col_width / 2 + (1 - col_width)],
                        [b, b],
                        ls="dotted",
                        color="k",
                        lw=lwd,
                    )
                else:
                    ax.plot(
                        [i + col_width / 2, i + col_width / 2 + (1 - col_width)],
                        [r, r],
                        ls="dotted",
                        color="k",
                        lw=lwd,
                    )
            height = r
            max_height = max(max_height, height)
            final_i = i

        error = None
        if dict_stats_mc and len(dict_stats_mc) > 0:
            impact = dict_stats_mc[list(dict_stats_mc.keys())[1]].index.tolist()[ite]
            error = (
                ipcc_res[-1]
                * dict_stats_mc[list(dict_stats_mc.keys())[1]].loc[impact, "QCD [%]"]
                / 100
            )
        ax.bar(
            final_i + 1,
            ipcc_res[-1],
            bottom=0,
            width=col_width,
            label=label_list[-1],
            edgecolor="k",
            yerr=error,
            error_kw=(
                {}
                if error == None
                else {
                    "capsize": 1.0,
                    "linewidth": 0.4,
                    "capthick": 0.4,
                    "ecolor": "magenta",
                }
            ),
            linewidth=lwd,
            color=COLORS[-1],
            alpha=1.0,
            zorder=2.0,
        )
        if b < height:
            ax.plot(
                [i + col_width / 2, i + col_width / 2 + (1 - col_width)],
                [b, b],
                ls="dotted",
                color="k",
                lw=lwd,
            )
        else:
            ax.plot(
                [i + col_width / 2, i + col_width / 2 + (1 - col_width)],
                [r, r],
                ls="dotted",
                color="k",
                lw=lwd,
            )

        # customization
        ax.plot(
            np.linspace(-1, len(label_list), 10),
            np.ones(10) * ipcc_res[0],
            ls="--",
            color=COLORS[0],
            lw=lwd,
            alpha=0.5,
            zorder=0.1,
        )
        # ax.set_ylim([1.0e12, max_height])
        ax.set_xlim([-0.5, len(label_list) - 0.5])
        ax.set_xticks(range(len(label_list)))
        ax.set_xticklabels([lab for lab in label_list], size=fontsize_axs-2)
        ax.xaxis.set_tick_params(labelbottom=True, rotation=30)
        ax.xaxis.set_tick_params(pad=1.0, bottom="on")
        plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

        title_tmp = (
            scenarios[0].lca_methods[ite][0]
            + " - "
            + scenarios[0].lca_methods[ite][2].replace(")", "a").split("(")[-1]
        )
        ax.set_title(title_tmp)  # [0], scenarios[0].lca_methods[ite][2], sep=' ')

        # annotating the gap
        delta_mark = len(label_list) - 2 - 0.5 + col_width / 2 - 0.2
        ax.annotate(
            f"{(height-ipcc_res[0])*1000:0.1f} Mt CO$_2$-eq year$^-$$^1$",
            # xy=(delta_mark, (height + ipcc_res[0]) / 2),
            xy=(delta_mark, height),
            xycoords="data",
            xytext=(len(label_list) / 2 - 0.5, ylim_low + 0.1),
            ha="center",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="->",
                lw=0.3,
                connectionstyle="angle3",
                color=[0, 0, 0],
            ),
            fontsize=fontsize_axs,
            fontfamily="calibri",
        )
        ax.plot(
            delta_mark * np.ones(2),
            [ipcc_res[0], height],
            ls="-",
            color="black",
            lw=lwd,
            alpha=1.0,
        )
        ax.plot(
            [delta_mark - 0.05, delta_mark + 0.05],
            ipcc_res[0] * np.ones(2),
            ls="-",
            color="black",
            lw=lwd,
            alpha=1.0,
        )
        ax.plot(
            [delta_mark - 0.05, delta_mark + 0.05],
            [height, height],
            ls="-",
            color="black",
            lw=lwd,
            alpha=1.0,
        )

        # annotations on top of the bars
        rects = ax.patches
        print(rects, len(rects))
        top_bar_label = [ipcc_res[0]]
        top_bar_label += [
            (ipcc_ - ipcc_res[i]) for i, ipcc_ in enumerate(ipcc_res[1::])
        ]
        top_bar_label += [ipcc_res[-1]]
        for ite, (rect, label) in enumerate(zip(rects, top_bar_label)):
            text_top_bar = (
                f"{1000*label:0.0f}"
                if (label < 0 or ite == 0 or ite == len(rects) - 1)
                else f"+{1000*label:0.0f}"
            )
            color_top_bar = "blue" if label < 0 else "red"
            color_top_bar = color_top_bar if ite != 0 else "black"
            color_top_bar = color_top_bar if ite != len(rects) - 1 else "black"
            height = max(
                [
                    rect.get_patch_transform().transform(
                        [(0, 0), (1, 0), (1, 1), (0, 1)]
                    )[i, 1]
                    for i in range(4)
                ]
            )
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                1.001 * height,
                text_top_bar,
                ha="center",
                va="bottom",
                # rotation=45,
                fontsize=top_bar_font_size,
                color=color_top_bar,
                fontfamily="calibri",
            )

        max_height = max(
            [
                max(
                    [
                        rect.get_patch_transform().transform(
                            [(0, 0), (1, 0), (1, 1), (0, 1)]
                        )[i, 1]
                        for i in range(4)
                    ]
                )
                for rect in rects
            ]
        )
        ax.text(
            rects[-1].get_x() + rects[-1].get_width() * 1.0,
            1.05 * max_height,
            r"[Mt CO$_2$-eq year$^{-1}$]",
            ha="right",
            va="bottom",
            fontsize=top_bar_font_size,
            color="black",
            fontfamily="calibri",
        )

        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="minor", length=2)
        ax.tick_params(which="major", length=4)

    # plt.xlabel('Impact of each measure of REPower in the European NG system')
    ax.set_ylim(
        [
            ylim_low,
            max_height * 1.075,
        ]
    )

    axs[0].set_ylabel("Global warming potential [Gt CO$_2$-eq year$^{-1}$]")

    if save_fig:
        plt.savefig(path2save + ".png", bbox_inches="tight")
        plt.savefig(path2save + ".svg", bbox_inches="tight")

    if "fig" in locals():
        return fig, axs
    else:
        return 0, axs


def plot_cc_hist(df_mc_lca, save_fig=False, N_mc=1024, fig=None, ax=None):

    q_quantile = 0.01
    v_no_outlier = {}
    category = "IPCC 2021 | Life cycle CO2 emissions"

    for j, sce_name in enumerate(df_mc_lca.keys()):
        df_mc_lca_tmp = df_mc_lca[sce_name][category]
        q_low = df_mc_lca_tmp.quantile(q_quantile)
        q_hi = df_mc_lca_tmp.quantile(1 - q_quantile)
        v_no_outlier[sce_name] = df_mc_lca_tmp.loc[
            (df_mc_lca_tmp <= q_hi) & (df_mc_lca_tmp >= q_low)
        ]

    df_mc_lca_no_outlier = pd.DataFrame(v_no_outlier)

    df_mc_lca_no_outlier_tmp = []
    for col in df_mc_lca_no_outlier.columns:
        df_mc_lca_no_outlier[f"{col}_name"] = col
        df_mc_lca_no_outlier_tmp.append(
            df_mc_lca_no_outlier[[col, f"{col}_name"]].rename(
                columns={col: category, f"{col}_name": "scenario"}
            )
        )

    df_box_plot_cc = pd.concat(df_mc_lca_no_outlier_tmp, axis=0)
    df_box_plot_cc.reset_index(drop=True, inplace=True)
    df_box_plot_cc[category] = 100 * df_box_plot_cc[category] / (501 * 447.0e6)

    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(fig_length[2] / 3, 0.3 * fig_height), dpi=300 if save_fig else 120)
    current_xlim = (np.Inf, -np.Inf)

    sns.histplot(
        data=df_box_plot_cc,
        y=category,
        hue="scenario",
        stat="frequency",
        multiple="layer",
        palette=COLORS,
        bins=50,
        alpha=0.5,
        # color=COLORS[j],
        # label=sce_name.capitalize() if i != 1 else SCENARIOS[1],
        edgecolor="black",
        linewidth=0.2,
        # orientation="horizontal",
        ax=ax,
    )
    stored_xlim = ax.get_xlim()

    for i, col in enumerate(df_mc_lca_no_outlier.columns):
        if df_mc_lca_no_outlier[col].dtype != float:
            continue
        ax.axhline(
            y=100 * df_mc_lca_no_outlier[col].median() / (501 * 447.0e6),
            ls="--",
            lw=0.75,
            color=COLORS[i],
        )

    ax2 = ax.twinx()
    df_box_plot_cc_2C = pd.concat(df_mc_lca_no_outlier_tmp, axis=0)
    df_box_plot_cc_2C.reset_index(drop=True, inplace=True)
    df_box_plot_cc_2C[category] = (
        100 * df_box_plot_cc_2C[category] / (1440.375 * 447.0e6)
    )
    sns.histplot(
        data=df_box_plot_cc_2C,
        y=category,
        hue="scenario",
        stat="frequency",
        multiple="layer",
        palette=COLORS,
        bins=1,
        alpha=0.0,
        # color=COLORS[j],
        # label=sce_name.capitalize() if i != 1 else SCENARIOS[1],
        edgecolor="white",
        linewidth=0.0,
        # orientation="horizontal",
        ax=ax2,
        legend=False,
    )
    ax.set_xlim(stored_xlim)

    ax2.set_yticks(np.round(ax.get_yticks() * 501 / 1440.375, 0))
    ax2.set_ylim((ax.get_ylim()[0] * 501 / 1440.375, ax.get_ylim()[1] * 501 / 1440.375))

    # ax.set_ylim(0, 520)
    ax.set_title(category)

    ax.set_ylabel("Share of carbon budget 1.5°C [%]", labelpad=1.0)
    ax2.set_ylabel("Share of carbon budget 2.0°C [%]", labelpad=1.0)

    ax.tick_params(which="major", length=4)

    handles = ax.get_legend().legend_handles
    labels = SCENARIOS
    leg = ax.legend(handles, labels, ncols=2, loc="upper center")
    leg.get_frame().set_linewidth(0.3)

    # plt.tight_layout()

    if save_fig:
        plt.savefig(f"./figs/mc_hist_cc_{N_mc}_ite.svg", bbox_inches="tight")

    return fig, ax


def plot_cc_violin(
    df_mc_lca,
    category="IPCC 2021 | Life cycle CO2 emissions",
    save_fig=False,
    N_mc=1024,
    fig=None,
    ax=None,
    budget15=501 * 447.0e6,
    budget20=1440.375 * 447.0e6,
):
    q_quantile = 0.01
    v_no_outlier = {}

    if category == "IPCC 2021 | Life cycle CO2 emissions" or category.endswith("global warming potential (GWP100)"):
        # budget15 = 501 * 447.0e6
        # budget20 = 1440.375 * 447.0e6
        pass
    else:
        print(f"Budget not found for {category}")
        return fig, ax

    for j, sce_name in enumerate(df_mc_lca.keys()):
        df_mc_lca_tmp = df_mc_lca[sce_name][category]
        q_low = df_mc_lca_tmp.quantile(q_quantile)
        q_hi = df_mc_lca_tmp.quantile(1 - q_quantile)
        v_no_outlier[sce_name] = df_mc_lca_tmp.loc[
            (df_mc_lca_tmp <= q_hi) & (df_mc_lca_tmp >= q_low)
        ]

    df_mc_lca_no_outlier = pd.DataFrame(v_no_outlier)

    df_mc_lca_no_outlier_tmp = []
    for col in df_mc_lca_no_outlier.columns:
        df_mc_lca_no_outlier[f"{col}_name"] = col
        df_mc_lca_no_outlier_tmp.append(
            df_mc_lca_no_outlier[[col, f"{col}_name"]].rename(
                columns={col: category, f"{col}_name": "scenario"}
            )
        )

    df_box_plot_cc = pd.concat(df_mc_lca_no_outlier_tmp, axis=0)
    df_box_plot_cc.reset_index(drop=True, inplace=True)
    df_box_plot_cc[category] = 100 * df_box_plot_cc[category] / budget15

    if fig == None and ax == None:
        fig, ax = plt.subplots(figsize=(fig_length[2] / 3, 0.3 * fig_height), dpi=300 if save_fig else 120)
    current_xlim = (np.Inf, -np.Inf)

    sns.violinplot(
        data=df_box_plot_cc,
        y=category,
        x="scenario",
        hue="scenario",
        palette=COLORS[0 : len(df_mc_lca)],
        # alpha=0.5,
        edgecolor="black",
        linewidth=0.3,
        inner=None,
        ax=ax,
        gap=0.0,
        density_norm="count",
        saturation=0.5,
    )
    sns.boxplot(
        data=df_box_plot_cc,
        y=category,
        x="scenario",
        hue="scenario",
        palette=COLORS[0 : len(df_mc_lca)],
        boxprops={"zorder": 2},
        ax=ax,
        width=0.3,
        linewidth=0.3,
        linecolor="black",
        fliersize=0.0,
    )
    stored_xlim = ax.get_xlim()

    for i, col in enumerate(df_mc_lca_no_outlier.columns):
        if df_mc_lca_no_outlier[col].dtype != float:
            continue
        ax.axhline(
            y=100 * df_mc_lca_no_outlier[col].median() / budget15,
            ls="--",
            lw=0.5,
            color=COLORS[i],
        )

    ax2 = ax.twinx()
    df_box_plot_cc_2C = pd.concat(df_mc_lca_no_outlier_tmp, axis=0)
    df_box_plot_cc_2C.reset_index(drop=True, inplace=True)
    df_box_plot_cc_2C[category] = 100 * df_box_plot_cc_2C[category] / budget20
    sns.histplot(
        data=df_box_plot_cc_2C,
        y=category,
        hue="scenario",
        palette=COLORS[0 : len(df_mc_lca)],
        alpha=0.0,
        edgecolor="white",
        linewidth=0.0,
        ax=ax2,
        legend=False,
    )
    ax.set_xlim(stored_xlim)

    ax2.set_yticks(np.round(ax.get_yticks() * budget15 / budget20, 0))
    ax2.set_ylim(
        (ax.get_ylim()[0] * budget15 / budget20, ax.get_ylim()[1] * budget15 / budget20)
    )

    ax.set_title(category.split(" | ")[-1].capitalize().replace("co2", "CO$_2$"))
    ax.set_xlabel(None)
    ax.xaxis.set_tick_params(labelbottom=True, rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    ax.set_ylabel("Share of annual carbon budget 1.5°C [%]", labelpad=1.0)
    ax2.set_ylabel("Share of annual carbon budget 2.0°C [%]", labelpad=1.0)

    ax.tick_params(which="major", length=4)

    if save_fig:
        plt.savefig(f"./figs/mc_violin-box_cc_{N_mc}_ite.svg", bbox_inches="tight")

    return fig, ax


# histogram of Base - REPowerEU
def plot_cc_hist_comparison(
    base_minus_repower,
    df_prob_mc_numeric,
    scenario=SCENARIOS[1],
    base=SCENARIOS[0],
    save_fig=False,
    N_mc=1024,
    fig=None,
    ax=None,
):
    if (fig, ax) == (None, None):
        fig, ax = plt.subplots(figsize=(fig_length[2] / 3, 0.3 * fig_height), dpi=300 if save_fig else 120)

    xlabels = "IPCC 2021 - GWP 100a"
    col = "IPCC 2021 | climate change: including SLCFs | global warming potential (GWP100)"

    ax.set_title(xlabels)

    q_quantile = 0.00
    q_low = base_minus_repower[col].quantile(q_quantile)
    q_hi = base_minus_repower[col].quantile(1 - q_quantile)
    base_minus_repower_no_outlier = base_minus_repower[
        (base_minus_repower[col] < q_hi) & (base_minus_repower[col] > q_low)
    ][[col, f"color_{col}", "dummy_color"]]

    pallete = [COLORS[SCENARIOS.index(base)], COLORS[SCENARIOS.index(scenario)]]

    ax_lt0_leg_handles = []
    if base_minus_repower_no_outlier[col].min() < 0:
        ax_lt0 = sns.histplot(
            data=base_minus_repower_no_outlier.loc[
                base_minus_repower_no_outlier[col] < 0
            ].sort_values(by=f"color_{col}", ascending=False),
            x=col,
            stat="frequency",
            hue=f"color_{col}",
            palette=[pallete[1]],
            # bins=15,
            # binwidth=0.5,
            fill=True,
            alpha=1.0,
            ax=ax,
            linewidth=0.2,
            edgecolor="black",
            label="less",
            legend=True,
        )
        ax_lt0_leg_handles = ax_lt0.get_legend().legend_handles
    ax_gt0_leg_handles = []
    if base_minus_repower_no_outlier[col].max() > 0:
        ax_gt0 = sns.histplot(
            data=base_minus_repower_no_outlier.loc[
                base_minus_repower_no_outlier[col] >= 0
            ].sort_values(by=f"color_{col}", ascending=True),
            x=col,
            stat="frequency",
            hue=f"color_{col}",
            palette=[pallete[0]],
            # bins=15,
            # binwidth=0.5,
            fill=True,
            alpha=1.0,
            ax=ax,
            linewidth=0.2,
            edgecolor="black",
            label="more",
            legend=True,
        )
        ax_gt0_leg_handles = ax_gt0.get_legend().legend_handles
    ax.set_xlim(
        min(0, base_minus_repower_no_outlier[col].min()),
        max(0, base_minus_repower_no_outlier[col].max()),
    )

    line_treshold = ax.axvline(
        x=0.0,
        ls="--",
        lw=0.7,
        color="black",
        label="Threshold",
        alpha=0.75,
        # legend=True,
    )

    line_mean = ax.axvline(
        x=base_minus_repower_no_outlier[col].mean(),
        ls="-.",
        lw=0.7,
        color="magenta",
        label="Mean",
        alpha=0.75,
        # legend=True,
    )
    # plot an x at the mean and y=0
    ax.plot(
        [base_minus_repower_no_outlier[col].mean()],
        [0],
        marker="x",
        markersize=4,
        color="magenta",
        lw=0.7,
        label=None,
    )

    xaxis_len = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
    yaxis_len = abs(ax.get_ylim()[1] - ax.get_ylim()[0])

    # extend 20% of y-axis upwards to make room for the annotation
    ax.set_ylim(
        ax.get_ylim()[0],
        ax.get_ylim()[1] + 0.2 * abs(ax.get_ylim()[1] - ax.get_ylim()[0]),
    )

    # add a coloured box below x-axis above the x=0 line to indicate the probability of A>B (called burden shifting)
    right_side_annotation = True
    is_yellow = False
    if (
        df_prob_mc_numeric.loc[col, f"P[{base} > {scenario}]"] <= 0.75
        and df_prob_mc_numeric.loc[col, f"P[{base} > {scenario}]"] >= 0.25
    ):
        # set x-axis to be centered at 0
        ax.set_xlim(
            -1 * max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1])),
            max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1])),
        )
        # is_yellow = True

    box_width = 180.0
    if df_prob_mc_numeric.loc[col, f"{base} == {scenario}?"] > 0.05:
        annotation_txt = ax.text(
            0.0 + 0.0 * xaxis_len,  # x-coordinate of the text
            ax.get_ylim()[1] - 0.075 * yaxis_len,  # y-coordinate of the text
            "Statistically indistringuishable means"
            + "\np-value "
            + f"{df_prob_mc_numeric.loc[col, f'{base} == {scenario}?']:0.2f}",
            ha="center",  # horizontal alignment
            va="top",  # vertical alignment
            fontfamily="calibri",
            fontsize=fontsize_axs - 2,
            # transform=ax.transAxes,
            wrap=True,
            bbox=dict(
                facecolor="#ffeda0",
                alpha=0.85,
                edgecolor="black",
                linewidth=0.3,
                boxstyle="darrow,pad=0.3",
            ),
        )
        annotation_txt._get_wrap_line_width = lambda: 2 * box_width
    else:
        if df_prob_mc_numeric.loc[col, f"P[{base} > {scenario}]"] < 0.75:
            annotation_txt = ax.text(
                0.0 + 0.015 * xaxis_len,  # x-coordinate of the text
                ax.get_ylim()[1] - 0.075 * yaxis_len,  # y-coordinate of the text
                f"Increase    \nP = {100-100*df_prob_mc_numeric.loc[col, f'P[{base} > {scenario}]']:0.0f}%   ",  # text to display
                ha="left",  # horizontal alignment
                va="top",  # vertical alignment
                fontfamily="calibri",  # font family
                fontsize=fontsize_axs - 2,  # font size
                wrap=True,
                bbox=dict(
                    facecolor="red" if not is_yellow else "#ffeda0",
                    edgecolor="black",
                    alpha=0.75 if not is_yellow else 0.85,
                    linewidth=0.3,
                    boxstyle="rarrow,pad=0.3",
                ),
            )
            annotation_txt._get_wrap_line_width = lambda: box_width
        if df_prob_mc_numeric.loc[col, f"P[{base} > {scenario}]"] > 0.25:
            annotation_txt = ax.text(
                0.0 - 0.015 * xaxis_len,  # x-coordinate of the text
                ax.get_ylim()[1] - 0.075 * yaxis_len,  # y-coordinate of the text
                f"   Reduction\nP = {100*df_prob_mc_numeric.loc[col, f'P[{base} > {scenario}]']:0.0f}%",  # text to display
                ha="right",  # horizontal alignment
                va="top",  # vertical alignment
                fontfamily="calibri",  # font family
                fontsize=fontsize_axs - 2,  # font size
                wrap=True,
                bbox=dict(
                    facecolor="green" if not is_yellow else "#ffeda0",
                    edgecolor="black",
                    alpha=0.75 if not is_yellow else 0.85,
                    linewidth=0.3,
                    boxstyle="larrow,pad=0.3",
                ),
            )
            right_side_annotation = False
            annotation_txt._get_wrap_line_width = lambda: box_width

    xaxis_len = abs(ax.get_xlim()[1] - ax.get_xlim()[0])

    ax.set(xlabel=None)

    ax.xaxis.set_major_formatter("{x:.0f}%")
    ax.xaxis.set_minor_locator(AutoMinorLocator(3))
    ax.tick_params(which="minor", length=2)
    ax.tick_params(which="major", length=4)

    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.set_xlabel(
        f"Distribution of climate change difference\nfrom {base} to {scenario} [%]",
        y=0.02,
    )

    leg = ax.legend(
        ax_lt0_leg_handles
        + ax_gt0_leg_handles
        + [line_treshold, line_mean],
        [
            "Impact\nreduction",
            "Impact\nincrease",
            # "Cumulative\ndistribution",
            "Threshold",
            "Mean",
        ],
        ncol=1,
        loc="center right",
        # loc=0,
        borderaxespad=0,
    )

    leg.get_frame().set_linewidth(0.3)

    if save_fig:
        plt.savefig(f"./figs/mc_hist_A-B_pct_IPCC_{N_mc}_ite.svg", bbox_inches="tight")

    return fig, [ax]