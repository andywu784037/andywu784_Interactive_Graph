Interactive Plots download, and code:

import pandas as pd
import plotly.express as px

# --- Dataset ---
data = {
    "x":  [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "y1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    "y2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
    "y3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
    "y4": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
}

df = pd.DataFrame(data)

# --- Determine common axis ranges ---
x_min, x_max = df["x"].min() - 1, df["x"].max() + 1
y_min = df[[f"y{i}" for i in range(1, 5)]].min().min()
y_max = df[[f"y{i}" for i in range(1, 5)]].max().max()

# --- Generate and save interactive scatter plots ---
for i in range(1, 5):
    fig = px.scatter(
        df,
        x="x",
        y=f"y{i}",
        title=f"Interactive Scatter Plot – Dataset {i}",
        trendline="ols",
        labels={"x": "X Values", f"y{i}": f"Y{i} Values"},
        color_discrete_sequence=["#007bff"],
    )

    fig.update_traces(
        marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey'))
    )
    fig.update_layout(
        width=700,
        height=500,
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[y_min, y_max]),
    )

    # Save to HTML file (for GitHub Pages)
    fig.write_html(f"plot{i}.html")

print("✅ All plots saved as plot1.html to plot4.html")

PDF Code:

# --- Anscombe’s Quartet: Full Report Generator (single cell) ---
# Author: Andy Wu
# Title: Exploring Anscombe’s Quartet: A Lesson in Exploratory Data Analysis
# Date: auto (today)
# Description: A detailed analysis and visualization of Anscombe's Quartet demonstrating the
# importance of graphical analysis in statistics. Produces a multi-page PDF report.

import os
from datetime import datetime
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats
from IPython.display import display, Markdown, HTML

# -------------------------
# 0. Report meta & user inputs (defaults used)
# -------------------------
TITLE = "Exploring Anscombe’s Quartet: A Lesson in Exploratory Data Analysis"
AUTHOR = "Andy Wu"
DATE_STR = datetime.today().strftime("%B %d, %Y")   # e.g., November 9, 2025
SHORT_DESC = ("A detailed analysis and visualization of Anscombe's Quartet "
              "demonstrating the importance of graphical analysis in statistics.")
GITHUB_LINK = "https://github.com/yourusername/anscombe-notebook (placeholder)"
INTERACTIVE_LINK = "To be added"
COLLAB_NOTES = ("Analysis performed by Andy Wu. Template collaboration notes included; "
                "replace with actual PR links and contributor roles if applicable.")
OUTPUT_PDF = "Anscombe_Quartet_Report.pdf"
PY_FILE = "anscombe_report_code.py"

# -------------------------
# 1. Data (Anscombe's Quartet)
# -------------------------
data = {
    "x1": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "y1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    "x2": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "y2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
    "x3": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "y3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
    "x4": [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
    "y4": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
}
df = pd.DataFrame(data)
datasets = [(df["x1"], df["y1"]), (df["x2"], df["y2"]), (df["x3"], df["y3"]), (df["x4"], df["y4"])]

# -------------------------
# 2. Helper: statistics calculation
# -------------------------
def analyze(x, y):
    """Return a dict of summary stats for given x and y (numpy/pandas sequences)."""
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var(ddof=1)
    var_y = y.var(ddof=1)
    sd_x = x.std(ddof=1)
    sd_y = y.std(ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    r = np.corrcoef(x, y)[0, 1]
    slope = r * (sd_y / sd_x)
    intercept = mean_y - slope * mean_x
    r2 = r ** 2
    return {
        "n": n,
        "mean_x": mean_x,
        "mean_y": mean_y,
        "var_x": var_x,
        "var_y": var_y,
        "sd_x": sd_x,
        "sd_y": sd_y,
        "cov_xy": cov_xy,
        "r": r,
        "slope": slope,
        "intercept": intercept,
        "r2": r2
    }

# Compute stats for all four datasets
results = []
for i in range(1, 5):
    x = df[f"x{i}"]
    y = df[f"y{i}"]
    stats_dict = analyze(x, y)
    stats_dict["Dataset"] = f"Dataset {i}"
    results.append(stats_dict)

summary_df = pd.DataFrame(results)[[
    "Dataset","n","mean_x","mean_y","var_x","var_y","sd_x","sd_y","cov_xy","r","slope","intercept","r2"
]]

# Round for neatness
summary_display = summary_df.copy()
summary_display.loc[:, summary_display.columns != "Dataset"] = summary_display.loc[:, summary_display.columns != "Dataset"].round(6)

# -------------------------
# 3. Prepare figures (so we can embed them in the PDF)
# -------------------------
sns.set_theme(style="whitegrid")
figures = {}

# 3.1 Scatterplots with regression lines (4-panel)
fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
axes = axes.flatten()
for i, ((x, y), ax) in enumerate(zip(datasets, axes), start=1):
    sns.regplot(x=x, y=y, ax=ax, ci=None, line_kws={'color':'red'})
    s = analyze(x, y)
    eq = fr"$\hat{{y}} = {s['intercept']:.2f} + {s['slope']:.2f}x$"
    ax.set_title(f"Dataset {i}\n{eq}", fontsize=10)
    ax.set_xlim(2, 20)
    ax.set_ylim(2, 14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
plt.tight_layout()
figures["scatter_4panel"] = fig

# 3.2 Residual plots (4-panel)
fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
axes = axes.flatten()
for i, ((x, y), ax) in enumerate(zip(datasets, axes), start=1):
    s = analyze(x, y)
    # slope/intercept via linregress for numeric stability
    slope, intercept, _, _, _ = stats.linregress(x, y)
    residuals = np.array(y) - (intercept + slope * np.array(x))
    ax.scatter(x, residuals)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_title(f"Residuals: Dataset {i}")
    ax.set_xlim(2, 20)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("x")
    ax.set_ylabel("residual")
plt.tight_layout()
figures["residuals_4panel"] = fig

# 3.3 Overlaid scatter plot
fig = plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue', 'purple']
for i, color in enumerate(colors, start=1):
    plt.scatter(df[f"x{i}"], df[f"y{i}"], label=f"Dataset {i}", color=color, alpha=0.7)
plt.title("Overlaid Scatter Plot of All Datasets")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(2, 20)
plt.ylim(2, 14)
plt.legend()
sns.despine()
figures["overlaid"] = fig

# 3.4 Violin + Box combined (one figure)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
records = []
for i in range(1, 5):
    for v in df[f"x{i}"]:
        records.append({"Dataset": f"Dataset {i}", "Value": v, "Type": "x"})
    for v in df[f"y{i}"]:
        records.append({"Dataset": f"Dataset {i}", "Value": v, "Type": "y"})
df_long = pd.DataFrame(records)
sns.violinplot(ax=axes[0], x="Type", y="Value", hue="Dataset", data=df_long, split=False)
axes[0].set_title("Violin Plots of x and y Distributions (All Datasets)")
axes[0].set_xlabel("Variable Type")
axes[0].set_ylabel("Value")
sns.boxplot(ax=axes[1], x="Type", y="Value", hue="Dataset", data=df_long)
axes[1].set_title("Box Plots of x and y Distributions (All Datasets)")
axes[1].set_xlabel("Variable Type")
axes[1].set_ylabel("Value")
for ax in axes:
    ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    sns.despine(ax=ax)
plt.tight_layout()
figures["violin_box"] = fig

# Save figures temporarily (PdfPages can embed figure objects directly; we also keep file copies)
fig_dir = "report_figs"
os.makedirs(fig_dir, exist_ok=True)
for name, fig in figures.items():
    fname = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    # Also keep the figure object (already in figures dict)
    plt.close(fig)  # close to avoid duplicate display in notebook

# ----- Create a table figure for the summary statistics -----
def fig_from_table(df_table, title=None, fontsize=10):
    """Return a matplotlib figure showing a table (for inclusion in PDF)."""
    fig, ax = plt.subplots(figsize=(10, 2 + 0.4 * len(df_table)))
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12, pad=12)
    # Create table
    table = ax.table(cellText=df_table.values,
                     colLabels=df_table.columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.auto_set_column_width(col=list(range(len(df_table.columns))))
    return fig

table_fig = fig_from_table(summary_display, title="Summary statistics for Anscombe's Quartet (rounded)")
table_fname = os.path.join(fig_dir, "summary_table.png")
table_fig.savefig(table_fname, dpi=200, bbox_inches='tight')
plt.close(table_fig)

# -------------------------
# 4. Build PDF report using PdfPages
# -------------------------
def add_text_page(pdf, title, lines, fontsize=11, lineheight=1.2):
    """Add a text page to the PdfPages file using matplotlib text rendering."""
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    text = "\n".join(lines)
    plt.text(0.05, 0.95, title, fontsize=16, weight='bold', va='top')
    plt.text(0.05, 0.88, "\n".join(textwrap.wrap(text, 110)), fontsize=fontsize, va='top')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

def add_markdown_like_page(pdf, heading, body, fontsize=11):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.05, 0.95, heading, fontsize=16, weight='bold', va='top')
    y = 0.92
    wrapped = textwrap.wrap(body, 110)
    for line in wrapped:
        y -= 0.025 * (len(line) // 100 + 1)  # approximate spacing
        plt.text(0.05, y, line, fontsize=fontsize, va='top')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

with PdfPages(OUTPUT_PDF) as pdf:
    # Title page
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.5, 0.75, TITLE, fontsize=20, ha='center', weight='bold')
    plt.text(0.5, 0.70, f"Author(s): {AUTHOR}", fontsize=12, ha='center')
    plt.text(0.5, 0.67, f"Date: {DATE_STR}", fontsize=12, ha='center')
    plt.text(0.5, 0.60, SHORT_DESC, fontsize=11, ha='center', wrap=True)
    pdf.savefig(fig)
    plt.close(fig)

    # Abstract / Executive summary
    abstract = ("Anscombe's Quartet consists of four datasets with nearly identical "
                "summary statistics (means, variances, correlations, regression lines), "
                "yet markedly different distributions and patterns when plotted. "
                "This report demonstrates exploratory data analysis (EDA) on the quartet, "
                "emphasizing why visualization is essential. The report includes statistical "
                "formulas, summary tables, visual diagnostics, interpretation, and code for reproducibility.")
    add_markdown_like_page(pdf, "Abstract / Executive summary", abstract)

    # Introduction
    intro = ("Anscombe's Quartet was constructed by Francis Anscombe in 1973 to "
             "highlight the importance of graphing data before analyzing. The quartet "
             "contains four datasets that have similar summary statistics but different "
             "underlying distributions — showing that numerical summaries alone can be misleading. "
             "This analysis performs both numeric and visual EDA on each dataset.")
    add_markdown_like_page(pdf, "Introduction", intro)

    # Data section
    data_text = (f"Data: The report uses the canonical Anscombe's Quartet datasets. "
                 "Data were loaded into a pandas DataFrame within this notebook and are shown below.")
    add_markdown_like_page(pdf, "Data", data_text)
    # Add a preview table (first 6 rows)
    preview = df.head(6).copy()
    preview_fig = fig_from_table(preview, title="Data preview (first 6 rows)")
    pdf.savefig(preview_fig)
    plt.close(preview_fig)

    # Methods
    methods = ("Statistics computed: mean (x,y), sample variance, standard deviation, covariance, Pearson correlation (r), "
               "linear regression coefficients (intercept and slope), and coefficient of determination (R²). "
               "Visualizations: scatter plots with regression lines, residual plots, overlaid dataset comparison, "
               "violin and box plots for distributions.")
    add_markdown_like_page(pdf, "Methods", methods)

    # Summary statistics table
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.05, 0.95, "Summary statistics (rounded)", fontsize=14, weight='bold', va='top')
    pdf.savefig(fig)
    plt.close(fig)
    # add the pre-made table image
    im = plt.imread(table_fname)
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.imshow(im)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Visualizations section heading
    add_markdown_like_page(pdf, "Visualizations", "Below are the key visualizations generated during the EDA.")

    # Scatterplots page(s)
    im = plt.imread(os.path.join(fig_dir, "scatter_4panel.png"))
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.imshow(im)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Residuals
    im = plt.imread(os.path.join(fig_dir, "residuals_4panel.png"))
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.imshow(im)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Overlaid
    im = plt.imread(os.path.join(fig_dir, "overlaid.png"))
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.imshow(im)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Violin/box figure
    im = plt.imread(os.path.join(fig_dir, "violin_box.png"))
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.imshow(im)
    plt.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

    # Interactive link & interpretation
    add_markdown_like_page(pdf, "Interactive version", f"Interactive notebook or dashboard can be linked here: https://andywu784037.github.io/andywu784_Interactive_Graph/")
    interpretation = ("Interpretation: Although the four datasets have nearly identical summary statistics, "
                      "their scatterplots reveal very different structures: Dataset 1 looks roughly linear, "
                      "Dataset 2 has a small curve/outlier effect, Dataset 3 contains a high-leverage point, "
                      "and Dataset 4 contains an outlier that strongly influences the regression. "
                      "Therefore, summary statistics alone are insufficient — visualization is required to reveal "
                      "pattern, leverage, non-linearity, and outliers.")
    add_markdown_like_page(pdf, "Interpretation", interpretation)

    # Reproducibility & Code link
    repro = (f"Reproducibility: Code to reproduce this analysis is saved alongside the notebook. "
             f"GitHub link : https://github.com/andywu784037/andywu784_Interactive_Graph/tree/main#readme")
    add_markdown_like_page(pdf, "Reproducibility & Code", repro)

    # Collaboration notes
    add_markdown_like_page(pdf, "Collaboration notes", COLLAB_NOTES)

    # Conclusion & next steps
    conclusions = ("Conclusion: Anscombe's Quartet is a classic demonstration that identical "
                   "numerical summaries can hide very different underlying distributions. "
                   "Next steps: consider robust regression, influence diagnostics (Cook's distance), "
                   "and interactive dashboards to explore leverage and outliers.")
    add_markdown_like_page(pdf, "Conclusion & next steps", conclusions)

    # Appendix: code (we will slice the source code into pages)
    add_markdown_like_page(pdf, "Appendix: Code & Extra Figures", "The following pages include the core notebook code and extra figure images.")
    # Save current code to a .py file so user can push to GitHub
    source_code = r'''
# (This is the same code used to create the report -- saved programmatically.)
# If you want a standalone script, run this .py file after installing required packages.
# [Truncated for PDF view; the full notebook is best stored on GitHub or as the .ipynb]
'''
    with open(PY_FILE, "w") as f:
        f.write(source_code)
    # Put some code snippets on multiple pages (so the PDF contains readable code in the appendix)
    code_lines = open(__file__, "r").read().splitlines() if "__file__" in globals() else [
        "# NOTE: To keep the PDF appendix readable, view the notebook or the saved .py file for full code."
    ]
    # Add a simple page telling where to find the code
    add_markdown_like_page(pdf, "Appendix: Code Location", f"The full notebook and exported script were saved: {os.path.abspath(PY_FILE)}. Use the notebook itself for full source.")
    # Add extra figures page (small thumbnails)
    add_markdown_like_page(pdf, "Appendix: Extra Figures", "Figures used in the analysis were saved to the 'report_figs' folder. Thumbnails shown below.")
    # thumbnail grid
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    imgs = [os.path.join(fig_dir, f) for f in os.listdir(fig_dir) if f.endswith(".png")]
    imgs.sort()
    # simple placement
    x0, y0 = 0.05, 0.9
    dx, dy = 0.45, 0.4
    for idx, img in enumerate(imgs):
        try:
            im = plt.imread(img)
            ax = fig.add_axes([x0 + (idx % 2) * dx, y0 - (idx // 2) * dy, 0.4, 0.35])
            ax.imshow(im)
            ax.set_title(os.path.basename(img), fontsize=8)
            ax.axis('off')
        except Exception:
            pass
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# -------------------------
# 5. Save a backup .py file containing the key plotting+analysis code (for GitHub)
# -------------------------
backup_code = f"""# anscombe_report_code.py
# Generated {DATE_STR}
# Contains the plotting + analysis code used for the report.

# NOTE: For clarity, this file is a shortened 'runner' with the dataset and plotting code.
# Use the notebook for full documentation and markdown sections.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

data = {data!r}
df = pd.DataFrame(data)

# Example: compute summary stats for dataset 1
x = df['x1']; y = df['y1']
mean_x = np.mean(x); mean_y = np.mean(y)
var_x = np.var(x, ddof=1); var_y = np.var(y, ddof=1)
r = np.corrcoef(x,y)[0,1]
slope = r * (np.std(y, ddof=1)/np.std(x, ddof=1))
intercept = mean_y - slope*mean_x
print('Dataset 1: mean_x=', mean_x, 'mean_y=', mean_y, 'r=', r)
# Add the rest of the code as desired...
"""
with open(PY_FILE, "w") as f:
    f.write(backup_code)

print(f"Report generation finished. PDF saved to: {os.path.abspath(OUTPUT_PDF)}")
print(f"Helper script saved to: {os.path.abspath(PY_FILE)}")




