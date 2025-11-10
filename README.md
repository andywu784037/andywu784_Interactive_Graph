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

