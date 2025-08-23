import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def create_swath_plots(merged_df, primary_element, use_log_scale):
    """Create swath plots for Easting, Northing, and Elevation."""
    st.header("Swath Plots")
    st.subheader("Swath Plot Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        easting_bins = st.number_input("Number of Easting bins", min_value=2, max_value=50, value=5)
    with col2:
        northing_bins = st.number_input("Number of Northing bins", min_value=2, max_value=50, value=5)
    with col3:
        elevation_bins = st.number_input("Number of Elevation bins", min_value=2, max_value=50, value=5)

    tab1, tab2, tab3 = st.tabs(["Easting Swath", "Northing Swath", "Elevation Swath"])

    # This function is defined in analysis.py, so we need to import it in app.py and pass it down
    # For simplicity in this refactoring, we'll redefine it here as it's only used here.
    # A better approach would be to import it.
    def create_swath_data(df, coord_col, value_col, num_bins=3):
        df = df.sort_values(coord_col)
        bins = np.linspace(df[coord_col].min(), df[coord_col].max(), num_bins + 1)
        df['bin'] = pd.cut(df[coord_col], bins)
        swath_stats = df.groupby('bin')[value_col].agg(['mean', 'count', 'std']).reset_index()
        swath_stats['bin_centre'] = [(x.left + x.right)/2 for x in swath_stats['bin']]
        swath_stats['bin_width'] = [(x.right - x.left) for x in swath_stats['bin']]
        swath_stats = swath_stats[swath_stats['count'] > 0]
        return swath_stats

    def plot_swath(swath_stats, title, x_title, y_title, reverse_x=False):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=swath_stats['bin_centre'],
            y=swath_stats['count'],
            name='Number of Samples',
            yaxis='y2',
            opacity=0.3,
            marker_color='lightblue',
            width=swath_stats['bin_width'] * 0.9
        ))
        fig.add_trace(go.Scatter(
            x=swath_stats['bin_centre'],
            y=swath_stats['mean'],
            mode='markers+lines',
            name='Mean Grade',
            line=dict(color='blue', width=2),
            error_y=dict(
                type='data',
                array=swath_stats['std'],
                visible=True,
                color='red',
                thickness=1,
                width=4
            )
        ))
        layout_dict = {
            'xaxis_title': x_title,
            'yaxis_title': y_title,
            'yaxis2': dict(
                title="Number of Samples",
                overlaying='y',
                side='right'
            ),
            'height': 700,  
            'width': 2000,  
            'title': title
        }
        if reverse_x:
            layout_dict['xaxis'] = {'autorange': 'reversed'}
        fig.update_layout(**layout_dict)
        return fig

    with tab1:
        easting_swath = create_swath_data(merged_df, 'x', primary_element, easting_bins)
        fig = plot_swath(easting_swath, "Grade Distribution by Easting", "Easting", primary_element)
        st.plotly_chart(fig)

    with tab2:
        northing_swath = create_swath_data(merged_df, 'y', primary_element, northing_bins)
        fig = plot_swath(northing_swath, "Grade Distribution by Northing", "Northing", primary_element)
        st.plotly_chart(fig)

    with tab3:
        elevation_swath = create_swath_data(merged_df, 'z', primary_element, elevation_bins)
        fig = plot_swath(elevation_swath, "Grade Distribution by Elevation", "Elevation", primary_element, reverse_x=True)
        st.plotly_chart(fig)

def plot_scree(wcss=None, explained_variance_ratio=None, is_pca=False):
    """Create scree plot for clustering or PCA."""
    fig = go.Figure()
    if is_pca and explained_variance_ratio is not None:
        cumulative = np.cumsum(explained_variance_ratio)
        fig.add_trace(go.Bar(
            x=list(range(1, len(explained_variance_ratio) + 1)),
            y=explained_variance_ratio,
            name='Individual'
        ))
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative) + 1)),
            y=cumulative,
            name='Cumulative',
            line=dict(color='red')
        ))
        fig.update_layout(
            title='PCA Scree Plot',
            xaxis_title='Principal Component',
            yaxis_title='Explained Variance Ratio',
            showlegend=True
        )
    elif wcss is not None:
        wcss_normalised = [w / wcss[0] for w in wcss]
        wcss_decrease = [-((wcss[i] - wcss[i-1]) / wcss[i-1]) * 100 if i > 0 else 0 
                         for i in range(len(wcss))]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(wcss) + 1)),
                y=wcss_normalised,
                mode='lines+markers',
                name='Normalised WCSS',
                line=dict(color='blue')
            ), secondary_y=False
        )
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(wcss) + 1)),
                y=wcss_decrease,
                name='% Decrease',
                marker_color='rgba(255, 165, 0, 0.5)'
            ), secondary_y=True
        )
        fig.update_layout(
            title='Clustering Scree Plot',
            xaxis_title='Number of Clusters',
            showlegend=True
        )
        fig.update_yaxes(title_text="Normalised WCSS", secondary_y=False)
        fig.update_yaxes(title_text="Percentage Decrease", secondary_y=True)
    return fig

def plot_pca_biplot(X_pca, pca, feature_names):
    """Create PCA biplot (2D or 3D)."""
    fig = go.Figure()
    n_components = X_pca.shape[1]
    if n_components >= 3:
        fig.add_trace(go.Scatter3d(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            z=X_pca[:, 2],
            mode='markers',
            marker=dict(size=6),
            name='Samples',
            opacity=0.7
        ))
    else:
        fig.add_trace(go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            marker=dict(size=6),
            name='Samples',
            opacity=0.7
        ))

    data_range = np.max(np.abs(X_pca))
    loading_range = np.max(np.abs(pca.components_))
    scaling_factor = (data_range / loading_range) * 0.8

    for i, feature in enumerate(feature_names):
        if n_components >= 3:
            fig.add_trace(go.Scatter3d(
                x=[0, pca.components_[0, i] * scaling_factor],
                y=[0, pca.components_[1, i] * scaling_factor],
                z=[0, pca.components_[2, i] * scaling_factor],
                mode='lines+text',
                line=dict(color='red', width=3),
                text=['', feature],
                textposition='top center',
                textfont=dict(size=12),
                name=feature,
                showlegend=True
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[0, pca.components_[0, i] * scaling_factor],
                y=[0, pca.components_[1, i] * scaling_factor],
                mode='lines+text',
                line=dict(color='red', width=3),
                text=['', feature],
                textposition='top center',
                textfont=dict(size=12),
                name=feature,
                showlegend=True
            ))

    if n_components >= 3:
        fig.update_layout(
            title='3D PCA Biplot',
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%})',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1600,
            height=1200
        )
    else:
        fig.update_layout(
            title='2D PCA Biplot',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%})',
            width=1200,
            height=800
        )
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
            font=dict(size=12)
        ),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.add_annotation(
        text="Red lines show feature loadings",
        xref="paper", yref="paper",
        x=0, y=1.05,
        showarrow=False,
        font=dict(size=14)
    )
    return fig

def plot_eigenvectors(pca, feature_names):
    """Plot eigenvectors for each principal component."""
    n_components = pca.components_.shape[0]
    n_rows = (n_components + 1) // 2
    fig = make_subplots(rows=n_rows, cols=2)
    for i in range(n_components):
        row = i // 2 + 1
        col = i % 2 + 1
        eigenvector = pca.components_[i]
        sorted_idx = np.argsort(eigenvector)
        pos = np.arange(len(eigenvector))
        fig.add_trace(
            go.Scatter(
                x=pos,
                y=eigenvector[sorted_idx],
                mode='lines+markers',
                name=f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})',
                line=dict(color='blue')
            ),
            row=row, col=col
        )
        fig.update_xaxes(
            ticktext=[feature_names[idx] for idx in sorted_idx],
            tickvals=pos,
            tickangle=45,
            row=row, col=col
        )
        fig.update_yaxes(title_text=f'PC{i+1} Loading', row=row, col=col)
    fig.update_layout(
        height=300 * n_rows,
        width=1600,
        showlegend=True,
        title_text="Principal Component Loadings"
    )
    return fig

def get_cluster_summary(df, cluster_features, primary_element):
    """Get statistics for each feature by cluster."""
    features = list(dict.fromkeys(cluster_features + [primary_element]))
    summary_dict = {}
    for feature in features:
        cluster_stats = {}
        for cluster in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster][feature]
            cluster_stats[f'Cluster {cluster}'] = {
                'mean': cluster_data.mean(),
                'median': cluster_data.median(),
                'std': cluster_data.std(),
                'min': cluster_data.min(),
                'max': cluster_data.max()
            }
        summary_dict[feature] = cluster_stats
    index = pd.MultiIndex.from_product([features, ['mean', 'median', 'std', 'min', 'max']])
    columns = [f'Cluster {i}' for i in sorted(df['Cluster'].unique())]
    summary_df = pd.DataFrame(index=index, columns=columns)
    for feature in features:
        for stat in ['mean', 'median', 'std', 'min', 'max']:
            for cluster in sorted(df['Cluster'].unique()):
                summary_df.loc[(feature, stat), f'Cluster {cluster}'] = summary_dict[feature][f'Cluster {cluster}'][stat]
    return summary_df

def plot_cluster_boxplots(df, cluster_features, primary_element, use_log_scale=True):
    """Create boxplots showing element distributions by cluster."""
    fig = go.Figure()
    cluster_colours = px.colors.qualitative.Set1
    legend_added = set()
    for feature in cluster_features + [primary_element]:
        for i, cluster in enumerate(sorted(df['Cluster'].unique())):
            cluster_data = df[df['Cluster'] == cluster]
            show_legend = cluster not in legend_added
            if show_legend:
                legend_added.add(cluster)
            fig.add_trace(go.Box(
                x=[feature] * len(cluster_data),
                y=cluster_data[feature],
                name=f'Cluster {cluster}',
                marker_color=cluster_colours[i % len(cluster_colours)],
                boxpoints='outliers',
                jitter=0.3,
                pointpos=0,
                offsetgroup=str(cluster),
                showlegend=show_legend
            ))
    fig.update_layout(
        xaxis_title='Elements',
        yaxis_title='Values',
        boxmode='group',
        height=1000,
        width=max(1800, 200 * len(cluster_features)),
        showlegend=True,
        yaxis=dict(
            type='log' if use_log_scale else 'linear',
            tickformat='.3f',
            dtick='D1' if use_log_scale else None,
            showgrid=True,
            tickmode='auto',
            tick0=0,
            ticks='outside'
        ),
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        margin=dict(r=150)
    )
    if use_log_scale:
        all_values = []
        for feature in cluster_features + [primary_element]:
            all_values.extend(df[feature].dropna().values)
        min_val = min(v for v in all_values if v > 0)
        max_val = max(all_values)
        log_min = np.floor(np.log10(min_val))
        log_max = np.ceil(np.log10(max_val))
        tick_vals = []
        tick_text = []
        for i in range(int(log_min), int(log_max) + 1):
            tick_vals.extend([10**i, 2*10**i, 5*10**i])
            tick_text.extend([f'{10**i:.3g}', f'{2*10**i:.3g}', f'{5*10**i:.3g}'])
        tick_vals = [v for v in tick_vals if min_val <= v <= max_val]
        tick_text = tick_text[:len(tick_vals)]
        tick_vals, tick_text = zip(*sorted(zip(tick_vals, tick_text)))
        tick_vals = list(tick_vals)
        tick_text = list(tick_text)
        fig.update_yaxes(
            tickvals=tick_vals,
            ticktext=tick_text
        )
    return fig

def plot_lithology_cluster_comparison(df):
    """Create heatmap comparing lithology vs cluster."""
    lith_cluster_counts = df.groupby(['LITHO', 'Cluster']).size().unstack(fill_value=0)
    fig = px.imshow(lith_cluster_counts,
                    labels=dict(x="Cluster", y="Lithology", color="Count"),
                    title="Lithology vs Cluster Heatmap")
    fig.update_traces(
        text=lith_cluster_counts.values.astype(int),
        texttemplate="%{text}",
        textfont={"size": 12},
        showscale=True
    )
    fig.update_xaxes(
        tickmode='array',
        ticktext=list(range(len(lith_cluster_counts.columns))),
        tickvals=list(range(len(lith_cluster_counts.columns)))
    )
    return fig

def create_lithology_analysis(merged_df, primary_element, use_log_scale, litho_dict=None):
    """Create lithology analysis plots and stats."""
    st.header("Lithology Analysis")
    st.subheader(f"Summary Statistics by Lithology - {primary_element}")
    
    litho_stats = merged_df.groupby('LITHO').agg({
        primary_element: [
            'count',
            'mean',
            'median',
            'std',
            'min',
            lambda x: x.quantile(0.25),
            lambda x: x.quantile(0.75),
            'max',
            lambda x: x.std()/x.mean() if x.mean() != 0 else np.nan
        ]
    })
    litho_stats.columns = [
        'Count', 'Mean', 'Median', 'Std Dev',
        'Min', 'Q1', 'Q3', 'Max', 'CV'
    ]
    litho_stats = litho_stats.round(3)
    litho_stats = litho_stats.reset_index()

    if litho_dict:
        litho_stats['Description'] = litho_stats['LITHO'].map(lambda x: litho_dict.get(x, ""))
        litho_stats['LITHO_LABEL'] = litho_stats.apply(
            lambda row: f"{row['LITHO']} {row['Description']}".strip(), axis=1
        )
    else:
        litho_stats['LITHO_LABEL'] = litho_stats['LITHO']

    litho_stats = litho_stats.sort_values('Count', ascending=False)
    display_columns = ['LITHO']
    if litho_dict:
        display_columns.append('Description')
    display_columns.extend(['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Q1', 'Q3', 'Max', 'CV'])
    st.dataframe(litho_stats[display_columns])

    st.subheader(f"Grade Distribution by Lithology - {primary_element}")
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "Sort Lithologies by:",
            ['Median', 'Mean', 'Count', 'Alphabetical']
        )
    with col2:
        min_samples = st.number_input(
            "Minimum samples per lithology:",
            min_value=1,
            value=2,
            step=1
        )
    valid_lithos = litho_stats[litho_stats['Count'] >= min_samples]['LITHO']
    plot_df = merged_df[merged_df['LITHO'].isin(valid_lithos)].copy()
    if not plot_df.empty:
        if sort_by == 'Median':
            litho_order = litho_stats[litho_stats['Count'] >= min_samples].sort_values('Median', ascending=False)['LITHO']
        elif sort_by == 'Mean':
            litho_order = litho_stats[litho_stats['Count'] >= min_samples].sort_values('Mean', ascending=False)['LITHO']
        elif sort_by == 'Count':
            litho_order = litho_stats[litho_stats['Count'] >= min_samples].sort_values('Count', ascending=False)['LITHO']
        else: 
            litho_order = sorted(valid_lithos)

        fig = go.Figure()
        litho_label_map = dict(zip(litho_stats['LITHO'], litho_stats['LITHO_LABEL']))
        for litho in litho_order:
            litho_data = plot_df[plot_df['LITHO'] == litho][primary_element]
            fig.add_trace(go.Box(
                y=litho_data,
                name=litho_label_map[litho],
                boxpoints='outliers',
                jitter=0.3,
                pointpos=0
            ))
        y_min = plot_df[primary_element].min()
        y_max = plot_df[primary_element].max()
        if use_log_scale and y_min > 0:
            log_y_min = np.floor(np.log10(y_min))
            log_y_max = np.ceil(np.log10(y_max))
            tick_values = [y_min]
            tick_texts = [f'{y_min:.2f}']
            for i in range(int(log_y_min), int(log_y_max) + 1):
                current = 10**i
                if y_min <= current <= y_max:
                    if current not in tick_values:
                        tick_values.append(current)
                        tick_texts.append(f'{current:.0f}')
                if 10 * current <= y_max and 10 * current >= y_min:
                    if 10 * current not in tick_values:
                        tick_values.append(10 * current)
                        tick_texts.append(f'{10 * current:.0f}')
            if y_max not in tick_values:
                tick_values.append(y_max)
                tick_texts.append(f'{y_max:.2f}')
            tick_values, tick_texts = zip(*sorted(zip(tick_values, tick_texts)))
            fig.update_layout(
                title=f"{primary_element} Distribution by Lithology",
                yaxis=dict(
                    title=f"{primary_element} (log scale)",
                    type='log',
                    range=[np.log10(y_min), np.log10(y_max)],
                    showgrid=True,
                    tickmode='array',
                    tickvals=tick_values,
                    ticktext=tick_texts
                ),
                showlegend=True,
                height=600,
                boxmode='group'
            )
        else:
            y_range = y_max - y_min
            if y_range <= 10:
                tick_interval = 1
            elif y_range <= 20:
                tick_interval = 2
            elif y_range <= 50:
                tick_interval = 5
            else:
                tick_interval = 10 ** np.floor(np.log10(y_range / 10))
            tick_min = np.floor(y_min / tick_interval) * tick_interval
            tick_max = np.ceil(y_max / tick_interval) * tick_interval
            tick_values = np.arange(tick_min, tick_max + tick_interval, tick_interval)
            if y_min not in tick_values:
                tick_values = np.sort(np.append(tick_values, y_min))
            if y_max not in tick_values:
                tick_values = np.sort(np.append(tick_values, y_max))
            if tick_interval >= 1:
                tick_format = '.0f'
            elif tick_interval >= 0.1:
                tick_format = '.1f'
            else:
                tick_format = '.2f'
            tick_texts = [f'{v:.2f}' if v in (y_min, y_max) else f'{v:{tick_format}}' for v in tick_values]
            fig.update_layout(
                title=f"{primary_element} Distribution by Lithology",
                yaxis=dict(
                    title=primary_element,
                    type='linear',
                    range=[tick_values[0], tick_values[-1]],
                    showgrid=True,
                    tickmode='array',
                    tickvals=tick_values,
                    ticktext=tick_texts
                ),
                showlegend=True,
                height=600,
                boxmode='group'
            )
        fig.update_layout(
            xaxis=dict(
                tickangle=-45,
                tickfont=dict(size=10)
            )
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for plotting after applying filters.")

def add_grade_visualisation(fig, df, element, use_log_scale, viz_mode, color_by, x_offset=0, colorbar_x=1.02):
    """
    Adds grade visualisation to a 3D plot.
    Accepts a colorbar_x parameter to control the legend position.
    """
    df_vis = df.copy()
    
    if use_log_scale and df_vis[element].min() > 0:
        df_vis[element] = np.log10(df_vis[element])
        colorbar_title = f"Log10({element})"
    else:
        colorbar_title = element

    hover_text = [
        f"<b>Hole ID:</b> {row['HOLE_ID']}<br>" +
        f"<b>{element}:</b> {row[element]:.4f}<br>" +
        f"<b>From:</b> {row['FROM']:.2f}m<br>" +
        f"<b>To:</b> {row['TO']:.2f}m"
        for _, row in df.iterrows()
    ]

    fig.add_trace(go.Scatter3d(
        x=df_vis['x'] + x_offset, 
        y=df_vis['y'], 
        z=df_vis['z'],
        mode='markers',
        marker=dict(
            size=8,
            color=df_vis[element],
            colorscale='viridis',
            colorbar=dict(
                title=colorbar_title,
                x=colorbar_x,
                len=0.7,
                yanchor='middle',
                y=0.5
            ),
            showscale=True
        ),
        name=element,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        visible=True if viz_mode == "Combined" else 'legendonly' if viz_mode == "Separate" else True
    ))

def add_lithology_visualisation(fig, viz_litho_df, viz_mode, selected_lithos=None, litho_dict=None, x_offset=0):
    """Add lithology visualisation to the figure as intervals."""
    if viz_litho_df is None or viz_litho_df.empty:
        st.warning("No lithology data available for display.")
        return
    
    unique_lithos = viz_litho_df['LITHO'].unique()
    colour_palette = px.colors.qualitative.Alphabet
    litho_colours = [colour_palette[i % len(colour_palette)] for i in range(len(unique_lithos))]
    litho_colour_map = dict(zip(unique_lithos, litho_colours))
    legend_added = set()
    
    for hole_id in viz_litho_df['HOLE_ID'].unique():
        hole_data = viz_litho_df[viz_litho_df['HOLE_ID'] == hole_id].sort_values('FROM')
        
        for _, interval in hole_data.iterrows():
            litho_code = interval['LITHO']
            litho_desc = litho_dict.get(litho_code, "") if litho_dict else ""
            legend_name = f"{litho_code} {litho_desc}".strip() if litho_dict else litho_code
            show_legend = legend_name not in legend_added
            if show_legend:
                legend_added.add(legend_name)
            
            azimuth_rad = np.radians(90 - interval['AZIMUTH'])
            dip_rad = np.radians(-interval['DIP'])
            
            from_depth = interval['FROM']
            from_dx = from_depth * np.cos(dip_rad) * np.cos(azimuth_rad)
            from_dy = from_depth * np.cos(dip_rad) * np.sin(azimuth_rad)
            from_dz = from_depth * np.sin(dip_rad)
            from_x = interval['EASTING'] + from_dx + x_offset
            from_y = interval['NORTHING'] + from_dy
            from_z = interval['ELEVATION'] - from_dz
            
            to_depth = interval['TO']
            to_dx = to_depth * np.cos(dip_rad) * np.cos(azimuth_rad)
            to_dy = to_depth * np.cos(dip_rad) * np.sin(azimuth_rad)
            to_dz = to_depth * np.sin(dip_rad)
            to_x = interval['EASTING'] + to_dx + x_offset
            to_y = interval['NORTHING'] + to_dy
            to_z = interval['ELEVATION'] - to_dz
            
            fig.add_trace(go.Scatter3d(
                x=[from_x, to_x],
                y=[from_y, to_y],
                z=[from_z, to_z],
                mode='lines',
                line=dict(
                    width=20,
                    color=litho_colour_map[litho_code]
                ),
                name=legend_name,
                legendgroup=legend_name,
                hovertemplate=(
                    f"<b>Hole ID:</b> {hole_id}<br>" +
                    f"<b>Lithology:</b> {legend_name}<br>" +
                    f"<b>From:</b> {interval['FROM']:.2f}m<br>" +
                    f"<b>To:</b> {interval['TO']:.2f}m<br>" +
                    f"<b>Length:</b> {interval['TO'] - interval['FROM']:.2f}m<br>" +
                    "<extra></extra>"
                ),
                showlegend=show_legend
            ))
            
def add_collar_points(fig, collar_df, x_offset=0):
    """Add collar points to the figure."""
    fig.add_trace(go.Scatter3d(
        x=collar_df['EASTING'] + x_offset,
        y=collar_df['NORTHING'],
        z=collar_df['ELEVATION'],
        mode='markers',
        marker=dict(
            size=4,
            color='red',
            symbol='circle'
        ),
        name='Collar Points',
        hovertemplate=(
            "<b>Hole ID:</b> %{customdata}<br>" +
            "<b>Easting:</b> %{x:.2f}<br>" +
            "<b>Northing:</b> %{y:.2f}<br>" +
            "<b>Elevation:</b> %{z:.2f}<br>"
        ),
        customdata=collar_df['HOLE_ID']
    ))

def create_combined_3d_visualisation(df, collar_df, viz_litho_df, litho_dict, selected_viz, primary_element, use_log_scale, vertical_exaggeration, legend_title="Legend", use_log_scale_anomaly=True):
    """
    Creates a combined 3D visualisation with optional grade, lithology, cluster, and anomaly score views.
    Now includes a separate log scale control for anomaly scores.
    """
    fig = go.Figure()
    offsets = {"Anomaly Score": 0, "Grade": 10, "Clusters": -10, "Lithology": 20}

    active_holes = df['HOLE_ID'].unique()
    filtered_collar_df = collar_df[collar_df['HOLE_ID'].isin(active_holes)]
    filtered_litho_df = viz_litho_df[viz_litho_df['HOLE_ID'].isin(active_holes)] if viz_litho_df is not None else None
    
    for viz_type in selected_viz:
        offset = offsets.get(viz_type, 0)
        
        if viz_type == "Grade" and primary_element and not df.empty:
            add_grade_visualisation(fig, df, primary_element, use_log_scale, "Combined", color_by='grade', x_offset=offset, colorbar_x=1.1)
        
        elif viz_type == "Anomaly Score" and 'Anomaly_Score' in df.columns:
            add_grade_visualisation(fig, df, 'Anomaly_Score', use_log_scale=use_log_scale_anomaly, viz_mode="Combined", color_by='grade', x_offset=offset, colorbar_x=0.99)
            
        elif viz_type == "Lithology" and filtered_litho_df is not None and not filtered_litho_df.empty:
            add_lithology_visualisation(fig, filtered_litho_df, "Combined", None, litho_dict, x_offset=offset)
            
        elif viz_type == "Clusters" and 'Cluster' in df.columns:
            cluster_viz_df = df[df['Cluster'] >= 0].copy()
            if not cluster_viz_df.empty:
                for cluster in sorted(cluster_viz_df['Cluster'].unique()):
                    cluster_data = cluster_viz_df[cluster_viz_df['Cluster'] == cluster]
                    hover_text = [f"<b>Hole ID:</b> {row['HOLE_ID']}<br><b>Cluster:</b> {cluster}<br><b>From:</b> {row['FROM']:.2f}m<br><b>To:</b> {row['TO']:.2f}m" for _, row in cluster_data.iterrows()]
                    fig.add_trace(go.Scatter3d(
                        x=cluster_data['x'] + offset, y=cluster_data['y'], z=cluster_data['z'], mode='markers', 
                        marker=dict(size=8, color=px.colors.qualitative.Set1[cluster % len(px.colors.qualitative.Set1)]), 
                        name=f'Cluster {cluster}', text=hover_text, hovertemplate="%{text}<extra></extra>"
                    ))

    required_offsets = {offsets[viz] for viz in selected_viz if viz in offsets}
    for offset in required_offsets:
        if not df.empty:
            for hole in active_holes:
                hole_data = df[df['HOLE_ID'] == hole]
                collar_point = filtered_collar_df[filtered_collar_df['HOLE_ID'] == hole]
                if not collar_point.empty:
                    collar_point = collar_point.iloc[0]
                    x_line = [collar_point['EASTING'] + offset] + (hole_data['x'] + offset).tolist()
                    y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
                    z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
                    fig.add_trace(go.Scatter3d(x=x_line, y=y_line, z=z_line, mode='lines', line=dict(color='gray', width=1), showlegend=False))
            
            if not filtered_collar_df.empty:
                add_collar_points(fig, filtered_collar_df, x_offset=offset)

    update_figure_layout(fig, vertical_exaggeration, legend_title)
    return fig

def update_figure_layout(fig, vertical_exaggeration=1.0, legend_title="Legend"):
    """Update figure layout with proper aspect ratios."""
    x_min, x_max = float('inf'), float('-inf')
    y_min, y_max = float('inf'), float('-inf')
    z_min, z_max = float('inf'), float('-inf')

    for trace in fig.data:
        if hasattr(trace, 'x') and trace.x is not None and len(trace.x) > 0:
            x_min = min(x_min, min(trace.x))
            x_max = max(x_max, max(trace.x))
        if hasattr(trace, 'y') and trace.y is not None and len(trace.y) > 0:
            y_min = min(y_min, min(trace.y))
            y_max = max(y_max, max(trace.y))
        if hasattr(trace, 'z') and trace.z is not None and len(trace.z) > 0:
            z_min = min(z_min, min(trace.z))
            z_max = max(z_max, max(trace.z))

    if x_min == float('inf'): x_min, x_max = 0, 1
    if y_min == float('inf'): y_min, y_max = 0, 1
    if z_min == float('inf'): z_min, z_max = 0, 1

    x_diff = x_max - x_min
    y_diff = y_max - y_min
    z_diff = z_max - z_min
    max_diff = max(x_diff, y_diff, z_diff * vertical_exaggeration)
    if max_diff == 0: max_diff = 1

    fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(
                x=x_diff / max_diff,
                y=y_diff / max_diff,
                z=(z_diff * vertical_exaggeration) / max_diff
            ),
            xaxis_title="Easting",
            yaxis_title="Northing",
            zaxis_title="Elevation"
        ),
        width=1800,
        height=1200,
        margin=dict(l=0, r=200, b=0, t=40),
        legend=dict(
            yanchor="top",
            y=0.9,
            xanchor="left",
            x=1.15,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            title=legend_title
        )
    )
    fig.update_scenes(
        xaxis_range=[x_min, x_max],
        yaxis_range=[y_min, y_max],
        zaxis_range=[z_min, z_max]
    )

def create_drill_fence_cross_section(merged_df, viz_litho_df, collar_df, section_line_start, section_line_end, section_width, primary_element=None, use_log_scale=True, litho_dict=None):
    """
    Create a drill fence cross-section with lithology and grade on opposite sides of drill traces.
    """
    start_x, start_y = section_line_start
    end_x, end_y = section_line_end
    
    line_dx = end_x - start_x
    line_dy = end_y - start_y
    line_length = np.sqrt(line_dx**2 + line_dy**2)
    
    if line_length == 0:
        st.error("Section line start and end points cannot be the same")
        return None
    
    line_unit_x = line_dx / line_length
    line_unit_y = line_dy / line_length
    
    perp_unit_x = -line_unit_y
    perp_unit_y = line_unit_x
    
    def point_to_line_distance_and_position(px, py):
        """Calculate perpendicular distance from point to section line and position along line."""
        to_point_x = px - start_x
        to_point_y = py - start_y
        along_line = to_point_x * line_unit_x + to_point_y * line_unit_y
        perp_distance = to_point_x * perp_unit_x + to_point_y * perp_unit_y
        return along_line, perp_distance
    
    section_data = []
    for idx, row in merged_df.iterrows():
        along_line, perp_dist = point_to_line_distance_and_position(row['x'], row['y'])
        
        if (abs(perp_dist) <= section_width/2 and 
            0 <= along_line <= line_length):
            
            row_dict = row.to_dict()
            row_dict['section_distance'] = along_line
            row_dict['perp_distance'] = perp_dist
            section_data.append(row_dict)
    
    if not section_data:
        return None
    
    section_df = pd.DataFrame(section_data)
    
    fig = go.Figure()
    
    grade_offset = 5
    litho_offset = -5
    
    collar_section_data = []
    for idx, row in collar_df.iterrows():
        along_line, perp_dist = point_to_line_distance_and_position(row['EASTING'], row['NORTHING'])
        
        if (abs(perp_dist) <= section_width/2 and 
            0 <= along_line <= line_length):
            collar_section_data.append({
                'HOLE_ID': row['HOLE_ID'],
                'section_distance': along_line,
                'elevation': row['ELEVATION'],
                'perp_distance': perp_dist
            })
    
    collar_section_dict = {}
    if collar_section_data:
        collar_section_dict = {c['HOLE_ID']: c for c in collar_section_data}
    
    if primary_element and primary_element in section_df.columns:
        hover_text = []
        for _, row in section_df.iterrows():
            info = [
                f"<b>Hole ID:</b> {row['HOLE_ID']}",
                f"<b>{primary_element}:</b> {row[primary_element]:.2f}",
                f"<b>From:</b> {row['FROM']:.2f}m",
                f"<b>To:</b> {row['TO']:.2f}m",
                f"<b>Distance along section:</b> {row['section_distance']:.1f}m"
            ]
            if 'LITHO' in row:
                info.append(f"<b>Lithology:</b> {row['LITHO']}")
            hover_text.append("<br>".join(info))
        
        colour_values = section_df[primary_element]
        min_val = section_df[primary_element].min()
        max_val = section_df[primary_element].max()
        
        if use_log_scale and min_val > 0:
            log_min = np.floor(np.log10(min_val))
            log_max = np.ceil(np.log10(max_val))
            tick_vals = []
            tick_text = []
            for i in range(int(log_min), int(log_max) + 1):
                current = 10**i
                if min_val <= current <= max_val:
                    tick_vals.append(current)
                    tick_text.append(f'{current:.3g}')
            if min_val not in tick_vals:
                tick_vals.insert(0, min_val)
                tick_text.insert(0, f'{min_val:.2f}')
            if max_val not in tick_vals:
                tick_vals.append(max_val)
                tick_text.append(f'{max_val:.2f}')
        else:
            tick_vals = np.linspace(min_val, max_val, 6)
            tick_text = [f'{v:.3f}' for v in tick_vals]
        
        fig.add_trace(go.Scatter(
            x=section_df['section_distance'] + grade_offset,
            y=section_df['z'],
            mode='markers',
            marker=dict(
                size=15,
                color=colour_values,
                colorscale='rdylbu',
                reversescale=True,
                showscale=True,
                colorbar=dict(
                    title=f"{primary_element} Grade",
                    len=0.75,
                    tickmode='array',
                    tickvals=tick_vals,
                    ticktext=tick_text,
                    x=1.02
                ),
                cmin=min_val,
                cmax=max_val,
                symbol='square',
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name=f'{primary_element} Grade'
        ))
    
    if viz_litho_df is not None:
        litho_section_data = []
        for idx, row in viz_litho_df.iterrows():
            along_line, perp_dist = point_to_line_distance_and_position(row['x'], row['y'])
            
            if (abs(perp_dist) <= section_width/2 and 
                0 <= along_line <= line_length):
                
                row_dict = row.to_dict()
                row_dict['section_distance'] = along_line
                row_dict['perp_distance'] = perp_dist
                litho_section_data.append(row_dict)
        
        if litho_section_data:
            litho_section_df = pd.DataFrame(litho_section_data)
            
            unique_lithos = litho_section_df['LITHO'].unique()
            colour_palette = px.colors.qualitative.Set1
            litho_colours = {litho: colour_palette[i % len(colour_palette)] for i, litho in enumerate(unique_lithos)}
            
            legend_added = set()
            
            for hole_id in litho_section_df['HOLE_ID'].unique():
                hole_litho_data = litho_section_df[litho_section_df['HOLE_ID'] == hole_id].sort_values('FROM')
                
                for _, interval in hole_litho_data.iterrows():
                    litho_code = interval['LITHO']
                    litho_desc = litho_dict.get(litho_code, "") if litho_dict else ""
                    legend_name = f"{litho_code} {litho_desc}".strip() if litho_dict else litho_code
                    
                    show_legend = legend_name not in legend_added
                    if show_legend:
                        legend_added.add(legend_name)
                    
                    azimuth_rad = np.radians(90 - interval['AZIMUTH'])
                    dip_rad = np.radians(-interval['DIP'])
                    
                    from_depth = interval['FROM']
                    from_dx = from_depth * np.cos(dip_rad) * np.cos(azimuth_rad)
                    from_dy = from_depth * np.cos(dip_rad) * np.sin(azimuth_rad)
                    from_dz = from_depth * np.sin(dip_rad)
                    from_x = interval['EASTING'] + from_dx
                    from_y = interval['NORTHING'] + from_dy
                    from_z = interval['ELEVATION'] - from_dz
                    
                    to_depth = interval['TO']
                    to_dx = to_depth * np.cos(dip_rad) * np.cos(azimuth_rad)
                    to_dy = to_depth * np.cos(dip_rad) * np.sin(azimuth_rad)
                    to_dz = to_depth * np.sin(dip_rad)
                    to_x = interval['EASTING'] + to_dx
                    to_y = interval['NORTHING'] + to_dy
                    to_z = interval['ELEVATION'] - to_dz
                    
                    from_along, _ = point_to_line_distance_and_position(from_x, from_y)
                    to_along, _ = point_to_line_distance_and_position(to_x, to_y)
                    
                    fig.add_trace(go.Scatter(
                        x=[from_along + litho_offset, to_along + litho_offset],
                        y=[from_z, to_z],
                        mode='lines',
                        line=dict(width=20, color=litho_colours[litho_code]),
                        name=legend_name,
                        legendgroup=f"litho_{legend_name}",
                        showlegend=show_legend,
                        hovertemplate=(
                            f"<b>Hole:</b> {hole_id}<br>" +
                            f"<b>Lithology:</b> {legend_name}<br>" +
                            f"<b>From:</b> {interval['FROM']:.1f}m<br>" +
                            f"<b>To:</b> {interval['TO']:.1f}m<br>" +
                            "<extra></extra>"
                        )
                    ))
    
    for hole_id in section_df['HOLE_ID'].unique():
        hole_data = section_df[section_df['HOLE_ID'] == hole_id].sort_values('FROM')
        
        if hole_id in collar_section_dict:
            collar_point = collar_section_dict[hole_id]
            trace_points = [(collar_point['section_distance'], collar_point['elevation'])]
            for _, row in hole_data.iterrows():
                trace_points.append((row['section_distance'], row['z']))
            trace_x, trace_y = zip(*trace_points)
            
            fig.add_trace(go.Scatter(
                x=trace_x,
                y=trace_y,
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip',
                name=f'Trace {hole_id}'
            ))
        else:
            if len(hole_data) > 1:
                fig.add_trace(go.Scatter(
                    x=hole_data['section_distance'],
                    y=hole_data['z'],
                    mode='lines',
                    line=dict(color='black', width=2),
                    showlegend=False,
                    hoverinfo='skip',
                    name=f'Trace {hole_id}'
                ))
    
    if collar_section_data:
        collar_section_df = pd.DataFrame(collar_section_data)
        fig.add_trace(go.Scatter(
            x=collar_section_df['section_distance'],
            y=collar_section_df['elevation'],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='triangle-up'),
            text=collar_section_df['HOLE_ID'],
            textposition='top center',
            textfont=dict(size=10, color='red'),
            name='Collars',
            hovertemplate="<b>Hole:</b> %{customdata}<br><b>Collar</b><extra></extra>",
            customdata=collar_section_df['HOLE_ID']
        ))
    
    section_azimuth = np.degrees(np.arctan2(line_dx, line_dy))
    if section_azimuth < 0:
        section_azimuth += 360
    
    fig.update_layout(
        title=f"Drill Fence Cross Section - Azimuth: {section_azimuth:.1f}° - Length: {line_length:.1f}m - Width: {section_width:.1f}m",
        xaxis_title="Distance along section (m)",
        yaxis_title="Elevation (m)",
        height=900,
        width=1600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.15,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        margin=dict(r=300, l=50, t=80, b=50)
    )
    
    return fig

def create_section_line_map(merged_df, collar_df, section_line_start, section_line_end, section_width):
    """Create a map showing the section line and data points."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged_df['x'],
        y=merged_df['y'],
        mode='markers',
        marker=dict(size=4, color='lightblue', opacity=0.6),
        name='All Samples',
        hovertemplate="<b>Hole:</b> %{customdata}<extra></extra>",
        customdata=merged_df['HOLE_ID']
    ))
    
    fig.add_trace(go.Scatter(
        x=collar_df['EASTING'],
        y=collar_df['NORTHING'],
        mode='markers+text',  
        marker=dict(size=8, color='red', symbol='circle'),  
        text=collar_df['HOLE_ID'],
        textposition='top center',  
        textfont=dict(size=10, color='red'),  
        name='Collars',
        hovertemplate="<b>Hole:</b> %{customdata}<extra></extra>",
        customdata=collar_df['HOLE_ID']
    ))
    
    fig.add_trace(go.Scatter(
        x=[section_line_start[0], section_line_end[0]],
        y=[section_line_start[1], section_line_end[1]],
        mode='lines',
        line=dict(color='red', width=4),
        name='Section Line'
    ))
    
    start_x, start_y = section_line_start
    end_x, end_y = section_line_end
    line_dx = end_x - start_x
    line_dy = end_y - start_y
    line_length = np.sqrt(line_dx**2 + line_dy**2)
    
    if line_length > 0:
        perp_unit_x = -line_dy / line_length
        perp_unit_y = line_dx / line_length
        
        boundary1_start = [start_x + perp_unit_x * section_width/2, start_y + perp_unit_y * section_width/2]
        boundary1_end = [end_x + perp_unit_x * section_width/2, end_y + perp_unit_y * section_width/2]
        boundary2_start = [start_x - perp_unit_x * section_width/2, start_y - perp_unit_y * section_width/2]
        boundary2_end = [end_x - perp_unit_x * section_width/2, end_y - perp_unit_y * section_width/2]
        
        for i, (b_start, b_end) in enumerate([(boundary1_start, boundary1_end), (boundary2_start, boundary2_end)]):
            fig.add_trace(go.Scatter(
                x=[b_start[0], b_end[0]],
                y=[b_start[1], b_end[1]],
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                name='Section Width' if i == 0 else None,
                showlegend=i == 0
            ))
    
    fig.update_layout(
        title="Section Line Location",
        xaxis_title="Easting",
        yaxis_title="Northing",
        height=600,
        width=800,
        xaxis=dict(scaleanchor="y", scaleratio=1)
    )
    
    return fig

def show_statistical_analysis(merged_df, primary_element, use_log_scale):
    """Show basic stats and histogram."""
    st.header("Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Summary Statistics")
        stats_dict = {
            'Statistic': [
                'Count', 'Mean', 'Median', 'Std Dev', 'CV', 
                'Min', 'Q1', 'Q3', 'Max', 'Skewness', 'Kurtosis'
            ],
            'Value': [
                len(merged_df[primary_element]),
                merged_df[primary_element].mean(),
                merged_df[primary_element].median(),
                merged_df[primary_element].std(),
                merged_df[primary_element].std() / merged_df[primary_element].mean(),
                merged_df[primary_element].min(),
                merged_df[primary_element].quantile(0.25),
                merged_df[primary_element].quantile(0.75),
                merged_df[primary_element].max(),
                merged_df[primary_element].skew(),
                merged_df[primary_element].kurtosis()
            ]
        }
        stats_df = pd.DataFrame(stats_dict)
        stats_df['Value'] = stats_df['Value'].round(2)
        st.dataframe(stats_df.set_index('Statistic'), width=400, height=420)
    with col2:
        st.subheader("Histogram")
        if use_log_scale and merged_df[primary_element].min() > 0:
            log_data = np.log10(merged_df[primary_element])
            fig = px.histogram(log_data, nbins=30, title=f"{primary_element} Distribution (log scale)")
            min_val = merged_df[primary_element].min()
            q1_val = merged_df[primary_element].quantile(0.25)
            median_val = merged_df[primary_element].median()
            q3_val = merged_df[primary_element].quantile(0.75)
            max_val = merged_df[primary_element].max()
            tick_vals = [np.log10(val) for val in [min_val, q1_val, median_val, q3_val, max_val]]
            tick_text = [f'{val:,.2f}' for val in [min_val, q1_val, median_val, q3_val, max_val]]
            fig.update_xaxes(
                title_text=primary_element,
                tickmode='array',
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[np.log10(min_val), np.log10(max_val)]
            )
        else:
            fig = px.histogram(merged_df[primary_element], nbins=30, title=f"{primary_element} Distribution")
            min_val = merged_df[primary_element].min()
            q1_val = merged_df[primary_element].quantile(0.25)
            median_val = merged_df[primary_element].median()
            q3_val = merged_df[primary_element].quantile(0.75)
            max_val = merged_df[primary_element].max()
            fig.update_xaxes(
                title_text=primary_element,
                tickmode='array',
                tickvals=[min_val, q1_val, median_val, q3_val, max_val],
                ticktext=[f'{v:.2f}' for v in [min_val, q1_val, median_val, q3_val, max_val]]
            )
        st.plotly_chart(fig, use_container_width=True)

def create_cluster_visualisation(merged_df, viz_df, collar_df, primary_element, use_log_scale, vertical_exaggeration=1.0):
    """Create 3D visualisation of clusters with the same options as in the visuals tab."""
    fig = go.Figure()
    if 'Cluster' in viz_df.columns:
        for cluster in sorted(viz_df['Cluster'].unique()):
            if cluster >= 0:
                cluster_data = viz_df[viz_df['Cluster'] == cluster]
                if not cluster_data.empty:
                    hover_text = []
                    for _, row in cluster_data.iterrows():
                        info = [
                            f"<b>Hole ID:</b> {row['HOLE_ID']}",
                            f"<b>Cluster:</b> {cluster}"
                        ]
                        if primary_element:
                            info.append(f"<b>{primary_element}:</b> {row[primary_element]:.2f}")
                        info.append(f"<b>From:</b> {row['FROM']:.2f}")
                        info.append(f"<b>To:</b> {row['TO']:.2f}")
                        if 'LITHO' in row:
                            info.append(f"<b>Lithology:</b> {row['LITHO']}")
                        hover_text.append("<br>".join(info))
                    fig.add_trace(go.Scatter3d(
                        x=cluster_data['x'],
                        y=cluster_data['y'],
                        z=cluster_data['z'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=px.colors.qualitative.Set1[cluster % len(px.colors.qualitative.Set1)]
                        ),
                        name=f'Cluster {cluster}',
                        hovertemplate="%{text}<br>" +
                                    "<b>X:</b> %{x:.2f}<br>" +
                                    "<b>Y:</b> %{y:.2f}<br>" +
                                    "<b>Z:</b> %{z:.2f}<extra></extra>",
                        text=hover_text
                    ))
        for hole in viz_df['HOLE_ID'].unique():
            hole_data = viz_df[viz_df['HOLE_ID'] == hole]
            collar_point = collar_df[collar_df['HOLE_ID'] == hole].iloc[0]
            x_line = [collar_point['EASTING']] + hole_data['x'].tolist()
            y_line = [collar_point['NORTHING']] + hole_data['y'].tolist()
            z_line = [collar_point['ELEVATION']] + hole_data['z'].tolist()
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))
        add_collar_points(fig, collar_df)
    update_figure_layout(fig, vertical_exaggeration)
    return fig