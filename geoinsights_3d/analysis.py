import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from scipy.interpolate import Rbf
from skimage.measure import marching_cubes
import tensorflow as tf
from tensorflow import keras

def apply_filters(df, selected_holes, selected_lithos, selected_clusters=None, primary_element=None, min_cutoff=None, max_cutoff=None):
    """
    Filters the dataframe based on user selections for holes, lithologies, clusters, and grade range.
    """
    filtered_df = df.copy()
    if selected_holes:
        filtered_df = filtered_df[filtered_df['HOLE_ID'].isin(selected_holes)]
    if selected_lithos and 'LITHO' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['LITHO'].isin(selected_lithos)]
    
    if selected_clusters and 'Cluster' in filtered_df.columns:
        selected_clusters_int = [int(c) for c in selected_clusters]
        filtered_df = filtered_df[filtered_df['Cluster'].isin(selected_clusters_int)]

    if primary_element and min_cutoff is not None and max_cutoff is not None and primary_element in filtered_df.columns:
        numeric_mask = pd.to_numeric(filtered_df[primary_element], errors='coerce').notna()
        filtered_df = filtered_df[numeric_mask]
        filtered_df = filtered_df[
            (filtered_df[primary_element] >= min_cutoff) &
            (filtered_df[primary_element] <= max_cutoff)
        ]
    return filtered_df

def create_swath_data(df, coord_col, value_col, num_bins=3):
    """
    Create swath data by binning the dataframe along a specified coordinate column.
    """
    df = df.sort_values(coord_col)
    bins = np.linspace(df[coord_col].min(), df[coord_col].max(), num_bins + 1)
    df['bin'] = pd.cut(df[coord_col], bins)
    swath_stats = df.groupby('bin')[value_col].agg(['mean', 'count', 'std']).reset_index()
    swath_stats['bin_centre'] = [(x.left + x.right) / 2 for x in swath_stats['bin']]
    swath_stats['bin_width'] = [x.right - x.left for x in swath_stats['bin']]
    swath_stats = swath_stats[swath_stats['count'] > 0]
    return swath_stats

def perform_clustering(df, features, max_clusters=10, use_log_transform=False):
    """Run k-means clustering analysis and return scaled data and WCSS."""
    X = df[features]
    if use_log_transform:
        X = np.log(X + 1e-10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    return X_scaled, scaler, wcss

def perform_pca_analysis(X, n_components, use_log_transform=False):
    """Run PCA with optional log transform."""
    if use_log_transform:
        X = np.log(X + 1e-10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, scaler

def calculate_significant_intervals(df, element, cutoff, min_length, max_internal_waste, litho_dict=None):
    """Find significant intervals based on cutoff grade."""
    results = []
    for hole_id, hole_data in df.groupby('HOLE_ID'):
        hole_data = hole_data.sort_values('FROM')
        current_interval = {
            'start_depth': None, 'end_depth': None, 'grades': [], 'lengths': [], 
            'lithos': [], 'waste_lengths': [], 'last_significant_to': None
        }
        for idx, row in hole_data.iterrows():
            try:
                interval_length = float(row['TO']) - float(row['FROM'])
                current_grade = float(row[element])
                if interval_length <= 0 or pd.isna(current_grade):
                    continue
                if current_grade >= cutoff:
                    if (current_interval['start_depth'] is None or
                       (current_interval['last_significant_to'] is not None and 
                        row['FROM'] - current_interval['last_significant_to'] <= max_internal_waste)):
                        if current_interval['start_depth'] is None:
                            current_interval['start_depth'] = row['FROM']
                        if (current_interval['last_significant_to'] is not None and 
                            row['FROM'] > current_interval['last_significant_to']):
                            waste_length = row['FROM'] - current_interval['last_significant_to']
                            current_interval['waste_lengths'].append(waste_length)
                        current_interval['grades'].append(current_grade)
                        current_interval['lengths'].append(interval_length)
                        if 'LITHO' in row:
                            current_interval['lithos'].append(row['LITHO'])
                        current_interval['end_depth'] = row['TO']
                        current_interval['last_significant_to'] = row['TO']
                    else:
                        if current_interval['start_depth'] is not None:
                            total_length = sum(current_interval['lengths'])
                            total_waste = sum(current_interval['waste_lengths'])
                            if total_length >= min_length:
                                weighted_grade = np.average(current_interval['grades'], weights=current_interval['lengths'])
                                interval_dict = {
                                    'HOLE_ID': hole_id, 'FROM': current_interval['start_depth'], 'TO': current_interval['end_depth'],
                                    'LENGTH': total_length + total_waste, f'{element}_GRADE': weighted_grade,
                                    'INTERNAL_WASTE': total_waste, 'LITHOLOGY': ' / '.join(set(current_interval['lithos']))
                                }
                                if litho_dict is not None and current_interval['lithos']:
                                    descriptions = [desc for lith in set(current_interval['lithos']) if (desc := litho_dict.get(lith))]
                                    if descriptions:
                                        interval_dict['DESCRIPTION'] = ' / '.join(descriptions)
                                results.append(interval_dict)
                        current_interval = {
                            'start_depth': row['FROM'], 'end_depth': row['TO'], 'grades': [current_grade],
                            'lengths': [interval_length], 'lithos': [row['LITHO']] if ('LITHO' in row and pd.notna(row['LITHO'])) else [],
                            'waste_lengths': [], 'last_significant_to': row['TO']
                        }
                elif current_interval['start_depth'] is not None:
                    if row['FROM'] - current_interval['last_significant_to'] > max_internal_waste:
                        total_length = sum(current_interval['lengths'])
                        total_waste = sum(current_interval['waste_lengths'])
                        if total_length >= min_length:
                            weighted_grade = np.average(current_interval['grades'], weights=current_interval['lengths'])
                            interval_dict = {
                                'HOLE_ID': hole_id, 'FROM': current_interval['start_depth'], 'TO': current_interval['end_depth'],
                                'LENGTH': total_length + total_waste, f'{element}_GRADE': weighted_grade,
                                'INTERNAL_WASTE': total_waste, 'LITHOLOGY': ' / '.join(set(current_interval['lithos']))
                            }
                            if litho_dict is not None and current_interval['lithos']:
                                descriptions = [desc for lith in set(current_interval['lithos']) if (desc := litho_dict.get(lith))]
                                if descriptions:
                                    interval_dict['DESCRIPTION'] = ' / '.join(descriptions)
                            results.append(interval_dict)
                        current_interval = {'start_depth': None, 'end_depth': None, 'grades': [], 'lengths': [], 'lithos': [], 'waste_lengths': [], 'last_significant_to': None}
            except (ValueError, TypeError):
                continue
        if current_interval['start_depth'] is not None:
            total_length = sum(current_interval['lengths'])
            total_waste = sum(current_interval['waste_lengths'])
            if total_length >= min_length:
                weighted_grade = np.average(current_interval['grades'], weights=current_interval['lengths'])
                interval_dict = {
                    'HOLE_ID': hole_id, 'FROM': current_interval['start_depth'], 'TO': current_interval['end_depth'],
                    'LENGTH': total_length + total_waste, f'{element}_GRADE': weighted_grade,
                    'INTERNAL_WASTE': total_waste, 'LITHOLOGY': ' / '.join(set(current_interval['lithos']))
                }
                if litho_dict is not None and current_interval['lithos']:
                    descriptions = [desc for lith in set(current_interval['lithos']) if (desc := litho_dict.get(lith))]
                    if descriptions:
                        interval_dict['DESCRIPTION'] = ' / '.join(descriptions)
                results.append(interval_dict)
    return pd.DataFrame(results)

def build_and_train_autoencoder(data, bottleneck_size=2, epochs=50):
    """
    Builds, compiles, and trains an autoencoder. Returns the model, scaler, and training history.
    """
    if data.shape[1] <= bottleneck_size:
        st.error(f"Bottleneck size ({bottleneck_size}) must be smaller than the number of features ({data.shape[1]}). Please adjust the bottleneck size.")
        return None, None, None

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    n_features = data.shape[1]

    encoder = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(n_features * 2, activation='relu'),
        keras.layers.Dense(n_features, activation='relu'),
        keras.layers.Dense(bottleneck_size, activation='relu')
    ], name="encoder")

    decoder = keras.Sequential([
        keras.layers.Input(shape=(bottleneck_size,)),
        keras.layers.Dense(n_features, activation='relu'),
        keras.layers.Dense(n_features * 2, activation='relu'),
        keras.layers.Dense(n_features, activation='sigmoid')
    ], name="decoder")

    autoencoder = keras.Sequential([encoder, decoder], name="autoencoder")
    autoencoder.compile(optimizer='adam', loss='mse')

    with st.spinner(f"Training autoencoder for up to {epochs} epochs..."):
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
        history = autoencoder.fit(
            data_scaled, data_scaled, epochs=epochs, batch_size=32,
            validation_split=0.1, verbose=0, callbacks=[early_stopping]
        )

    return autoencoder, scaler, history

def calculate_reconstruction_error(model, data, scaler):
    """Calculates the mean squared error for each sample's reconstruction."""
    data_scaled = scaler.transform(data)
    reconstructed_data = model.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructed_data, 2), axis=1)
    return mse

def composite_for_shell(df, element_col, composite_length):
    """Composites drillhole data to a fixed length, including coordinates."""
    df = df.sort_values(by=['HOLE_ID', 'FROM']).copy()
    all_composites = []
    for dhid, hole_data in df.groupby('HOLE_ID'):
        start_depth = hole_data['FROM'].min()
        end_depth_hole = hole_data['TO'].max()
        while start_depth < end_depth_hole:
            end_depth_comp = start_depth + composite_length
            mask = (hole_data['FROM'] < end_depth_comp) & (hole_data['TO'] > start_depth)
            interval_samples = hole_data[mask]
            if not interval_samples.empty:
                comp_from = np.maximum(interval_samples['FROM'], start_depth)
                comp_to = np.minimum(interval_samples['TO'], end_depth_comp)
                comp_len = comp_to - comp_from
                total_len = np.sum(comp_len)
                if total_len > 0.01:
                    weighted_grade = np.sum(comp_len * interval_samples[element_col]) / total_len
                    weighted_x = np.sum(comp_len * interval_samples['x']) / total_len
                    weighted_y = np.sum(comp_len * interval_samples['y']) / total_len
                    weighted_z = np.sum(comp_len * interval_samples['z']) / total_len
                    all_composites.append({
                        'HOLE_ID': dhid, 'x': weighted_x, 'y': weighted_y,
                        'z': weighted_z, element_col: weighted_grade
                    })
            start_depth = end_depth_comp
    return pd.DataFrame(all_composites)

class GradeShellGenerator:
    def __init__(self, element, use_log_transform, grid_resolution=50, model_type='anisotropic'):
        self.element = element
        self.use_log_transform = use_log_transform
        self.grid_resolution = grid_resolution
        self.raw_data = None
        self.model_type = model_type.lower()
        log_status = "Log-Transformed" if self.use_log_transform else "Standard"
        method_map = {'isotropic': 'Isotropic RBF', 'anisotropic': 'Anisotropic RBF (Rotation + Scaling)'}
        self.method_name = method_map.get(self.model_type, 'Unknown Method')
        st.info(f"Initialised model with method: {self.method_name} ({log_status})")

    def create_grade_grid(self, modeling_data, full_data_bounds):
        st.write(f"--- Starting grade grid creation with method: {self.method_name} ---")
        st.write(f"Original modeling points: {len(modeling_data)}")
        modeling_data = modeling_data.drop_duplicates(subset=['x', 'y', 'z'], keep='first').copy()
        st.write(f"Points after removing duplicates: {len(modeling_data)}")

        if len(modeling_data) > 8000:
            st.warning(f"Large dataset ({len(modeling_data)} points). Subsampling to 8000 for RBF performance.")
            modeling_data = modeling_data.sample(8000, random_state=42)

        points = modeling_data[['x', 'y', 'z']].values
        values = modeling_data[self.element].values
        
        if self.use_log_transform:
            st.write("Applying log transform to grade values.")
            values = np.log1p(values)

        if self.model_type == 'anisotropic':
            st.write("Determining anisotropy using PCA on high-grade data...")
            high_grade_cutoff = modeling_data[self.element].quantile(0.75)
            high_grade_data = modeling_data[modeling_data[self.element] > high_grade_cutoff]
            if len(high_grade_data) < 3:
                st.warning("Not enough high-grade data for PCA. Reverting to isotropic.")
                mean_coord, rotation_matrix, scaling_factors = np.mean(points, axis=0), np.identity(3), np.ones(3)
            else:
                pca = PCA(n_components=3)
                mean_coord = np.mean(high_grade_data[['x', 'y', 'z']].values, axis=0)
                pca.fit(high_grade_data[['x', 'y', 'z']].values - mean_coord)
                rotation_matrix = pca.components_
                explained_variance = pca.explained_variance_ratio_
                epsilon = 1e-6
                scaling_factors = np.sqrt(explained_variance[0] / (explained_variance + epsilon))
                st.write(f"Anisotropy Ratios (1:Sec:Tert): 1.0 : {1/scaling_factors[1]:.2f} : {1/scaling_factors[2]:.2f}")
        elif self.model_type == 'isotropic':
            st.write("Isotropic method selected. No rotation or scaling will be applied.")
            mean_coord, rotation_matrix, scaling_factors = np.mean(points, axis=0), np.identity(3), np.ones(3)
        else:
            raise ValueError(f"Invalid model_type: '{self.model_type}'. Choose 'isotropic' or 'anisotropic'.")

        points_transformed = ((points - mean_coord) @ rotation_matrix.T) * scaling_factors

        min_coords = full_data_bounds[['x', 'y', 'z']].min()
        max_coords = full_data_bounds[['x', 'y', 'z']].max()
        ranges = max_coords - min_coords
        padding = ranges * 0.20
        expanded_min = min_coords - padding
        expanded_max = max_coords + padding
        
        grid_x_coords, grid_y_coords, grid_z_coords = np.mgrid[
            expanded_min['x']:expanded_max['x']:complex(0, self.grid_resolution), 
            expanded_min['y']:expanded_max['y']:complex(0, self.grid_resolution), 
            expanded_min['z']:expanded_max['z']:complex(0, self.grid_resolution)
        ]
        
        grid_points_flat = np.vstack([grid_x_coords.ravel(), grid_y_coords.ravel(), grid_z_coords.ravel()]).T
        grid_transformed_flat = ((grid_points_flat - mean_coord) @ rotation_matrix.T) * scaling_factors

        st.write("Setting up and executing RBF interpolator... (this may take a while)")
        rbf_interpolator = Rbf(
            points_transformed[:, 0], points_transformed[:, 1], points_transformed[:, 2], values, 
            function='thin_plate', smooth=0.01
        )
        predicted_values_flat = rbf_interpolator(
            grid_transformed_flat[:, 0], grid_transformed_flat[:, 1], grid_transformed_flat[:, 2]
        )
        predicted_grades = predicted_values_flat.reshape(grid_x_coords.shape)

        if self.use_log_transform:
            st.write("Back-transforming log-space predictions.")
            predicted_grades = np.expm1(predicted_grades)

        st.write("Grade grid creation complete.")
        return predicted_grades, grid_x_coords[:,0,0], grid_y_coords[0,:,0], grid_z_coords[0,0,:]

    def visualise(self, predicted_grades, grid_x, grid_y, grid_z, vertical_exaggeration=1.0):
        fig = go.Figure()
        slider_min = self.raw_data[self.element].quantile(0.50)
        slider_max = self.raw_data[self.element].quantile(0.99)
        st.write(f"Setting grade slider range to: [{slider_min:.2f}, {slider_max:.2f}]")
        slider_cutoffs = np.linspace(slider_min, slider_max, 15)

        for i, cutoff in enumerate(slider_cutoffs):
            try:
                verts, faces, _, _ = marching_cubes(
                    volume=predicted_grades, level=cutoff, 
                    spacing=(grid_x[1]-grid_x[0], grid_y[1]-grid_y[0], grid_z[1]-grid_z[0])
                )
                verts += [grid_x[0], grid_y[0], grid_z[0]]
                fig.add_trace(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], opacity=0.3, color='cyan', name=f'Cutoff: {cutoff:.2f}', visible=(i == 0)))
            except (ValueError, RuntimeError):
                st.warning(f"Could not generate a surface for cutoff {cutoff:.2f}. Skipping.")
                fig.add_trace(go.Mesh3d(visible=False))
        
        points = self.raw_data.sample(min(len(self.raw_data), 5000))
        cmax_value = self.raw_data[self.element].quantile(0.98)
        fig.add_trace(go.Scatter3d(x=points['x'], y=points['y'], z=points['z'], mode='markers', text=points[self.element], hovertemplate=(f'<b>{self.element} Grade: %{{text:.2f}}</b><br><br>X: %{{x:.1f}}<br>Y: %{{y:.1f}}<br>Z: %{{z:.1f}}<extra></extra>'), marker=dict(size=6, color=points[self.element], colorscale='Viridis', colorbar=dict(title=f'{self.element} Grade'), showscale=True, cmin=0, cmax=cmax_value), name='Drill Samples'))
        
        grade_steps = [dict(method="update", args=[{"visible": [i == j for j in range(len(slider_cutoffs))] + [True]}], label=f"{cutoff:.2f}") for i, cutoff in enumerate(slider_cutoffs)]
        grade_slider = dict(active=0, currentvalue={"prefix": "Cutoff Grade: "}, pad={"t": 50}, steps=grade_steps, y=0)

        opacity_levels = np.linspace(0.1, 1.0, 10)
        opacity_steps = [dict(method="restyle", args=[{"opacity": op}], label=f"{op:.1f}") for op in opacity_levels]
        opacity_slider = dict(active=2, currentvalue={"prefix": "Opacity: "}, pad={"t": 50}, steps=opacity_steps, y=0.1)

        log_status = "Log-Transformed" if self.use_log_transform else "Standard"
        
        x_min, x_max = grid_x.min(), grid_x.max()
        y_min, y_max = grid_y.min(), grid_y.max()
        z_min, z_max = grid_z.min(), grid_z.max()

        x_diff = x_max - x_min if x_max > x_min else 1
        y_diff = y_max - y_min if y_max > y_min else 1
        z_diff = z_max - z_min if z_max > z_min else 1
        max_diff = max(x_diff, y_diff, z_diff * vertical_exaggeration)
        if max_diff == 0: max_diff = 1

        fig.update_layout(
            title=f"Interactive Grade Shell ({self.method_name}, {log_status}) for {self.element}",
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title='Easting (X)', yaxis_title='Northing (Y)', zaxis_title='Elevation (Z)',
                aspectmode='manual',
                aspectratio=dict(x=x_diff/max_diff, y=y_diff/max_diff, z=(z_diff*vertical_exaggeration)/max_diff),
                xaxis_range=[x_min, x_max], yaxis_range=[y_min, y_max], zaxis_range=[z_min, z_max]
            ),
            sliders=[grade_slider, opacity_slider]
        )
        return fig