import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from streamlit import session_state as state
from PIL import Image
from google import genai
import io
import sys
import os
import matplotlib.pyplot as plt
import base64
from sklearn.ensemble import RandomForestRegressor
import importlib.resources


# Import functions from custom modules
from geoinsights_3d.utils import create_html_download
from geoinsights_3d.data_processing import (
    process_collar_data, process_assay_data, process_litho_data,
    process_litho_dict, process_and_merge_data
)
from geoinsights_3d.plotting import (
    create_swath_plots, show_statistical_analysis, create_lithology_analysis,
    create_combined_3d_visualisation, create_drill_fence_cross_section,
    create_section_line_map, plot_scree, plot_pca_biplot, plot_eigenvectors,
    get_cluster_summary, plot_cluster_boxplots, plot_lithology_cluster_comparison,
    create_cluster_visualisation, update_figure_layout
)
from geoinsights_3d.analysis import (
    perform_clustering, perform_pca_analysis, calculate_significant_intervals, GradeShellGenerator,
    build_and_train_autoencoder, apply_filters, composite_for_shell, create_swath_data
)



def main():


    # =============================================================================
    # INIT SESSION STATE VARIABLES
    # =============================================================================
    if 'X_scaled' not in st.session_state:
        st.session_state.X_scaled = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'wcss' not in st.session_state:
        st.session_state.wcss = None
    if 'n_clusters' not in st.session_state:
        st.session_state.n_clusters = 3
    if 'selected_cluster_features' not in st.session_state:
        st.session_state.selected_cluster_features = None
    if 'apply_filters_globally' not in st.session_state:
        st.session_state.apply_filters_globally = False
    if 'previous_collar_file' not in st.session_state:
        st.session_state.previous_collar_file = None
    if 'previous_assay_file' not in st.session_state:
        st.session_state.previous_assay_file = None
    if 'merged_df' not in st.session_state:
        st.session_state.merged_df = None
    if 'viz_df' not in st.session_state:
        st.session_state.viz_df = None
    if 'viz_litho_df' not in st.session_state:
        st.session_state.viz_litho_df = None
    if 'collar_df' not in st.session_state:
        st.session_state.collar_df = None
    if 'element_cols' not in st.session_state:
        st.session_state.element_cols = []
    if 'litho_dict' not in st.session_state:
        st.session_state.litho_dict = None
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = None
    if 'significant_intervals' not in st.session_state:
        st.session_state.significant_intervals = None
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = None
    if 'viz_litho_df' not in st.session_state:
        st.session_state.viz_litho_df = None
    if 'litho_dict' not in st.session_state:
        st.session_state.litho_dict = {}
    # Add state for persisting generated figures
    if 'grade_shell_fig' not in st.session_state:
        st.session_state.grade_shell_fig = None
    if 'solid_model_fig' not in st.session_state:
        st.session_state.solid_model_fig = None


    # =============================================================================
    # PAGE CONFIGURATION
    # =============================================================================
    st.set_page_config(
        page_title="GeoInsights 3D",
        page_icon="🪨",
        layout="wide"
    )
    st.markdown('<div style="position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: gray;">Version 1.1</div>', unsafe_allow_html=True)

    # =============================================================================
    # HEADER AND LOGO
    # =============================================================================
    try:

        logo_path = importlib.resources.files('geoinsights_3d.assets').joinpath('3DA logo.png')

        with importlib.resources.as_file(logo_path) as actual_filepath:
            logo = Image.open(actual_filepath)
        
        buffered = io.BytesIO()
        logo.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        st.markdown(
            f'<div style="display: flex; justify-content: center; margin-bottom: 20px;"><img src="data:image/png;base64,{img_str}" width="600"></div>',
            unsafe_allow_html=True
        )

    except FileNotFoundError:
        st.warning("Logo file not found. Please check the package installation.")
        st.markdown('<div style="text-align: center; margin-bottom: 20px;">3DA</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load logo due to an error: {e}")
        st.markdown('<div style="text-align: center; margin-bottom: 20px;">3DA</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Exploratory Data Analysis and Visualisation</h1>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; margin-bottom: 20px;'>For a full tutorial on this application, please read the <a href='https://tinyurl.com/geoinsights3d' target='_blank'>instructional blog post</a>.</div>",
        unsafe_allow_html=True
    )
    with st.sidebar:
        st.markdown("<h3>LLM Integration</h3>", unsafe_allow_html=True)

        # First, try to get the key from Streamlit's secrets
        try:
            st.session_state.google_api_key = st.secrets["GOOGLE_API_KEY"]
            st.success("API Key loaded from secrets!")
        except:
            # If not found in secrets, ask the user for it
            google_api_key = st.text_input(
                "Enter your Google API Key",
                type="password",
                help="Enter your Google API Key for LLM functionality",
                key="google_api_key_input"
            )
            if google_api_key:
                st.session_state.google_api_key = google_api_key

        google_model = st.text_input(
            "Enter the Google AI Model", 
            help="Enter the Google AI model you want to use for LLM integration",
            key="google_model_input"
        )
        if google_model:
            st.session_state.google_model = google_model
        
        st.markdown("<hr>", unsafe_allow_html=True)

    # =============================================================================
    # LLM PROMPT GENERATION FUNCTION
    # =============================================================================
    def generate_summary_prompt(user_context=""):
        """Generates the initial summary prompt for the LLM."""
        prompt = "Geochemical Analysis Summary:\n"
        if st.session_state.merged_df is not None:
            df = st.session_state.merged_df
            num_holes = df['HOLE_ID'].nunique()
            num_samples = len(df)
            prompt += f"- Drillholes analysed: {num_holes}, Total samples: {num_samples}.\n"
            
            primary_element = None
            if st.session_state.element_cols:
                primary_element = st.session_state.element_cols[0]
                mean_val = df[primary_element].mean()
                median_val = df[primary_element].median()
                std_val = df[primary_element].std()
                prompt += f"- Primary Element Analysed: {primary_element}.\n"
                prompt += f"  - Overall Stats: Mean = {mean_val:.2f}, Median = {median_val:.2f}, Std Dev = {std_val:.2f}.\n"
                
                try:
                    east_stats = create_swath_data(df, 'x', primary_element, num_bins=5)
                    prompt += "- Easting Swath Statistics (Grade Trend):\n"
                    for entry in east_stats.to_dict('records'):
                        prompt += f"   * Bin centred at {entry['bin_centre']:.1f}: Mean={entry['mean']:.2f} ({entry['count']} samples)\n"
                except Exception as e:
                    prompt += "- (Swath plot statistics calculation failed)\n"

            if 'LITHO' in df.columns:
                prompt += "\n- Lithology Analysis:\n"
                litho_counts = df['LITHO'].value_counts()
                top_n_lithos = 3
                
                prompt += "  - Dominant Lithologies Encountered:\n"
                for i, (litho_code, count) in enumerate(litho_counts.head(top_n_lithos).items()):
                    litho_desc = ""
                    if st.session_state.litho_dict and litho_code in st.session_state.litho_dict:
                        litho_desc = f" ({st.session_state.litho_dict[litho_code]})"
                    percentage = (count / num_samples) * 100
                    prompt += f"    * {litho_code}{litho_desc}: {count} samples ({percentage:.1f}%)\n"
                if len(litho_counts) > top_n_lithos:
                    prompt += f"    * (Plus {len(litho_counts) - top_n_lithos} other lithologies)\n"

                if primary_element and primary_element in df.columns:
                    try:
                        litho_grade_stats = df.groupby('LITHO')[primary_element].agg(['median', 'count']).reset_index()
                        min_samples_for_stat = 5
                        reliable_litho_stats = litho_grade_stats[litho_grade_stats['count'] >= min_samples_for_stat]
                        
                        if not reliable_litho_stats.empty:
                            reliable_litho_stats = reliable_litho_stats.sort_values('median', ascending=False)
                            
                            highest_grade_lithos = reliable_litho_stats.head(2)
                            lowest_grade_lithos = reliable_litho_stats.tail(2)
                            
                            prompt += f"  - Grade ({primary_element}) Relationship:\n"
                            prompt += f"    * Highest Median Grades often in: "
                            hg_list = []
                            for _, row in highest_grade_lithos.iterrows():
                                desc = f" ({st.session_state.litho_dict.get(row['LITHO'], '')})" if st.session_state.litho_dict else ""
                                hg_list.append(f"{row['LITHO']}{desc} (Median: {row['median']:.2f})")
                            prompt += ", ".join(hg_list) + "\n"
                            
                            prompt += f"    * Lowest Median Grades often in: "
                            lg_list = []
                            for _, row in lowest_grade_lithos.iterrows():
                                desc = f" ({st.session_state.litho_dict.get(row['LITHO'], '')})" if st.session_state.litho_dict else ""
                                lg_list.append(f"{row['LITHO']}{desc} (Median: {row['median']:.2f})")
                            prompt += ", ".join(lg_list) + "\n"
                        else:
                            prompt += f"  - (Not enough samples per lithology to reliably determine grade relationships for {primary_element})\n"
                    except Exception as e:
                        prompt += f"  - (Error calculating grade/lithology relationship for {primary_element})\n"

            if "Cluster" in df.columns:
                clusters = sorted(df["Cluster"].unique())
                num_clusters = len(clusters)
                prompt += f"\n- Cluster Analysis ({num_clusters} Clusters Identified):\n"
                
                prompt += "  - Geochemical Differences:\n"
                for cluster in clusters:
                    cluster_df = df[df["Cluster"] == cluster]
                    prompt += f"    * Cluster {cluster} ({len(cluster_df)} samples):\n"
                    elements_to_summarise = st.session_state.element_cols
                    stats_list = []
                    for element in elements_to_summarise:
                        if element in cluster_df.columns:
                            med_e = cluster_df[element].median()
                            stats_list.append(f"{element} median={med_e:.2f}")
                    prompt += f"        - Key Elements: {'; '.join(stats_list)}\n"

                if 'LITHO' in df.columns:
                    prompt += "  - Cluster-Lithology Association:\n"
                    for cluster in clusters:
                        cluster_df = df[df["Cluster"] == cluster]
                        if not cluster_df.empty and 'LITHO' in cluster_df.columns:
                            litho_counts_cluster = cluster_df['LITHO'].value_counts()
                            if not litho_counts_cluster.empty:
                                top_litho_code = litho_counts_cluster.index[0]
                                top_litho_count = litho_counts_cluster.iloc[0]
                                percentage = (top_litho_count / len(cluster_df)) * 100
                                litho_desc = ""
                                if st.session_state.litho_dict and top_litho_code in st.session_state.litho_dict:
                                    litho_desc = f" ({st.session_state.litho_dict[top_litho_code]})"
                                prompt += f"    * Cluster {cluster}: Dominated by {top_litho_code}{litho_desc} ({percentage:.1f}%)\n"
                            else:
                                prompt += f"    * Cluster {cluster}: No dominant lithology found.\n"
                        else:
                            prompt += f"    * Cluster {cluster}: Lithology data missing for this cluster.\n"

            if st.session_state.significant_intervals is not None and not st.session_state.significant_intervals.empty:
                sig_intervals_df = st.session_state.significant_intervals
                num_intervals = len(sig_intervals_df)
                prompt += f"\n- Significant Intervals ({primary_element} > Cutoff):\n"
                prompt += f"  - {num_intervals} significant intervals detected.\n"
                if 'LITHOLOGY' in sig_intervals_df.columns and sig_intervals_df['LITHOLOGY'].notna().any():
                    all_lithos_in_intervals = set()
                    for lith_string in sig_intervals_df['LITHOLOGY'].dropna():
                        codes = [code.strip() for code in lith_string.split('/')]
                        all_lithos_in_intervals.update(codes)
                    
                    if all_lithos_in_intervals:
                        litho_list_str = []
                        for code in sorted(list(all_lithos_in_intervals))[:5]:
                            desc = f" ({st.session_state.litho_dict.get(code, '')})" if st.session_state.litho_dict else ""
                            litho_list_str.append(f"{code}{desc}")
                        prompt += f"  - Primarily hosted within: {', '.join(litho_list_str)}"
                        if len(all_lithos_in_intervals) > 5:
                            prompt += " (and others)."
                        prompt += "\n"

        else:
            prompt += "No processed data available for analysis.\n"
            
        if user_context.strip():
            prompt += "\nAdditional Geological Context Provided by User:\n" + user_context.strip() + "\n"
        
        prompt += "\nInstructions for LLM:\n"
        prompt += "Based on the summary statistics, spatial trends (swath plots), cluster analysis, and available lithological information provided above, please provide a concise yet detailed geological interpretation. Focus on:\n"
        prompt += "1. Key geochemical characteristics and element associations.\n"
        prompt += "2. Interpretation of the geochemical clusters: What might they represent in terms of geological processes, alteration, or rock types? Consider their distinct geochemical signatures and lithological associations (if provided).\n"
        prompt += "3. Spatial distribution patterns of grades and clusters.\n"
        prompt += "4. Significance of the high-grade intervals and their geological context (lithology, location).\n"
        prompt += "5. Integrate the user-provided context (if any) into your interpretation.\n"
        prompt += "Aim for a geologist-to-geologist level summary, highlighting potential implications for mineral exploration or geological understanding."
        
        return prompt

    # =============================================================================
    # MAIN APP: TABS
    # =============================================================================
    tab_data, tab_viz, tab_gradeshell, tab_solid_model, tab_stats, tab_clustering, tab_ml_explain, tab_anomaly, tab_llm, tab_qa, tab_download = st.tabs([
        "📁 Data Loading", "📏 3D Visualisations and Cross-Sections", "🩸 Grade Shell", "🧊 3D Solid Model", "📈 Statistics", "⚇ Clustering", "🏷️ ML Explain", "🔎 Anomaly Detection", "🤖 AI GEO", "📋 Data Analysis Playground",  "💾 Export Data"
    ])

    # =============================================================================
    # DATA LOADING TAB
    # =============================================================================
    with tab_data:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>📁 Data Loading</h2>", unsafe_allow_html=True)
        st.markdown("Upload your data files to begin analysis or load demo data to quickly explore the app's features.")

        if "demo_files_loaded" not in st.session_state:
            st.session_state.demo_files_loaded = False

        if not st.session_state.demo_files_loaded:
            if st.button("Load Demo Data"):
                try:
                    data_files_path = importlib.resources.files('geoinsights_3d.demo_data')

                    def load_demo_file(filename):
                        file_path = data_files_path.joinpath(filename)
                        file_bytes = file_path.read_bytes() # Read file content directly
                        bytes_io_obj = io.BytesIO(file_bytes)
                        bytes_io_obj.name = filename # Mimic the FileUploader object
                        return bytes_io_obj

                    # Load each demo file using the packaged path
                    st.session_state.demo_collar_file = load_demo_file("Drill_hole_location.csv")
                    st.session_state.demo_assay_file = load_demo_file("Drill_hole_geochemistry.csv")
                    st.session_state.demo_litho_file = load_demo_file("Drill_hole_lithology.csv")
                    st.session_state.demo_litho_dict_file = load_demo_file("Drill_hole_lithology_dictionary.csv")
                    
                    st.session_state.demo_files_loaded = True
                    st.success("Demo data loaded successfully!")
                    st.rerun()
                
                except FileNotFoundError:
                    st.error("Demo data files not found. The package might be corrupted or installed incorrectly.")
                except Exception as e:
                    st.error(f"An unexpected error occurred while loading demo data: {e}")


        if st.session_state.get("demo_files_loaded", False):
            collar_file = st.session_state.demo_collar_file
            assay_file = st.session_state.demo_assay_file
            litho_file = st.session_state.demo_litho_file
            litho_dict_file = st.session_state.demo_litho_dict_file
        else:
            collar_file = st.file_uploader("Upload Collar File (CSV)", type=["csv"], key="collar_uploader")
            assay_file = st.file_uploader("Upload Assay File (CSV)", type=["csv"], key="assay_uploader")
            litho_file = st.file_uploader("Upload Lithology File (CSV)", type=["csv"], key="litho_uploader")
            litho_dict_file = st.file_uploader("Upload Lithology Dictionary File (CSV)", type=["csv"], key="litho_dict_uploader")
        
        file_format = st.radio(
            "Select CSV File Format",
            ("Standard CSV (Headers in row 1)", "Geological Survey Format (Headers in H1000)"),
            index=1
        )

        files_changed = (
            (collar_file is not None and collar_file != st.session_state.get("previous_collar_file")) or
            (assay_file is not None and assay_file != st.session_state.get("previous_assay_file"))
        )

        if files_changed or st.session_state.get("merged_df") is None:
            # Clear all session state variables related to analysis and plots
            for key in ['X_scaled', 'scaler', 'wcss', 'merged_df', 'viz_df', 'viz_litho_df', 
                        'significant_intervals', 'anomaly_analysis_complete', 'autoencoder_model', 
                        'anomaly_scaler', 'reconstructed_df', 'clustering_completed', 
                        'grade_shell_fig', 'solid_model_fig']:
                if key in st.session_state:
                    st.session_state[key] = None
            
            st.session_state.previous_collar_file = collar_file
            st.session_state.previous_assay_file = assay_file

            valid_data_combinations = False
            if collar_file:
                if assay_file and not litho_file:
                    valid_data_combinations = True
                    st.session_state.analysis_mode = "collar_assay"
                elif litho_file and not assay_file:
                    valid_data_combinations = True
                    st.session_state.analysis_mode = "collar_litho"
                elif assay_file and litho_file:
                    valid_data_combinations = True
                    st.session_state.analysis_mode = "all"
            
            if valid_data_combinations:
                with st.spinner("Processing and merging data..."):
                    if hasattr(collar_file, 'seek'): collar_file.seek(0)
                    st.session_state.collar_df = process_collar_data(collar_file, file_format)

                    assay_df, litho_df = None, None
                    st.session_state.element_cols = []
                    composite_enabled = st.sidebar.checkbox("Composite geochemical data", key="data_loading_composite_checkbox")
                    composite_length = 1.0
                    if composite_enabled:
                        composite_length = st.sidebar.slider("Composite Interval (m)", 1.0, 10.0, 2.0, 0.1, key="data_loading_composite_slider")

                    if st.session_state.collar_df is not None:
                        if st.session_state.analysis_mode in ["collar_assay", "all"]:
                            if hasattr(assay_file, 'seek'): assay_file.seek(0)
                            assay_df, st.session_state.element_cols = process_assay_data(assay_file, file_format)
                        
                        if st.session_state.analysis_mode in ["collar_litho", "all"]:
                            if hasattr(litho_file, 'seek'): litho_file.seek(0)
                            litho_df = process_litho_data(litho_file, file_format)
                            if litho_dict_file:
                                if hasattr(litho_dict_file, 'seek'): litho_dict_file.seek(0)
                                st.session_state.litho_dict = process_litho_dict(litho_dict_file, file_format)

                        st.session_state.merged_df, st.session_state.viz_litho_df = process_and_merge_data(
                            st.session_state.collar_df, assay_df, litho_df, st.session_state.element_cols, composite_enabled, composite_length
                        )

                        st.session_state.original_df = st.session_state.merged_df.copy()
                        
                        if st.session_state.merged_df is not None:
                            st.rerun()
                        else:
                            st.error("Failed to process or merge data.")
        
        if st.session_state.get("merged_df") is not None:
            st.success("Data is loaded and ready for analysis.")
            st.write("Preview of Processed Data:")
            st.dataframe(st.session_state.merged_df.head())
        else:
            st.info("Please upload your data files or load the demo data to begin.")
    # =============================================================================
    # 3D VISUALISATIONS AND CROSS-SECTIONS TAB
    # =============================================================================
    with tab_viz:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>📏 3D Visualisation</h2>", unsafe_allow_html=True)
        if st.session_state.merged_df is not None:
            viz_df = st.session_state.merged_df.copy()

            with st.expander("Filter Visualisation Data", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    all_holes = sorted(viz_df['HOLE_ID'].unique())
                    selected_holes = st.multiselect("Filter by Hole ID", options=all_holes, key="viz_hole_filter")

                    selected_lithos = []
                    if 'LITHO' in viz_df.columns:
                        all_lithos = sorted(viz_df['LITHO'].dropna().unique())
                        selected_lithos = st.multiselect("Filter by Lithology", options=all_lithos, key="viz_litho_filter")

                with col2:
                    primary_element = None
                    min_cutoff, max_cutoff = None, None
                    if st.session_state.analysis_mode in ["collar_assay", "all"] and st.session_state.element_cols:
                        primary_element = st.selectbox("Select element for Grade filter", st.session_state.element_cols, key="viz_element_filter")
                        min_val = float(viz_df[primary_element].min())
                        max_val = float(viz_df[primary_element].max())
                        min_cutoff, max_cutoff = st.slider(
                            f"Filter by {primary_element} range", min_val, max_val, (min_val, max_val), key="viz_grade_slider"
                        )
                    
                    use_log_scale = st.checkbox("Use log scale for grade colour", value=True, key="viz_log_scale")

                with col3:
                    selected_clusters = []
                    if 'Cluster' in viz_df.columns:
                        all_clusters = sorted([c for c in viz_df['Cluster'].unique() if c >= 0])
                        if all_clusters:
                            selected_clusters = st.multiselect("Filter by Cluster", options=all_clusters, key="viz_cluster_filter")

            viz_df = apply_filters(
                viz_df, selected_holes, selected_lithos, selected_clusters, primary_element, min_cutoff, max_cutoff
            )
            
            viz_collar_df = st.session_state.collar_df.copy()
            if selected_holes:
                viz_collar_df = viz_collar_df[viz_collar_df['HOLE_ID'].isin(selected_holes)]
            
            viz_litho_df_filtered = st.session_state.viz_litho_df.copy() if st.session_state.viz_litho_df is not None else None
            if viz_litho_df_filtered is not None:
                if selected_lithos:
                    viz_litho_df_filtered = viz_litho_df_filtered[viz_litho_df_filtered['LITHO'].isin(selected_lithos)]
                if selected_holes:
                    viz_litho_df_filtered = viz_litho_df_filtered[viz_litho_df_filtered['HOLE_ID'].isin(selected_holes)]

            vertical_exaggeration = st.slider("Vertical Exaggeration", 1.0, 10.0, 1.0, 0.1, key="viz_exaggeration")

            viz_options = []
            if st.session_state.analysis_mode in ["collar_assay", "all"] and st.session_state.element_cols:
                viz_options.append("Grade")
            if 'LITHO' in st.session_state.merged_df.columns:
                viz_options.append("Lithology")
            if 'Cluster' in st.session_state.merged_df.columns:
                viz_options.append("Clusters")
                
            selected_viz = st.multiselect(
                "Select visualisation types. (If multiple are selected, they will be offset.) Elements can be toggled on and off using the legend",
                viz_options,
                default=viz_options[0] if viz_options else None
            )

            if selected_viz:
                if not viz_df.empty:
                    fig = create_combined_3d_visualisation(
                        df=viz_df,
                        collar_df=viz_collar_df,
                        viz_litho_df=viz_litho_df_filtered,
                        litho_dict=st.session_state.litho_dict,
                        selected_viz=selected_viz,
                        primary_element=primary_element,
                        use_log_scale=use_log_scale,
                        vertical_exaggeration=vertical_exaggeration,
                        legend_title="Combined Plot"
                    )
                    
                    st.plotly_chart(fig)
                    html_string, filename = create_html_download(fig, "3D_visualisation")
                    st.download_button(label="📊 Download 3D Visualisation (HTML)", data=html_string, file_name=filename, mime="text/html", key="download_3d_viz")
                else:
                    st.warning("No data remains after applying filters. Please adjust your filter settings.")
            
            else:
                st.warning("Please select at least one visualisation type.")
                
            st.markdown("---")
            st.markdown("<h3 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>✂️ Drill Fence Cross Sections</h3>", unsafe_allow_html=True)
            
            st.markdown("### Define Drill Fence Section")
            st.write("Select drill holes to create a fence section. Lithology will display on the left side of drill traces, grades on the right side.")
            
            hole_coords = st.session_state.collar_df[['HOLE_ID', 'EASTING', 'NORTHING']].copy()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Select holes to create drill fence:**")
                selected_holes_cross = st.multiselect(
                    "Choose the first and last hole in the fence line (minimum 2 required):",
                    hole_coords['HOLE_ID'].tolist(),
                    default=hole_coords['HOLE_ID'].tolist()[:3] if len(hole_coords) >= 3 else hole_coords['HOLE_ID'].tolist(),
                    key="cross_section_holes"
                )
            
            with col2:
                section_width = st.slider("Section Width (m)", min_value=10.0, max_value=200.0, value=50.0, step=10.0, key="cross_section_width")
                
                primary_element_cross = None
                if st.session_state.analysis_mode in ["collar_assay", "all"] and st.session_state.element_cols:
                    primary_element_cross = st.selectbox("Grade element:", st.session_state.element_cols, key="cross_section_element")
                    use_log_scale_cross = st.checkbox("Use log scale for grades", value=True, key="cross_section_log")
            
            if len(selected_holes_cross) >= 2:
                fence_holes = hole_coords[hole_coords['HOLE_ID'].isin(selected_holes_cross)]
                
                first_hole = fence_holes[fence_holes['HOLE_ID'] == selected_holes_cross[0]].iloc[0]
                last_hole = fence_holes[fence_holes['HOLE_ID'] == selected_holes_cross[-1]].iloc[0]
                
                section_line_start = [first_hole['EASTING'], first_hole['NORTHING']]
                section_line_end = [last_hole['EASTING'], last_hole['NORTHING']]
                
                st.success(f"Drill fence from **{first_hole['HOLE_ID']}** to **{last_hole['HOLE_ID']}** ({len(selected_holes_cross)} holes selected)")
                
                st.subheader("Drill Fence Location")
                map_fig = create_section_line_map(
                    viz_df,
                    st.session_state.collar_df,
                    section_line_start, 
                    section_line_end, 
                    section_width
                )
                
                selected_hole_coords = hole_coords[hole_coords['HOLE_ID'].isin(selected_holes_cross)]
                map_fig.add_trace(go.Scatter(
                    x=selected_hole_coords['EASTING'],
                    y=selected_hole_coords['NORTHING'],
                    mode='markers+text',
                    marker=dict(size=12, color='blue', symbol='triangle-up'),
                    name='Selected Holes',
                    hovertemplate="<b>Selected Hole:</b> %{text}<extra></extra>"
                ))
                
                st.plotly_chart(map_fig, use_container_width=True)
                
                if st.button("Generate Drill Fence Cross Section", type="primary", key="generate_cross_section"):
                    with st.spinner("Generating drill fence cross-section..."):
                        fig_cross = create_drill_fence_cross_section(
                            viz_df,
                            st.session_state.viz_litho_df,
                            st.session_state.collar_df,
                            section_line_start,
                            section_line_end,
                            section_width,
                            primary_element_cross,
                            use_log_scale_cross if 'use_log_scale_cross' in locals() else True,
                            st.session_state.litho_dict
                        )
                        
                        if fig_cross:
                            st.subheader("Drill Fence Cross Section")
                            st.plotly_chart(fig_cross, use_container_width=True)
                            st.info("📍 **Reading the Cross Section:**\n"
                                "- **Black lines** = Drill hole traces\n"
                                "- **Left side (coloured bars)** = Lithology intervals\n" 
                                "- **Right side (coloured dots)** = Grade values\n"
                                "- **Red triangles** = Collar locations with hole IDs")
                        else:
                            st.warning("No data found in the specified drill fence area. Try adjusting the section width or selecting different holes.")
            else:
                st.warning("⚠️ Please select at least 2 holes to create a drill fence section.")

            if st.session_state.analysis_mode in ["collar_assay", "all"] and primary_element:
                if not viz_df.empty:
                    create_swath_plots(viz_df, primary_element, use_log_scale)
                
        else:
            st.warning("Please load data in the Data Loading tab first.")

    # =============================================================================
    # GRADE SHELL TAB
    # =============================================================================
    with tab_gradeshell:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>🩸 Grade Shell Generation</h2>", unsafe_allow_html=True)
        st.markdown("This tool generates a 3D grade shell using a Radial Basis Function (RBF) interpolator. Adjust the parameters and click the button to begin.")

        if st.session_state.merged_df is not None and st.session_state.analysis_mode in ["collar_assay", "all"]:
            st.subheader("Grade Shell Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                element_of_interest = st.selectbox("Select Element for Shell", st.session_state.element_cols, key="shell_element")
                model_type_shell = st.selectbox(
                    "Select Interpolation Method", 
                    ['anisotropic', 'isotropic'], 
                    index=0, 
                    key="shell_model_type",
                    help="Anisotropic uses PCA to model directional grade trends. Isotropic assumes grade influence is equal in all directions."
                )
            with col2:
                use_log_transform_shell = st.checkbox("Use Log Transform", value=True, key="shell_log")
                do_compositing_shell = st.checkbox("Composite Data for Modelling", value=True, key="shell_composite")
                if do_compositing_shell:
                    composite_length_shell = st.number_input("Composite Length (m)", min_value=1.0, value=5.0, step=1.0, key="shell_comp_len")
            with col3:
                grid_resolution_shell = st.slider("Grid Resolution", min_value=20, max_value=100, value=50, step=5, key="shell_grid_res", help="Higher values increase detail but are much slower.")
                vertical_exaggeration_shell = st.slider(
                    "Vertical Exaggeration", 
                    min_value=1.0, 
                    max_value=10.0, 
                    value=1.0, 
                    step=0.1,
                    key="shell_exaggeration"
                )

            if st.button("Generate Grade Shell", type="primary"):
                st.session_state.grade_shell_fig = None # Clear previous figure
                with st.spinner("Generating grade shell... This may take a few minutes."):
                    try:
                        required_cols = ['HOLE_ID', 'FROM', 'TO', 'x', 'y', 'z', element_of_interest]
                        if not all(col in st.session_state.merged_df.columns for col in required_cols):
                            st.error(f"Missing required columns for grade shell generation. Needed: {required_cols}")
                        else:
                            modeling_df = st.session_state.merged_df.copy()
                            
                            if do_compositing_shell:
                                st.write(f"Compositing data to {composite_length_shell}m intervals...")
                                modeling_df = composite_for_shell(modeling_df, element_of_interest, composite_length_shell)

                            if modeling_df.empty or modeling_df[element_of_interest].isnull().all():
                                st.error("No valid data available for modelling after preparation. Check compositing parameters and element selection.")
                            else:
                                shell_generator = GradeShellGenerator(
                                    element=element_of_interest,
                                    use_log_transform=use_log_transform_shell,
                                    grid_resolution=grid_resolution_shell,
                                    model_type=model_type_shell
                                )
                                shell_generator.raw_data = modeling_df
                                
                                predicted_grades, grid_x, grid_y, grid_z = shell_generator.create_grade_grid(
                                    modeling_data=modeling_df,
                                    full_data_bounds=st.session_state.merged_df
                                )
                                
                                fig_shell = shell_generator.visualise(predicted_grades, grid_x, grid_y, grid_z)
                                update_figure_layout(fig_shell, vertical_exaggeration=vertical_exaggeration_shell)
                                
                                # Store the generated figure in session state
                                st.session_state.grade_shell_fig = fig_shell
                                st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred during grade shell generation: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Display the figure if it exists in session state
            if st.session_state.grade_shell_fig:
                st.success("Grade shell generated successfully!")
                st.plotly_chart(st.session_state.grade_shell_fig, use_container_width=True)
                html_string, filename = create_html_download(st.session_state.grade_shell_fig, f"gradeshell_{element_of_interest}")
                st.download_button(
                    label="📥 Download Grade Shell (HTML)",
                    data=html_string,
                    file_name=filename,
                    mime="text/html"
                )

        else:
            st.warning("Please load collar and assay data in the 'Data Loading' tab first.")

    # =============================================================================
    # 3D SOLID MODEL TAB
    # =============================================================================
    with tab_solid_model:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>🧊 3D Solid Model Generation</h2>", unsafe_allow_html=True)
        st.markdown("Generate 3D solids for categorical data like lithology or clusters using an anisotropic RBF interpolator.")

        if st.session_state.merged_df is not None:
            source_options = []
            if 'LITHO' in st.session_state.merged_df.columns:
                source_options.append('Lithology')
            if "Cluster" in st.session_state.merged_df.columns:
                source_options.append('Clusters')

            if not source_options:
                st.warning("No lithology or cluster data available for modelling. Please load the appropriate data or run clustering first.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    model_source = st.selectbox("Select data source for solids", source_options, key="solid_model_source")
                    
                    unit_display_map = {}
                    if model_source == 'Lithology':
                        column_name = 'LITHO'
                        available_units = sorted(st.session_state.merged_df[column_name].dropna().unique())
                        for unit in available_units:
                            desc = st.session_state.litho_dict.get(unit, "") if st.session_state.litho_dict else ""
                            unit_display_map[unit] = f"{unit} - {desc}" if desc else str(unit)
                    else:
                        column_name = 'Cluster'
                        available_units = sorted([c for c in st.session_state.merged_df[column_name].dropna().unique() if c >= 0])
                        for unit in available_units:
                            unit_display_map[unit] = f"Cluster {int(unit)}"

                    display_unit_map = {v: k for k, v in unit_display_map.items()}
                    
                    selected_display_names = st.multiselect(
                        f"Select {model_source} units to model",
                        options=list(unit_display_map.values()),
                        default=list(unit_display_map.values())[:min(3, len(unit_display_map))]
                    )
                
                with col2:
                    grid_res_solid = st.slider("Grid Resolution", min_value=20, max_value=80, value=35, step=5, key="solid_grid_res", help="Higher values increase detail but are much slower.")
                    vert_exag_solid = st.slider("Vertical Exaggeration for View", min_value=1.0, max_value=10.0, value=1.0, step=0.1, key="solid_vert_exag")
                    surface_opacity = st.slider("Surface Opacity", min_value=0.1, max_value=1.0, value=0.5, step=0.1, key="solid_opacity")

                st.info("💡 **Tip:** After generating the model, you can click on items in the legend on the right to toggle the visibility of each solid.")

                if st.button("Generate 3D Solid Model", type="primary"):
                    st.session_state.solid_model_fig = None # Clear previous figure
                    if not selected_display_names:
                        st.error("Please select at least one unit to model.")
                    else:
                        selected_units = [display_unit_map[name] for name in selected_display_names]

                        with st.spinner("Generating 3D solids... This can be slow for high resolution or many units."):
                            try:
                                model_df = st.session_state.merged_df[st.session_state.merged_df[column_name].isin(selected_units)].copy()
                                model_df.dropna(subset=['x', 'y', 'z', column_name], inplace=True)

                                if len(model_df) < 10:
                                    st.error("Not enough data points in the selected units to build a model.")
                                else:
                                    from scipy.interpolate import Rbf
                                    from skimage.measure import marching_cubes
                                    from sklearn.decomposition import PCA

                                    fig_solid = go.Figure()
                                    colours = px.colors.qualitative.Plotly

                                    min_coords, max_coords = model_df[['x', 'y', 'z']].min(), model_df[['x', 'y', 'z']].max()
                                    ranges = max_coords - min_coords
                                    padding = ranges * 0.15
                                    expanded_min, expanded_max = min_coords - padding, max_coords + padding

                                    grid_x_coords, grid_y_coords, grid_z_coords = np.mgrid[
                                        expanded_min['x']:expanded_max['x']:complex(0, grid_res_solid),
                                        expanded_min['y']:expanded_max['y']:complex(0, grid_res_solid),
                                        expanded_min['z']:expanded_max['z']:complex(0, grid_res_solid)
                                    ]
                                    grid_points_flat = np.vstack([grid_x_coords.ravel(), grid_y_coords.ravel(), grid_z_coords.ravel()]).T

                                    for i, unit in enumerate(selected_units):
                                        st.write(f"--- Modelling unit: {unit_display_map[unit]} ({i+1}/{len(selected_units)}) ---")
                                        
                                        indicator_values = (model_df[column_name] == unit).astype(float)
                                        points = model_df[['x', 'y', 'z']].values

                                        if len(points) > 6000:
                                            st.write(f"Subsampling {len(points)} points to 6000 for performance.")
                                            sample_indices = np.random.choice(len(points), 6000, replace=False)
                                            points, indicator_values = points[sample_indices], indicator_values.iloc[sample_indices]

                                        unit_points = model_df[model_df[column_name] == unit][['x', 'y', 'z']].values
                                        if len(unit_points) < 5:
                                            st.warning(f"Unit '{unit_display_map[unit]}' has < 5 points. Using isotropic model for this unit.")
                                            mean_coord, rotation_matrix, scaling_factors = np.mean(points, axis=0), np.identity(3), np.ones(3)
                                        else:
                                            pca = PCA(n_components=3)
                                            mean_coord = np.mean(unit_points, axis=0)
                                            pca.fit(unit_points - mean_coord)
                                            rotation_matrix, explained_variance = pca.components_, pca.explained_variance_ratio_
                                            scaling_factors = np.sqrt(explained_variance[0] / (explained_variance + 1e-6))
                                            st.write(f"Anisotropy Ratios (1:Sec:Tert): 1.0 : {1/scaling_factors[1]:.2f} : {1/scaling_factors[2]:.2f}")

                                        points_transformed = ((points - mean_coord) @ rotation_matrix.T) * scaling_factors
                                        grid_transformed_flat = ((grid_points_flat - mean_coord) @ rotation_matrix.T) * scaling_factors

                                        rbf = Rbf(points_transformed[:, 0], points_transformed[:, 1], points_transformed[:, 2], indicator_values.values, function='thin_plate', smooth=0.1)
                                        predicted_indicator = rbf(grid_transformed_flat[:, 0], grid_transformed_flat[:, 1], grid_transformed_flat[:, 2]).reshape(grid_x_coords.shape)

                                        try:
                                            spacing = (grid_x_coords[1,0,0]-grid_x_coords[0,0,0], grid_y_coords[0,1,0]-grid_y_coords[0,0,0], grid_z_coords[0,0,1]-grid_z_coords[0,0,0])
                                            verts, faces, _, _ = marching_cubes(predicted_indicator, level=0.5, spacing=spacing)
                                            verts += [grid_x_coords[0,0,0], grid_y_coords[0,0,0], grid_z_coords[0,0,0]]

                                            fig_solid.add_trace(go.Mesh3d(
                                                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                                                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                                color=colours[i % len(colours)],
                                                opacity=surface_opacity,
                                                flatshading=True,
                                                name=unit_display_map[unit],
                                                showlegend=True,
                                                hoverinfo='name',
                                                lighting=dict(
                                                    ambient=0.3, diffuse=1, specular=0.5,
                                                    roughness=0.5, fresnel=0.2
                                                ),
                                                lightposition=dict(x=1000, y=1000, z=1000)
                                            ))
                                        except (ValueError, RuntimeError) as e:
                                            st.warning(f"Could not generate a surface for unit '{unit_display_map[unit]}'. It might be too small or not form a coherent body. Error: {e}")

                                    st.write("Plotting coloured drill traces...")
                                    unit_colour_map = {unit: colours[i % len(colours)] for i, unit in enumerate(selected_units)}
                                    legend_traces_added = set()

                                    for hole_id, hole_data in model_df.groupby('HOLE_ID'):
                                        hole_data = hole_data.sort_values('FROM')
                                        for i in range(len(hole_data) - 1):
                                            p1 = hole_data.iloc[i]
                                            p2 = hole_data.iloc[i+1]
                                            unit = p1[column_name]

                                            if unit in selected_units:
                                                color = unit_colour_map.get(unit, 'grey')
                                                unit_name = unit_display_map.get(unit, str(unit))
                                                legend_group = f"trace_{unit_name}"

                                                show_legend_for_trace = legend_group not in legend_traces_added
                                                if show_legend_for_trace:
                                                    legend_traces_added.add(legend_group)

                                                fig_solid.add_trace(go.Scatter3d(
                                                    x=[p1['x'], p2['x']], y=[p1['y'], p2['y']], z=[p1['z'], p2['z']],
                                                    mode='lines', line=dict(color=color, width=15),
                                                    name=f"Trace: {unit_name}", legendgroup=legend_group,
                                                    showlegend=show_legend_for_trace, hoverinfo='none'
                                                ))
                                    
                                    update_figure_layout(fig_solid, vertical_exaggeration=vert_exag_solid)
                                    fig_solid.update_layout(title=f"3D Solid Model - Source: {model_source}", legend_title=model_source)
                                    
                                    # Store the generated figure in session state
                                    st.session_state.solid_model_fig = fig_solid
                                    st.rerun()

                            except Exception as e:
                                st.error(f"An unexpected error occurred during model generation: {e}")
                                import traceback
                                st.code(traceback.format_exc())

                # Display the figure if it exists in session state
                if st.session_state.solid_model_fig:
                    st.success("Model generation complete!")
                    st.plotly_chart(st.session_state.solid_model_fig, use_container_width=True)
                    html_string, filename = create_html_download(st.session_state.solid_model_fig, f"solid_model_{model_source}")
                    st.download_button(label="📥 Download Solid Model (HTML)", data=html_string, file_name=filename, mime="text/html")

        else:
            st.warning("Please load data in the 'Data Loading' tab first.")

    # The rest of the tabs (Statistics, Clustering, ML Explain, Anomaly, AI GEO, QA, Download) remain unchanged.
    # I am including them here for completeness of the single app.py file.

    # =============================================================================
    # STATISTICS TAB
    # =============================================================================
    with tab_stats:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>📈 Statistical Analysis</h2>", unsafe_allow_html=True)
        
        if st.session_state.merged_df is not None:
            stats_df = st.session_state.merged_df.copy()

            with st.expander("Filter Data for Statistical Analysis", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    selected_lithos_stats = []
                    if 'LITHO' in stats_df.columns:
                        all_lithos = sorted(stats_df['LITHO'].dropna().unique())
                        selected_lithos_stats = st.multiselect("Filter by Lithology", all_lithos, key="stats_litho_filter")
                
                with col2:
                    selected_clusters_stats = []
                    if 'Cluster' in stats_df.columns:
                        all_clusters = sorted([c for c in stats_df['Cluster'].unique() if c >= 0])
                        if all_clusters:
                            selected_clusters_stats = st.multiselect("Filter by Cluster", all_clusters, key="stats_cluster_filter")

                if selected_lithos_stats:
                    stats_df = stats_df[stats_df['LITHO'].isin(selected_lithos_stats)]
                if selected_clusters_stats:
                    stats_df = stats_df[stats_df['Cluster'].isin(selected_clusters_stats)]

            if not stats_df.empty:
                if st.session_state.analysis_mode in ["collar_assay", "all"] and st.session_state.element_cols:
                    primary_element = st.selectbox("Select element for statistical analysis:", 
                                                st.session_state.element_cols,
                                                key="stats_primary_element")
                    use_log_scale = st.checkbox("Use log scale for visualisations", value=True, key="stats_log_scale")
                    
                    show_statistical_analysis(stats_df, primary_element, use_log_scale)
                    
                    st.header("Correlation Analysis")
                    correlation_matrix = stats_df[st.session_state.element_cols].corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=correlation_matrix,
                        x=st.session_state.element_cols,
                        y=st.session_state.element_cols,
                        text=np.round(correlation_matrix, 2),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        hoverongaps=False,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    fig.update_layout(title="Correlation Matrix", height=500, width=500)
                    st.plotly_chart(fig)
                    
                    st.subheader("Element selection for Correlation Analysis and Scatter Diagrams")
                    scatter_elements = st.multiselect(
                        "Select elements for scatter plot (minimum 2)",
                        st.session_state.element_cols,
                        default=st.session_state.element_cols[:min(3, len(st.session_state.element_cols))]
                    )
                    if len(scatter_elements) >= 2:
                        st.subheader("Selected Elements Correlation Matrix")
                        corr_stats = pd.DataFrame(
                            [[stats_df[e1].corr(stats_df[e2]) for e2 in scatter_elements]
                                for e1 in scatter_elements],
                            columns=scatter_elements,
                            index=scatter_elements
                        )
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_stats,
                            x=scatter_elements,
                            y=scatter_elements,
                            text=np.round(corr_stats, 2),
                            texttemplate='%{text}',
                            textfont={"size": 10},
                            hoverongaps=False,
                            colorscale='RdBu',
                            zmid=0
                        ))
                        fig.update_layout(
                            height=max(400, len(scatter_elements) * 40),
                            width=max(500, len(scatter_elements) * 50),
                            xaxis=dict(tickangle=-45),
                            margin=dict(l=50, r=50, t=50, b=50)
                        )
                        st.plotly_chart(fig)
                        st.subheader("Selected Elements Scatter Diagrams")
                        pairs = [(i, j) for i in scatter_elements for j in scatter_elements if i < j]
                        n_pairs = len(pairs)
                        n_cols = min(3, n_pairs)
                        n_rows = (n_pairs + n_cols - 1) // n_cols
                        fig = make_subplots(rows=n_rows, cols=n_cols)
                        def format_tick_label(value):
                            if value >= 1:
                                return f'{value:.0f}'
                            elif value >= 0.1:
                                return f'{value:.2f}'
                            elif value >= 0.01:
                                return f'{value:.3f}'
                            else:
                                return f'{value:.4f}'
                        idx = 0
                        for elem1, elem2 in pairs:
                            row = idx // n_cols + 1
                            col = idx % n_cols + 1
                            scatter = go.Scatter(
                                x=stats_df[elem1], y=stats_df[elem2],
                                mode='markers',
                                marker=dict(
                                    size=6,
                                    color=stats_df[primary_element],
                                    colorscale='Viridis',
                                    showscale=True if idx == 0 else False,
                                    colorbar=dict(title=primary_element) if idx == 0 else None,
                                    opacity=0.7
                                ),
                                name=f'{elem1} vs {elem2}'
                            )
                            fig.add_trace(scatter, row=row, col=col)
                            fig.update_xaxes(title_text=elem1, row=row, col=col)
                            fig.update_yaxes(title_text=elem2, row=row, col=col)
                            if use_log_scale:
                                def log_tick_values(data):
                                    min_val = max(data.min(), 1e-10)
                                    max_val = data.max()
                                    return np.logspace(np.log10(min_val), np.log10(max_val), num=6)
                                x_ticks = log_tick_values(stats_df[elem1])
                                y_ticks = log_tick_values(stats_df[elem2])
                                fig.update_xaxes(
                                    type="log",
                                    tickmode='array',
                                    tickvals=x_ticks,
                                    ticktext=[format_tick_label(x) for x in x_ticks],
                                    row=row, col=col
                                )
                                fig.update_yaxes(
                                    type="log",
                                    tickmode='array',
                                    tickvals=y_ticks,
                                    ticktext=[format_tick_label(y) for y in y_ticks],
                                    row=row, col=col
                                )
                            idx += 1
                        fig.update_layout(height=500 * n_rows, width=500 * n_cols, showlegend=False)
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select at least 2 elements for scatter plot analysis")

                    st.header("Significant Intervals")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        min_length = st.number_input("Minimum Interval Length (m)", value=2.0, min_value=0.1, step=0.5)
                    with col2:
                        max_internal_waste = st.number_input("Maximum Internal Waste (m)", value=2.0, min_value=0.0, step=0.5)
                    with col3:
                        interval_cutoff = st.number_input(
                            f"Minimum {primary_element} Grade",
                            value=float(stats_df[primary_element].median()),
                            min_value=float(stats_df[primary_element].min()),
                            max_value=float(stats_df[primary_element].max()),
                            step=0.1
                        )
                    if st.button("Calculate Significant Intervals"):
                        st.session_state.significant_intervals = calculate_significant_intervals(
                            stats_df, primary_element, interval_cutoff, min_length, max_internal_waste, st.session_state.litho_dict
                        )
                        if not st.session_state.significant_intervals.empty:
                            st.write(st.session_state.significant_intervals)
                        else:
                            st.warning("No significant intervals found with current parameters.")
                
                if st.session_state.analysis_mode in ["collar_litho", "all"] and 'LITHO' in stats_df.columns:
                    if st.session_state.analysis_mode == "collar_litho":
                        stats_df['DUMMY'] = 1.0
                        primary_element = 'DUMMY'
                    else:
                        primary_element = st.selectbox("Select element for lithology analysis:", 
                                                    st.session_state.element_cols,
                                                    key="litho_primary_element")
                    use_log_scale = st.checkbox("Use log scale for lithology analysis", value=True, key="litho_log_scale")
                    create_lithology_analysis(stats_df, primary_element, use_log_scale, st.session_state.litho_dict)
            else:
                st.warning("No data available for analysis after applying filters.")
        else:
            st.warning("Please load data in the Data Loading tab first.")

    # =============================================================================
    # CLUSTERING TAB
    # =============================================================================
    with tab_clustering:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>⚇ Geochemical Clustering</h2>", unsafe_allow_html=True)
        
        if st.session_state.merged_df is not None and st.session_state.analysis_mode in ["collar_assay", "all"]:
            if 'selected_cluster_features' not in state or state.selected_cluster_features is None:
                state.selected_cluster_features = st.session_state.element_cols[:min(5, len(st.session_state.element_cols))]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("Select All Features"):
                    state.selected_cluster_features = st.session_state.element_cols.copy()
            with col2:
                cluster_features = st.multiselect(
                    "Select features for clustering",
                    st.session_state.element_cols,
                    default=state.selected_cluster_features,
                    key="cluster_features_select"
                )
            
            st.subheader("Select Data Subset for Clustering")
            subset_option_cluster = st.radio(
                "Perform clustering on:",
                ("All Data", "Specific Lithology"),
                key="cluster_subset_option"
            )
            
            cluster_df_for_analysis = st.session_state.merged_df.copy()
            
            if subset_option_cluster == "Specific Lithology":
                if 'LITHO' in cluster_df_for_analysis.columns:
                    all_lithos = sorted(cluster_df_for_analysis['LITHO'].dropna().unique())
                    selected_lithos_cluster = st.multiselect(
                        "Select lithologies to include in clustering", 
                        all_lithos, 
                        default=all_lithos, 
                        key="cluster_litho_select"
                    )
                    if selected_lithos_cluster:
                        cluster_df_for_analysis = cluster_df_for_analysis[cluster_df_for_analysis['LITHO'].isin(selected_lithos_cluster)]
                else:
                    st.warning("No lithology data available. Clustering will use all data.")

            use_pca = st.checkbox("Perform PCA before clustering", value=False)
            use_log_transform = st.checkbox("Apply natural log transform", value=False)
            
            if use_pca:
                n_components = st.slider("Maximum number of components to consider", 2, len(cluster_features), min(3, len(cluster_features)))
            
            max_clusters = st.slider("Maximum number of clusters to consider", min_value=2, max_value=15, value=5)

            if st.button("Analyse Clusters"):
                state.selected_cluster_features = cluster_features
                
                if len(cluster_features) < 2:
                    st.warning("Please select at least 2 features for clustering.")
                elif cluster_df_for_analysis.empty:
                    st.warning("No data available for clustering after applying filters.")
                else:
                    X = cluster_df_for_analysis[cluster_features]
                    if use_pca:
                        X_pca, pca, scaler = perform_pca_analysis(X, n_components, use_log_transform)
                        st.session_state.pca_results = {'X_pca': X_pca, 'pca': pca, 'scaler': scaler}
                        
                        wcss = []
                        for i in range(1, max_clusters + 1):
                            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
                            kmeans.fit(X_pca)
                            wcss.append(kmeans.inertia_)
                        state.X_scaled = X_pca
                        state.wcss = wcss
                    else:
                        X_scaled, scaler, wcss = perform_clustering(cluster_df_for_analysis, cluster_features, max_clusters, use_log_transform)
                        state.X_scaled = X_scaled
                        state.scaler = scaler
                        state.wcss = wcss
                    
                    state.clustered_data_index = cluster_df_for_analysis.index
                    st.session_state.clustering_completed = False # Reset to force re-clustering
                    st.rerun()

            if use_pca and 'pca_results' in st.session_state:
                st.subheader("PCA Scree Plot")
                st.plotly_chart(plot_scree(explained_variance_ratio=st.session_state.pca_results['pca'].explained_variance_ratio_, is_pca=True))
                st.plotly_chart(plot_pca_biplot(st.session_state.pca_results['X_pca'], st.session_state.pca_results['pca'], cluster_features))
                st.subheader("Principal Component Loadings")
                eigenvector_fig = plot_eigenvectors(st.session_state.pca_results['pca'], cluster_features)
                st.plotly_chart(eigenvector_fig)

            if state.get('wcss'):
                st.subheader("Clustering Scree Plot")
                st.plotly_chart(plot_scree(wcss=state.wcss, is_pca=False))
                
                state.n_clusters = st.number_input(
                    "Select final number of clusters", 
                    min_value=2, 
                    max_value=max_clusters, 
                    value=state.n_clusters
                )

                kmeans = KMeans(n_clusters=state.n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(state.X_scaled)
                
                st.session_state.merged_df['Cluster'] = -1
                labels_series = pd.Series(cluster_labels, index=state.clustered_data_index, name='Cluster')
                st.session_state.merged_df.update(labels_series)
                
                st.session_state.clustering_completed = True
                
                if not use_pca and state.scaler:
                    if use_log_transform:
                        state.cluster_centers = pd.DataFrame(np.exp(state.scaler.inverse_transform(kmeans.cluster_centers_)) - 1e-10, columns=cluster_features)
                    else:
                        state.cluster_centers = pd.DataFrame(state.scaler.inverse_transform(kmeans.cluster_centers_), columns=state.selected_cluster_features)
                
                st.success(f"Clustering complete with {state.n_clusters} clusters.")
                
                if 'cluster_centers' in state and state.cluster_centers is not None:
                    st.write("Cluster Centres:")
                    st.write(state.cluster_centers)
                
                st.subheader("Cluster Summary Statistics")
                primary_element_for_cluster = st.selectbox("Select element for cluster comparison:", st.session_state.element_cols, key="cluster_summary_element_final")
                
                summary_df = st.session_state.merged_df[st.session_state.merged_df['Cluster'] >= 0]
                
                if not summary_df.empty:
                    summary_stats = get_cluster_summary(summary_df, state.selected_cluster_features, primary_element_for_cluster)
                    st.write(summary_stats.round(3))

                    st.subheader("Feature Distribution by Cluster")
                    use_log_scale_cluster = st.checkbox("Use log scale for boxplots", value=True, key="cluster_boxplot_log_final")
                    fig_boxplots = plot_cluster_boxplots(summary_df, state.selected_cluster_features, primary_element_for_cluster, use_log_scale_cluster)
                    st.plotly_chart(fig_boxplots)
                    
                    if 'LITHO' in summary_df.columns:
                        st.subheader("Lithology vs Cluster Comparison")
                        fig_lith_cluster = plot_lithology_cluster_comparison(summary_df)
                        st.plotly_chart(fig_lith_cluster)
                    
                    st.subheader("3D Visualisation of Clusters")
                    viz_options = ["Clusters"]
                    if primary_element_for_cluster:
                        viz_options.append("Grade")
                    if 'LITHO' in summary_df.columns:
                        viz_options.append("Lithology")
                    
                    selected_viz_cluster = st.multiselect(
                        "Select visualisation types:", viz_options, default=["Clusters"], key="cluster_viz_type_selection"
                    )
                    
                    vertical_exaggeration_cluster = st.slider(
                        "Vertical Exaggeration", 1.0, 10.0, 1.0, 0.1, key="cluster_viz_exaggeration_slider"
                    )

                    active_collars = st.session_state.collar_df[st.session_state.collar_df['HOLE_ID'].isin(summary_df['HOLE_ID'].unique())]
                    active_litho_viz = st.session_state.viz_litho_df[st.session_state.viz_litho_df['HOLE_ID'].isin(summary_df['HOLE_ID'].unique())] if st.session_state.viz_litho_df is not None else None

                    fig_3d_cluster = create_combined_3d_visualisation(
                        df=summary_df, collar_df=active_collars, viz_litho_df=active_litho_viz,
                        litho_dict=st.session_state.litho_dict, selected_viz=selected_viz_cluster,
                        primary_element=primary_element_for_cluster, use_log_scale=use_log_scale_cluster,
                        vertical_exaggeration=vertical_exaggeration_cluster, legend_title="Cluster Plot"
                    )
                    st.plotly_chart(fig_3d_cluster)
                    
                    html_string, filename = create_html_download(fig_3d_cluster, "cluster_visualisation")
                    st.download_button(
                        label="🔬 Download Cluster Plot (HTML)", data=html_string, file_name=filename,
                        mime="text/html", key="download_cluster_viz"
                    )
                else:
                    st.warning("No data with assigned clusters to display.")
        else:
            st.warning("Please load collar and assay data in the 'Data Loading' tab first.")

    # =============================================================================
    # ML EXPLAIN TAB
    # =============================================================================
    with tab_ml_explain:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>🏷️ ML Explain</h2>", unsafe_allow_html=True)

        st.write("This tab allows you to run SHAP analysis for model explanations using an element as the target.")
        
        if "merged_df" not in st.session_state or st.session_state.merged_df is None:
            st.warning("Please load data first in the Data Loading tab.")
        else:
            st.subheader("SHAP Analysis Options")
            
            target_element = None
            if (hasattr(st.session_state, 'analysis_mode') and 
                st.session_state.analysis_mode in ["collar_assay", "all"] and 
                hasattr(st.session_state, 'element_cols') and st.session_state.element_cols):
                target_element = st.selectbox("Select target element for SHAP analysis", st.session_state.element_cols)
            else:
                st.error("Elemental data is not available or analysis mode is not set appropriately for SHAP analysis.")
            
            subset_option = st.radio("Select data subset for SHAP analysis", ("All Data", "Specific Cluster", "Specific Lithology"))
            shap_df = st.session_state.merged_df.copy()
            
            if subset_option == "Specific Cluster":
                if "Cluster" in shap_df.columns:
                    valid_clusters = [c for c in shap_df["Cluster"].unique() if c >= 0]
                    if valid_clusters:
                        clusters = sorted([str(c) for c in valid_clusters])
                        selected_clusters = st.multiselect("Select clusters", clusters, default=clusters)
                        shap_df = shap_df[shap_df["Cluster"].astype(str).isin(selected_clusters)]
                    else:
                        st.warning("No valid clusters found (all samples are unassigned). Running SHAP on all data.")
                else:
                    st.warning("No clustering data available. Running SHAP on all data.")
            elif subset_option == "Specific Lithology":
                if "LITHO" in shap_df.columns:
                    lithos = sorted(shap_df["LITHO"].astype(str).unique())
                    selected_lithos = st.multiselect("Select lithologies", lithos, default=lithos)
                    shap_df = shap_df[shap_df["LITHO"].astype(str).isin(selected_lithos)]
                else:
                    st.warning("No lithology data available. Running SHAP on all data.")
            
            if target_element:
                if st.button("Run SHAP Analysis"):
                    exclude_cols = ['HOLE_ID', 'FROM', 'TO', 'EASTING', 'NORTHING', 'ELEVATION', 'DIP', 'AZIMUTH',
                                    'MIDPOINT', 'dx', 'dy', 'dz', 'x', 'y', 'z', 'LITHO', 'Cluster']
                    
                    model_cols = [col for col in shap_df.columns
                                if col not in exclude_cols
                                and pd.api.types.is_numeric_dtype(shap_df[col])
                                and col != target_element]
                    
                    if not model_cols:
                        st.error("No suitable features available for SHAP analysis after filtering.")
                    elif target_element not in shap_df.columns:
                        st.error(f"Target element '{target_element}' not found in the dataframe.")
                    elif shap_df[model_cols].empty or shap_df[target_element].empty:
                        st.error("Feature set or target data is empty after subsetting. Cannot train model.")
                    else:
                        st.subheader("Training Model for SHAP Analysis")
                        st.write(f"Target: {target_element}")
                        st.write(f"Number of features: {len(model_cols)}")
                        st.write(f"Number of samples: {len(shap_df)}")
                        
                        
                        
                        X = shap_df[model_cols].fillna(0)
                        y = shap_df[target_element].fillna(0)
                        
                        if X.empty:
                            st.error("Feature data (X) is empty. Cannot train model.")
                        else:
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X, y)
                            st.success("Model trained successfully.")
                            
                            st.subheader("Computing SHAP Values")
                            import shap
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X)
                            
                            st.subheader("SHAP Summary Plot (Bar)")
                            fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                            st.pyplot(fig_bar)
                            
                            st.subheader("SHAP Beeswarm Plot")
                            fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_values, X, show=False)
                            st.pyplot(fig_beeswarm)
            elif st.session_state.merged_df is not None:
                st.info("Please select a target element to proceed with SHAP analysis.")

    # =============================================================================
    # ANOMALY DETECTION TAB
    # =============================================================================
    with tab_anomaly:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>🔎 Geochemical Anomaly Detection</h2>", unsafe_allow_html=True)
        st.markdown("""
        This tool uses an Autoencoder neural network to identify geochemically anomalous samples.
        It learns the 'normal' geochemical signature of your dataset and flags samples that don't fit this pattern.
        The resulting Anomaly Score measures how poorly the model could reconstruct a sample. A high score indicates a strong anomaly.
        """)

        if 'anomaly_analysis_complete' not in st.session_state:
            st.session_state.anomaly_analysis_complete = False
        if 'autoencoder_model' not in st.session_state:
            st.session_state.autoencoder_model = None
        if 'anomaly_scaler' not in st.session_state:
            st.session_state.anomaly_scaler = None
        if 'reconstructed_df' not in st.session_state:
            st.session_state.reconstructed_df = None
        if 'anomaly_features' not in st.session_state:
            st.session_state.anomaly_features = []
        if 'anomaly_subset_option' not in st.session_state:
            st.session_state.anomaly_subset_option = "All Data"
        if 'anomaly_subset_selection' not in st.session_state:
            st.session_state.anomaly_subset_selection = []

        if st.session_state.get("merged_df") is not None and st.session_state.analysis_mode in ["collar_assay", "all"]:
            st.subheader("1. Anomaly Detection Setup")

            with st.expander("Select Geochemical Features", expanded=True):
                if 'anomaly_multiselect_features' not in st.session_state:
                    st.session_state.anomaly_multiselect_features = st.session_state.element_cols[:min(10, len(st.session_state.element_cols))]

                if st.button("Select All Features", key="anomaly_select_all_btn"):
                    st.session_state.anomaly_multiselect_features = st.session_state.element_cols
                
                anomaly_features_selection = st.multiselect(
                    "Select elements to define the geochemical signature:",
                    st.session_state.element_cols,
                    key="anomaly_multiselect_features"
                )

            st.markdown("#### Model Parameters")
            col1, col2 = st.columns(2)
            with col1:
                max_bottleneck = max(1, len(anomaly_features_selection) - 1) if anomaly_features_selection else 1
                bottleneck_val = max(1, int(len(anomaly_features_selection) / 2)) if anomaly_features_selection else 1
                is_slider_disabled = (max_bottleneck <= 1)
                bottleneck_size = st.slider("Bottleneck Size", 1, max_bottleneck, bottleneck_val, 1, key="anomaly_bottleneck", help="Size of the compressed data representation.", disabled=is_slider_disabled)
            with col2:
                training_epochs = st.slider("Max Training Epochs", 10, 200, 50, 10, key="anomaly_epochs", help="Max number of training cycles. Early stopping is used.")

            st.subheader("2. Select Data for Model Training")
            st.info("Select the data subset to train the model on. The model will learn this subset's signature as 'normal'. It will then be applied to the entire dataset to find anomalies relative to this baseline.")

            training_options = ("All Data", "Specific Cluster", "Specific Lithology")
            saved_option_index = training_options.index(st.session_state.anomaly_subset_option) if st.session_state.anomaly_subset_option in training_options else 0
            
            subset_option = st.radio(
                "Select data subset for training the anomaly model:",
                training_options,
                index=saved_option_index,
                key="anomaly_subset_option_input"
            )
            
            df_for_training = st.session_state.original_df.copy()
            if 'Cluster' in st.session_state.merged_df.columns:
                df_for_training['Cluster'] = st.session_state.merged_df['Cluster']

            subset_selection = []
            
            if subset_option == "Specific Cluster":
                if "Cluster" in df_for_training.columns:
                    valid_clusters = [c for c in df_for_training["Cluster"].unique() if c >= 0]
                    if valid_clusters:
                        clusters = sorted([str(c) for c in valid_clusters])
                        default_selection = clusters
                        if st.session_state.anomaly_subset_option == "Specific Cluster":
                            default_selection = st.session_state.anomaly_subset_selection
                        
                        subset_selection = st.multiselect("Select clusters for training", clusters, default=default_selection, key="anomaly_cluster_select")
                        df_for_training = df_for_training[df_for_training["Cluster"].astype(str).isin(subset_selection)]
                    else:
                        st.warning("No valid clusters found. Training will use all data.")
                else:
                    st.warning("No clustering data available. Training will use all data.")
            
            elif subset_option == "Specific Lithology":
                if "LITHO" in df_for_training.columns:
                    lithos = sorted(df_for_training["LITHO"].astype(str).unique())
                    default_selection = lithos
                    if st.session_state.anomaly_subset_option == "Specific Lithology":
                        default_selection = st.session_state.anomaly_subset_selection
                    
                    subset_selection = st.multiselect("Select lithologies for training", lithos, default=default_selection, key="anomaly_litho_select")
                    df_for_training = df_for_training[df_for_training["LITHO"].astype(str).isin(subset_selection)]
                else:
                    st.warning("No lithology data available. Training will use all data.")

            st.success(f"The model will be trained on **{len(df_for_training)}** samples.")

            if st.button("Train Model and Detect Anomalies", type="primary", key="anomaly_train_btn"):
                st.session_state.anomaly_subset_option = subset_option
                st.session_state.anomaly_subset_selection = subset_selection
                
                if len(anomaly_features_selection) < 2:
                    st.error("Please select at least two features for analysis.")
                else:
                    st.session_state.anomaly_features = anomaly_features_selection
                    data_to_train = df_for_training[st.session_state.anomaly_features].dropna()

                    if len(data_to_train) < 40:
                        st.error(f"Not enough data ({len(data_to_train)} samples) to train. At least 40 are recommended.")
                        st.session_state.anomaly_analysis_complete = False
                    else:
                        model, scaler, history = build_and_train_autoencoder(data_to_train, bottleneck_size, training_epochs)
                        if model and scaler and history:
                            st.success("Model trained successfully!")
                            
                            st.write("Model Training History:")
                            history_df = pd.DataFrame(history.history)
                            fig_history = px.line(history_df, y=['loss', 'val_loss'], title="Training & Validation Loss")
                            fig_history.update_layout(xaxis_title="Epoch", yaxis_title="Mean Squared Error (Loss)")
                            st.plotly_chart(fig_history, use_container_width=True)

                            full_data_for_scoring = st.session_state.original_df[st.session_state.anomaly_features].fillna(0)
                            scaled_data = scaler.transform(full_data_for_scoring)
                            reconstructed_scaled_data = model.predict(scaled_data)
                            
                            st.session_state.autoencoder_model = model
                            st.session_state.anomaly_scaler = scaler
                            st.session_state.reconstructed_df = pd.DataFrame(scaler.inverse_transform(reconstructed_scaled_data), index=full_data_for_scoring.index, columns=st.session_state.anomaly_features)
                            
                            anomaly_scores = np.mean(np.power(scaled_data - reconstructed_scaled_data, 2), axis=1)
                            
                            st.session_state.original_df['Anomaly_Score'] = anomaly_scores
                            st.session_state.merged_df['Anomaly_Score'] = anomaly_scores
                            
                            st.session_state.anomaly_analysis_complete = True
                            st.rerun()
                        else:
                            st.error("Model training failed. Please check parameters and data.")
                            st.session_state.anomaly_analysis_complete = False
            
            if st.session_state.anomaly_analysis_complete:
                st.subheader("3. Inspect Anomalies")

                df_display = st.session_state.original_df.copy()
                if 'Cluster' in st.session_state.merged_df.columns:
                    df_display['Cluster'] = st.session_state.merged_df['Cluster']

                st.info(f"Displaying results for the training subset: **{st.session_state.anomaly_subset_option}**.")
                if st.session_state.anomaly_subset_option == "Specific Cluster":
                    df_display = df_display[df_display["Cluster"].astype(str).isin(st.session_state.anomaly_subset_selection)]
                elif st.session_state.anomaly_subset_option == "Specific Lithology":
                    df_display = df_display[df_display["LITHO"].astype(str).isin(st.session_state.anomaly_subset_selection)]

                if 'Anomaly_Score' not in df_display.columns or df_display.empty:
                    st.warning("Anomaly scores not found in the current view or the subset is empty.")
                    st.stop()

                anomaly_scores = df_display['Anomaly_Score']

                st.markdown("#### Anomaly Score Distribution")
                fig_hist = px.histogram(x=anomaly_scores, nbins=50, title="Distribution of Anomaly Scores", log_y=True)
                fig_hist.update_layout(xaxis_title="Anomaly Score (Reconstruction Error)", yaxis_title="Count (Log Scale)")
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown("#### Inspect Anomalous Samples")
                default_cutoff = np.percentile(anomaly_scores, 95) if not anomaly_scores.empty else 0
                min_score = float(anomaly_scores.min()) if not anomaly_scores.empty else 0
                max_score = float(anomaly_scores.max()) if not anomaly_scores.empty else 1
                
                anomaly_cutoff = st.slider("Set Anomaly Score Cutoff", min_score, max_score, float(default_cutoff))
                
                anomalous_samples_df = df_display[df_display['Anomaly_Score'] >= anomaly_cutoff].sort_values('Anomaly_Score', ascending=False)
                st.write(f"Found {len(anomalous_samples_df)} samples with an Anomaly Score >= {anomaly_cutoff:.4f} in the current view.")
                cols_to_show = ['HOLE_ID', 'FROM', 'TO', 'Anomaly_Score'] + st.session_state.anomaly_features
                st.dataframe(anomalous_samples_df[cols_to_show])

                if not anomalous_samples_df.empty:
                    st.markdown("#### Reconstruction Comparison")
                    sample_options = [f"Index: {idx} (Hole: {row['HOLE_ID']}, From: {row['FROM']:.2f}m)" for idx, row in anomalous_samples_df.iterrows()]
                    selected_sample_str = st.selectbox("Choose a sample to inspect:", sample_options, key="anomaly_sample_select")
                    
                    if selected_sample_str:
                        selected_index = int(selected_sample_str.split(" ")[1])
                        original_vals = st.session_state.original_df.loc[selected_index][st.session_state.anomaly_features]
                        reconstructed_vals = st.session_state.reconstructed_df.loc[selected_index]
                        comparison_df = pd.DataFrame({'Original': original_vals, 'Reconstructed': reconstructed_vals})
                        comparison_df['Difference'] = comparison_df['Original'] - comparison_df['Reconstructed']
                        st.dataframe(comparison_df.style.format('{:.4f}'))

                        fig_comp = go.Figure(data=[go.Bar(name='Original', x=comparison_df.index, y=comparison_df['Original']), go.Bar(name='Reconstructed', x=comparison_df.index, y=comparison_df['Reconstructed'])])
                        fig_comp.update_layout(barmode='group', title=f"Original vs. Reconstructed Values for Sample {selected_index}")
                        st.plotly_chart(fig_comp, use_container_width=True)

                st.subheader("4. 3D Visualisation of Anomalies")
                
                viz_options = ["Anomaly Score"]
                if st.session_state.element_cols:
                    viz_options.append("Grade")
                if 'Cluster' in df_display.columns:
                    viz_options.append("Clusters")
                if 'LITHO' in df_display.columns:
                    viz_options.append("Lithology")

                selected_viz_anomaly = st.multiselect(
                    "Select data to display:",
                    viz_options,
                    default=["Anomaly Score"],
                    key="anomaly_viz_type"
                )
                
                primary_element_for_viz = None
                use_log_scale_for_grade = True
                if "Grade" in selected_viz_anomaly:
                    col1, col2 = st.columns(2)
                    with col1:
                        primary_element_for_viz = st.selectbox("Select element for Grade view:", st.session_state.element_cols, key="anomaly_grade_select")
                    with col2:
                        use_log_scale_for_grade = st.checkbox("Use log scale for grade", value=True, key="anomaly_grade_log")
                
                col1_viz, col2_viz = st.columns(2)
                with col1_viz:
                    vertical_exaggeration_anomaly = st.slider("Vertical Exaggeration", 1.0, 10.0, 1.0, 0.1, key="anomaly_3d_exag")
                with col2_viz:
                    use_log_scale_for_anomaly = st.checkbox("Use log scale for Anomaly Score", value=True, key="anomaly_score_log")

                fig_3d_anomaly = create_combined_3d_visualisation(
                    df=df_display,
                    collar_df=st.session_state.collar_df,
                    viz_litho_df=st.session_state.viz_litho_df,
                    litho_dict=st.session_state.litho_dict,
                    selected_viz=selected_viz_anomaly,
                    primary_element=primary_element_for_viz,
                    use_log_scale=use_log_scale_for_grade,
                    use_log_scale_anomaly=use_log_scale_for_anomaly,
                    vertical_exaggeration=vertical_exaggeration_anomaly,
                    legend_title="Anomaly Plot"
                )
                
                st.plotly_chart(fig_3d_anomaly)
                html_string, filename = create_html_download(fig_3d_anomaly, "anomaly_detection_3d_plot")
                st.download_button(label="📥 Download 3D Anomaly Plot (HTML)", data=html_string, file_name=filename, mime="text/html")
        
        else:
            st.warning("Please load collar and assay data in the 'Data Loading' tab to use this feature.")

    # =============================================================================
    # AI GEO TAB
    # =============================================================================
    with tab_llm:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>🤖 AI GEO Analysis</h2>", unsafe_allow_html=True)

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'clear_input' not in st.session_state:
            st.session_state.clear_input = False
        
        if st.session_state.clear_input:
            st.session_state.clear_input = False
        
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                role = message["role"]
                content = message["content"]
                
                if role == "user":
                    st.markdown(f"**You**: {content}")
                    st.markdown("---")
                else:
                    st.markdown(f"**AI**: {content}")
                    st.markdown("---")
        
        if not st.session_state.chat_history:
            additional_context = st.text_area(
                "Enter additional geological context (e.g., known rock types, mineralisation style, weathering):",
                key="llm_context",
                height=100
            )
        
        def submit_callback():
            user_question = st.session_state.llm_followup
            if user_question:
                st.session_state.current_question = user_question
                st.session_state.submit_question = True
        
        user_input = st.text_area(
            "Ask a follow-up question or provide additional information:",
            key="llm_followup",
            height=100,
            on_change=submit_callback if st.session_state.chat_history else None
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if not st.session_state.chat_history and st.session_state.google_api_key:
                if st.button("Generate Initial Summary"):
                    with st.spinner("Generating summary..."):
                        try:
                            prompt = generate_summary_prompt(user_context=additional_context)
                            initial_context = f"Additional geological context: {additional_context}" if additional_context else "No additional context provided."
                            
                            st.session_state.chat_history.append({
                                "role": "user",
                                "content": initial_context
                            })
                            
                            client = genai.Client(api_key=st.session_state.google_api_key)
                            response = client.models.generate_content(
                                model=st.session_state.google_model,
                                contents=prompt
                            )
                            
                            if response is not None and hasattr(response, "text") and response.text:
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response.text
                                })
                                
                                if 'summary_prompt' not in st.session_state:
                                    st.session_state.summary_prompt = prompt
                                    
                                st.rerun()
                            else:
                                st.error("Received an empty response from the LLM.")
                        except Exception as e:
                            st.error(f"Error generating summary: {e}")
        
        with col2:
            if st.session_state.chat_history and st.session_state.google_api_key and user_input:
                send_button = st.button("Send Follow-up")
                
                submit_question = send_button or st.session_state.get('submit_question', False)
                
                if submit_question:
                    st.session_state.submit_question = False
                    question = user_input or st.session_state.get('current_question', '')
                    
                    if question:
                        with st.spinner("Processing..."):
                            try:
                                st.session_state.chat_history.append({
                                    "role": "user",
                                    "content": question
                                })
                                
                                conversation_context = "You are an expert geologist analysing drill hole data."
                                conversation_context += "\n\nOriginal analysis context:\n" + st.session_state.get('summary_prompt', 'No original context available.')
                                conversation_context += "\n\nConversation history:\n"
                                
                                history_messages = st.session_state.chat_history[:-1]
                                for msg in history_messages:
                                    role_name = "User" if msg["role"] == "user" else "You (AI Assistant)"
                                    conversation_context += f"\n{role_name}: {msg['content']}\n"
                                
                                follow_up_prompt = f"{conversation_context}\n\nUser's follow-up question: {question}\n\nProvide a detailed, helpful response while maintaining your role as a geological expert."
                                
                                model_for_followup = st.session_state.get('google_model')
                                response = None

                                if not model_for_followup:
                                    st.error("Please ensure a Google AI Model name is entered in the sidebar for follow-up questions.")
                                    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                                        st.session_state.chat_history.pop()
                                else:
                                    client = genai.Client(api_key=st.session_state.google_api_key)
                                    response = client.models.generate_content(
                                        model=model_for_followup,
                                        contents=follow_up_prompt
                                    )
                                
                                if response is not None and hasattr(response, "text") and response.text:
                                    st.session_state.chat_history.append({
                                        "role": "assistant",
                                        "content": response.text
                                    })
                                    
                                    st.session_state.clear_input = True
                                    st.session_state.current_question = ""
                                    
                                    st.rerun()
                                elif model_for_followup:
                                    st.error("Received an empty response from the LLM.")
                                    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                                        st.session_state.chat_history.pop()

                            except Exception as e:
                                st.error(f"Error processing follow-up: {e}")
                                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
                                    st.session_state.chat_history.pop()
        
        with col3:
            if st.session_state.chat_history:
                if st.button("Clear Conversation"):
                    st.session_state.chat_history = []
                    if 'summary_prompt' in st.session_state:
                        del st.session_state.summary_prompt
                    if 'current_question' in st.session_state:
                        del st.session_state.current_question
                    if 'submit_question' in st.session_state:
                        del st.session_state.submit_question
                    st.rerun()
        
        if not st.session_state.google_api_key:
            st.info("Please enter your Google API Key in the sidebar to generate an LLM summary and ask follow-up questions.")
        
        with st.expander("How does this work?"):
            st.markdown("""
            1. First, provide your Google Gemini API key and the specific Google AI Model name in the sidebar. You can also add optional geological context.
            2. Click 'Generate Initial Summary' to analyse your drill hole data using the specified model.
            3. The AI will create an initial interpretation based on your data.
            4. You can then ask follow-up questions or provide additional information.
            5. The system maintains the conversation context throughout your session.
            6. Click 'Clear Conversation' to start over with a fresh analysis.
            """)

    # =============================================================================
    # DATA ANALYSIS PLAYGROUND TAB
    # =============================================================================
    with tab_qa:
        st.header("")
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>📋 Data Analysis Playground</h2>", unsafe_allow_html=True)

        df = st.session_state.get("merged_df")
        if df is None or df.empty:
            st.warning("Please load and process data first.")
        else:
            question = st.text_input("Ask a question in plain English about your drillhole data (e.g. What's the average Au grade for Hole B1? or Plot a histogram of gold values)", key="qa_question")
            
            if question and st.session_state.google_api_key:
                if st.button("Get Answer", key="qa_button"):
                    df_head = df.head(5).to_csv(index=False)
                    cols = ", ".join(df.columns.tolist())
                    prompt = f"""
    You are a Python data analyst. Given a pandas DataFrame `df` with columns: {cols}
    and the first rows:
    {df_head}

    Write ONLY Python code (no markdown, no code fences) that:
    1) answers the question: "{question}"
    2) assigns the result to a variable named answer AND prints the answer variable to the console if generating text output
    3) All numeric values should be rounded to 2 decimal places

    If the question asks for a visualisation:
    - Use matplotlib or plotly to create the visualisation
    - For matplotlib: Create the figure using plt.figure(), make the plot, and end with plt.tight_layout()
    - For plotly: Assign the figure to a variable named 'fig'
    - Make sure visualisations have proper titles, axis labels, and legends

    DO NOT include any imports, markdown formatting, or code fence delimiters.
    The following packages are already available: pandas as pd, numpy as np, matplotlib.pyplot as plt, plotly.express as px, and plotly.graph_objects as go.
    Just write clean Python code that can be directly executed.
    """
                    try:
                        model_to_use_qa = st.session_state.get('google_model')
                        if not model_to_use_qa:
                            st.error("Please enter a Google AI Model name in the sidebar to use this feature.")
                        else:
                            client = genai.Client(api_key=st.session_state.google_api_key)
                            response = client.models.generate_content(
                                model=model_to_use_qa, 
                                contents=prompt
                            )
                            raw_code = response.text or ""
                            
                            code_clean = raw_code
                            if "```" in code_clean:
                                parts = code_clean.split("```")
                                for i, part in enumerate(parts):
                                    if i % 2 == 1:
                                        if part.startswith("python\n"):
                                            parts[i] = part[7:]
                                        elif part.startswith("python"):
                                            parts[i] = part[6:]
                                cleaned_parts = []
                                for i, part in enumerate(parts):
                                    if i % 2 == 1:
                                        cleaned_parts.append(part)
                                if cleaned_parts:
                                    code_clean = "\n".join(cleaned_parts)
                            
                            code_lines = []
                            for line in code_clean.splitlines():
                                if (not line.strip().startswith('import ') and 
                                    not line.strip().startswith('from ') and
                                    not line.strip().startswith('#')):
                                    code_lines.append(line)
                            
                            code_clean = '\n'.join(code_lines).strip()
                            
                            st.subheader("Generated Code")
                            st.code(code_clean, language="python")
                            
                            local_env = {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "px": px, "go": go}
                            old_stdout = sys.stdout
                            sys.stdout = mystdout = io.StringIO()
                            
                            plt.switch_backend('Agg')
                            
                            content_generated = False
                            
                            try:
                                exec(code_clean, {}, local_env)
                            except Exception as exec_error:
                                sys.stdout = old_stdout
                                st.error(f"Error executing the code: {exec_error}")
                                st.error("Attempting to further clean the code...")
                                
                                more_clean = raw_code
                                if "```python" in more_clean:
                                    more_clean = more_clean.split("```python", 1)[1]
                                elif "```" in more_clean:
                                    more_clean = more_clean.split("```", 1)[1]
                                
                                if "```" in more_clean:
                                    more_clean = more_clean.rsplit("```", 1)[0]
                                
                                clean_lines_retry = []
                                for line_retry in more_clean.strip().splitlines():
                                    if (not line_retry.strip().startswith('import ') and 
                                        not line_retry.strip().startswith('from ') and
                                        not line_retry.strip().startswith('#')):
                                        clean_lines_retry.append(line_retry)
                                
                                more_clean = '\n'.join(clean_lines_retry).strip()
                                
                                st.subheader("Retrying with further cleaned code:")
                                st.code(more_clean, language="python")
                                
                                try:
                                    sys.stdout = mystdout = io.StringIO()
                                    exec(more_clean, {}, local_env)
                                except Exception as retry_error:
                                    sys.stdout = old_stdout
                                    st.error(f"Retry also failed: {retry_error}")
                                    st.error("Traceback for retry:")
                                    import traceback
                                    st.code(traceback.format_exc())
                            
                            sys.stdout = old_stdout
                            output = mystdout.getvalue().strip()
                            
                            if output:
                                st.subheader("Text Output")
                                st.write(output)
                                content_generated = True
                            
                            fig_count = len(plt.get_fignums()) 
                            if fig_count > 0:
                                st.subheader("Visualisation (Matplotlib)")
                                for i in plt.get_fignums():
                                    fig_mpl = plt.figure(i)
                                    buf = io.BytesIO()
                                    fig_mpl.savefig(buf, format='png', bbox_inches='tight')
                                    buf.seek(0)
                                    st.image(buf)
                                plt.close('all')
                                content_generated = True
                            
                            if 'fig' in local_env and hasattr(local_env['fig'], 'update_layout'):
                                st.subheader("Interactive Visualisation (Plotly)")
                                st.plotly_chart(local_env['fig'])
                                content_generated = True
                            
                            if not content_generated and model_to_use_qa:
                                st.warning("No output or visualisation was generated. The generated code might not produce direct output or the question was ambiguous. Try rephrasing your question.")
                                
                    except Exception as e:
                        if 'old_stdout' in locals() and sys.stdout != old_stdout :
                            sys.stdout = old_stdout
                        st.error(f"An error occurred in the Data Analysis Playground: {e}")
                        st.error("Traceback:")
                        import traceback
                        st.code(traceback.format_exc())

            elif not st.session_state.google_api_key:
                st.info("Please enter your Google API Key in the sidebar to use this feature.")
            elif question and not st.session_state.google_model:
                if st.button("Get Answer", key="qa_button_no_model_check"):
                    st.error("Please enter a Google AI Model name in the sidebar to use this feature.")

            with st.expander("Example Visualisation Questions"):
                st.markdown("""
                Try these example questions for generating visualisations:
                
                - **Plot a histogram of gold (Au) values**
                - **Create a scatter plot of copper vs gold**
                - **Create a box plot of gold values by lithology**
                - **Make a bar chart showing average gold by HOLE_ID**
                - **Plot a correlation heatmap of all elemental values**
                - **Create a 3D scatter plot of x, y, and z coordinates coloured by gold values**
                
                For more complex analysis:
                
                - **Find and visualise the relationship between depth and gold values**
                - **Compare gold distribution across different clusters**
                - **Show the spatial distribution of samples with gold values above 0.5**
                """)

    # =============================================================================
    # DOWNLOAD TAB
    # =============================================================================
    with tab_download:
        st.markdown("<h2 style='color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 0.5rem;'>💾 Download Data</h2>", unsafe_allow_html=True)

        if st.session_state.merged_df is not None:
            download_cols = st.columns(4)
            with download_cols[0]:
                csv_processed = st.session_state.merged_df.to_csv(index=False)
                st.download_button(
                    label="Download Processed Data",
                    data=csv_processed,
                    file_name="processed_drillhole_data.csv",
                    mime="text/csv"
                )
            if st.session_state.analysis_mode in ["collar_assay", "all"] and st.session_state.element_cols:
                with download_cols[1]:
                    primary_element_for_stats = st.session_state.element_cols[0] if len(st.session_state.element_cols) > 0 else None
                    if primary_element_for_stats and primary_element_for_stats in st.session_state.merged_df.columns:
                        stats_dict = {
                            'Statistic': [
                                'Count', 'Mean', 'Median', 'Std Dev', 'CV', 
                                'Min', 'Q1', 'Q3', 'Max', 'Skewness', 'Kurtosis'
                            ],
                            'Value': [
                                len(st.session_state.merged_df[primary_element_for_stats]),
                                st.session_state.merged_df[primary_element_for_stats].mean(),
                                st.session_state.merged_df[primary_element_for_stats].median(),
                                st.session_state.merged_df[primary_element_for_stats].std(),
                                st.session_state.merged_df[primary_element_for_stats].std() / st.session_state.merged_df[primary_element_for_stats].mean(),
                                st.session_state.merged_df[primary_element_for_stats].min(),
                                st.session_state.merged_df[primary_element_for_stats].quantile(0.25),
                                st.session_state.merged_df[primary_element_for_stats].quantile(0.75),
                                st.session_state.merged_df[primary_element_for_stats].max(),
                                st.session_state.merged_df[primary_element_for_stats].skew(),
                                st.session_state.merged_df[primary_element_for_stats].kurtosis()
                            ]
                        }
                        stats_df = pd.DataFrame(stats_dict)
                        stats_df['Value'] = stats_df['Value'].round(3)
                        csv_stats = stats_df.to_csv(index=False)
                        st.download_button(
                            label=f"Download {primary_element_for_stats} Statistics",
                            data=csv_stats,
                            file_name=f"{primary_element_for_stats}_statistics.csv",
                            mime="text/csv"
                        )
            if st.session_state.significant_intervals is not None and not st.session_state.significant_intervals.empty:
                with download_cols[2]:
                    csv_intervals = st.session_state.significant_intervals.to_csv(index=False)
                    st.download_button(
                        label="Download Significant Intervals",
                        data=csv_intervals,
                        file_name="significant_intervals.csv",
                        mime="text/csv"
                    )
            if 'LITHO' in st.session_state.merged_df.columns:
                with download_cols[3]:
                    litho_stats = st.session_state.merged_df.groupby('LITHO').size().reset_index(name='Count')
                    if st.session_state.litho_dict:
                        litho_stats['Description'] = litho_stats['LITHO'].map(lambda x: st.session_state.litho_dict.get(x, ""))
                    csv_litho_stats = litho_stats.to_csv(index=False)
                    st.download_button(
                        label="Download Lithology Statistics",
                        data=csv_litho_stats,
                        file_name="lithology_statistics.csv",
                        mime="text/csv"
                    )
        else:
            st.warning("Please load data in the Data Loading tab first.")

if __name__ == "__main__":
    main()