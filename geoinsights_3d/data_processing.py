import streamlit as st
import pandas as pd
import numpy as np

def read_csv(file, format_type):
    """Read CSV files with standard or mining format."""
    try:
        if format_type == "Standard CSV (Headers in row 1)":
            return pd.read_csv(file)
        else:  # Geological Survey Format
            df = pd.read_csv(file, header=None)
            header_row_idx = df[df.iloc[:, 0] == 'H1000'].index[0]
            headers = df.iloc[header_row_idx].values.tolist()
            data_rows = df[df.iloc[:, 0] == 'D']
            result_df = pd.DataFrame(data_rows.values, columns=headers)
            return result_df.iloc[:, 1:]
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def process_collar_data(collar_file, format_type):
    """Process collar file and return formatted DataFrame."""
    try:
        collar_df = read_csv(collar_file, format_type)
        if collar_df is not None:
            st.write("Collar Data Preview:")
            st.write(collar_df.head())

            st.subheader("Select Collar Columns")
            hole_id_col = next((col for col in collar_df.columns if 'hole' in col.lower()), None)
            easting_col = next((col for col in collar_df.columns if any(x in col.lower() for x in ['easting', 'mga_e', 'x','long','longitude'])), None)
            northing_col = next((col for col in collar_df.columns if any(x in col.lower() for x in ['northing', 'mga_n', 'y','lat','latitude'])), None)
            elevation_col = next((col for col in collar_df.columns if any(x in col.lower() for x in ['elevation', 'rl', 'z'])), None)
            dip_col = next((col for col in collar_df.columns if 'dip' in col.lower()), None)
            azimuth_col = next((col for col in collar_df.columns if any(x in col.lower() for x in ['azi', 'azimuth'])), None)

            hole_id_col = st.selectbox("Select HOLE_ID column", collar_df.columns, 
                                       index=(collar_df.columns.get_loc(hole_id_col) if hole_id_col else 0))
            easting_col = st.selectbox("Select EASTING column", collar_df.columns,
                                       index=(collar_df.columns.get_loc(easting_col) if easting_col else 0))
            northing_col = st.selectbox("Select NORTHING column", collar_df.columns,
                                        index=(collar_df.columns.get_loc(northing_col) if northing_col else 0))
            elevation_col = st.selectbox("Select ELEVATION column", collar_df.columns,
                                         index=(collar_df.columns.get_loc(elevation_col) if elevation_col else 0))
            dip_col = st.selectbox("Select DIP column", collar_df.columns,
                                   index=(collar_df.columns.get_loc(dip_col) if dip_col else 0))
            azimuth_col = st.selectbox("Select AZIMUTH column", collar_df.columns,
                                       index=(collar_df.columns.get_loc(azimuth_col) if azimuth_col else 0))

            collar_df = collar_df.rename(columns={
                hole_id_col: 'HOLE_ID',
                easting_col: 'EASTING',
                northing_col: 'NORTHING',
                elevation_col: 'ELEVATION',
                dip_col: 'DIP',
                azimuth_col: 'AZIMUTH'
            })
            for col in ['EASTING', 'NORTHING', 'ELEVATION', 'DIP', 'AZIMUTH']:
                collar_df[col] = pd.to_numeric(collar_df[col], errors='coerce')
            
            return collar_df
    except Exception as e:
        st.error(f"Error processing collar file: {str(e)}")
        return None

def process_assay_data(assay_file, format_type):
    """Process assay file and select columns for geochemical data."""
    try:
        assay_df = read_csv(assay_file, format_type)
        if assay_df is not None:
            st.write("Assay Data Preview:")
            st.write(assay_df.head())

            st.subheader("Select Assay Columns")
            hole_id_col = next((col for col in assay_df.columns if 'hole' in col.lower()), None)
            from_col = next((col for col in assay_df.columns if 'from' in col.lower()), None)
            to_col = next((col for col in assay_df.columns if 'to' in col.lower()), None)

            hole_id_col = st.selectbox("Select HOLE_ID column (Assay)", assay_df.columns, 
                                       index=(assay_df.columns.get_loc(hole_id_col) if hole_id_col else 0))
            from_col = st.selectbox("Select FROM column", assay_df.columns, 
                                    index=(assay_df.columns.get_loc(from_col) if from_col else 0))
            to_col = st.selectbox("Select TO column", assay_df.columns, 
                                  index=(assay_df.columns.get_loc(to_col) if to_col else 0))

            available_elements = [col for col in assay_df.columns if col not in [hole_id_col, from_col, to_col]]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                select_all = st.checkbox("Select all (Remember to remove non-assay columns)", value=True)
            with col2:
                if select_all:
                    element_cols = st.multiselect(
                        "Select element columns",
                        available_elements,
                        default=available_elements
                    )
                else:
                    element_cols = st.multiselect(
                        "Select element columns",
                        available_elements
                    )

            assay_df = assay_df.rename(columns={
                hole_id_col: 'HOLE_ID',
                from_col: 'FROM',
                to_col: 'TO'
            })
            
            numeric_cols = ['FROM', 'TO'] + element_cols
            for col in numeric_cols:
                assay_df[col] = assay_df[col].astype(str).str.replace('<', '-')
                assay_df[col] = pd.to_numeric(assay_df[col], errors='coerce')
                assay_df.loc[assay_df[col] < 0, col] = abs(assay_df[col]) / 2

            assay_df = assay_df[['HOLE_ID', 'FROM', 'TO'] + element_cols]
            return assay_df, element_cols
    except Exception as e:
        st.error(f"Error processing assay file: {str(e)}")
        return None, None

def process_litho_data(litho_file, format_type):
    """Process lithology file and return formatted DataFrame."""
    try:
        litho_df = read_csv(litho_file, format_type)
        if litho_df is not None:
            st.write("Lithology Data Preview:")
            st.write(litho_df.head())

            st.subheader("Select Lithology Columns")
            hole_id_col = next((col for col in litho_df.columns if 'hole' in col.lower()), None)
            from_col = next((col for col in litho_df.columns if 'from' in col.lower()), None)
            to_col = next((col for col in litho_df.columns if 'to' in col.lower()), None)
            litho_col = next((col for col in litho_df.columns if any(x in col.lower() for x in ['lith', 'rock', 'geol'])), None)

            hole_id_col = st.selectbox("Select HOLE_ID column (Lithology)", litho_df.columns, 
                                       index=(litho_df.columns.get_loc(hole_id_col) if hole_id_col else 0))
            from_col = st.selectbox("Select FROM column (Lithology)", litho_df.columns, 
                                    index=(litho_df.columns.get_loc(from_col) if from_col else 0))
            to_col = st.selectbox("Select TO column (Lithology)", litho_df.columns, 
                                  index=(litho_df.columns.get_loc(to_col) if to_col else 0))
            litho_col = st.selectbox("Select LITHOLOGY column", litho_df.columns, 
                                     index=(litho_df.columns.get_loc(litho_col) if litho_col else 0))

            litho_df = litho_df.rename(columns={
                hole_id_col: 'HOLE_ID',
                from_col: 'FROM',
                to_col: 'TO',
                litho_col: 'LITHO'
            })
            litho_df['FROM'] = pd.to_numeric(litho_df['FROM'], errors='coerce')
            litho_df['TO'] = pd.to_numeric(litho_df['TO'], errors='coerce')
            litho_df = litho_df[['HOLE_ID', 'FROM', 'TO', 'LITHO']]
            return litho_df
    except Exception as e:
        st.error(f"Error processing lithology file: {str(e)}")
        return None

def process_litho_dict(litho_dict_file, format_type):
    """Process lithology dictionary file and return a dictionary code->description."""
    try:
        litho_dict_df = read_csv(litho_dict_file, format_type)
        if litho_dict_df is not None:
            st.write("Lithology Dictionary Preview:")
            st.write(litho_dict_df.head())

            st.subheader("Select Lithology Dictionary Columns")
            code_col = next((col for col in litho_dict_df.columns if any(x in col.lower() for x in ['code', 'lith', 'rock'])), None)
            desc_col = next((col for col in litho_dict_df.columns if any(x in col.lower() for x in ['desc', 'name', 'type'])), None)

            code_col = st.selectbox("Select Lithology Code column", litho_dict_df.columns,
                                    index=(litho_dict_df.columns.get_loc(code_col) if code_col else 0))
            desc_col = st.selectbox("Select Lithology Description column", litho_dict_df.columns,
                                    index=(litho_dict_df.columns.get_loc(desc_col) if desc_col else 0))

            litho_dict = dict(zip(litho_dict_df[code_col], litho_dict_df[desc_col]))
            return litho_dict
    except Exception as e:
        st.error(f"Error processing lithology dictionary file: {str(e)}")
        return None

def composite_geochemical_data(df, element_cols, composite_length):
    """Create composites of geochemical intervals at a fixed length."""
    if 'HOLE_ID' not in df.columns or 'FROM' not in df.columns or 'TO' not in df.columns:
        st.error("DataFrame must have columns 'HOLE_ID', 'FROM', 'TO' for compositing.")
        return df

    composited_rows = []
    for hole_id, hole_data in df.groupby('HOLE_ID', sort=False):
        hole_data = hole_data.sort_values('FROM')
        hole_start = hole_data['FROM'].min()
        hole_end = hole_data['TO'].max()
        composite_top = hole_start

        while composite_top < hole_end:
            composite_bot = composite_top + composite_length
            if composite_bot > hole_end:
                composite_bot = hole_end
            overlap = hole_data[
                (hole_data['FROM'] < composite_bot) &
                (hole_data['TO'] > composite_top)
            ].copy()
            if overlap.empty:
                composite_top = composite_bot
                continue

            overlap['interval_start'] = overlap['FROM'].clip(lower=composite_top)
            overlap['interval_end'] = overlap['TO'].clip(upper=composite_bot)
            overlap['interval_length'] = overlap['interval_end'] - overlap['interval_start']

            composited_values = {}
            total_length = overlap['interval_length'].sum()

            for elem in element_cols:
                composited_values[elem] = np.average(overlap[elem], weights=overlap['interval_length'])

            composited_row = {
                'HOLE_ID': hole_id,
                'FROM': composite_top,
                'TO': composite_bot
            }
            composited_row.update(composited_values)
            composited_rows.append(composited_row)
            composite_top = composite_bot

    composite_df = pd.DataFrame(composited_rows)
    return composite_df

def process_and_merge_data(collar_df, assay_df, litho_df, element_cols, composite_enabled, composite_length):
    """Processes and merges collar, assay, and lithology data."""
    merged_df = None
    viz_litho_df = None
    
    if collar_df is not None:
        # First, process assay data if available
        if assay_df is not None:
            # Apply compositing if needed
            if composite_enabled and element_cols:
                assay_df = composite_geochemical_data(assay_df, element_cols, composite_length)

            # Merge collar and assay data
            merged_df = pd.merge(collar_df, assay_df, on='HOLE_ID', how='inner')
            
            # Calculate 3D coordinates for assay data
            if 'FROM' in merged_df.columns and 'TO' in merged_df.columns:
                merged_df['MIDPOINT'] = (merged_df['FROM'] + merged_df['TO']) / 2
                merged_df['AZIMUTH_RAD'] = np.radians(90 - merged_df['AZIMUTH'])
                merged_df['DIP_RAD'] = np.radians(merged_df['DIP'])
                merged_df['dx'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.cos(merged_df['AZIMUTH_RAD'])
                merged_df['dy'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.sin(merged_df['AZIMUTH_RAD'])
                merged_df['dz'] = merged_df['MIDPOINT'] * np.sin(merged_df['DIP_RAD'])
                merged_df['x'] = merged_df['EASTING'] + merged_df['dx']
                merged_df['y'] = merged_df['NORTHING'] + merged_df['dy']
                merged_df['z'] = merged_df['ELEVATION'] + merged_df['dz']

        # Process lithology data if available
        if litho_df is not None:
            # Create visualisation dataframe for lithology
            viz_litho_df = pd.merge(litho_df, collar_df[['HOLE_ID','EASTING','NORTHING','ELEVATION','DIP','AZIMUTH']], on='HOLE_ID')
            viz_litho_df['MIDPOINT'] = (viz_litho_df['FROM'] + viz_litho_df['TO']) / 2
            viz_litho_df['AZIMUTH_RAD'] = np.radians(90 - viz_litho_df['AZIMUTH'])
            viz_litho_df['DIP_RAD'] = np.radians(viz_litho_df['DIP'])
            viz_litho_df['dx'] = viz_litho_df['MIDPOINT'] * np.cos(viz_litho_df['DIP_RAD']) * np.cos(viz_litho_df['AZIMUTH_RAD'])
            viz_litho_df['dy'] = viz_litho_df['MIDPOINT'] * np.cos(viz_litho_df['DIP_RAD']) * np.sin(viz_litho_df['AZIMUTH_RAD'])
            viz_litho_df['dz'] = viz_litho_df['MIDPOINT'] * np.sin(viz_litho_df['DIP_RAD'])
            viz_litho_df['x'] = viz_litho_df['EASTING'] + viz_litho_df['dx']
            viz_litho_df['y'] = viz_litho_df['NORTHING'] + viz_litho_df['dy']
            viz_litho_df['z'] = viz_litho_df['ELEVATION'] + viz_litho_df['dz']
            
            # If we have no assay data, use lithology data as the main dataset
            if merged_df is None:
                merged_df = pd.merge(collar_df, litho_df, on='HOLE_ID', how='inner')
                merged_df['MIDPOINT'] = (merged_df['FROM'] + merged_df['TO']) / 2
                merged_df['AZIMUTH_RAD'] = np.radians(90 - merged_df['AZIMUTH'])
                merged_df['DIP_RAD'] = np.radians(merged_df['DIP'])
                merged_df['dx'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.cos(merged_df['AZIMUTH_RAD'])
                merged_df['dy'] = merged_df['MIDPOINT'] * np.cos(merged_df['DIP_RAD']) * np.sin(merged_df['AZIMUTH_RAD'])
                merged_df['dz'] = merged_df['MIDPOINT'] * np.sin(merged_df['DIP_RAD'])
                merged_df['x'] = merged_df['EASTING'] + merged_df['dx']
                merged_df['y'] = merged_df['NORTHING'] + merged_df['dy']
                merged_df['z'] = merged_df['ELEVATION'] + merged_df['dz']
            else:
                # If we have both assay and lithology data, we need to join them carefully
                # First, ensure both dataframes have the same HOLE_ID, FROM, TO structure
                assay_intervals = merged_df[['HOLE_ID', 'FROM', 'TO']].copy()
                
                # Create a function to find the best matching lithology for each assay interval
                def find_matching_litho(row, litho_data):
                    hole_lithos = litho_data[litho_data['HOLE_ID'] == row['HOLE_ID']]
                    if hole_lithos.empty:
                        return None
                    
                    # Find lithologies that overlap with this interval
                    overlaps = hole_lithos[
                        ((hole_lithos['FROM'] <= row['FROM']) & (hole_lithos['TO'] > row['FROM'])) |
                        ((hole_lithos['FROM'] < row['TO']) & (hole_lithos['TO'] >= row['TO'])) |
                        ((hole_lithos['FROM'] >= row['FROM']) & (hole_lithos['TO'] <= row['TO']))
                    ]
                    
                    if overlaps.empty:
                        # Find nearest lithology if no direct overlap
                        hole_lithos['distance'] = np.minimum(
                            np.abs(hole_lithos['FROM'] - row['MIDPOINT']),
                            np.abs(hole_lithos['TO'] - row['MIDPOINT'])
                        )
                        return hole_lithos.loc[hole_lithos['distance'].idxmin()]['LITHO']
                    else:
                        # Get the lithology with the most overlap
                        overlaps['overlap'] = np.minimum(overlaps['TO'], row['TO']) - np.maximum(overlaps['FROM'], row['FROM'])
                        return overlaps.loc[overlaps['overlap'].idxmax()]['LITHO']
                
                # Apply the function to each assay interval
                merged_df['LITHO'] = merged_df.apply(
                    lambda row: find_matching_litho(row, litho_df), axis=1
                )
    
    return merged_df, viz_litho_df