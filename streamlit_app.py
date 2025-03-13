import streamlit as st
import os
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
from tslearn.metrics import dtw

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates["empty"] = {}
pio.templates.default = "empty"

#################################### CODE WEB #################################################

st.set_page_config(
    page_title="Eurozone Divergence Monitor",
    layout="wide",
)

st.markdown(
    """
    <style>
    .title {
        text-align: center; 
        margin-bottom: -20px;
        font-family: 'Futura', sans-serif; 
        font-size: 2.5em;
        color: black;
    }
    .sub-title {
        text-align: center; 
        margin-bottom: 10px;
        font-family: 'Futura', sans-serif; 
        font-size: 2em; 
        color: black;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebarContent"] {
        background-color: white !important;
        color: black !important;
    }
    [data-testid="stMarkdown"] {
        max-width: 75vw !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use the styled title
st.markdown('<div class="title">Eurozone Divergence Monitor</div>', unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([0.125, 0.75, 0.125])
with col_center:
    fig_divergence_monitor = st.empty() 
button_index = st.empty()


st.markdown(
    """
    Our divergence monitor measures the weighted synchronicity of the business and financial cycles of eurozone member countries. 
    The basic idea—first put forth by <a href="#ref-mundell-1961">Robert Mundell (1961)</a>—behind the monitor is that monetary policy 
    cannot function effeciently in a currency union when cycles are asymmetric. If, for example, Germany is in crisis and Spain is booming, 
    as was the case after the turn of the millennium, the European Central Bank (ECB) cannot set the right interest rate for both countries. 
    A lower interest rate would lead to overheating the economy in Spain, and a higher interest rate would exacerbate the crisis in Germany. 
    Since Mario Draghi's <a href="https://www.ecb.europa.eu/press/key/date/2012/html/sp120726.en.html">"Whatever it takes"</a> speech, 
    monetary policy in the Eurozone is driven by measures seeking to hold the eurozone together. Among other things, we hope that our 
    monitor can help economists and policymakers in the assessment of such measures. The indicator is measured on a quarterly basis and 
    will be updated regularly. The <b>next update</b> is scheduled for <b>August, 2025</b>.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="sub-title">Methodology</div>', unsafe_allow_html=True)
st.markdown(
    """
    Our measure separates cycle extraction from synchronization measurement. First, we estimate cycles based on the co-movement of relevant 
    indicators using a non-parametric method that preserves turning points without imposing fixed cycle durations. The business and financial 
    cycle models are based on four fundamental variables each: real quarterly growth rates of GDP, private consumption, gross fixed capital 
    formation, and unemployment for the business cycle and real quarterly growth rates of outstanding credit, house prices, stock prices and 
    bond prices for the financial cyc For more details, see our paper <a href="#ref-bugdalle-pfeifer-2025">Bugdalle &amp; Pfeifer (2025)</a>. 
    """,
    unsafe_allow_html=True, 
)

col1_2d, col2_2d = st.columns(2)  
with col1_2d:
    fig_bc = st.empty()  

with col2_2d:
    fig_fc = st.empty()

st.markdown(
    """   
    Second, once the cycles (above) are identified, we compute pairwise dynamic time warping (DTW) distances to assess the similarity of their 
    shapes, regardless of phase shifts or timing differences. DTW is non-parametric technique introduced by <a href="#ref-sakoe-1978">Sakoe &amp; Shiba (1978)</a> 
    that adjusts for time lags or leads between two time series, providing a measure of inter-temporal similarity.
    """,
    unsafe_allow_html=True,
)

math_container = st.container()
with math_container:
    st.markdown(
        r"""
        In our case, DTW is applied to each type of the smoothed cycle indices, meaning one measure of similarity is estimated for each type of cycle.  
        Within each cycle category, DTW computes the alignment path $\mathbf{\pi}_{ij}$ for each pair of countries $i$ and $j$, that minimizes the 
        cumulative distance between two cycles:
        """
    )

st.latex(r'''
    D(\mathbf{x}, \mathbf{y}) = \min_{\pi_{ij}} \sum_{(t, s) \in \pi_{ij}} \left| \mathbf{x_t} - \mathbf{y_s} \right|^2,
''')

st.markdown(
    r"""
    where, $\mathbf{x}_{t}$ and $\mathbf{y}_{s}$ are the smoothed cycle values at time $t$ and time $s$ for countries $i$ and $j$, respectively.
    The resulting distance $D(x,y)$ captures the degree of similarity, with smaller values indicating greater similarity between the cycles. 
    To ensure that the DTW comparison reflects the timing of cyclical movements, the alignment is performed over a local window defined by the average cycle duration.
    To aggregate the pairwise DTW distances into a single indicator, we compute a gdp-weighted mean DTW distance. The result is the above indicator.
    """
)


col1_3d, col2_3d = st.columns(2)  
with col1_3d:
    bus_cycle_3d = st.empty()  

with col2_3d:
    fin_cycle_3d = st.empty()  

st.markdown(
    """   
Each curve in the 3D plots above represents, for each country, the average weighted DTW distance computed across all pairwise comparisons with every other countries’ cycles.
A Gaussian filter is applied for visualization along the time axis to smooth short-term fluctuations. The dotted business and financial divergences in our monitor are the 
un-filtered averages of these distances. 
"""
)


st.markdown('<div class="sub-title">References</div>', unsafe_allow_html=True)
st.markdown(
    """
        <p>
        <a id="ref-bugdalle-pfeifer-2025"></a>
        Bugdalle, T., Pfeifer, M. (2025). A Tale of Two Cycles: Business and Financial Cycle
        Synchronization in the Euro Area. arXiv preprint.
        </p>
        <p>
        <a id="ref-sakoe-1978"></a>
        Sakoe, H., Chiba, S. (1978). Dynamic programming algorithm optimization 
        for spoken word recognition. <em>IEEE Transactions on Acoustics, Speech, 
        and Signal Processing, 26</em>(1), 43–49.
        </p>
        <p>
        <a id="ref-mundell-1961"></a>
        Mundell, R. (1961). A Theory of Optimal Currency Areas. 
        <em>American Economic Review, 51</em>(4), 657–665.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



#################################### CODE PLOTS #################################################
base_dir = os.path.dirname(__file__)

# Build the path to the Excel file relative to base_dir
excel_path = os.path.join(base_dir, 'Data_Final', 'Country_Cycle_Data', 'Cycle_Data.xlsx')

# Load the Excel file
excel_data = pd.ExcelFile(excel_path, engine='openpyxl')

# Build the path to "Country_Cycle_Summary.xlsx"
cycle_avg_path = os.path.join(base_dir, 'Data_Final', 'Country_Cycle_Data', 'Country_Cycle_Summary.xlsx')

# Load the summary data
cycle_avg = pd.read_excel(cycle_avg_path, index_col='Country')

sheet_names = excel_data.sheet_names

resampled_series_data_fc = {}
resampled_series_data_bc = {}

# Find the global time range for quarterly dates
global_start_date = min(pd.to_datetime(dates.iloc[0]) for dates in [pd.to_datetime(pd.read_excel(excel_data, sheet_name=sheet).iloc[:, 0]) for sheet in sheet_names])
global_end_date = max(pd.to_datetime(dates.iloc[-1]) for dates in [pd.to_datetime(pd.read_excel(excel_data, sheet_name=sheet).iloc[:, 0]) for sheet in sheet_names])

global_dates_gdp = pd.date_range(global_start_date, global_end_date, freq='QS')

# Load CPI data (one sheet, with each country's CPI in its own column)
cpi_path = os.path.join(base_dir, 'Data_Final', 'CPI.xlsx')
gdp_path = os.path.join(base_dir, 'Data_Final', 'business_cycle_data_final_ERMcut.xlsx')

# Load the CPI data using the relative path and set the index
cpi_data = pd.read_excel(cpi_path, index_col='Unnamed: 0')
cpi_data.index = pd.to_datetime(cpi_data.index)

# Load GDP data for each country and compute real GDP
gdp_data = {}

for sheet in sheet_names:
    # Load nominal GDP data for the country
    data_gdp = pd.read_excel(gdp_path, sheet_name=sheet)
    country_dates_gdp = pd.to_datetime(data_gdp.iloc[:, 0])
    gdp_values = data_gdp['GDP'].values

    # Create a Series for nominal GDP indexed by country-specific dates
    nominal_gdp_series = pd.Series(data=gdp_values, index=country_dates_gdp)

    # Reindex to align with the global quarterly dates using forward fill
    nominal_gdp_series = nominal_gdp_series.reindex(global_dates_gdp, method='ffill')

    # Extract the CPI series for this country (assumes column name equals sheet name)
    cpi_series = cpi_data[sheet]
    cpi_series.index = pd.to_datetime(cpi_series.index)
    cpi_series = cpi_series.reindex(global_dates_gdp, method='ffill')

    # Convert nominal GDP to real GDP (assuming CPI base = 100)
    real_gdp_series = nominal_gdp_series / (cpi_series / 100.0)

    # Store the real GDP series in the dictionary
    gdp_data[sheet] = real_gdp_series

### Business Cycle

# Define the global quarterly dates based on the common time range
global_dates_bc = pd.date_range(global_start_date, global_end_date, freq='QS')

# Resample each country's data to align with global quarterly dates
for sheet in sheet_names:
    # Load the data for the current sheet (country)
    data_bc = pd.read_excel(excel_data, sheet_name=sheet)

    # Extract the dates and FCycle columns
    country_dates_bc = pd.to_datetime(data_bc.iloc[:, 0])  # Date column
    bcycle_values = data_bc['BCycle'].values  # FCycle column

    # Convert FCycle values to a pandas Series indexed by country-specific dates
    country_series_bc = pd.Series(data=bcycle_values, index=country_dates_bc)

    # Reindex to align with the global quarterly dates, forward filling to handle missing data
    resampled_series_bc = country_series_bc.reindex(global_dates_bc, method='ffill')

    # Store the resampled series in the dictionary
    resampled_series_data_bc[sheet] = resampled_series_bc.values

# Convert the resampled data into a matrix for easier processing
num_countries_bc = len(sheet_names)
num_quarters_bc = len(global_dates_bc)
time_series_matrix_bc = np.vstack([resampled_series_data_bc[sheet] for sheet in sheet_names])

# Initialize matrices to store mean DTW-based asymmetry for each country at each time point
asymmetry_matrix_bc = np.full((num_countries_bc, num_quarters_bc), np.nan)
pairwise_asymmetry_matrix_bc = np.full((num_countries_bc, num_countries_bc, num_quarters_bc), np.nan)

for country_idx_bc in range(num_countries_bc):
    # First valid data point for this country
    country_start_idx = np.where(~np.isnan(time_series_matrix_bc[country_idx_bc, :]))[0][0]

    for t in range(country_start_idx, num_quarters_bc):
        weighted_dtw_distances_bc = []
        weights_bc = []

        for other_country_idx_bc in range(num_countries_bc):
            if other_country_idx_bc == country_idx_bc:
                continue

            # Get the start index for the other country
            other_valid_indices = np.where(~np.isnan(time_series_matrix_bc[other_country_idx_bc, :]))[0]
            if len(other_valid_indices) == 0:
                continue
            other_start_idx = other_valid_indices[0]

            # Retrieve cycle durations (assumed to be in quarters)
            avg_cycle_dur1 = cycle_avg.loc[sheet_names[country_idx_bc], "Min_Distance_BCycle"]
            avg_cycle_dur2 = cycle_avg.loc[sheet_names[other_country_idx_bc], "Min_Distance_BCycle"]

            # Compute average cycle (in quarters)
            avg_cycle_bc = int(((avg_cycle_dur1 * 4) + (avg_cycle_dur2 * 4)) / 2)

            # Determine the earliest eligible time index for computing DTW:
            required_t = max(country_start_idx, other_start_idx) + avg_cycle_bc

            # Only compute DTW if we've reached the required time index:
            if t < required_t:
                continue

            # Define the window as the last 'avg_cycle' observations ending at time t.
            window_length = avg_cycle_bc
            window_start = t - window_length + 1

            # Extract the local windows from both series and remove NaNs
            ts_country = time_series_matrix_bc[country_idx_bc, window_start: t + 1]
            ts_other = time_series_matrix_bc[other_country_idx_bc, window_start: t + 1]
            ts_country = ts_country[~np.isnan(ts_country)]
            ts_other = ts_other[~np.isnan(ts_other)]

            # Compute DTW only if both windows have data
            if len(ts_country) > 0 and len(ts_other) > 0:
                dtw_distance = dtw(ts_country, ts_other)

                # Store the individual pairwise DTW distance
                pairwise_asymmetry_matrix_bc[country_idx_bc, other_country_idx_bc, t] = dtw_distance

                # Store the weighted distance
                weighted_dtw_distances_bc.append(dtw_distance)

                # Use GDP for weighting
                gdp_focal = gdp_data[sheet_names[country_idx_bc]].iloc[t]
                gdp_other = gdp_data[sheet_names[other_country_idx_bc]].iloc[t]
                combined_weight = gdp_focal + gdp_other
                weights_bc.append(combined_weight)

        # Normalize the weights and store the weighted average
        if weights_bc:
            weights_array = np.array(weights_bc)
            normalized_weights_bc = np.array(weights_array) / np.sum(weights_array)
            dtw_array = np.array(weighted_dtw_distances_bc).flatten()
            asymmetry_matrix_bc[country_idx_bc, t] = np.sum(dtw_array * normalized_weights_bc)

# Find the first non-NaN entry for each country in the asymmetry_matrix
first_data_indices_bc = [
    np.where(~np.isnan(asymmetry_matrix_bc[country_idx_bc, :]))[0][0]
    for country_idx_bc in range(num_countries_bc)
]

# Sort countries by their first available DTW date
sorted_indices_bc = np.argsort(first_data_indices_bc)

# Reorder the asymmetry matrix and country names for plotting based on sorted indices
sorted_asymmetry_matrix_bc = asymmetry_matrix_bc[sorted_indices_bc]
sorted_sheet_names_bc = [sheet_names[idx_bc] for idx_bc in sorted_indices_bc]

smoothed_matrix_bc = np.copy(sorted_asymmetry_matrix_bc)

# Apply a Gaussian filter along the time axis
for i in range(smoothed_matrix_bc.shape[0]):
    row = smoothed_matrix_bc[i, :]
    smoothed_row = gaussian_filter(row, sigma=8, mode='nearest')
    smoothed_matrix_bc[i, :] = smoothed_row

asymmetry_matrix_plot_bc = np.nan_to_num(sorted_asymmetry_matrix_bc, nan=0)
asymmetry_matrix_plot_bc = np.ma.masked_equal(asymmetry_matrix_plot_bc, 0)

# Convert dates to years for labeling on the x-axis
year_labels_bc = [d.year for d in global_dates_bc]  # Extract only the year for each quarter
xaxis_ticks_bc = np.arange(0, num_quarters_bc, 10)
xaxis_labels_bc = [year_labels_bc[i] for i in xaxis_ticks_bc]

# Set up for 3D surface plot with sorted data
x_axis_bc = np.arange(num_quarters_bc)  # Time on the x-axis
y_axis_bc = sorted_sheet_names_bc  # Use sorted country names on the y-axis

# Create the meshgrid for the 3D surface plot
x_grid_bc, y_grid_bc = np.meshgrid(np.arange(num_quarters_bc), np.arange(num_countries_bc))

darker_blues = [
    [0.0, 'rgb(158, 202, 225)'],
    [0.2, 'rgb(107, 174, 214)'],
    [0.4, 'rgb(66, 146, 198)'],
    [0.6, 'rgb(33, 113, 181)'],
    [0.8, 'rgb(8, 69, 148)'],
    [1.0, 'rgb(8, 48, 107)']
]

# Create the 3D surface plot using Plotly
fig_bus = go.Figure(data=[go.Surface(z=smoothed_matrix_bc,
                                 x=x_grid_bc, y=y_grid_bc,
                                 colorscale=darker_blues,
                                 colorbar=dict(
                                        title=dict(text="",
                                                   font=dict(size=14, color='grey')
                                        ),  
                                        tickfont=dict(size=12, color='grey'),   
                                        thickness=15,  
                                        len=0.4       
                                    ),
                                 )],
                                 template=None
                   )
# Remove the 'titleside' property if it exists
cb = fig_bus.data[0].colorbar
if "titleside" in cb:
    del cb["titleside"]

# Customize layout
fig_bus.update_layout(
    plot_bgcolor='white',    
    paper_bgcolor='white', 
    title=dict(
        text="Business Cycle Divergences",
        x=0.5,              
        xanchor="center",    
        font=dict(color='black')
    ),
    scene=dict(
        xaxis_title=dict(text=""),
        xaxis=dict(
            tickvals=xaxis_ticks_bc, 
            ticktext=xaxis_labels_bc,
            showgrid=True,
            gridcolor='lightgrey',
            tickfont=dict(color='grey') 
        ),
        yaxis_title=dict(text=""),
        yaxis=dict(
            tickvals=np.arange(num_countries_bc), 
            ticktext=y_axis_bc,
            showgrid=True,
            gridcolor='lightgrey',
            tickfont=dict(color='grey') 
        ),
        zaxis=dict(
            title="Mean DTW Distance",  
            titlefont=dict(color="grey"),
            showgrid=True,
            gridcolor='lightgrey',
            tickfont=dict(color='grey')  
        )
    ),
    autosize=False,
    margin=dict(l=0, r=0, t=0, b=0),
    width=600,
    height=500,
)

bus_cycle_3d.plotly_chart(fig_bus, use_container_width=False)

### Financial Cycle 

# Define the global quarterly dates based on the common time range
global_dates_fc = pd.date_range(global_start_date, global_end_date, freq='QS')

# Resample each country's data to align with global quarterly dates
for sheet in sheet_names:
    # Load the data for the current sheet (country)
    data_fc = pd.read_excel(excel_data, sheet_name=sheet)

    # Extract the dates and FCycle columns
    country_dates_fc = pd.to_datetime(data_fc.iloc[:, 0])  # Date column
    fcycle_values = data_fc['FCycle'].values  # FCycle column

    # Convert FCycle values to a pandas Series indexed by country-specific dates
    country_series_fc = pd.Series(data=fcycle_values, index=country_dates_fc)

    # Reindex to align with the global quarterly dates, forward filling to handle missing data
    resampled_series_fc = country_series_fc.reindex(global_dates_fc, method='ffill')

    # Store the resampled series in the dictionary
    resampled_series_data_fc[sheet] = resampled_series_fc.values

# Convert the resampled data into a matrix for easier processing
num_countries_fc = len(sheet_names)
num_quarters_fc = len(global_dates_fc)
time_series_matrix_fc = np.vstack([resampled_series_data_fc[sheet] for sheet in sheet_names])

# Initialize matrices to store mean DTW-based asymmetry for each country at each time point
asymmetry_matrix_fc = np.full((num_countries_fc, num_quarters_fc), np.nan)
pairwise_asymmetry_matrix_fc = np.full((num_countries_fc, num_countries_fc, num_quarters_fc), np.nan)

for country_idx_fc in range(num_countries_fc):
    # First valid data point for this country
    country_start_idx = np.where(~np.isnan(time_series_matrix_fc[country_idx_fc, :]))[0][0]

    for t in range(country_start_idx, num_quarters_fc):
        weighted_dtw_distances_fc = []
        weights_fc = []

        for other_country_idx_fc in range(num_countries_fc):
            if other_country_idx_fc == country_idx_fc:
                continue

            # Get the start index for the other country
            other_valid_indices = np.where(~np.isnan(time_series_matrix_fc[other_country_idx_fc, :]))[0]
            if len(other_valid_indices) == 0:
                continue
            other_start_idx = other_valid_indices[0]

            # Retrieve cycle durations (assumed to be in quarters)
            avg_cycle_dur1 = cycle_avg.loc[sheet_names[country_idx_fc], "Min_Distance_FCycle"]
            avg_cycle_dur2 = cycle_avg.loc[sheet_names[other_country_idx_fc], "Min_Distance_FCycle"]

            # Compute average cycle (in quarters)
            avg_cycle_fc = int(((avg_cycle_dur1 * 4) + (avg_cycle_dur2 * 4)) / 2)

            # Determine the earliest eligible time index for computing DTW:
            required_t = max(country_start_idx, other_start_idx) + avg_cycle_fc

            # Only compute DTW if we've reached the required time index:
            if t < required_t:
                continue

            # Define the window as the last 'avg_cycle' observations ending at time t.
            window_length = avg_cycle_fc  # window_length is the average cycle in quarters
            window_start = t - window_length + 1

            # Extract the local windows from both series and remove NaNs
            ts_country = time_series_matrix_fc[country_idx_fc, window_start: t + 1]
            ts_other = time_series_matrix_fc[other_country_idx_fc, window_start: t + 1]
            ts_country = ts_country[~np.isnan(ts_country)]
            ts_other = ts_other[~np.isnan(ts_other)]

            # Compute DTW only if both windows have data
            if len(ts_country) > 0 and len(ts_other) > 0:
                dtw_distance = dtw(ts_country, ts_other)

                 # Store the individual pairwise DTW distance
                pairwise_asymmetry_matrix_fc[country_idx_fc, other_country_idx_fc, t] = dtw_distance

                # Store the weighted distance
                weighted_dtw_distances_fc.append(dtw_distance)

                # Use GDP for weighting
                gdp_focal = gdp_data[sheet_names[country_idx_fc]].iloc[t]
                gdp_other = gdp_data[sheet_names[other_country_idx_fc]].iloc[t]
                combined_weight = gdp_focal + gdp_other
                weights_fc.append(combined_weight)

        # Normalize the weights and store the weighted average
        if weights_fc:
            weights_array = np.array(weights_fc)
            normalized_weights_fc = np.array(weights_array) / np.sum(weights_array)
            dtw_array = np.array(weighted_dtw_distances_fc).flatten()
            asymmetry_matrix_fc[country_idx_fc, t] = np.sum(dtw_array * normalized_weights_fc)

# Find the first non-NaN entry for each country in the asymmetry_matrix
first_data_indices_fc = [
    np.where(~np.isnan(asymmetry_matrix_fc[country_idx_fc, :]))[0][0]
    for country_idx_fc in range(num_countries_fc)
]

# Sort countries by their first available DTW date
sorted_indices_fc = np.argsort(first_data_indices_fc)

# Reorder the asymmetry matrix and country names for plotting based on sorted indices
sorted_asymmetry_matrix_fc = asymmetry_matrix_fc[sorted_indices_fc]
sorted_sheet_names_fc = [sheet_names[idx_f] for idx_f in sorted_indices_fc]

asymmetry_matrix_plot_fc = np.nan_to_num(sorted_asymmetry_matrix_fc, nan=0)
asymmetry_matrix_plot_fc = np.ma.masked_equal(asymmetry_matrix_plot_fc, 0)

smoothed_matrix_fc = np.copy(sorted_asymmetry_matrix_fc)

# Apply a Gaussian filter along the time axis
for i in range(smoothed_matrix_fc.shape[0]):
    row = smoothed_matrix_fc[i, :]
    smoothed_row = gaussian_filter(row, sigma=8, mode='nearest')
    smoothed_matrix_fc[i, :] = smoothed_row

# Convert dates to years for labeling on the x-axis
year_labels_fc = [d.year for d in global_dates_fc]  # Extract only the year for each quarter
xaxis_ticks_fc = np.arange(0, num_quarters_fc, 10)
xaxis_labels_fc = [year_labels_fc[i] for i in xaxis_ticks_fc]

# Set up for 3D surface plot with sorted data
x_axis_fc = np.arange(num_quarters_fc)  # Time on the x-axis
y_axis_fc = sorted_sheet_names_fc  # Use sorted country names on the y-axis

# Create the meshgrid for the 3D surface plot
x_grid_fc, y_grid_fc = np.meshgrid(np.arange(num_quarters_fc), np.arange(num_countries_fc))

# Create the 3D surface plot using Plotly
darker_purples = [
    [0.0, 'rgb(188, 189, 220)'],
    [0.2, 'rgb(158, 154, 200)'],
    [0.4, 'rgb(128, 125, 186)'],
    [0.6, 'rgb(106, 81, 163)'],
    [0.8, 'rgb(92, 53, 153)'],
    [1.0, 'rgb(74, 20, 134)']
]

fig_fin = go.Figure(data=[go.Surface(z=smoothed_matrix_fc,
                                 x=x_grid_fc, y=y_grid_fc,
                                 colorscale=darker_purples,
                                 colorbar=dict(
                                        title=dict(text="",
                                                   font=dict(size=14, color='grey'), 
                                        ),
                                        tickfont=dict(size=12, color='grey'),   
                                        thickness=15,  
                                        len=0.4   
                                 )],
                                 template=None
                   ))

# Customize layout
fig_fin.update_layout(
    plot_bgcolor='white',    
    paper_bgcolor='white', 
    title=dict(
        text="Financial Cycle Divergences",
        x=0.5,              
        xanchor="center",    
        font=dict(color='black')
    ),
    scene=dict(
        xaxis_title=dict(text=""),
        xaxis=dict(
            tickvals=xaxis_ticks_bc, 
            ticktext=xaxis_labels_bc,
            showgrid=True,
            gridcolor='lightgrey',
            tickfont=dict(color='grey') 
        ),
        yaxis_title=dict(text=""),
        yaxis=dict(
            tickvals=np.arange(num_countries_bc), 
            ticktext=y_axis_bc,
            showgrid=True,
            gridcolor='lightgrey',
            tickfont=dict(color='grey') 
        ),
        zaxis=dict(
            title="Mean DTW Distance",  
            titlefont=dict(color="grey"),
            showgrid=True,
            gridcolor='lightgrey',
            tickfont=dict(color='grey')
        )  
    ),
    autosize=False,
    width=600,
    height=500,
    margin=dict(l=0, r=0, t=0, b=0)
)

# Show the plot
fin_cycle_3d.plotly_chart(fig_fin, use_container_width=False)  

# Compute the symmetry indices by averaging
index_bc   = np.nanmean(asymmetry_matrix_bc, axis=0)
index_fc   = np.nanmean(asymmetry_matrix_fc, axis=0)

composite_divergence_series_mean = np.nanmean([index_fc,
                                               index_bc,
                                               ], axis=0)

## Create a mask for dates >= 1983-01-01
start_date = pd.Timestamp('1985-01-01')
mask = global_dates_bc >= start_date

# Slice each array so we only keep data from Q1 1983 onward
time_axis = global_dates_bc[mask]
index_fc_subset = index_fc[mask]
index_bc_subset = index_bc[mask]
divergence_index_subset = composite_divergence_series_mean[mask]

# Plot the sliced data
fig_index = go.Figure()

fig_index.add_trace(go.Scatter(
    x=time_axis,
    y=index_fc_subset,
    mode='lines',
    line=dict(color='rgb(106, 81, 163)', dash='dot'),
    opacity=0.4,
    name='Financial Divergence',
))

fig_index.add_trace(go.Scatter(
    x=time_axis,
    y=index_bc_subset,
    mode='lines',
    line=dict(color='rgb(33, 113, 181)', dash='dot'),
    opacity=0.4,
    name='Business Divergence'
))

fig_index.add_trace(go.Scatter(
    x=time_axis,
    y=divergence_index_subset,
    mode='lines',
    line=dict(color='rgb(0, 0, 0)'),
    name='Divergence Index',
))

df_events = pd.read_excel(path+"/Events.xlsx")
df_events.columns = df_events.columns.str.strip().str.replace(" ", "")


df_events['Date'] = pd.to_datetime(df_events['Date'])
df_events = df_events[df_events['Date'] >= start_date]

esm_df = df_events[df_events['ESM_Short'].notnull() & (df_events['ESM_Short'] != "")]
esm_df = esm_df[esm_df['Date'] >= start_date]

ump_df = df_events[df_events['UMP_Short'].notnull() & (df_events['UMP_Short'] != "")]
ump_df = ump_df[ump_df['Date'] >= start_date]

# (From the min to max of all three series in your subset)
y_min = min(index_fc_subset.min(), index_bc_subset.min(), divergence_index_subset.min())
y_max = max(index_fc_subset.max(), index_bc_subset.max(), divergence_index_subset.max())

# ---------------------------
# Build a single scatter trace for ESM events
# ---------------------------
x_esm = []
y_esm = []
text_esm = []
hovertext_esm = []
marker_sizes_esm = []
textpositions_esm = []


for row in esm_df.itertuples():
    event_date = row.Date
    short_label = row.ESM_Short        # no separate short vs. long in your data
    long_label = row.ESM_Long        # same as short, or customize if you have another column

    # Each event = two points + None to break the line
    # First point (bottom), second point (top), None to start new line
    x_esm += [event_date, event_date, None]
    y_esm += [y_min, y_max, None]

    # Decide text position based on label
    if short_label == "PT, ES":
        pos = "top right"
    else:
        pos = "top center" 

    # Put the label on the top point only
    text_esm      += ["", short_label, None]
    textpositions_esm += ["top center", pos, "top center"]
    hovertext_esm += ["", long_label, None]
    marker_sizes_esm += [0, 0, 0]

esm_trace = go.Scatter(
    x=x_esm,
    y=y_esm,
    mode='lines+markers+text',
    line=dict(color='lightgrey', dash='dash'),
    marker=dict(color=None, size=marker_sizes_esm),
    text=text_esm,
    textfont=dict(color='black'),
    textposition=textpositions_esm,
    hovertext=hovertext_esm,
    hovertemplate="%{hovertext}<extra></extra>",
    name='ESM Events',
    visible=False  # start hidden
)

x_ump = []
y_ump = []
text_ump = []
hovertext_ump = []
marker_sizes_ump = []
textpositions_ump = []


for row in ump_df.itertuples():
    event_date = row.Date
    # We retrieve columns by attribute or asdict:
    short_label = row._asdict()['UMP_Short']
    long_label = row._asdict()['UMP_Long']

    x_ump += [event_date, event_date, None]
    y_ump += [y_min, y_max, None]

    # Decide text position based on label
    if short_label == "APP":
        pos = "top left"
    elif short_label == "PSPP":
        pos = "top right"
    else:
        pos = "top center"  

    # Put the label on the top point only
    text_ump      += ["", short_label, None]
    textpositions_ump += ["top center", pos, "top center"]
    hovertext_ump += ["", long_label, None]
    marker_sizes_ump += [0, 0, 0]


ump_trace = go.Scatter(
    x=x_ump,
    y=y_ump,
    mode='lines+markers+text',
    line=dict(color='lightgrey', dash='dash'),
    text=text_ump,
    textfont=dict(color='black'),
    marker=dict(color=None, size=marker_sizes_ump),
    textposition=textpositions_ump,
    hovertext=hovertext_ump,
    hovertemplate="%{hovertext}<extra></extra>",
    name='UMP Events',
    visible=False  # start hidden
)

# Add the two event traces
fig_index.add_trace(esm_trace)
fig_index.add_trace(ump_trace)

df_recession = pd.read_excel(path + "/Recession_OCED.xlsx")
df_recession.columns = df_recession.columns.str.strip().str.replace(" ", "")
df_recession["observation_date"] = pd.to_datetime(df_recession["observation_date"])

# Filter to include only recession periods (assuming EUROREC==1) and dates after start_date
df_rec = df_recession[(df_recession["EUROREC"] == 1) & 
                      (df_recession["observation_date"] >= start_date)].copy()

if not df_rec.empty:
    df_rec['group'] = (df_rec['observation_date'].diff() > pd.Timedelta(days=95)).cumsum()
    
    recession_periods = df_rec.groupby('group')['observation_date'].agg(['min','max']).reset_index()
    
    recession_shapes = []
    for _, row in recession_periods.iterrows():
        recession_shapes.append(dict(
            type="rect",
            xref="x",
            yref="paper",  # spans the full vertical area of the plot
            x0=row['min'],
            x1=row['max'],
            y0=0,
            y1=1,
            fillcolor="grey",
            opacity=0.1,
            layer="below",
            line_width=0,
        ))
    # Add the recession shapes to your figure's layout
    fig_index.update_layout(shapes=recession_shapes)

recession_indicator_trace = go.Scatter(
    xaxis='x2', 
    yaxis='y2',
    x=[-1.5],               
    y=[1], 
    mode='lines',
    line=dict(color='lightgrey', width=6),  
    name='OECD Recession Indicator',
    showlegend=True,     
    visible=True        
)

# Add the dummy trace to your figure so it appears in the legend
fig_index.add_trace(recession_indicator_trace)

fig_index.update_layout(
    updatemenus=[
        dict(
            active=2,  
            type="buttons",
            direction="right",
            x=0,
            xanchor="left",
            y=1.1,
            showactive=True,
            font=dict(color='black'),
            buttons=[
                dict(
                    label="Countries joining ESM",
                    method="update",
                    args=[{"visible": [True, True, True, True, False,]}],
                ),
                dict(
                    label="Unconventional Monetary Policy",
                    method="update",
                    args=[{"visible": [True, True, True, False, True,]}],
                ),
                dict(
                    label="Hide Events",
                    method="update",
                    args=[{"visible": [True, True, True, False, False,]}],
                ),
            ],
        )
    ]
)


fig_index.update_layout(
        xaxis2=dict(
        visible=False,         
        overlaying='x',        
        side='top',           
        range=[0,1]           
    ),
    yaxis2=dict(
        visible=False,
        overlaying='y',
        side='right',
        range=[0,1]
    ),
    legend=dict(
        x=0.5,
        y=-0.2,
        xanchor="center",
        yanchor="bottom",
        orientation="h",
        bgcolor='rgba(255,255,255,0.2)',
        font=dict(color='black')  
    ),
    title="",
    xaxis_title="",
    autosize=False,
    width=1000,
    height=600,
    xaxis=dict(
        tickformat='%Y',
        tickfont=dict(color='black'),  
        showgrid=True,
        gridcolor='whitesmoke',
    ),
    yaxis=dict(
        tickfont=dict(color='black'), 
        showgrid=True,
        gridcolor='whitesmoke', 
        title="Mean DTW Distance",  
        titlefont=dict(color="grey")
    ),
    plot_bgcolor='white',  
    paper_bgcolor='white',  
    #template='plotly_white',
)

fig_divergence_monitor.plotly_chart(fig_index, use_container_width=True)


# Create a DataFrame
index_data = pd.DataFrame({
        'Date': global_dates_bc,
        'Financial Cycle': index_fc,
        'Business Cycle': index_bc,
        'Divergence Index': composite_divergence_series_mean
})

csv_index_data = index_data.to_csv(index=False).encode('utf-8')

# Add CSS to center the button
st.markdown(
    """
    <style>
    .stDownloadButton > button {
            background-color: #f0f0f0 !important; 
            color: #555555 !important;            
            border-radius: 5px;                   
            border: 1px solid #ccc;               
        }
    </style>
    """,
    unsafe_allow_html=True
)

with button_index.container():
    col1, col2, col3 = st.columns([2.5, 2, 1])  
    with col2:
        st.download_button(
            label="Download Data",
            data=csv_index_data,
            file_name="divergence_data.csv",
            mime="text/csv"
        )


# Load the cycle data for all countries (from all sheets in the Excel file)
cycle_data = {}
for country in sheet_names:
    df_cycle = pd.read_excel(excel_data, sheet_name=country)
    # Convert the first column to datetime (assumed to be the date column)
    df_cycle['Date'] = pd.to_datetime(df_cycle.iloc[:, 0])
    cycle_data[country] = df_cycle

# Determine the number of countries
n = len(sheet_names)

if n > 1:
    sample_positions = [0.2 + 0.7 * i/(n-1) for i in range(n)]
else:
    sample_positions = [0.5]

blue_colors = px.colors.sample_colorscale("Blues", sample_positions)
purple_colors = px.colors.sample_colorscale("Purples", sample_positions)

with col1_2d:
    fig_bc = go.Figure()
    for idx, country in enumerate(sheet_names):
        df = cycle_data[country]
        fig_bc.add_trace(go.Scatter(
            x=df['Date'],
            y=df['BCycle'],
            mode='lines',
            name=country,
            line=dict(color=blue_colors[idx])
        ))
    fig_bc.update_layout(
        title=dict(
        text="Business Cycles",
        x=0.5,              
        xanchor="center",    
        font=dict(color='black')
    ),
        xaxis=dict(
            tickfont=dict(color='black'),
            showgrid=True,      
            gridcolor='lightgrey',
            gridwidth=1,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Deviation from historical median growth", font=dict(color='grey')),
            tickfont=dict(color='grey'),
            showgrid=True,      
            gridcolor='lightgrey',
            gridwidth=1,
            zeroline=False
        ),
        #template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        title_font=dict(color='black'),
        legend=dict(font=dict(color='black'))
    )
    st.plotly_chart(fig_bc, use_container_width=True)

with col2_2d:
    fig_fc = go.Figure()
    for idx, country in enumerate(sheet_names):
        df = cycle_data[country]
        fig_fc.add_trace(go.Scatter(
            x=df['Date'],
            y=df['FCycle'],
            mode='lines',
            name=country,
            line=dict(color=purple_colors[idx])
        ))
    fig_fc.update_layout(
        title=dict(
        text="Financial Cycles",
        x=0.5,              
        xanchor="center",    
        font=dict(color='black')
    ),
        xaxis=dict(
            tickfont=dict(color='black'),
            showgrid=True,       
            gridcolor='lightgrey',
            gridwidth=1,
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Deviation from historical median growth", font=dict(color='grey')),
            tickfont=dict(color='grey'),
            showgrid=True,      
            gridcolor='lightgrey',
            gridwidth=1,
            zeroline=False
        ),
        #template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        title_font=dict(color='black'),
        legend=dict(font=dict(color='black'))
    )
    st.plotly_chart(fig_fc, use_container_width=False)
