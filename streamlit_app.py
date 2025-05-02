import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

#################################### CODE WEB #################################################

st.set_page_config(
    page_title="Eurozone Divergence Monitor",
    layout="wide",
)

st.markdown(
    """
     <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
    .title {
        text-align: center; 
        margin-bottom: 10px;
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
        max-width: 65vw !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    @media (max-width: 400px) {
        [data-testid="stMarkdown"] {
            max-width: 85vw !important;
        }
    }
    [data-testid="column"] {
    overflow: visible !important;
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
    
    Since Mario Draghi's <a href="https://www.ecb.europa.eu/press/key/date/2012/html/sp120726.en.html">"Whatever it takes"</a> speech—when divergence 
    was at a peak in mid-2012 (see OMT marker above)—monetary policy in the Eurozone is driven by measures seeking to hold the eurozone together. 
    Among other things, we hope that our monitor can help economists and policymakers in the assessment of such measures. The monitor is measured on a 
    quarterly basis and will be updated regularly. The <b>next update</b> is scheduled for <b>August, 2025</b>.
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
    bond prices for the financial cycle. For more details, see our paper <a href="#ref-bugdalle-pfeifer-2025">Bugdalle &amp; Pfeifer (2025)</a>. 
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
        In our case, DTW is applied to each type of the smoothed cycle indices, meaning one measure of similarity is estimated for each type of cycle. Within each cycle category, 
        DTW computes the alignment path $\mathbf{\pi}_{ij}$ for each pair of countries $i$ and $j$, that minimizes the cumulative distance between two cycles:
        """
    )

st.latex(r'''
D(\mathbf{x}_i, \mathbf{x}_j) = \min_{\pi_{ij}} \sum_{(t, s) \in \pi_{ij}} \left| \mathbf{x}_{i,t} - \mathbf{x}_{j,s} \right|^2,
''')

st.markdown(
    r"""
    where, $\mathbf{x}_{i,t}$ and $\mathbf{x}_{j,s}$ are the smoothed cycle values at time $t$ and time $s$ for countries $i$ and $j$, respectively.
    The resulting distance $D(\mathbf{x}_i, \mathbf{x}_j)$ captures the degree of similarity, with smaller values indicating greater similarity between the cycles. 
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
        Bugdalle, T., Pfeifer, M. (2025). Warpings in time: Business and financial cycle
        synchronization in the euro area. arXiv preprint.
        </p>
        <p>
        <a id="ref-sakoe-1978"></a>
        Sakoe, H., Chiba, S. (1978). Dynamic programming algorithm optimization 
        for spoken word recognition. <em>IEEE Transactions on Acoustics, Speech, 
        and Signal Processing, 26</em>(1), 43–49.
        </p>
        <p>
        <a id="ref-mundell-1961"></a>
        Mundell, R. (1961). A theory of optimal currency areas. 
        <em>American Economic Review, 51</em>(4), 657–665.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)



#################################### CODE PLOTS #################################################
# Load files
base_dir = os.path.dirname(__file__)

precomputed_path = os.path.join(base_dir, 'Data_Final', 'Data_web', 'precomputed_data.pkl')

if os.path.exists(precomputed_path):
    with open(precomputed_path, 'rb') as f:
        precomputed = pickle.load(f)
    asymmetry_matrix_bc = precomputed["asymmetry_matrix_bc"]
    asymmetry_matrix_fc = precomputed["asymmetry_matrix_fc"]
    index_bc   = precomputed["index_bc"]
    index_fc   = precomputed["index_fc"]
    composite_divergence_series_mean = precomputed["composite_divergence_series_mean"]
    global_dates_bc = precomputed["global_dates_bc"]
    sorted_sheet_names_bc = precomputed["sorted_sheet_names_bc"]
    smoothed_matrix_bc = precomputed["smoothed_matrix_bc"] 
    smoothed_matrix_fc = precomputed["smoothed_matrix_fc"] 
    x_grid_fc = precomputed["x_grid_fc"]
    x_grid_bc = precomputed["x_grid_bc"] 
    y_grid_fc = precomputed["y_grid_fc"] 
    y_grid_bc = precomputed["y_grid_bc"] 
    xaxis_ticks_bc = precomputed["xaxis_ticks_bc"]
    xaxis_ticks_fc = precomputed["xaxis_ticks_fc"]
    xaxis_labels_bc = precomputed["xaxis_labels_bc"]
    xaxis_labels_fc = precomputed["xaxis_labels_fc"]
    num_countries_bc = precomputed["num_countries_bc"]
    num_countries_fc = precomputed["num_countries_fc"]
    num_quarters_bc = precomputed["num_quarters_bc"]
    num_quarters_fc = precomputed["num_quarters_fc"]
    y_axis_bc = precomputed["y_axis_bc"]
    y_axis_fc = precomputed["y_axis_fc"]


# Business cycle 
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
                                    ))])
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
            title=dict(text="Mean DTW Distance",  
                      font=dict(color="grey")),
            showgrid=True,
            gridcolor='lightgrey',
            tickfont=dict(color='grey')  
        )
    ),
    width=1000,
    height=700,
)

bus_cycle_3d.plotly_chart(fig_bus, use_container_width=False)

### Financial Cycle 
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
                                 ))])

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
            title=dict(text="Mean DTW Distance", 
                       font=dict(color="grey")),
            showgrid=True,
            gridcolor='lightgrey',
            tickfont=dict(color='grey')
        )  
    ),
    width=1000,
    height=700,
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
    opacity=0.5,
    name='Financial Divergence',
))

fig_index.add_trace(go.Scatter(
    x=time_axis,
    y=index_bc_subset,
    mode='lines',
    line=dict(color='rgb(33, 113, 181)', dash='dot'),
    opacity=0.5,
    name='Business Divergence'
))

fig_index.add_trace(go.Scatter(
    x=time_axis,
    y=divergence_index_subset,
    mode='lines',
    line=dict(color='rgb(0, 0, 0)'),
    name='Divergence Index',
))

df_events_path = os.path.join(base_dir, 'Data_Final', 'Events.xlsx')
df_events = pd.read_excel(df_events_path)
df_events.columns = df_events.columns.str.strip().str.replace(" ", "")


df_events['Date'] = pd.to_datetime(df_events['Date'])
df_events = df_events[df_events['Date'] >= start_date]

erm_df = df_events[df_events['ERM_Short'].notnull() & (df_events['ERM_Short'] != "")]
erm_df = erm_df[erm_df['Date'] >= start_date]

ump_df = df_events[df_events['UMP_Short'].notnull() & (df_events['UMP_Short'] != "")]
ump_df = ump_df[ump_df['Date'] >= start_date]

# (From the min to max of all three series in your subset)
y_min = min(index_fc_subset.min(), index_bc_subset.min(), divergence_index_subset.min())
y_max = max(index_fc_subset.max(), index_bc_subset.max(), divergence_index_subset.max())

# ---------------------------
# Build a single scatter trace for ERM events
# ---------------------------
x_erm = []
y_erm = []
text_erm = []
hovertext_erm = []
marker_sizes_erm = []
textpositions_erm = []


for row in erm_df.itertuples():
    event_date = row.Date
    short_label = row.ERM_Short        # no separate short vs. long in your data
    long_label = row.ERM_Long        # same as short, or customize if you have another column

    # Each event = two points + None to break the line
    # First point (bottom), second point (top), None to start new line
    x_erm += [event_date, event_date, None]
    y_erm += [y_min, y_max, None]

    # Decide text position based on label
    if short_label == "PT, ES":
        pos = "top right"
    else:
        pos = "top center" 

    # Put the label on the top point only
    text_erm      += ["", short_label, None]
    textpositions_erm += ["top center", pos, "top center"]
    hovertext_erm += ["", long_label, None]
    marker_sizes_erm += [0, 0, 0]

erm_trace = go.Scatter(
    x=x_erm,
    y=y_erm,
    mode='lines+markers+text',
    line=dict(color='lightgrey', dash='dash'),
    marker=dict(color=None, size=marker_sizes_erm),
    text=text_erm,
    textfont=dict(color='black'),
    textposition=textpositions_erm,
    hovertext=hovertext_erm,
    hovertemplate="%{hovertext}<extra></extra>",
    name='ERM Events',
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
fig_index.add_trace(erm_trace)
fig_index.add_trace(ump_trace)

df_recession_path = os.path.join(base_dir, 'Data_Final', 'Recession_OCED.xlsx')
df_recession = pd.read_excel(df_recession_path)
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
                    label="Countries joining ERM",
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
    title=dict(text=""),
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
        title=dict(text="Mean DTW Distance",  
                  font=dict(color="grey")
        )
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
excel_path = os.path.join(base_dir, 'Data_Final', 'Country_Cycle_Data', 'Cycle_Data.xlsx')
# Load the Excel file
excel_data = pd.ExcelFile(excel_path, engine='openpyxl')
sheet_names = excel_data.sheet_names


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
        annotations=[
        dict(
            x=1.15,
            y=1.05,
            xref="paper",
            yref="paper",
            text="Click to select/deselect countries",
            showarrow=False,
            font=dict(size=12, color="black")
        )
        ],
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
        annotations=[
        dict(
            x=1.15,
            y=1.05,
            xref="paper",
            yref="paper",
            text="Click to select/deselect countries",
            showarrow=False,
            font=dict(size=12, color="black")
        )
    ],
        #template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='black'),
        title_font=dict(color='black'),
        legend=dict(font=dict(color='black'))
    )
    st.plotly_chart(fig_fc, use_container_width=False)


