import base64
import io
from io import StringIO
import pandas as pd
import holoviews as hv
import panel as pn
from holoviews import opts
from holoviews.operation.datashader import datashade
from holoviews.streams import Selection1D
from bokeh.models import ColumnDataSource


hv.extension('bokeh')
pn.extension()

def export_callback():

    # Dynamic map to update selected points
    selected_points = hv.DynamicMap(lambda index: get_selected_points(index), streams=[selection_stream])

    selected_df = df.iloc[selection_stream.index]

    # Build filename
    base, ext = file_input.filename.rsplit('.', 1)
    output_filename = f"{base}_selectedregion_{counter['index']}.{ext}"
    counter['index'] += 1
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    selected_df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    # Store processed bytes
    processed_data['bytes'] = buffer.read()

    return io.BytesIO(processed_data['bytes'])
    
    status.object = f"✅ Ready to download {len(selected_df)} points"

# Widgets
file_input = pn.widgets.FileInput(accept='.csv')

download_button = pn.widgets.FileDownload(
    label='Download processed file',
    button_type='success',
    auto=False,
    callback=export_callback
)

status = pn.pane.Markdown("")

# Display pane for the plot
plot_pane = pn.pane.HoloViews(height=500)

# State variables
df = None
selection_stream = None

counter = {'index': 1}  # Keeps track of the X in selectedregion_X
processed_data = {'bytes': None}  

def process_file(event):
    global df, selection_stream

    if file_input.value is None:
        return

    try:

        # Directly read file content (no Base64 decoding)
        decoded_bytes = file_input.value
        if isinstance(decoded_bytes, str):
            decoded_bytes = decoded_bytes.encode('utf-8')  # Ensure it's in byte format

        # Use BytesIO to load it into pandas (bypassing Base64 handling)
        file_like = io.BytesIO(decoded_bytes)

        # Try reading the CSV with latin1 encoding
        df = pd.read_csv(file_like, encoding='latin1', on_bad_lines='skip')

        # Clean column names (strip spaces, invisible characters)
        df.columns = [col.strip() for col in df.columns]
        df.columns = [col.replace('Âµ', 'µ') for col in df.columns]
        # Print cleaned column names for debugging
        print("Cleaned column names from loaded CSV:")
        print(df.columns.tolist())

        
        # Step 6: Check if required columns exist in the CSV
        if not {'Centroid X µm', 'Centroid Y µm'}.issubset(df.columns):
            status.object = "❌ CSV must contain 'Centroid X µm' and 'Centroid Y µm' columns."
            return


        base_points = hv.Points(df, kdims=["Centroid X µm", "Centroid Y µm"]).opts(
            alpha=0, size=1, tools=['box_select', 'lasso_select']
        )

        # Shaded (visual) points
        shaded = datashade(base_points).opts(
            opts.RGB(width=800, height=600)
        )

        # Selection stream tied to the base (transparent) points
        selection_stream = Selection1D(source=base_points)

        def get_selected_points(index):
            if index:
                selected = df.iloc[index]
                return hv.Points(selected, kdims=["Centroid X µm", "Centroid Y µm"]).opts(
                    color='red', size=8
                )
            return hv.Points([])

        # Dynamic map to update selected points
        selected_points = hv.DynamicMap(lambda index: get_selected_points(index), streams=[selection_stream])

        # Compose the plot: shaded + interactive layer + selected points
        plot_pane.object = shaded * base_points * selected_points
        
        status.object = "✅ File loaded. Use box/lasso to select points."

        # Build filename
        base, ext = file_input.filename.rsplit('.', 1)
        output_filename = f"{base}_selectedregion_{counter['index']}.{ext}"
        #counter['index'] = counter['index'] + 1

        # Update download button
        download_button.filename = output_filename

        
    except Exception as e:
        status.object = f"❌ Error loading file: {e}"

file_input.param.watch(process_file, 'value')

def export_callback(event):

    # Dynamic map to update selected points
    selected_points = hv.DynamicMap(lambda index: get_selected_points(index), streams=[selection_stream])

    selected_df = df.iloc[selection_stream.index]

    # Build filename
    base, ext = file_input.filename.rsplit('.', 1)
    output_filename = f"{base}_selectedregion_{counter['index']}.{ext}"
    counter['index'] += 1
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    selected_df.to_csv(buffer, index=False)
    buffer.seek(0)
    
    # Store processed bytes
    processed_data['bytes'] = buffer.read()
    
    # Update download button
    download_button.filename = output_filename
    
    
    
    status.object = f"✅ Exported {len(selected_df)} points to `selected_points.csv`."

#export_button.on_click(export_callback)

# Layout
app = pn.Column(
    "## Upload CSV and Select Points",
    file_input,
    plot_pane,
    status,
    download_button
)

app.servable()
