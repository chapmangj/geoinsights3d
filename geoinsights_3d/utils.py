import pandas as pd

def create_html_download(fig, filename_prefix):
    """Create HTML download for plotly figures."""
    html_string = fig.to_html(include_plotlyjs='cdn')
    return html_string, f"{filename_prefix}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html"