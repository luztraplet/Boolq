import os

from dash import Dash

from src.domain.models import load_models
from src.domain.stories import load_stories
from src.framework.callbacks import add_callbacks
from src.framework.ui import add_layout

app = Dash(__name__,
           update_title='Loading models...',
           external_stylesheets=[os.path.join(".", "assets", "sandstone.css"),
                                 os.path.join(".", "assets", "custom.css")],
           meta_tags=[{'name': 'viewport',
                       'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}])

server = app.server

app.stories = load_stories()
add_layout(app)
add_callbacks(app)
app.models = load_models()

if __name__ == "__main__":
    app.run_server(host="127.0.0.1")
