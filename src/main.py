import os

import dash.exceptions
import dash_bootstrap_components as dbc
from dash import Dash
from dash import Output, Input, State
from dash import html, dcc
from transformers import RobertaForSequenceClassification, RobertaTokenizer, BertForSequenceClassification, \
    BertTokenizer

from src.domain.models import boolq_model, para_model, is_solution, predict
from src.domain.stories import stories
from src.framework.ui import getChatBubble
from src.framework.ui import header, content, disclaimer

app = Dash(__name__,
           update_title='Loading models...',
           external_stylesheets=[os.path.join(".", "assets", "sandstone.css"),
                                 os.path.join(".", "assets", "custom.css")],
           meta_tags=[{'name': 'viewport',
                       'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}])

server = app.server

app.layout = dbc.Container(
    [
        header,
        content,
        disclaimer,
        dcc.Store(id='storage'),
        html.Div(id="garbage-output"),
        html.Div(id="garbage-input"),
    ]
)

app.clientside_callback(
    """
    function(clicks, elemid) {
        document.getElementById(elemid).scrollIntoView(alignToTop=false, {
          behavior: 'smooth'
        });
    }
    """,
    Output('garbage-output', 'children'),
    [Input('chat-content', 'children')],
    [State('scroll', 'id')]
)


@app.callback(
    Output("modal", "is_open"),
    [Input("garbage-input", "children"), Input("modal-close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(_, __, ___):
    ctx = dash.callback_context.triggered[0]['prop_id']
    if ctx == "modal-close" + ".n_clicks":
        return False
    return True


@app.callback(
    Output('update', 'disabled'),
    Input('garbage-input', 'children'),
)
def model_callback(_):
    if all([boolq_model['model'] is not None, boolq_model['tokenizer'] is not None, para_model['model'] is not None,
            para_model['model'] is not None]):
        return False

    boolq_model['tokenizer'] = RobertaTokenizer.from_pretrained(boolq_model['token_path'])
    boolq_model['model'] = RobertaForSequenceClassification.from_pretrained(boolq_model['model_path'])

    para_model['tokenizer'] = BertTokenizer.from_pretrained(para_model['token_path'])
    para_model['model'] = BertForSequenceClassification.from_pretrained(para_model['model_path'])

    return False


@app.callback(
    Output('chat-content', 'children'),
    Output('input', 'value'),
    Output('storage', 'data'),
    Input('update', 'n_clicks'),
    Input('input', 'n_submit'),
    Input('send', 'n_clicks'),
    State('select-story', 'value'),
    State('input', 'value'),
    State('chat-content', 'children'),
    State('storage', 'data')
)
def update_output_div(_, __, ___, story, input, chat, storage):
    ctx = dash.callback_context.triggered[0]['prop_id']

    story = int(story)
    if ctx == "update" + ".n_clicks":
        return [
                   getChatBubble(
                       f"I am a RoBERTa base model finetuned on the BoolQ dataset.",
                       False),
                   getChatBubble(f"Ask me yes-no questions about the following story to find out what happened.",
                                 False),
                   getChatBubble(stories[story]['question'], False)
               ], None, {"story": story}

    if ctx == "send" + ".n_clicks" or ctx == "input" + ".n_submit":
        if input is None:
            raise dash.exceptions.PreventUpdate
        if not input.strip() or not chat:
            raise dash.exceptions.PreventUpdate
        if not all([boolq_model['model'] is not None, boolq_model['tokenizer'] is not None,
                    para_model['model'] is not None, para_model['model'] is not None]):
            raise dash.exceptions.PreventUpdate
        if is_solution(input, stories[storage['story']]['solution']):
            return chat + [getChatBubble(input, True),
                           getChatBubble("Yes! Well done, this is the solution.", False)], None, dash.no_update
        return chat + [getChatBubble(input, True),
                       getChatBubble(
                           predict(input, stories[storage['story']]['passage']), False)], None, dash.no_update

    raise dash.exceptions.PreventUpdate()


if __name__ == "__main__":
    # app.run_server(host="0.0.0.0")
    app.run_server(debug=True)