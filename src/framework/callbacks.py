import dash.exceptions
from dash import Output, Input, State

from src.domain.models import is_solution, predict
from src.framework.ui import get_chat_bubble


def add_callbacks(app):
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
        stories = app.stories
        if ctx == "update" + ".n_clicks":
            return [
                       get_chat_bubble(
                           f"I am a RoBERTa base model finetuned on the BoolQ dataset.",
                           False),
                       get_chat_bubble(f"Ask me yes-no questions about the following story to find out what happened.",
                                       False),
                       get_chat_bubble(stories[story]['question'], False)
                   ], None, {"story": story}

        if ctx == "send" + ".n_clicks" or ctx == "input" + ".n_submit":
            if input is None:
                raise dash.exceptions.PreventUpdate
            if not input.strip() or not chat:
                raise dash.exceptions.PreventUpdate

            boolq_model, para_model = app.models

            if not all([boolq_model['model'] is not None, boolq_model['tokenizer'] is not None,
                        para_model['model'] is not None, para_model['model'] is not None]):
                raise dash.exceptions.PreventUpdate

            if is_solution(para_model, input, stories[storage['story']]['solution']):
                return chat + [get_chat_bubble(input, True),
                               get_chat_bubble("Yes! Well done, this is the solution.", False)], None, dash.no_update
            return chat + [get_chat_bubble(input, True),
                           get_chat_bubble(
                               predict(boolq_model, input, stories[storage['story']]['passage']),
                               False)], None, dash.no_update

        raise dash.exceptions.PreventUpdate()
