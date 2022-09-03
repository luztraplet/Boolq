import string

import dash.exceptions
import dash_bootstrap_components as dbc
import torch
from dash import Dash, html, Output, Input, State, dcc
from transformers import RobertaForSequenceClassification, RobertaTokenizer, BertForSequenceClassification, \
    BertTokenizer

app = Dash(__name__,
           update_title='Loading models...',
           external_stylesheets=["assets/sandstone.css", "assets/custom.css"],
           meta_tags=[{'name': 'viewport',
                       'content': 'width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,'}])

server = app.server

stories = [
    {
        "title": "The falling man",
        "question": "A man was pushed out of an airplane, without a parachute. How was he able to survive?",
        "solution": "The plane was sitting on the runway",
        "passage": "A man was pushed out of an airplane. He did not wear a parachute. He survived the fall because "
                   "the plane was still on the runway and he didn't fall that far."
    },
    {
        "title": "The mysterious light",
        "question": "A man woke up in the middle of the night and turned off the light."
                    "Ten people died as result. Why?",
        "solution": "He was the lighthouse keeper",
        "passage": "A man woke up in the middle of the night and turned off the light. He was the lighthouse keeper. "
                   "Without the light, a ship didn't know where the land was and therefore wrecked on rocks and all "
                   "the passengers died."
    },
    {
        "title": "Overtaking",
        "question": "Why was the man able to pass three cars going 70 miles-per-hour, while he was going only 60 "
                    "miles-per-hour?",
        "solution": "The cars he passed were going in the opposite direction",
        "passage": "A man was able to pass three cars. He was driving at 70 miles-per-hour. The other cars were going "
                   "60 miles-per-hour and in the opposite direction. "
    },
    {
        "title": "The dead brother",
        "question": "A man wakes up and sees that his brother is dead. Though they are alone in a quiet room, "
                    "he immediately realizes that he will be dead soon too. How does he know this?",
        "solution": "His brother is his Siamese twin",
        "passage": "A man wakes up and sees that his brother is dead. Though they are alone in a quiet room, "
                   "he immediately realizes that he will be dead soon too. His brother is his Siamese twin, "
                   "and since they share common organs, he cannot survive alone "
    }
]

models_path = r'C:\Users\Luzian Trapletti\Documents\Important\schule\MA\Programmieren\ProductionCode\MLModels'

para_model = {
    "model_path":  models_path + '\ParaModel',
    "token_path": models_path + '\ParaModelTokenizer',
    "model": None,
    "tokenizer": None,
}

boolq_model = {
    "model_path": models_path + '\BoolQModel',
    "token_path": models_path + '\BoolQModelTokenizer',
    "model": None,
    "tokenizer": None,
}

header = html.Div(
    [
        html.H1("Blackstories Chat Bot"),
        html.H4("Try out what this neural network can do.", className="text-muted")
    ], className="my-5 text-center"
)


def getChatBubble(message, is_user):
    return dbc.Row(
        dbc.Card(
            dbc.CardBody(
                message,
                className="w-auto"
            ),
            color="info" if is_user else "success",
            outline=True,
            className="chat-card user-card" if is_user else "chat-card bot-card"
        ),
        justify="end" if is_user else "start"
    )


content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Label("Select story", html_for="select-story"), width=4
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Select(
                        id="select-story",
                        value=0,
                        options=[
                            {"label": s["title"], "value": int(i)} for i, s in enumerate(stories)]
                    )
                ),
                dbc.Col(
                    dbc.Button("Update and clear chat", id="update", disabled=True, className="w-100 h-100")
                )
            ],
            className="mb-3"
        ),
        dbc.Card(
            id="chat-content",
            className="my-3 gap-3 flex-column min-vh-50 p-custom"
        ),
        dbc.Row(
            dbc.InputGroup(
                [
                    dbc.Input(placeholder="Enter question...", value="", size="lg", id="input", debounce=True,
                              autocomplete="off", className="input-border-custom"),
                    dbc.Button("Send", color="info", id="send", className="button-border-custom")
                ], className="rounded-pill w-75 "
            ), justify="center"
        ),
        html.Div(className="min-vh-10", id="scroll")
    ]
)

footer = html.Div(
    [
        html.A("Made by Luzian Trapletti | Copyright © 2022 Luzian Trapletti"),
    ], className="text-center my-5"
)

app.layout = dbc.Container(
    [
        header,
        content,
        footer,
        dcc.Store(id='storage'),
        html.Div(id="garbage-output"),
        html.Div(id="garbage-input"),
    ]
)


def is_solution(passage, solution):
    passage = passage.translate(str.maketrans('', '', string.punctuation))
    input = para_model['tokenizer'](passage, solution, return_tensors='pt')
    logits = para_model['model'](**input).logits
    p = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    return True if p[1] > 0.6 else False


def predict(question, passage):
    input = boolq_model['tokenizer'](question, passage, return_tensors='pt')
    logits = boolq_model['model'](**input).logits
    p = torch.argmax(logits, dim=1)
    return "Yes" if p == 1 else "No"


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
    app.run_server()
