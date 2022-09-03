import dash_bootstrap_components as dbc
from dash import html

from src.domain.stories import stories

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
