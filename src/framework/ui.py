import dash_bootstrap_components as dbc
from dash import html, dcc

from src.domain.stories import stories

disclaimer = dbc.Modal(
    [
        dbc.ModalHeader(
            dbc.ModalTitle("Terms and Conditions"), close_button=False
        ),
        dbc.ModalBody(
            [
                html.A(
                    """The user should be aware that systems with generative language
                    capabilities may produce offensive or
                    misinforming content, and could be potentially misused. This has been
                    reported in the literature.
                    Our model does not filter in real-time any of the input text (prompts)
                    thus cannot exclude the
                    generation of offensive or misinforming results. We do not endorse and
                    are not responsible for any
                    outputs produced by the model."""
                ),
                html.Br(),
                html.Br(),
                html.A(
                    """This site is provided on an as-is and as-available basis. You agree
                    that your use of the site and
                    our services will be at your sole risk. To the fullest extent
                    permitted by law, we disclaim all
                    warranties, express or implied, in connection with the site and your
                    use thereof, including, without
                    limitation, the implied warranties of merchantability, fitness for a
                    particular purpose, and
                    non-infringement. We make no warranties or representations about the
                    accuracy or completeness of the
                    site’s content or the content of any websites linked to the site and
                    we will assume no liability or
                    responsibility for any (1) errors, mistakes, or inaccuracies of
                    content and materials, (2) personal
                    injury or property damage, of any nature whatsoever, resulting from
                    your access to and use of the
                    site, (3) any unauthorized access to or use of our secure servers
                    and/or any and all personal
                    information and/or financial information stored therein, (4) any
                    interruption or cessation of
                    transmission to or from the site, (5) any bugs, viruses, trojan
                    horses, or the like which may be
                    transmitted to or through the site by any third party, and/or (6) any
                    errors or omissions in any
                    content and materials or for any loss or damage of any kind incurred
                    as a result of the use of any
                    content posted, transmitted, or otherwise made available via the site.
                    We do not warrant, endorse,
                    guarantee, or assume responsibility for any product or service
                    advertised or offered by a third party
                    through the site, any hyperlinked website, or any website or mobile
                    application featured in any banner
                    or other advertising, and we will not be a party to or in any way be
                    responsible for monitoring any
                    transaction between you and any third-party providers of products or
                    services. As with the purchase
                    of a product or service through any medium or in any environment, you
                    should use your best judgment
                    and exercise caution where appropriate."""
                ),
                html.Br(),
                html.Br(),
                html.A(
                    """In no event will we be liable to you or any third party for any
                    direct, indirect, consequential,
                    exemplary, incidental, special, or punitive damages, including lost
                    profit, lost revenue, loss of
                    data, or other damages arising from your use of the site, even if we
                    have been advised of the
                    possibility of such damages. Notwithstanding anything to the contrary
                    contained herein, our liability
                    to you for any cause whatsoever and regardless of the form of the
                    action, will at all times be
                    limited to the lesser of the amount paid, if any, by you to us or CHF
                    one Swiss Franc."""
                )
            ]),
        dbc.ModalFooter(dbc.Button("I agree", id="modal-close")),
    ],
    size="xl",
    is_open=True,
    id="modal",
    keyboard=False,
    backdrop="static",
)

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
        dcc.Link("Source code", href="https://github.com/luztraplet/Boolq", target="blank"),
    ], className="text-center my-5"
)


def add_layout(app):
    app.layout = dbc.Container(
        [
            header,
            content,
            disclaimer,
            footer,
            dcc.Store(id='storage'),
            html.Div(id="garbage-output"),
            html.Div(id="garbage-input"),
        ]
    )
