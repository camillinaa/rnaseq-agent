import dash
from dash import html, dcc, Input, Output, State
import dash_mantine_components as dmc
import os
import uuid
from rich.console import Console
from rich.markdown import Markdown
import mistune
import smtplib
from email.message import EmailMessage
from main import create_agent
import datetime

# Instantiate your agent
agent = create_agent()

app = dash.Dash(__name__, external_stylesheets=["https://cdn.jsdelivr.net/npm/@mantine/core@latest/dist/mantine.min.css"])
app.title = "RNA-seq Chatbot"

app.layout = dmc.MantineProvider([
    dmc.Container([
        # Header section - compact and clean
        dmc.Stack([
            dmc.Group([
                html.Img(
                    src="https://humantechnopole.it/wp-content/uploads/2020/10/01_HTlogo_pantone_colore.png", 
                    style={"width": "160px", "height": "auto"}
                ),
                html.Div([
                    html.H2("RNA-seq Data Analysis Assistant", 
                           style={"margin": "0", "fontSize": "24px", "fontWeight": "600", "color": "#1a1a1a"}),
                    html.P("Query your RNA-seq data and generate visual summaries", 
                          style={"margin": "5px 0 0 0", "fontSize": "14px", "color": "#666", "fontWeight": "400"})
                ])
            ], style={"justifyContent": "space-between", "alignItems": "center", "width": "100%"})
        ], style={"marginBottom": "20px"}),

        # Chat area with export and clear buttons
        dmc.Stack([
            # Chat header with buttons
            html.Div([
                html.Button(
                    "Clear",
                    id="clear-button",
                    style={
                        "background": "#f8f9fa",
                        "border": "1px solid #d6d9dc",
                        "outline": "none",
                        "color": "#495057",
                        "fontSize": "11px",
                        "cursor": "pointer",
                        "padding": "6px 10px",
                        "borderRadius": "8px",
                        "transition": "all 0.2s ease",
                        "marginBottom": "0px",
                        "minWidth": "60px"
                    }
                ),
                html.Button(
                    "Export",
                    id="export-button",
                    style={
                        "background": "#f8f9fa",
                        "border": "1px solid #d6d9dc",
                        "outline": "none",
                        "color": "#495057",
                        "fontSize": "11px",
                        "cursor": "pointer",
                        "padding": "6px 10px",
                        "borderRadius": "8px",
                        "transition": "all 0.2s ease",
                        "marginBottom": "0px",
                        "minWidth": "60px"     # Ensure width matches Export

                    }
                ),
                dcc.Download(id="download-chat")
            ], style={
                "display": "flex",
                "justifyContent": "flex-end",  # Align the button group to the right
                "gap": "8px",                  # Space between Clear and Export buttons
                "width": "100%"
            }),
            
            # Chat window container
            html.Div([
                html.Div(
                    id='chat-window',
                    style={
                        'height': '55vh',
                        'resize': 'vertical',
                        'overflowY': 'auto',
                        'padding': '15px',
                        'border': '1px solid #e0e0e0',
                        'borderRadius': '12px',
                        'backgroundColor': '#fafafa',
                        'position': 'relative',
                        'boxShadow': 'inset 0 1px 3px rgba(0,0,0,0.05)'
                    }
                ),

                dmc.LoadingOverlay(
                    id="chat-loading",
                    visible=False,
                    zIndex=100,
                    loaderProps={"variant": "dots", "color": "#007bff"},
                    style={
                        "position": "absolute",
                        "top": 0,
                        "left": 0,
                        "right": 0,
                        "bottom": 0,
                        "borderRadius": "12px"
                    }
                )
            ], style={"position": "relative", "marginBottom": "0"})
        ]),

        # Input area - attached to chat area
        html.Div([
            dmc.Group([
                dmc.TextInput(
                    id='user-input', 
                    placeholder='Ask a question about your RNA-seq data...',
                    debounce=True,
                    style={
                        "flex": "1",
                        "marginRight": "10px"
                    },
                    styles={
                        "input": {
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "8px",
                            "padding": "12px 16px",
                            "fontSize": "14px",
                            "transition": "all 0.2s ease",
                            "outline": "none"
                        }
                    }
                ),
                dmc.Button(
                    'Submit',
                    id='send-button',
                    color='blue',
                    n_clicks=0,
                    style={
                        "borderRadius": "8px",
                        "padding": "12px 24px",
                        "fontSize": "14px",
                        "fontWeight": "500",
                        "border": "none",
                        "outline": "none"
                    }
                )
            ], style={
                "display": "flex", 
                "alignItems": "stretch",
                "gap": "0",
                "marginTop": "-1px",  # Attach to chat area
                "padding": "15px",
                "backgroundColor": "white",
                "border": "1px solid #e0e0e0",
                "borderTop": "none",
                "borderBottomLeftRadius": "12px",
                "borderBottomRightRadius": "12px"
            })
        ]),

        # Storage and status
        dcc.Store(id='chat-history', data=[]),
        dcc.Store(id='trigger-bot-response', data=0),
        
        html.Div([
            html.P("Conversations are not saved and will reset if refreshed. Use the export button to download your chat history.", 
                  style={"fontSize": "12px", "color": "#888", "margin": "10px 0 0 0", "textAlign": "center"})
        ]),

        # Support drawer - styled to match
        dmc.Stack(
            id="support-drawer",
            children=[
                html.Div( # html.Div is more reliable for clickable elements in Dash
                    "💬 Support",
                    id="open-support-form",
                    style={
                        "height": "36px",
                        "textAlign": "center",
                        "padding": "8px 20px",
                        "fontSize": "13px",
                        "backgroundColor": "#f8f9fa",
                        "color": "#495057",
                        "borderTopLeftRadius": "12px",
                        "borderTopRightRadius": "12px",
                        "cursor": "pointer",
                        "userSelect": "none",
                        "border": "1px solid #e0e0e0",
                        "borderBottom": "none",
                        "transition": "all 0.2s ease"
                    }
                ),

                dmc.Stack([
                    html.H5("Support Request", 
                           style={"marginTop": "15px", "marginBottom": "15px", "fontSize": "16px", "fontWeight": "600"}),
                    dmc.TextInput(
                        id="support-email", 
                        placeholder="Your email", 
                        style={"marginBottom": "10px"},
                        styles={"input": {"borderRadius": "6px", "border": "1px solid #e0e0e0"}}
                    ),
                    dmc.Textarea(
                        id="support-message", 
                        placeholder="Describe your issue...", 
                        style={"marginBottom": "15px"},
                        styles={"input": {"borderRadius": "6px", "border": "1px solid #e0e0e0", "minHeight": "80px"}}
                    ),
                    dmc.Button(
                        "Send", 
                        id="send-support", 
                        color="blue", 
                        style={"marginBottom": "10px", "borderRadius": "6px"}
                    ),
                    dmc.Stack(
                        id="support-status", 
                        style={"fontSize": "13px", "color": "#666"}
                    )
                ],
                id="support-drawer-body",
                style={"display": "none", "padding": "0 20px 20px 20px", "backgroundColor": "white", "border": "1px solid #e0e0e0", "borderTop": "none"})
            ],
            style={
                "position": "fixed",
                "bottom": "0",
                "left": "50%",
                "transform": "translateX(-50%)",
                "width": "320px",
                "height": "36px",
                "backgroundColor": "white",
                "boxShadow": "0 -4px 12px rgba(0, 0, 0, 0.1)",
                "borderTopLeftRadius": "12px",
                "borderTopRightRadius": "12px",
                "overflow": "hidden",
                "zIndex": "1002",
                "transition": "height 0.3s ease"
            }
        )
    ], style={"maxWidth": "900px", "margin": "0 auto", "padding": "20px", "height": "100vh", "display": "flex", "flexDirection": "column"})
])


def create_bot_message(message, html_content=None):

    renderer = mistune.create_markdown(renderer='html')
    html_message = renderer(message)
    
    children = [
        dcc.Markdown(
            message,
            style={
                'backgroundColor': 'white', 
                'borderRadius': '16px', 
                'padding': '16px',
                'marginBottom': '8px', 
                'maxWidth': '80%', 
                'display': 'inline-block',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'border': '1px solid #f0f0f0',
                'fontSize': '14px',
                'lineHeight': '1.5',
                'animation': 'fadeIn 0.4s ease-in-out'
            }
        )
    ]    

    if html_content:
        children.append(html.Iframe(
            srcDoc=html_content, 
            height="500", 
            style={
                "width": "100%", 
                "border": "1px solid #e0e0e0", 
                "borderRadius": "12px", 
                "marginTop": "10px",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.1)"
            }
        ))
    return dmc.Stack(children, style={"textAlign": "left", "marginBottom": "20px"})


def create_user_message(message):
    return dmc.Stack([
        dmc.Stack(message, style={
            'backgroundColor': '#007bff', 
            'color': 'white', 
            'borderRadius': '16px', 
            'padding': '16px',
            'marginBottom': '8px', 
            'maxWidth': '80%', 
            'marginLeft': 'auto', 
            'display': 'inline-block',
            'boxShadow': '0 2px 8px rgba(0,123,255,0.3)',
            'fontSize': '14px',
            'lineHeight': '1.5',
            'animation': 'fadeIn 0.4s ease-in-out'
        })
    ], style={"textAlign": "right", "marginBottom": "20px"})


# First callback: Immediately show user message
@app.callback(
    [Output('chat-window', 'children'),
     Output('chat-loading', 'visible'),
     Output('chat-history', 'data'),
     Output('user-input', 'value'),
     Output('trigger-bot-response', 'data')],
    [Input('send-button', 'n_clicks'),
     Input('user-input', 'n_submit')],
    [State('user-input', 'value'),
     State('chat-history', 'data'),
     State('trigger-bot-response', 'data')],
    prevent_initial_call=True
)
def show_user_message(n_clicks, n_submit, user_input, chat_history, trigger_counter):
    if not user_input or user_input.strip() == "":
        return dash.no_update, False, dash.no_update, dash.no_update, dash.no_update

    # Immediately add user message to chat history
    displayed_chat = [*chat_history, {"role": "user", "content": user_input}]
    
    # Render all messages including the new user message
    rendered = []
    for msg in displayed_chat:
        if msg["role"] == "user":
            rendered.append(create_user_message(msg["content"]))
        elif msg["role"] == "bot":
            rendered.append(create_bot_message(msg["content"], msg.get("html_plot")))

    # Show loading and trigger bot response
    return rendered, True, displayed_chat, "", trigger_counter + 1


# Second callback: Process bot response
@app.callback(
    [Output('chat-window', 'children', allow_duplicate=True),
     Output('chat-loading', 'visible', allow_duplicate=True),
     Output('chat-history', 'data', allow_duplicate=True)],
    [Input('trigger-bot-response', 'data')],
    [State('chat-history', 'data')],
    prevent_initial_call=True
)
def process_bot_response(trigger_counter, chat_history):
    if not chat_history or len(chat_history) == 0:
        return dash.no_update, False, dash.no_update
    
    # Get the last user message
    last_message = chat_history[-1]
    if last_message["role"] != "user":
        return dash.no_update, False, dash.no_update
    
    user_input = last_message["content"]
    
    try:
        result = agent.ask(user_input)
        print("DEBUG: result from agent.ask:", result)

        if isinstance(result, tuple) and len(result) == 2:
            answer, plot_filename = result
        elif isinstance(result, str):
            answer = result
            plot_filename = None
        elif isinstance(result, dict):
            answer = result.get("output", "No answer found.")
            plot_filename = result.get("plot_filename")
        else:
            answer = f"Unexpected result type: {type(result)}"
            plot_filename = None

        if plot_filename and os.path.exists(plot_filename):
            with open(plot_filename, "r") as f:
                html_plot = f.read()
        else:
            html_plot = None

        # Add bot response to chat history
        updated_chat = [*chat_history, {"role": "bot", "content": answer, "html_plot": html_plot}]
        
    except Exception as e:
        answer = f"Error: {e}"
        updated_chat = [*chat_history, {"role": "bot", "content": answer, "html_plot": None}]

    # Render all messages including the new bot response
    rendered = []
    for msg in updated_chat:
        if msg["role"] == "user":
            rendered.append(create_user_message(msg["content"]))
        elif msg["role"] == "bot":
            rendered.append(create_bot_message(msg["content"], msg.get("html_plot")))

    return rendered, False, updated_chat


@app.callback(
    Output('chat-window', 'children', allow_duplicate=True),
    Output('chat-history', 'data', allow_duplicate=True),
    Input('clear-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_chat(n_clicks):
    return [], []


@app.callback(
    Output("download-chat", "data"),
    Input("export-button", "n_clicks"),
    State("chat-history", "data"), 
    prevent_initial_call=True
)
def export_chat(n_clicks, chat_data):
    if not chat_data:
        return None
    
    # Create the text content
    chat_text = f"Chat Export - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    chat_text += "=" * 50 + "\n\n"
    
    for message in chat_data:
        # Adjust these keys based on your chat message structure
        role = message.get("role", "user")  # or however you store the role
        content = message.get("content", "")  # or however you store the message content
        timestamp = message.get("timestamp", "")  # if you have timestamps
        
        if role == "user":
            chat_text += f"User: {content}\n\n"
        else:
            chat_text += f"Assistant: {content}\n\n"
    
    # Return the download
    return dict(
        content=chat_text,
        filename=f"chat_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    

@app.callback(
    [Output("support-drawer", "style"),
     Output("support-drawer-body", "style")],
    [Input("open-support-form", "n_clicks")],
    [State("support-drawer", "style")],
    prevent_initial_call=True
)
def slide_support_drawer(n_clicks, style):
    new_style = style.copy()
    
    if style["height"] == "36px":
        new_style["height"] = "310px"
        return new_style, {
            "display": "block", 
            "padding": "0 20px 20px 20px", 
            "backgroundColor": "white", 
            "border": "1px solid #e0e0e0", 
            "borderTop": "none"
        }
    else:
        new_style["height"] = "36px"
        return new_style, {"display": "none"}


@app.callback(
    Output("support-status", "children"),
    [Input("send-support", "n_clicks")],
    [State("support-email", "value"),
     State("support-message", "value")],
    prevent_initial_call=True
)
def send_support_email(n_clicks, email, message):
    if not email or not message:
        return "Please fill in both fields."

    try:
        msg = EmailMessage()
        msg.set_content(f"Support request from: {email}\n\nMessage:\n{message}")
        msg["Subject"] = "RNA-seq Chatbot Support Request"
        msg["From"] = email
        msg["To"] = "camilla.callierotti@fht.org"

        with smtplib.SMTP("localhost") as server:
            server.send_message(msg)

        return "✅ Your message has been sent."
    except Exception as e:
        return f"❌ Failed to send message: {str(e)}"


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


if __name__ == '__main__':
    app.run(debug=True)