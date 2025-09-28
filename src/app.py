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
import utils
import pandas as pd

# Instantiate your agent
agent = create_agent()
utils.reset_memory(agent) # Reset context and memory for fresh session

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
                    html.P("Query your RNA-seq data and generate visual summaries and reports", 
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
                        "minWidth": "60px"      # Ensure width matches Export

                    }
                ),
                # The two dDownload components to be used for both text and CSV files
                dcc.Download(id="download-chat"),
                dcc.Download(id="download-data-report")
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
                  style={"fontSize": "12px", "color": "#888", "margin": "10px 0 0 0", "textAlign": "center"}),
            html.P("Developed by Camilla Callierotti for the National Facility for Data Handling and Analysis.", 
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


def create_bot_message(message, html_content=None, csv_preview_html=None, csv_filename=None):
    print(f"DEBUG create_bot_message: html_content={html_content is not None}, csv_preview_html={csv_preview_html is not None}, csv_filename={csv_filename}")
    
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

    # Add HTML plot iframe if present
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
    
    # Add CSV preview if present
    if csv_preview_html and csv_filename:
        print("DEBUG: Adding CSV preview to children")
        csv_container = html.Div([
            html.Div([
                html.H6("CSV Report Preview", style={
                    "margin": "0 0 10px 0", 
                    "fontSize": "14px", 
                    "fontWeight": "600",
                    "color": "#333"
                }),
                html.Div([
                    html.Button(
                        id={'type': 'download-csv', 'filename': csv_filename},
                        children=[
                            html.Span("📁 ", style={"fontSize": "14px"}),
                            html.Span("Download CSV", style={"fontSize": "12px", "color": "#666"}),
                        ],
                        style={
                        "backgroundColor": "transparent",
                        "border": "none",
                        "cursor": "pointer",
                        "padding": "5px 10px",
                        "borderRadius": "4px",
                        "display": "flex",
                        "alignItems": "center",
                        "transition": "background-color 0.2s"
                    },
                        n_clicks=0
                    ),
                ], className="csv-download-button-container"), 
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "10px"}),
            
            html.Div([
                html.Iframe(
                    srcDoc=csv_preview_html,
                    style={
                        "width": "100%",
                        "height": "350px",
                        "border": "1px solid #e0e0e0",
                        "borderRadius": "6px",
                        "backgroundColor": "#fafafa"
                    }
                )
            ])
        ], style={
            "marginTop": "10px",
            "padding": "15px",
            "border": "1px solid #e0e0e0",
            "borderRadius": "12px",
            "backgroundColor": "#f8f9fa"
        })
        children.append(csv_container)
    
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
        # Get response from agent (now returns a dictionary)
        result = agent.ask(user_input)
        print("DEBUG: result from agent.ask:", result)
        
        # Extract components from the dictionary response
        answer = result.get("final_answer", "No response received.")
        plot_filename = result.get("plot_filename")
        report_filename = result.get("report_filename")
        print(f"DEBUG: Extracted - plot_filename={plot_filename}, report_filename={report_filename}")
        
        html_plot = None
        csv_preview_html = None

        # Handle plot file if one was generated
        if plot_filename:
            full_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "plots", plot_filename)
            if os.path.exists(full_path):
                with open(full_path, "r") as f:
                    html_plot = f.read()
                print(f"DEBUG: Plot loaded from {full_path}")
            else:
                print(f"DEBUG: Plot file not found at {full_path}")

        # Handle CSV report file if one was generated
        if report_filename:
            # Clean the filename - remove any path prefixes that might be included
            clean_filename = os.path.basename(report_filename)
            csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "reports", clean_filename)
            print(f"DEBUG: Looking for CSV at: {csv_path}")
            
            if os.path.exists(csv_path):
                try:
                    # Read CSV and create preview HTML
                    df = pd.read_csv(csv_path)
                    print(f"DEBUG: CSV loaded successfully, shape: {df.shape}")
                    
                    # Limit preview to first 10 rows for performance
                    preview_df = df.head(10)
                    
                    # Create HTML table with styling
                    table_html = preview_df.to_html(
                        classes='csv-preview-table',
                        table_id='csv-preview',
                        escape=False,
                        index=False
                    )
                    
                    # Add CSS styling to the HTML
                    csv_preview_html = f"""
                    <style>
                        body {{ margin: 10px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
                        .csv-preview-table {{
                            width: 100%;
                            border-collapse: collapse;
                            font-size: 12px;
                        }}
                        .csv-preview-table th {{
                            background-color: #f8f9fa;
                            border: 1px solid #e0e0e0;
                            padding: 8px 12px;
                            text-align: left;
                            font-weight: 600;
                            color: #333;
                        }}
                        .csv-preview-table td {{
                            border: 1px solid #e0e0e0;
                            padding: 8px 12px;
                            text-align: left;
                        }}
                        .csv-preview-table tr:nth-child(even) {{
                            background-color: #f9f9f9;
                        }}
                        .csv-preview-table tr:hover {{
                            background-color: #f0f8ff;
                        }}
                    </style>
                    {table_html}
                    {f'<p style="font-size: 11px; color: #666; margin-top: 10px; text-align: center;">Showing first 10 rows of {len(df)} total rows</p>' if len(df) > 10 else ''}
                    """
                    print("DEBUG: CSV preview HTML created successfully")
                except Exception as e:
                    print(f"DEBUG: Error reading CSV file: {e}")
                    csv_preview_html = None
            else:
                print(f"DEBUG: CSV file not found at {csv_path}")

        # Add bot response to chat history, including any filenames and CSV data
        updated_chat = [*chat_history, {
            "role": "bot", 
            "content": answer, 
            "html_plot": html_plot, 
            "report_filename": report_filename,
            "csv_preview_html": csv_preview_html
        }]

        print(f"DEBUG: Updated chat entry - has csv_preview_html: {csv_preview_html is not None}")

    except Exception as e:
        print(f"DEBUG: Exception in process_bot_response: {e}")
        answer = f"I encountered an error while processing your question: {str(e)}."
        updated_chat = [*chat_history, {
            "role": "bot", 
            "content": answer, 
            "html_plot": None, 
            "report_filename": None,
            "csv_preview_html": None
        }]

    # Render all messages including the new bot response
    rendered = []
    for msg in updated_chat:
        if msg["role"] == "user":
            rendered.append(create_user_message(msg["content"]))
        elif msg["role"] == "bot":
            # Pass the parameters in the correct order and names
            rendered.append(create_bot_message(
                msg["content"], 
                html_content=msg.get("html_plot"),
                csv_preview_html=msg.get("csv_preview_html"),
                csv_filename=msg.get("report_filename")
            ))

    return rendered, False, updated_chat


# Third callback for CSV downloads
@app.callback(
    Output("download-data-report", "data", allow_duplicate=True),
    # The first argument (n_clicks_list) corresponds to this Input
    [Input({"type": "download-csv", "filename": dash.dependencies.ALL}, "n_clicks")],
    [State("chat-history", "data")],
    prevent_initial_call=True
)
def download_csv_file(n_clicks_list, chat_history):
    # Check if any download button was clicked.
    if not any(n_clicks_list) or not chat_history:
        return dash.no_update
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    # 🌟 NEW ROBUST LOGIC 🌟
    # ctx.args_grouping is a dictionary of argument groups.
    # The key is the property being listened to ('n_clicks'), and the value is a list 
    # corresponding to the ALL wildcard, where each item contains 'id' and 'value'.
    
    triggered_clicks = ctx.args_grouping['n_clicks']
    
    filename = None
    
    # Loop through the list to find which button's 'n_clicks' value is non-zero (i.e., triggered)
    for trigger_info in triggered_clicks:
        # Check if the trigger's value (n_clicks) is 1 (or any positive number indicating a click)
        # Note: Dash usually sets the clicked button's n_clicks to a higher value than the others.
        if trigger_info['value'] is not None and trigger_info['value'] > 0:
            # The 'id' key here holds the dictionary {'type': 'download-csv', 'filename': '...'}
            button_id_dict = trigger_info['id']
            filename = button_id_dict.get('filename')
            
            # Reset the n_clicks to 0 for this button to allow future clicks if needed 
            # (though dcc.Download usually handles a one-time trigger)
            # We break after the first one is found, assuming only one button was clicked.
            break

    if not filename:
        print("DEBUG: No triggered button with a valid filename found.")
        return dash.no_update
    
    # The rest of your logic is correct for pathing and downloading
    
    # Find the CSV file path
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "reports", filename)
    
    if os.path.exists(csv_path):
        # dcc.send_file automatically uses the browser's default download location.
        return dcc.send_file(csv_path, filename=filename, type='text/csv')
    else:
        print(f"DEBUG: CSV file not found at {csv_path}")
        return dash.no_update
    

@app.callback(
    Output('chat-window', 'children', allow_duplicate=True),
    Output('chat-history', 'data', allow_duplicate=True),
    Input('clear-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_chat(n_clicks):
    utils.reset_memory(agent) # Reset context and memory
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
    
    # Check for the latest generated report filename in chat history
    report_filename = None
    for message in reversed(chat_data):
        if message.get("role") == "bot" and message.get("report_filename"):
            report_filename = message["report_filename"]
            break
            
    if report_filename:
        # If a report filename is found, download the CSV file
        # Assumes the report is in a 'reports' subdirectory of the 'assets' folder
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "reports", report_filename)
        if os.path.exists(file_path):
            return dcc.send_file(file_path, filename=report_filename, type='text/csv')
        else:
            print(f"DEBUG: Report file not found at {file_path}. Defaulting to chat export.")

    # Fallback to creating the text content of the chat history
    chat_text = f"Chat Export - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    chat_text += "=" * 50 + "\n\n"
    
    for message in chat_data:
        role = message.get("role", "user")
        content = message.get("content", "")
        
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
