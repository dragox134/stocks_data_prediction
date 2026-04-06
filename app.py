import dash
from dash import dcc, html, Output, Input, State, no_update, ctx
import plotly.graph_objects as go
import threading
import traceback

from training_4_0 import train

app = dash.Dash(__name__, title="Stock Price Prediction")


LABEL_STYLE = {"fontSize": "var(--font-size-label)", "marginTop": "var(--padding-inner)"}
GRAPH_PANEL_BASE = {"backgroundColor": "#0f172a", "borderRadius": "12px", "padding": "8px"}

STOCK_OPTIONS = [{"label": i, "value": i} for i in ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]]
MODEL_OPTIONS = [
    {"label": "LSTM", "value": "lstm"},
    {"label": "Transformer", "value": "trs"},
]
BATCH_OPTIONS = [{"label": str(x), "value": x} for x in [8, 16, 32, 64, 128]]

TRAINING_STATE = {
    "running": False,
    "done": False,
    "error": None,
    "message": "",
    "current_epoch": 0,
    "total_epochs": 0,
    "train_losses": [],
    "val_losses": [],
    "train_dates": [],
    "test_dates": [],
    "future_dates": [],
    "train_actual": [],
    "train_pred": [],
    "test_actual": [],
    "test_pred": [],
    "test_future_pred": [],
    "train_accuracy": 0.0,
    "test_accuracy": 0.0,
    "best_loss": None,
}
TRAINING_LOCK = threading.Lock()


def empty_figure(title):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        plot_bgcolor="#111",
        paper_bgcolor="#111",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
        autosize=True,
        margin=dict(l=36, r=18, t=46, b=36),
    )
    return fig


def placeholder_figure(title, subtitle="Graphs will be shown here"):
    fig = empty_figure(title)
    fig.add_annotation(
        text=subtitle,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16, "color": "#9ca3af"},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def line_figure(title, values, label, color):
    fig = empty_figure(title)
    if values:
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(values) + 1)),
                y=values,
                mode="lines",
                name=label,
                line={"color": color, "width": 2},
            )
        )
        fig.update_layout(showlegend=False)
    else:
        fig.add_annotation(
            text="Waiting for training updates...",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14, "color": "#9ca3af"},
        )
    return fig


def close_figure(title, actual, predicted, pred_color, x_values=None, future_pred=None, future_x=None, future_color="#a78bfa"):
    fig = empty_figure(title)
    x_axis_is_date = bool(x_values or future_x)

    if actual:
        fig.add_trace(
            go.Scatter(
                x=x_values[:len(actual)] if x_values else list(range(1, len(actual) + 1)),
                y=actual,
                mode="lines",
                name="Actual",
                line={"color": "rgba(148, 163, 184, 0.45)", "width": 1.5},
                hovertemplate="Actual: %{y:.2f}<extra></extra>" if x_axis_is_date else None,
            )
        )
    if predicted:
        fig.add_trace(
            go.Scatter(
                x=x_values[:len(predicted)] if x_values else list(range(1, len(predicted) + 1)),
                y=predicted,
                mode="lines",
                name="Predicted",
                line={"color": pred_color, "width": 2},
                hovertemplate="Predicted: %{y:.2f}<extra></extra>" if x_axis_is_date else None,
            )
        )
    if future_pred:
        if predicted and future_x:
            future_y = [predicted[-1]] + list(future_pred)
            future_trace_x = [x_values[-1]] + list(future_x) if x_values else list(future_x)
        elif predicted:
            future_y = [predicted[-1]] + list(future_pred)
            future_start = len(predicted)
            future_trace_x = list(range(future_start, future_start + len(future_y)))
        else:
            future_y = list(future_pred)
            future_trace_x = list(future_x) if future_x else list(range(1, len(future_y) + 1))
        fig.add_trace(
            go.Scatter(
                x=future_trace_x,
                y=future_y,
                mode="lines+markers",
                name="Future (next days)",
                line={"color": future_color, "width": 2, "dash": "dot"},
                hovertemplate="Future: %{y:.2f}<extra></extra>" if x_axis_is_date else None,
            )
        )

    if x_axis_is_date:
        fig.update_xaxes(
            type="date",
            hoverformat="%d %b %Y",
            tickformatstops=[
                {"dtickrange": [None, "M1"], "value": "%d %b\n%Y"},
                {"dtickrange": ["M1", "M12"], "value": "%b\n%Y"},
                {"dtickrange": ["M12", None], "value": "%Y"},
            ],
        )
        fig.update_layout(hovermode="x unified")

    return fig


def graph_panel(graph_id, title, border_color):
    return html.Div(
        className="graph-panels",
        style={**GRAPH_PANEL_BASE, "border": f"1px solid {border_color}"},
        children=[
            dcc.Graph(
                id=graph_id,
                figure=placeholder_figure(title),
                className="dash-graph",
                responsive=True,
                config={"displayModeBar": False, "responsive": True},
            ),
        ],
    )


def _set_training_state(**kwargs):
    with TRAINING_LOCK:
        TRAINING_STATE.update(kwargs)


def _snapshot_training_state():
    with TRAINING_LOCK:
        return {
            "running": TRAINING_STATE["running"],
            "done": TRAINING_STATE["done"],
            "error": TRAINING_STATE["error"],
            "message": TRAINING_STATE["message"],
            "current_epoch": TRAINING_STATE["current_epoch"],
            "total_epochs": TRAINING_STATE["total_epochs"],
            "train_losses": list(TRAINING_STATE["train_losses"]),
            "val_losses": list(TRAINING_STATE["val_losses"]),
            "train_dates": list(TRAINING_STATE["train_dates"]),
            "test_dates": list(TRAINING_STATE["test_dates"]),
            "future_dates": list(TRAINING_STATE["future_dates"]),
            "train_actual": list(TRAINING_STATE["train_actual"]),
            "train_pred": list(TRAINING_STATE["train_pred"]),
            "test_actual": list(TRAINING_STATE["test_actual"]),
            "test_pred": list(TRAINING_STATE["test_pred"]),
            "test_future_pred": list(TRAINING_STATE["test_future_pred"]),
            "train_accuracy": TRAINING_STATE["train_accuracy"],
            "test_accuracy": TRAINING_STATE["test_accuracy"],
            "best_loss": TRAINING_STATE["best_loss"],
        }


def _run_training_in_background(params):
    try:
        def on_progress(epoch, num_epochs, train_losses, val_losses):
            _set_training_state(
                current_epoch=epoch,
                total_epochs=num_epochs,
                train_losses=list(train_losses),
                val_losses=list(val_losses),
                message=f"Training in progress... epoch {epoch}/{num_epochs}",
            )

        result = train(
            params["stock"],
            params["model_name"],
            params["lr"],
            params["epochs"],
            params["batch"],
            params["lookback"],
            params["days_to_predict"],
            progress_callback=on_progress,
            update_every=10,
        )

        _set_training_state(
            running=False,
            done=True,
            error=None,
            message="Training finished successfully.",
            train_losses=list(result.get("train_losses", [])),
            val_losses=list(result.get("val_losses", [])),
            train_dates=list(result.get("train_dates", [])),
            test_dates=list(result.get("test_dates", [])),
            future_dates=list(result.get("future_dates", [])),
            train_actual=list(result.get("train_actual", [])),
            train_pred=list(result.get("train_pred", [])),
            test_actual=list(result.get("test_actual", [])),
            test_pred=list(result.get("test_pred", [])),
            test_future_pred=list(result.get("test_future_pred", [])),
            train_accuracy=float(result.get("train_accuracy", 0.0)),
            test_accuracy=float(result.get("test_accuracy", 0.0)),
            best_loss=float(result.get("best_loss", 0.0)),
        )
    except Exception as exc:
        _set_training_state(
            running=False,
            done=True,
            error=str(exc),
            message=f"Training failed: {exc}",
        )
        print(traceback.format_exc())


app.layout = html.Div(
    className="app-root",
    children=[
        html.H1("Stock Price Prediction", className="app-title"),
        html.Div(
            className="app-grid",
            children=[
                html.Div(
                    className="control-panel control-panel--tall",
                    children=[
                        html.H2("Training Configuration", className="panel-title"),
                        html.Label("Stock Name", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="stock-name",
                            options=STOCK_OPTIONS,
                            value="AAPL",
                            clearable=False,
                        ),
                        html.Label("Model Name", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="model-name",
                            options=MODEL_OPTIONS,
                            value="lstm",
                            clearable=False,
                        ),
                        html.Label("Learning Rate", style=LABEL_STYLE),
                        dcc.Input(
                            id="learning-rate",
                            type="number",
                            step=1e-4,
                            value=0.001,
                        ),
                        html.Label("Number of Epochs", style=LABEL_STYLE),
                        dcc.Input(
                            id="num-epochs",
                            type="number",
                            step=1,
                            value=10,
                        ),
                        html.Label("Batch Size", style=LABEL_STYLE),
                        dcc.Dropdown(
                            id="batch-size",
                            options=BATCH_OPTIONS,
                            value=16,
                            clearable=False,
                        ),
                        html.Label("Lookback", style=LABEL_STYLE),
                        dcc.Input(
                            id="lookback",
                            type="number",
                            step=1,
                            value=7,
                        ),
                        html.Label("Days to Predict", style=LABEL_STYLE),
                        dcc.Input(
                            id="days-to-predict",
                            type="number",
                            step=1,
                            value=7,
                        ),
                        html.Button(
                            "Train Model",
                            id="train-button",
                            n_clicks=0,
                            className="train-btn",
                        ),
                        html.Div(id="train-status", className="train-status"),
                    ],
                ),
                html.Div(
                    className="dashboard-graph-grid",
                    children=[
                        html.Div(
                            className="row-2x2",
                            children=[
                                graph_panel("loss-train", "Loss/Train", "#164e63"),
                                graph_panel("loss-val", "Loss/Val", "#165e37"),
                            ],
                        ),
                        html.Div(
                            className="row-2x2",
                            children=[
                                graph_panel("train-close", "Train/Close", "#92400e"),
                                graph_panel("test-close", "Test/Close", "#991b1b"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        dcc.Interval(id="training-poller", interval=1500, n_intervals=0, disabled=True),
    ],
)


@app.callback(
    Output("train-status", "children"),
    Output("loss-train", "figure"),
    Output("loss-val", "figure"),
    Output("train-close", "figure"),
    Output("test-close", "figure"),
    Output("train-button", "disabled"),
    Output("training-poller", "disabled"),
    Input("train-button", "n_clicks"),
    Input("training-poller", "n_intervals"),
    State("stock-name", "value"),
    State("model-name", "value"),
    State("learning-rate", "value"),
    State("num-epochs", "value"),
    State("batch-size", "value"),
    State("lookback", "value"),
    State("days-to-predict", "value"),
    prevent_initial_call=True,
)
def on_train(n_clicks, _n_intervals, stock, model_name, lr, epochs, batch, lookback, days_to_predict):
    triggered = ctx.triggered_id

    if triggered == "train-button":
        state = _snapshot_training_state()
        if state["running"]:
            return (
                [html.Span("Training is already running", style={"color": "#f59e0b"})],
                line_figure("Loss/Train", state["train_losses"], "train", "#22c55e"),
                line_figure("Loss/Val", state["val_losses"], "val", "#60a5fa"),
                placeholder_figure("Train/Close", "Close price on train data will be shown after the training finishes"),
                placeholder_figure("Test/Close", "Close price on test data will be shown after the training finishes"),
                True,
                False,
            )

        params = {
            "stock": stock,
            "model_name": model_name,
            "lr": float(lr),
            "epochs": int(epochs),
            "batch": int(batch),
            "lookback": int(lookback),
            "days_to_predict": int(days_to_predict),
        }

        _set_training_state(
            running=True,
            done=False,
            error=None,
            message=f"Training started for {stock} ({model_name})",
            current_epoch=0,
            total_epochs=int(epochs),
            train_losses=[],
            val_losses=[],
            train_dates=[],
            test_dates=[],
            future_dates=[],
            train_actual=[],
            train_pred=[],
            test_actual=[],
            test_pred=[],
            test_future_pred=[],
            train_accuracy=0.0,
            test_accuracy=0.0,
            best_loss=None,
        )

        worker = threading.Thread(target=_run_training_in_background, args=(params,), daemon=True)
        worker.start()

        return (
            [html.Span("Training started", style={"color": "#22c55e"}), html.Br(), f"stock={stock}, model={model_name}"],
            placeholder_figure("Loss/Train", "Train loss over epochs will be shown during training"),
            placeholder_figure("Loss/Val", "Validation loss over epochs will be shown during training"),
            placeholder_figure("Train/Close", "Close price on train data will be shown after the training finishes"),
            placeholder_figure("Test/Close", "Close price on test data will be shown after the training finishes"),
            True,
            False,
        )

    state = _snapshot_training_state()

    if state["running"]:
        return (
            [
                html.Span(state["message"], style={"color": "#22c55e"}),
                html.Br(),
                f"Updated every 10 epochs ({state['current_epoch']}/{state['total_epochs']})",
            ],
            line_figure("Loss/Train", state["train_losses"], "train", "#22c55e"),
            line_figure("Loss/Val", state["val_losses"], "val", "#60a5fa"),
            placeholder_figure("Train/Close", "Close price on train data will be shown after the training finishes"),
            placeholder_figure("Test/Close", "Close price on test data will be shown after the training finishes"),
            True,
            False,
        )

    if state["done"]:
        status_color = "#ef4444" if state["error"] else "#22c55e"
        status_text = state["message"] if state["message"] else "Training finished"
        train_acc = float(state.get("train_accuracy", 0.0))
        test_acc = float(state.get("test_accuracy", 0.0))
        best_loss = state.get("best_loss")

        def _acc_color(value):
            if value >= 80.0:
                return "#22c55e"
            if value >= 60.0:
                return "#f59e0b"
            return "#ef4444"

        return (
            [
                html.Span(status_text, style={"color": status_color}),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("Best Loss", style={"fontSize": "12px", "opacity": 0.85}),
                                html.Div("-" if best_loss is None else f"{float(best_loss):.6f}", style={"fontSize": "clamp(26px, 2vw, 36px)", "fontWeight": 700, "lineHeight": "1.0", "color": "#38bdf8"}),
                            ],
                            style={"marginTop": "14px"},
                        ),
                        html.Div(
                            [
                                html.Div("Train Fit (R²)", style={"fontSize": "12px", "opacity": 0.85}),
                                html.Div(f"{train_acc:.2f}%", style={"fontSize": "34px", "fontWeight": 700, "lineHeight": "1.0", "color": _acc_color(train_acc)}),
                            ],
                            style={"marginTop": "10px"},
                        ),
                        html.Div(
                            [
                                html.Div("Test Fit (R²)", style={"fontSize": "12px", "opacity": 0.85}),
                                html.Div(f"{test_acc:.2f}%", style={"fontSize": "34px", "fontWeight": 700, "lineHeight": "1.0", "color": _acc_color(test_acc)}),
                            ],
                            style={"marginTop": "10px"},
                        ),
                    ],
                ),
            ],
            line_figure("Loss/Train", state["train_losses"], "train", "#22c55e"),
            line_figure("Loss/Val", state["val_losses"], "val", "#60a5fa"),
            close_figure("Train/Close", state["train_actual"], state["train_pred"], "#f59e0b", x_values=state["train_dates"]),
            close_figure("Test/Close", state["test_actual"], state["test_pred"], "#f97316", x_values=state["test_dates"], future_pred=state["test_future_pred"], future_x=state["future_dates"], future_color="#a78bfa"),
            False,
            True,
        )

    return no_update, no_update, no_update, no_update, no_update, no_update, no_update


if __name__ == "__main__":
    app.run(debug=True)
