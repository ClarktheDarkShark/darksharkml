from flask import Blueprint, render_template_string, request

# pull trained artifacts from the predictor module
from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
)

dash_preds = Blueprint('dash_preds', __name__, url_prefix='/predictions')

TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Stream Predictions</title>
    <style>
      table { border-collapse: collapse; width: 60%; max-width: 600px; }
      th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: center; }
      th { background: #f5f5f5; }
      body { font-family: sans-serif; margin: 2rem; }
      form { margin-bottom: 1rem; }
      input { padding: 4px; }
    </style>
  </head>
  <body>
    <h1>Top {{ top_n }} Predictions{% if game %} for “{{ game }}”{% endif %}</h1>
    <form method="get">
      <label>Stream (channel): <input name="stream" value="{{ stream }}"></label>
      <label>Game: <input name="game" value="{{ game }}"></label>
      <label>Top N: <input name="top_n" type="number" value="{{ top_n }}" min="1" max="50" style="width:4em;"></label>
      <button type="submit">Go</button>
    </form>
    {% if not ready %}
      <p>Model not trained yet. Try again soon.</p>
    {% else %}
    <table>
      <thead>
        <tr>
          <th>Start Time</th>
          <th>Duration (hrs)</th>
          <th>Expected Subs</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody>
      {% for row in predictions %}
        <tr>
          <td>{{ row.Time }}</td>
          <td>{{ row.Duration }}</td>
          <td>{{ row.Expected_Subs }}</td>
          <td>{{ row.Confidence }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% endif %}
  </body>
</html>
"""

@dash_preds.route('/', methods=['GET'])
def show_predictions():
    pipe, df_for_inf, features, cat_opts, start_opts, dur_opts, metrics = get_predictor_artifacts()
    ready = pipe is not None and df_for_inf is not None

    # pull & normalize query params
    stream = request.args.get('stream', 'thelegendyagami').strip().lower()
    game   = request.args.get('game', '').strip().lower()
    try:
        top_n = int(request.args.get('top_n', 10))
    except ValueError:
        top_n = 10

    if not ready:
        return render_template_string(
            TEMPLATE,
            ready=False,
            stream=stream,
            game=game,
            top_n=top_n,
            predictions=[],
        )

    # ensure our stored df_for_inf uses lowercase for matching
    df_for_inf['stream_name']   = df_for_inf['stream_name'].str.lower()
    df_for_inf['game_category'] = df_for_inf['game_category'].str.lower()
    cat_opts = [c.lower() for c in (cat_opts or [])]

    # make sure the stream exists
    if stream not in df_for_inf['stream_name'].unique():
        return render_template_string(
            TEMPLATE,
            ready=True,
            stream=stream,
            game=game,
            top_n=top_n,
            predictions=[],
        )

    # choose a valid game
    if not game or game not in cat_opts:
        # default to that stream’s last recorded game
        last_game = df_for_inf[df_for_inf['stream_name'] == stream]['game_category'].iloc[-1]
        game = last_game.lower()

    # run inference restricted to that game
    top_df = _infer_grid_for_game(
        pipe,
        df_for_inf,
        features,
        stream_name=stream,
        start_times=start_opts,
        durations=dur_opts,
        category_options=[game],
        top_n=top_n,
        unique_scores=True,
    )

    # format for rendering
    disp = top_df.copy()
    disp['Time']          = disp['start_time_hour'].astype(int).map(lambda h: f"{h:02d}:00")
    disp['Duration']      = disp['stream_duration'].astype(int)
    disp['Expected_Subs'] = disp['y_pred'].round().astype(int)
    disp['Confidence']    = disp['conf'].round(1)

    records = disp[['Time','Duration','Expected_Subs','Confidence']].to_dict(orient='records')

    return render_template_string(
        TEMPLATE,
        ready=True,
        stream=stream,
        game=game,
        top_n=top_n,
        predictions=records,
    )
