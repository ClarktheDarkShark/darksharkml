import os, threading, sys, asyncio
sys.modules["main"] = sys.modules[__name__] 
from flask import Flask, send_file, abort
from dotenv import load_dotenv
import logging
from db import db


# print("PID:", os.getpid(), "WERKZEUG_RUN_MAIN:", os.getenv("WERKZEUG_RUN_MAIN"))

# Load environment variables (including DATABASE_URL)
load_dotenv()

def create_app(include_migrate: bool = False):
    from dashboard import dash 
    # from dashboard_predictions import dash_preds
    app = Flask(__name__)

    logging.getLogger('werkzeug').setLevel(logging.ERROR)

    # 1) Database configuration
    uri = os.getenv("DATABASE_URL", "sqlite:///local.db")
    if uri.startswith("postgres://"):
        uri = uri.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # 2) Initialize DB + migrations
    db.init_app(app)

    if include_migrate:
        from flask_migrate import Migrate
        Migrate(app, db)

    app.register_blueprint(dash)
    # app.register_blueprint(dash_preds)

    @app.route("/")
    def home():
        return "Bot is running."           # Bot is running."

    @app.route('/images/<filename>')
    def serve_image(filename):
        filepath = os.path.join('/tmp', filename)
        if not os.path.isfile(filepath):
            abort(404)
        return send_file(filepath, mimetype='image/png')

    return app

app = create_app()

_bot_holder: dict[str, object] = {}

# --- main.py --------------------------------------------------------------

def run_flask():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port,
            debug=False, use_reloader=False, threaded=False)

async def run_bots():
    """Create and start BOTH Twitch bots on the same asyncio loop."""
    if os.getenv("RUN_BOT", "1") != "1":
        await asyncio.Event().wait()          # RUN_BOT disabled → sleep forever

    from bot import Bot as ChatBot   # primary bot
    from stats_bot import StatsBot               # stats-collector bot
    import constants

    stats_bot_channels = constants.MAIN_CHANNELS + constants.TEST_USERS
    stats_bot = StatsBot(channels=stats_bot_channels)
    chat_bot  = ChatBot()

    _bot_holder["bot"]       = chat_bot
    _bot_holder["stats_bot"] = stats_bot

    # run until both are stopped (Ctrl-C / SIGTERM)
    await asyncio.gather(chat_bot.start(), stats_bot.start())
    # await asyncio.gather(chat_bot.start())
    
if __name__ == "__main__":
    # 1) start Flask in a background thread
    threading.Thread(target=run_flask, daemon=True).start()

    try:
        asyncio.run(run_bots())
    except KeyboardInterrupt:
        pass
