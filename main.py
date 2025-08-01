import os
from flask import Flask
from db import db
from flask_caching import Cache

from dashboard_predictions import dash_preds
from dashboard_v2 import dash_v2
from dashboard_v3 import dash_v3

def create_app():
    app = Flask(__name__)

    # ── Database configuration (pointed via DATABASE_URL) ───────────────────
    db_url = os.environ.get('DATABASE_URL', '')
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # ── Initialize DB and Blueprints ────────────────────────────────────────
    db.init_app(app)
    app.register_blueprint(dash_preds)
    app.register_blueprint(dash_v2)  # Remove prefix; route will be accessible at /v2
    
    app.register_blueprint(dash_v3)

    with app.app_context():
        # create tables if you’re not running migrations
        db.create_all()

        # no more train_predictor(app) here!

    return app

# expose the app to Gunicorn
app = create_app()
cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': os.environ['REDIS_URL'],  # set this in Heroku config vars
    'CACHE_DEFAULT_TIMEOUT': 3600,               # 1h default TTL
})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
