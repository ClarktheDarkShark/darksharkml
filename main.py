import os
from flask import Flask
from db import db
from predictor import train_predictor
from dashboard_predictions import dash_preds

def create_app():
    app = Flask(__name__)

    # ── Grab the URL and patch the scheme if needed ─────────────────────────
    db_url = os.environ.get('DATABASE_URL', '')
    if db_url.startswith('postgres://'):
        # SQLAlchemy expects postgresql://
        db_url = db_url.replace('postgres://', 'postgresql://', 1)

    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # ── Initialize DB & blueprints ────────────────────────────────────────
    db.init_app(app)
    app.register_blueprint(dash_preds)

    with app.app_context():
        # create tables if you’re not running migrations
        db.create_all()

        # train the predictor
        train_predictor(app)

    return app

app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
