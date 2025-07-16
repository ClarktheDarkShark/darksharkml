import os
from flask import Flask
from db import db
from predictor import train_predictor
from dashboard_predictions import dash_preds

def create_app():
    app = Flask(__name__)

    # ── Configure your database URI; Heroku provides DATABASE_URL env var
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'DATABASE_URL',
        'sqlite:///local.db'   # fallback for local dev
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # ── Initialize your SQLAlchemy db object
    db.init_app(app)

    # ── Register the predictions dashboard
    app.register_blueprint(dash_preds)

    # ── Train the predictor on startup (blocks until done)
    with app.app_context():
        train_predictor(app)

    return app

# expose app for gunicorn
app = create_app()

if __name__ == '__main__':
    # local dev
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
