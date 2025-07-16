# train_worker.py
from main import create_app
from predictor import train_predictor

if __name__ == "__main__":
    app = create_app()
    # this will create tables and then train
    train_predictor(app)
    print("Training complete.")
