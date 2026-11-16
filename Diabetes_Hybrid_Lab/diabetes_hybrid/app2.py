"""Run the English demo."""
from app import create_app, run

app = create_app("en")


if __name__ == "__main__":
    run(app)
