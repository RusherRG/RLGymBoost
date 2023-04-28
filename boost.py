import typer

from utils import get_logger

app = typer.Typer()
logger = get_logger(__name__)


@app.command()
def tuner():
    logger.info("Running Tuner")


@app.command()
def trainer():
    logger.info("Running Trainer")


if __name__ == "__main__":
    app()
