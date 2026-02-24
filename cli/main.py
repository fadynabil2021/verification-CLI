import typer

from model.inference import classify_log

app = typer.Typer(add_completion=False)


@app.command()
def classify(log: str = typer.Option(..., "--log", help="Path to simulation log")):
    with open(log, "r", encoding="utf-8") as f:
        content = f.read()
    result = classify_log(content)
    typer.echo(f"label: {result['label']}")
    typer.echo(f"confidence: {result['confidence']}")
    typer.echo(f"explanation: {result['explanation']}")


if __name__ == "__main__":
    app()
