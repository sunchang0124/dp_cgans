import glob
import sys

import pandas as pd
import pkg_resources
import typer
from dp_cgans import DP_CGAN

cli = typer.Typer()

# Variables to make the prints gorgeous:
BOLD = '\033[1m'
END = '\033[0m'
GREEN = '\033[32m'
# RED = '\033[91m'
# YELLOW = '\033[33m'
# CYAN = '\033[36m'
# PURPLE = '\033[95m'
# BLUE = '\033[34m'


@cli.command("gen")
def cli_gen(
    input_file: str,
    gen_size: int = typer.Option(100, help="Number of rows in the generated samples file"),
    epochs: int = typer.Option(100, help="Number of epochs"),
    batch_size: int = typer.Option(1000, help="Batch size"),
    output: str = typer.Option("synthetic_samples.csv", help="Path to the output"),
    verbose: bool = typer.Option(True, help="Display logs")
):

    tabular_data=pd.read_csv(input_file)

    model = DP_CGAN(
        epochs=epochs, # number of training epochs
        batch_size=batch_size, # the size of each batch
        log_frequency=True,
        verbose=verbose,
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-4, 
        discriminator_lr=2e-4,
        discriminator_steps=1, 
        private=False,
        wandb=False
    )

    if verbose: print(f'🗜️  Model instantiated, fitting...')
    model.fit(tabular_data)

    if verbose: print(f'🧪 Model fitted, sampling...')
    sample = model.sample(gen_size)

    sample.to_csv(output)
    if verbose: print(f'✅ Samples generated in {BOLD}{GREEN}{output}{END}')


@cli.command("version")
def cli_version():
    print(pkg_resources.get_distribution('dp_cgans').version)


if __name__ == "__main__":
    cli()