import glob
import sys

import typer
import pandas as pd
from dp_cgans import DP_CGAN
import pkg_resources


cli = typer.Typer()

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
        verbose=False,
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-4, 
        discriminator_lr=2e-4,
        discriminator_steps=1, 
        private=False,
    )

    if verbose: print(f'Model instantiated, fitting')
    model.fit(tabular_data)

    if verbose: print(f'Model fit')
    # Sample the generated synthetic data
    sample = model.sample(gen_size)

    sample.to_csv(output)
    if verbose: print(f'✔️ Samples generated in {output}')


@cli.command("version")
def cli_version():
    print(pkg_resources.get_distribution('dp_cgans').version)


if __name__ == "__main__":
    cli()