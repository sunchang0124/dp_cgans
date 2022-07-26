import glob
import sys

import click
import pandas as pd
from dp_cgans import DP_CGAN


@click.group()
def cli():
    """Generate synthetic data"""
    pass


@cli.command(
    help="Generate synthetic data"
)
@click.argument("input-file", nargs=1)
@click.option("--epochs", default=100, help="Number of epochs")
@click.option("--batch-size", default=1000, help="Batch size")
@click.option("--output", default="synthetic_samples.csv", help="Path to the output")

def gen(input_file, epochs, batch_size, output):

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

    model.fit(tabular_data)

    # Sample the generated synthetic data
    sample = model.sample(100)

    sample.to_csv(output)

    # asset sample[0]['score'] >= 0.8

if __name__ == "__main__":
    sys.exit(cli())