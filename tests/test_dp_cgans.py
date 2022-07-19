import pandas as pd
from dp_cgans import DP_CGAN


def test_dp_cgans():
    print('Testing')

    tabular_data=pd.read_csv("dataset/example_tabular_data_UCIAdult.csv")

    model = DP_CGAN(
        epochs=1, # number of training epochs
        batch_size=1000, # the size of each batch
        log_frequency=True,
        verbose=True,
        generator_dim=(128, 128, 128),
        discriminator_dim=(128, 128, 128),
        generator_lr=2e-4, 
        discriminator_lr=2e-4,
        discriminator_steps=1, 
        private=False,
    )

    print("Start training model")
    model.fit(tabular_data)

    # Sample the generated synthetic data
    model.sample(100)

    assert True == True