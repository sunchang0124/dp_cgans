# pos_weight in BCEWithLogitsLoss model #
# sigma = 5 #
# delta = 2e-6 #

import warnings

import numpy as np
import pandas as pd
import math
import random
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, BCEWithLogitsLoss, utils
from tqdm import tqdm

from dp_cgans.ontocgan.onto_data_sampler import Onto_DataSampler
from dp_cgans.synthesizers.data_transformer import DataTransformer
from dp_cgans.synthesizers.base import BaseSynthesizer


######## ADDED ########

from dp_cgans.functions.rdp_accountant import compute_rdp, get_privacy_spent

######## ADDED - wandb ########
from types import SimpleNamespace
from pathlib import Path
import copy

class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
            gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pac == 0
        return self.seq(input.view(-1, self.pacdim))


class Residual(Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):

    def __init__(self, input_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = input_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


class Onto_DPCGANSynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        log_file_path (str):
           Path to log the losses if verbose is True
        embedding (OntologyEmbedding):
            OntologyEmbedding instance to retrieve the ontology embeddings.
        noise_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        private (bool): 
            Whether to use differential privacy
        wandb_config (dict):
            whether to use weights and bias tool to monitor the training
        conditional_columns (float):
            a matrix of embeddings
    """

    def __init__(self, log_file_path, embedding=None, noise_dim=100,
                 generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, private=False,
                 wandb=False, conditional_columns=None):

        assert batch_size % 2 == 0

        self._embedding = embedding
        self._noise_dim = noise_dim

        self._log_file_path = log_file_path

        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        print(f'Verbose: {self._verbose}')

        self.private = private
        self.conditional_columns = conditional_columns
        self.wandb = wandb
  
            

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        if self._verbose:
            print(f'Using {device}')

        self._device = torch.device(device)

        # 👇 Add this block
        if self._device.type == "cuda":
            torch.cuda.init()                 # initialize CUDA runtime
            _ = torch.cuda.current_device()   # touch device 0
            torch.cuda.synchronize()          # ensure context is live
            print("CUDA context created on", torch.cuda.get_device_name(0))

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None
    
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])


    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss_pair(self, data, c_pair, m_pair):

        loss=[]

        data  = torch.from_numpy(data).to(self._device)

        for each_dim in range(0, len(data[0])):
  
            criterion = BCEWithLogitsLoss(reduction='none')
            each_loss = criterion(data[:,each_dim],c_pair[:, each_dim])
            loss.append(each_loss)

        loss = torch.stack(loss, dim=1)

        return loss.sum() / len(loss[0]) / len(loss)  #(loss.sum() / len(loss[0]) / len(loss)) * 100
        
    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError('Invalid columns found: {}'.format(invalid_columns))


    ############ Tensorflow Privacy Measurement ##############

    def fit(self, train_data, discrete_columns=tuple(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            epochs (int):
                Number of epochs to train the model for.
        """

        # if self.conditional_columns != None:
        #     if set(self.conditional_columns) <= set(discrete_columns):
        #         discrete_columns = self.conditional_columns
        #     else:
        #         raise NotImplementedError("Conditional columns are not in the valid columns.",discrete_columns)
        if self.wandb == True:
            config = SimpleNamespace(
                epochs=epochs, # number of training epochs
                batch_size=self._batch_size, # the size of each batch
                log_frequency=self._log_frequency,
                verbose=self._verbose,
                generator_dim=self._generator_dim,
                discriminator_dim=self._discriminator_dim,
                generator_lr=self._generator_lr,
                discriminator_lr=self._discriminator_lr,
                discriminator_steps=self._discriminator_steps, 
                private=self.private
                
            )


        real_data_columns = [col for col in train_data.columns if '.value' in col]
        real_data = copy.deepcopy(train_data[real_data_columns])


        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        full_transformer = DataTransformer()
        full_transformer.fit(train_data, discrete_columns)
        train_data_full = full_transformer.transform(train_data)

        # Getting the list of unique RDs, sorted by order of appearance to correspond to category IDs
        rds = train_data.iloc[:, 0].values
        _, idx = np.unique(rds, return_index=True)
        rds = rds[np.sort(idx)]

        self._data_sampler = Onto_DataSampler(
            train_data_full, ##### checking point
            rds,
            full_transformer.output_info_list,
            self._log_frequency,
            self._embedding)

        # removing the RDs column for ZSL
        train_data = train_data.drop(columns=train_data.columns[0], axis=1)

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)

        data_dim = self._transformer.output_dimensions


        self._generator = Generator(
            self._noise_dim + self._embedding.embed_size,
            self._generator_dim,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._embedding.embed_size,
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._noise_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])
        
        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))
            
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        ######## ADDED ########
        
        for i in epoch_iterator:
            for step in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec_pair = self._data_sampler.sample_condvec_pair(self._batch_size)
                    

                    c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair

                    if condvec_pair is None:
                        c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
                        real = self._data_sampler.sample_data_pair(self._batch_size, col_pair_1, opt_pair_1)
                    else:
                        # Getting ontology embeddings
                        real_embeddings = self._data_sampler.get_embeds_from_cat_ids(cat_ids=c_pair_1, batch_size=self._batch_size)
                        real_embeddings = torch.from_numpy(real_embeddings).to(self._device)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        c_pair_2 = c_pair_1[perm]
                        fakez = torch.cat([fakez, real_embeddings], dim=1)

                        real = self._data_sampler.sample_data_pair(self._batch_size, col_pair_1[perm], opt_pair_1[perm])
                    
                        real_embeddings_2 = self._data_sampler.get_embeds_from_cat_ids(cat_ids=c_pair_2, batch_size=self._batch_size)
                        real_embeddings_2 = torch.from_numpy(real_embeddings_2).to(self._device)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if col_pair_1 is not None:
                        fake_cat = torch.cat([fakeact, real_embeddings], dim=1)
                        real_cat = torch.cat([real, real_embeddings_2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fake
    

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))


                    #### DP ####
                    if self.private:
                        sigma = 5
                        weight_clip = 0.01 

                        if sigma is not None:
                            for parameter in discriminator.parameters():
                                parameter.register_hook(
                                    lambda grad: grad + (1 / self._batch_size) * sigma
                                    * torch.randn(parameter.shape)
                                )
                    #### DP ####

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)  # https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/ 
                    loss_d.backward()
                    optimizerD.step()

                    if self.private:
                        #### DP ####
                        # Weight clipping for privacy guarantee
                        for param in discriminator.parameters():
                            param.data.clamp_(-weight_clip, weight_clip)
                        #### DP ####

                fakez = torch.normal(mean=mean, std=std)
                condvec_pair = self._data_sampler.sample_condvec_pair(self._batch_size)

                if condvec_pair is None:
                    c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
                else:
                    c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair

                    # Getting ontology embeddings
                    real_embeddings = self._data_sampler.get_embeds_from_cat_ids(cat_ids=c_pair_1, batch_size=self._batch_size)
                    real_embeddings = torch.from_numpy(real_embeddings).to(self._device)

                    c_pair_1 = torch.from_numpy(c_pair_1).to(self._device)
                    m_pair_1 = torch.from_numpy(m_pair_1).to(self._device)
                    fakez = torch.cat([fakez, real_embeddings], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c_pair_1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, real_embeddings], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec_pair is None:
                    cross_entropy_pair = 0
                else:
                    ##### ONTO-CGAN embeddings for generated data #####
                    generated_data = []
                    generated_data.append(fakeact.detach().cpu().numpy())
                    generated_data = np.concatenate(generated_data, axis=0)
                    generated_data = generated_data[:len(fakeact)]

                    sampled_generated_data = self._transformer.inverse_transform(generated_data)
                    sampled_generated_data["icd_code"] = sampled_generated_data["icd_code"].str.replace('ORDO.', 'http://www.orpha.net/ORDO/',regex=True)
                    generated_embeddings = self._data_sampler.get_rd_embeds(sampled_generated_data["icd_code"]) ### hard code
                    ###############################
                    
                    # cross_entropy_pair = self._cond_loss_pair(fake, c_pair_1, m_pair_1)
                    cross_entropy_pair = self._cond_loss_pair(generated_embeddings, real_embeddings, m_pair_1)
                    # print(-torch.mean(y_fake), cross_entropy_pair)

                loss_g = -torch.mean(y_fake) + cross_entropy_pair # + rules_penalty


                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()


            generator_loss = loss_g.detach().cpu()
            discriminator_loss = loss_d.detach().cpu()

    
            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df


                
            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )


            if self.private:
                orders = [1 + x / 10. for x in range(1, 100)]
                sampling_probability = self._batch_size/len(train_data)
                delta = 2e-6
                rdp = compute_rdp(q=sampling_probability,
                                  noise_multiplier=sigma,
                                  steps=i * steps_per_epoch,
                                  orders=orders)
                epsilon, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta) # target_delta=1e-5

                # log_file.write(f'Differential privacy with eps = {epsilon:.3g} and delta = {delta}.\n')
                # log_file.write(f'The optimal RDP order is {opt_order}.\n')

                # if opt_order == max(orders) or opt_order == min(orders):
                #     log_file.write('The privacy estimate is likely to be improved by expanding the set of orders.\n')
            else:
                epsilon = np.nan

                    

                    

    def sample(self, n, unseen_rds=[], sort=True):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            unseen_rds (list-like):
                List-like object containing names of unseen RDs to sample.
                Does not sample from seen_rds if there is at least one item in it.
            sort (bool):
                Whether to sort the resulting dataframe on RDs (alphabetically). Defaults to True.
        Returns:
            (Pandas.DataFrame):
                The sampled data.
        """

        steps = n // self._batch_size + 1
        data = []
        sampled_rds = []

        # Using unseen RDs for sampling if a list was passed as argument
        if len(unseen_rds) > 0:
            # DUplicate the rds to match batch_size
            unseen_rds = [rd for rd in unseen_rds for repetitions in range(math.ceil(self._batch_size/len(unseen_rds)))]
            unseen_rds = unseen_rds[:self._batch_size]

        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._noise_dim)

            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if len(unseen_rds) > 0:
                # Note: unsure if shuffle is useful here
                random.shuffle(unseen_rds)
                # Getting ontology embeddings from the list of unseen RDs
                real_embeddings = self._data_sampler.get_rd_embeds(unseen_rds)
                real_embeddings = torch.from_numpy(real_embeddings).to(self._device)

                fakez = torch.cat([fakez, real_embeddings], dim=1)
                sampled_rds += unseen_rds
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

                if condvec is None:
                    pass
                else:
                    c1, m1 = condvec
                    # Getting ontology embeddings
                    rds = self._data_sampler.get_rds(cat_ids=c1, batch_size=self._batch_size)
                    real_embeddings = self._data_sampler.get_rd_embeds(rds)
                    real_embeddings = torch.from_numpy(real_embeddings).to(self._device)

                    fakez = torch.cat([fakez, real_embeddings], dim=1)
                    sampled_rds += rds

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)

            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        sampled_data = self._transformer.inverse_transform(data)

        return sampled_data

    def set_device(self, device):
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)

    def xai_discriminator(self, data_samples):

        # for exlain AI (SHAP) the single row from the pd.DataFrame needs to be transformed. 
        data_samples = pd.DataFrame(data_samples).T

        condvec_pair = self._data_sampler.sample_condvec_pair(len(data_samples))
        c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair

        if condvec_pair is None:
            c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
            real = self._data_sampler.sample_data_pair(len(data_samples), col_pair_1, opt_pair_1)
        else:
            c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair
            c_pair_1 = torch.from_numpy(c_pair_1).to(self._device)
            m_pair_1 = torch.from_numpy(m_pair_1).to(self._device)

            perm = np.arange(len(data_samples))
            np.random.shuffle(perm)

            real = self._data_sampler.sample_data_pair(len(data_samples), col_pair_1[perm], opt_pair_1[perm])
            c_pair_2 = c_pair_1[perm]

        real = torch.from_numpy(real.astype('float32')).to(self._device)

        if col_pair_1 is not None:
            real_cat = torch.cat([real, c_pair_2], dim=1)
        else:
            real_cat = real

        ### Wassertein distance?? (a data point from real training data's wassertain distance means what?)
        discriminator_predict_score = self._discriminator(real_cat)

        return discriminator_predict_score