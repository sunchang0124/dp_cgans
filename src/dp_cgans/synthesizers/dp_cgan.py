import os
import torch
import threading
import queue
import time

from datetime import datetime
import warnings
import joblib
import numpy as np
import pandas as pd
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, BCEWithLogitsLoss
from tqdm import tqdm

from dp_cgans.synthesizers.data_sampler import DataSampler
from dp_cgans.synthesizers.data_transformer import DataTransformer
from dp_cgans.synthesizers.base import BaseSynthesizer

######## ADDED ########
from contextlib import redirect_stdout
from dp_cgans.functions.rdp_accountant import compute_rdp, get_privacy_spent

######## ADDED - wandb ########
from types import SimpleNamespace
from pathlib import Path
import copy
import matplotlib.pyplot as plt

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

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

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

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


class DataPrefetcher:
    """Multi-threaded prefetcher with parallel workers to maximize data throughput."""

    def __init__(self, data_sampler, batch_size, device, num_workers=8, prefetch_count=16):
        self.data_sampler = data_sampler
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.queue = queue.Queue(maxsize=prefetch_count)
        self.threads = []
        self.stop_flag = False
        self.error_count = 0

    def _worker(self, worker_id):
        """Background thread worker that prefetches data."""
        while not self.stop_flag:
            try:
                # Sample conditional vector pair
                condvec_pair = self.data_sampler.sample_condvec_pair(self.batch_size)

                if condvec_pair is None:
                    c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
                    real = self.data_sampler.sample_data_pair(self.batch_size, col_pair_1, opt_pair_1)
                    perm = None
                else:
                    c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair
                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = self.data_sampler.sample_data_pair(self.batch_size, col_pair_1[perm], opt_pair_1[perm])

                # Put the prefetched data in queue (blocks if queue is full)
                if not self.stop_flag:
                    self.queue.put((condvec_pair, real, perm), timeout=1)
            except queue.Full:
                continue
            except Exception as e:
                if not self.stop_flag:
                    self.error_count += 1
                    if self.error_count <= 3:  # Only print first 3 errors
                        print(f"Prefetcher worker {worker_id} error: {e}")
                        import traceback
                        traceback.print_exc()
                break

    def start(self):
        """Start all prefetcher worker threads."""
        self.stop_flag = False
        self.error_count = 0
        for i in range(self.num_workers):
            thread = threading.Thread(target=self._worker, args=(i,), daemon=True)
            thread.start()
            self.threads.append(thread)
        print(f"Started {self.num_workers} parallel data prefetcher workers")

    def get(self, timeout=30):
        """Get the next prefetched batch."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            print(f"WARNING: Prefetcher queue empty! Workers may have crashed.")
            return None

    def stop(self):
        """Stop all prefetcher threads."""
        self.stop_flag = True
        # Clear the queue to unblock any waiting workers
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except queue.Empty:
            pass

        # Wait for all threads to finish
        for thread in self.threads:
            if thread and thread.is_alive():
                thread.join(timeout=2)
        self.threads = []
        print(f"Stopped all prefetcher workers")


class DPCGANSynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
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

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True, private=False,
                 wandb=False, conditional_columns=None):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
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

        self.private = private
        self.conditional_columns = conditional_columns
        self.wandb = wandb


        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self._discriminator = None


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
        
        output_info_all_columns = self._transformer.output_info_list
        loss = torch.zeros(
            (len(data)*int((m_pair.size()[1]*(m_pair.size()[1]-1))/2),m_pair.size()[1]),
            device=self._device
            )
        st_primary = 0
        st_primary_c = 0
        cnt = 0
        cnt_primary=0
        for index_primary in range(0, len(output_info_all_columns)):
            column_info_primary = output_info_all_columns[index_primary]
            for span_info_primary in column_info_primary:
                if len(column_info_primary) != 1 or span_info_primary.activation_fn != "softmax":
                    # not discrete column
                    st_primary += span_info_primary.dim
                else:
        
                    ed_primary = st_primary + span_info_primary.dim
                    ed_primary_c = st_primary_c + span_info_primary.dim

                    cnt_secondary=cnt_primary+1
                    st_secondary = ed_primary
                    st_secondary_c = ed_primary_c
                    for index_secondary in range(index_primary+1, len(output_info_all_columns)):
                        column_info_secondary = output_info_all_columns[index_secondary]
                        for span_info_secondary in column_info_secondary:
                            if len(column_info_secondary) != 1 or span_info_secondary.activation_fn != "softmax":
                                # not discrete column
                                st_secondary += span_info_secondary.dim
                            else:

                                ed_secondary = st_secondary + span_info_secondary.dim
                                ed_secondary_c = st_secondary_c + span_info_secondary.dim
                                
                                real_data_labels = torch.cat([data[:,st_primary:ed_primary], data[:,st_secondary:ed_secondary]], dim=1)
                                class_counts = real_data_labels.sum(axis=0)

                                criterion = BCEWithLogitsLoss(reduction='none').to(self._device)#, pos_weight=pos_weights)
                                calculate_loss = criterion(
                                    torch.cat([data[:,st_primary:ed_primary], data[:,st_secondary:ed_secondary]], dim=1),
                                    torch.cat([c_pair[:,st_primary_c:ed_primary_c], c_pair[:,st_secondary_c:ed_secondary_c]],dim=1)
                                    )

                                # calculate_loss = calculate_loss.detach().numpy()
                                loss[cnt*len(data):(cnt+1)*len(data),cnt_primary] = calculate_loss[:,:span_info_primary.dim].sum(axis=1) * m_pair[:,cnt_primary]
                                loss[cnt*len(data):(cnt+1)*len(data),cnt_secondary] = calculate_loss[:,span_info_primary.dim:].sum(axis=1) * m_pair[:,cnt_secondary]

                                st_secondary = ed_secondary
                                st_secondary_c = ed_secondary_c
                                cnt += 1
                                cnt_secondary += 1

                    cnt_primary += 1
                    st_primary = ed_primary
                    st_primary_c = ed_primary_c
        
        result = loss.sum() / len(loss)
        return result


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
        """
        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """

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

            # wandb_config = wandb.init(
            #     project="dp_cgans_training_monitor",
            #     anonymous="allow",
            #     config=config
            # )
 
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

        self._transformer = DataTransformer()

        print("Start transforming data...")
        
        if os.path.exists(os.getcwd()+'/fitted_transformer.pkl'):
            print("Loading fitted transformer...")
            self._transformer = joblib.load(os.getcwd()+'/fitted_transformer.pkl')
        else:
            print("Start fitting transformer ...")
            self._transformer.fit(train_data, discrete_columns)
            self._transformer.fit(train_data, discrete_columns)
            joblib.dump(self._transformer, os.getcwd()+'/fitted_transformer.pkl')
            print("Saving fitted transformer...")
        
        print("Finish transforming data / loading transformed data...")

        train_data = self._transformer.transform(train_data)
        
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions
        

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(), # number of categories in the whole dataset.
            self._generator_dim,
            data_dim
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])
        
        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

            
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)

        # DISABLED: Multi-threading causes severe CPU/memory contention that kills GPU performance
        # Python's GIL and numpy operations don't parallelize well
        # prefetcher = DataPrefetcher(...)
        prefetcher = None

        print(f"Training Configuration:")
        print(f"  Batch size: {self._batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Generator dim: {self._generator_dim}")
        print(f"  Discriminator dim: {self._discriminator_dim}")
        print(f"  Device: {self._device}")
        print(f"  Data shape: {train_data.shape}")
        print(f"  Prefetching: DISABLED (direct sampling - threading caused contention)")
        print(f"{datetime.now()}", 'epoch: ', epoch_iterator)
        filename = f'loss_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(filename, 'w') as loss_file:
            loss_file.write("Epoch,Generator_Loss,Discriminator_Loss,Timestamp\n")
            for i in epoch_iterator:
                if i == 0 and self._device.type == 'cuda':
                    torch.cuda.init()
                    torch.cuda.synchronize()
                    print("CUDA initialized:", torch.cuda.get_device_name(0))

                # Timing diagnostics for first epoch
                if i == 0:
                    timing_stats = {
                        'disc_data_fetch': [],
                        'disc_forward': [],
                        'disc_backward': [],
                        'gen_data_fetch': [],
                        'gen_forward': [],
                        'cond_loss_pair': [],
                        'gen_backward': [],
                        'step_total': [],
                        'queue_size': []
                    }

                for id_ in range(steps_per_epoch):
                    if i == 0:
                        step_start = time.time()
                        # Track queue health
                        if prefetcher:
                            timing_stats['queue_size'].append(prefetcher.queue.qsize())

                    for n in range(self._discriminator_steps):
                        if i == 0:
                            data_start = time.time()

                        fakez = torch.normal(mean=mean, std=std).to(self._device)

                        # Direct sampling (prefetcher disabled)
                        prefetch_result = prefetcher.get() if prefetcher else None
                        if prefetch_result is None:
                            # Fallback to direct sampling if prefetcher fails
                            condvec_pair = self._data_sampler.sample_condvec_pair(self._batch_size)
                            if condvec_pair is None:
                                c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
                                real = self._data_sampler.sample_data_pair(self._batch_size, col_pair_1, opt_pair_1)
                                perm = None
                            else:
                                c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair
                                perm = np.arange(self._batch_size)
                                np.random.shuffle(perm)
                                real = self._data_sampler.sample_data_pair(self._batch_size, col_pair_1[perm], opt_pair_1[perm])
                        else:
                            condvec_pair, real, perm = prefetch_result

                        if condvec_pair is None:
                            c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
                        else:
                            c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair
                            c_pair_1 = torch.from_numpy(c_pair_1).to(self._device, non_blocking=True)
                            m_pair_1 = torch.from_numpy(m_pair_1).to(self._device, non_blocking=True)
                            fakez = torch.cat([fakez, c_pair_1], dim=1)
                            c_pair_2 = c_pair_1[perm]


                        real = torch.from_numpy(real.astype('float32')).to(self._device, non_blocking=True)

                        if i == 0:
                            data_time = time.time() - data_start
                            forward_start = time.time()

                        fake = self._generator(fakez) # categories (unique value count) + continuous (1+n_components)
                        fakeact = self._apply_activate(fake)

                        
                        
                        if col_pair_1 is not None:
                            fake_cat = torch.cat([fakeact, c_pair_1], dim=1)
                            real_cat = torch.cat([real, c_pair_2], dim=1)
                        else:
                            real_cat = real
                            fake_cat = fake
                        

                        y_fake = self._discriminator(fake_cat)
                        y_real = self._discriminator(real_cat)
                        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                        if i == 0:
                            forward_time = time.time() - forward_start
                            backward_start = time.time()

                        #### DP ####
                        if self.private:
                            sigma = 1
                            weight_clip = 0.01

                            if sigma is not None:
                                for parameter in self._discriminator.parameters():
                                    parameter.register_hook(
                                        lambda grad: grad.to(self._device) + (1 / self._batch_size) * sigma
                                        * torch.randn(parameter.shape).to(self._device)
                                    )
                        #### DP ####
                        pen = self._discriminator.calc_gradient_penalty(
                            real_cat, fake_cat, self._device, self.pac)

                        optimizerD.zero_grad()
                        # https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
                        pen.backward(retain_graph=True)
                        loss_d.backward()
                        optimizerD.step()
                        if self.private:
                            #### DP ####
                            # Weight clipping for privacy guarantee
                            for param in self._discriminator.parameters():
                                param.data.clamp_(-weight_clip, weight_clip)
                            #### DP ####

                        if i == 0:
                            backward_time = time.time() - backward_start
                            timing_stats['disc_data_fetch'].append(data_time)
                            timing_stats['disc_forward'].append(forward_time)
                            timing_stats['disc_backward'].append(backward_time)

                    if i == 0:
                        gen_data_start = time.time()

                    fakez = torch.normal(mean=mean, std=std).to(self._device)

                    # Direct sampling (prefetcher disabled)
                    prefetch_result = prefetcher.get() if prefetcher else None
                    if prefetch_result is None:
                        # Direct sampling
                        condvec_pair = self._data_sampler.sample_condvec_pair(self._batch_size)
                    else:
                        condvec_pair, _, _ = prefetch_result  # Only need condvec_pair for generator

                    if condvec_pair is None:
                        c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = None, None, None, None
                    else:
                        c_pair_1, m_pair_1, col_pair_1, opt_pair_1 = condvec_pair

                        c_pair_1 = torch.from_numpy(c_pair_1).to(self._device, non_blocking=True)
                        m_pair_1 = torch.from_numpy(m_pair_1).to(self._device, non_blocking=True)
                        fakez = torch.cat([fakez, c_pair_1], dim=1)

                    if i == 0:
                        gen_data_time = time.time() - gen_data_start
                        gen_forward_start = time.time()

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    if c_pair_1 is not None:
                        y_fake = self._discriminator(torch.cat([fakeact, c_pair_1], dim=1))
                    else:
                        y_fake = self._discriminator(fakeact)

                    if i == 0:
                        gen_forward_time = time.time() - gen_forward_start
                        cond_loss_start = time.time()

                    if condvec_pair is None:
                        cross_entropy_pair = 0
                    else:
                        cross_entropy_pair = self._cond_loss_pair(fake, c_pair_1, m_pair_1)

                    if i == 0:
                        cond_loss_time = time.time() - cond_loss_start
                        gen_backward_start = time.time()

                    # loss_g_pure =  -torch.mean(y_fake)
                    loss_g = -torch.mean(y_fake) + cross_entropy_pair # + rules_penalty

                    optimizerG.zero_grad(set_to_none=False)
                    loss_g.backward()
                    optimizerG.step()

                    if i == 0:
                        gen_backward_time = time.time() - gen_backward_start
                        step_total_time = time.time() - step_start

                        # Record all timings
                        timing_stats['gen_data_fetch'].append(gen_data_time)
                        timing_stats['gen_forward'].append(gen_forward_time)
                        timing_stats['cond_loss_pair'].append(cond_loss_time)
                        timing_stats['gen_backward'].append(gen_backward_time)
                        timing_stats['step_total'].append(step_total_time)

                generator_loss = loss_g.detach().cpu().item()
                discriminator_loss = loss_d.detach().cpu().item()
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                loss_file.write(f"{i},{generator_loss:.3f},{discriminator_loss:.3f},{timestamp}\n")
                loss_file.flush()  # Write immediately to disk

                # Print detailed timing statistics after first epoch
                if i == 0:
                    total_time = np.sum(timing_stats['step_total'])
                    disc_data = np.sum(timing_stats['disc_data_fetch'])
                    disc_fwd = np.sum(timing_stats['disc_forward'])
                    disc_bwd = np.sum(timing_stats['disc_backward'])
                    gen_data = np.sum(timing_stats['gen_data_fetch'])
                    gen_fwd = np.sum(timing_stats['gen_forward'])
                    cond_loss = np.sum(timing_stats['cond_loss_pair'])
                    gen_bwd = np.sum(timing_stats['gen_backward'])

                    measured_total = disc_data + disc_fwd + disc_bwd + gen_data + gen_fwd + cond_loss + gen_bwd
                    unmeasured = total_time - measured_total

                    print("\n" + "="*60)
                    print("  DETAILED FIRST EPOCH TIMING BREAKDOWN")
                    print("="*60)
                    print(f"DISCRIMINATOR TRAINING:")
                    print(f"  Data fetching:  {disc_data:6.2f}s ({100*disc_data/total_time:5.1f}%) - avg {disc_data/len(timing_stats['disc_data_fetch'])*1000:6.1f}ms")
                    print(f"  Forward pass:   {disc_fwd:6.2f}s ({100*disc_fwd/total_time:5.1f}%) - avg {disc_fwd/len(timing_stats['disc_forward'])*1000:6.1f}ms")
                    print(f"  Backward pass:  {disc_bwd:6.2f}s ({100*disc_bwd/total_time:5.1f}%) - avg {disc_bwd/len(timing_stats['disc_backward'])*1000:6.1f}ms")
                    print(f"\nGENERATOR TRAINING:")
                    print(f"  Data fetching:  {gen_data:6.2f}s ({100*gen_data/total_time:5.1f}%) - avg {gen_data/len(timing_stats['gen_data_fetch'])*1000:6.1f}ms")
                    print(f"  Forward pass:   {gen_fwd:6.2f}s ({100*gen_fwd/total_time:5.1f}%) - avg {gen_fwd/len(timing_stats['gen_forward'])*1000:6.1f}ms")
                    print(f"  Cond loss calc: {cond_loss:6.2f}s ({100*cond_loss/total_time:5.1f}%) - avg {cond_loss/len(timing_stats['cond_loss_pair'])*1000:6.1f}ms")
                    print(f"  Backward pass:  {gen_bwd:6.2f}s ({100*gen_bwd/total_time:5.1f}%) - avg {gen_bwd/len(timing_stats['gen_backward'])*1000:6.1f}ms")
                    print(f"\nSUMMARY:")
                    print(f"  Total measured: {measured_total:6.2f}s ({100*measured_total/total_time:5.1f}%)")
                    print(f"  Unmeasured:     {unmeasured:6.2f}s ({100*unmeasured/total_time:5.1f}%)")
                    print(f"  Total epoch:    {total_time:6.2f}s")
                    print(f"\n  GPU compute:    {disc_fwd+disc_bwd+gen_fwd+gen_bwd:6.2f}s ({100*(disc_fwd+disc_bwd+gen_fwd+gen_bwd)/total_time:5.1f}%)")
                    print(f"  Data overhead:  {disc_data+gen_data:6.2f}s ({100*(disc_data+gen_data)/total_time:5.1f}%)")

                    # Queue health stats (only if prefetcher is enabled)
                    if prefetcher and len(timing_stats['queue_size']) > 0:
                        queue_sizes = timing_stats['queue_size']
                        print(f"\nPREFETCHER QUEUE HEALTH:")
                        print(f"  Queue size: min={min(queue_sizes)}, avg={np.mean(queue_sizes):.1f}, max={max(queue_sizes)} (out of 8)")
                        if min(queue_sizes) < 1:
                            print(f"  ⚠️  WARNING: Queue running low! Workers can't keep up with GPU.")
                        elif min(queue_sizes) > 5:
                            print(f"  ✓ Queue healthy: Workers easily keeping up with GPU")
                        else:
                            print(f"  ✓ Queue adequate: Workers keeping pace")
                    print("="*60 + "\n")

                if self._verbose:
                    epoch_iterator.set_description(
                        description.format(gen=generator_loss, dis=discriminator_loss)
                    )

                if self.wandb == True :
                    ## Add WB logs
                    metrics = {
                        # "train/loss_g_pure": loss_g_pure.detach().cpu(),
                        "train/loss_g": loss_g.detach().cpu(),
                        "train/loss_d": loss_d.detach().cpu(),
                        "train/epoch": i + 1,
                        #"train/example_ct": len(loss_g)
                    }
                    # wandb.log(metrics)

                    SAVE_DIR = Path('./data/weights/')
                    SAVE_DIR.mkdir(exist_ok=True, parents=True)


                if self.private:
                    orders = [1 + x / 10. for x in range(1, 100)]
                    sampling_probability = self._batch_size/len(train_data)
                    delta = 2e-6
                    rdp = compute_rdp(q=sampling_probability,
                                        noise_multiplier=sigma,
                                        steps=i * steps_per_epoch,
                                        orders=orders)
                    epsilon, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta) # target_delta=1e-5

                    print('differential privacy with eps = {:.3g} and delta = {}.'.format(
                        epsilon, delta))
                    print('The optimal RDP order is {}.'.format(opt_order))

                    if opt_order == max(orders) or opt_order == min(orders):
                        print('The privacy estimate is likely to be improved by expanding '
                            'the set of orders.')
                else:
                    epsilon = np.nan

                            
                    ######## ADDED ########
                if self.wandb == True:
                    wandb.finish()

            # Stop the prefetcher after training (if enabled)
            if prefetcher:
                prefetcher.stop()


    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):cd
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

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
