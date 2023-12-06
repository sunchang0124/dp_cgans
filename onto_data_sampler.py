import numpy as np
### Chang ###
from functools import reduce
import pickle as pkl


class Onto_DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN.

    Args:
    data (numpy.ndarray):
        The transformed data to use for training.
    rds (list-like):
        The list of values of the rare_disease column in the dataset.
    output_info (list-like):
        Output info from the data transformation.
    log_frequency (bool):
        Whether to use log to count category frequencies.
    embedding (OntologyEmbedding):
        Embedding class to get embeddings from.
    """

    def __init__(self, data, rds, output_info, log_frequency, embedding=None):
        self._data = data
        self._rds = rds
        self._rd_col_dim = output_info[0][0].dim
        self._embedding = embedding

        def is_discrete_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == "softmax")

        n_discrete_columns = sum(
            [1 for column_info in output_info if is_discrete_column(column_info)])

        self._discrete_column_matrix_st = np.zeros(
            n_discrete_columns, dtype="int32")

        ### Modified by Chang ###
        self.get_position = []
        position_cnt = 0
        self._categories_each_column = []

        for column_info in output_info:
            if is_discrete_column(column_info):
                self._categories_each_column.append(column_info[0].dim)
                self.get_position.append(position_cnt)
            position_cnt += 1
        max_category = max(self._categories_each_column)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(
            n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns

        ### Modified by Chang ###

        self.get_position = np.array(self.get_position)
        self._n_categories = sum(self._categories_each_column)

        print(f'N_categories: {self._n_categories}')

        self._categories_each_column = np.array(self._categories_each_column)
        remove_max_from_list = np.delete(self._categories_each_column, np.where(self._categories_each_column == max_category)).flatten()
        if len(remove_max_from_list) > 1 :
            second_max_category = np.partition(remove_max_from_list, -2)[-1]
        elif len(remove_max_from_list) == 1 :
            second_max_category = remove_max_from_list[0]
        
        #### Fix needed! ####
        self._discrete_pair_cond_st = np.zeros((int(((n_discrete_columns)*(n_discrete_columns-1))/2),int(max_category * second_max_category)),dtype='int32') # int((max_category+1) * (second_max_category+1))
        self._discrete_pair_n_category = np.zeros(int(((n_discrete_columns)*(n_discrete_columns-1))/2), dtype='int32')
        self._discrete_column_pair_prob = np.zeros((int(((n_discrete_columns)*(n_discrete_columns-1))/2),int(max_category * second_max_category))) # int((max_category+1) * (second_max_category+1))

        st = 0
        current_id = 0
        current_cond_st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)

                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)

                self._discrete_column_category_prob[current_id, :span_info.dim] = (
                    category_prob)

                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])

        ### Modified by Chang###
        # starting after the rare_disease column to exclude it
        st_primary = self._rd_col_dim

        current_id_pair = 0
        current_cond_st_pair = 0
        self.pair_id_dict = {}

        self._rid_by_cat_cols_pair = []

        # Starts at 1 to exclude the rare_disease column
        for index_primary in range(1, len(output_info)):
            column_info_primary = output_info[index_primary]

            if is_discrete_column(column_info_primary):
                span_info_primary = column_info_primary[0]
                ed_primary = st_primary + span_info_primary.dim
                st_secondary = ed_primary

                for index_secondary in range(index_primary+1, len(output_info)):
                    column_info_secondary = output_info[index_secondary]

                    if is_discrete_column(column_info_secondary):

                        span_info_secondary = column_info_secondary[0]
                        ed_secondary = st_secondary + span_info_secondary.dim

                        rid_by_cat_pair = []
                        
                        # concatenate the category columns of each triple of data columns
                        combine_pair_data = np.concatenate((data[:, 0:self._rd_col_dim], data[:, st_primary:ed_primary], data[:, st_secondary:ed_secondary]), axis=1)
                        # convert each triple to a single number
                        combine_pair_data_decimal = reduce(lambda a, b: 2*a+b, combine_pair_data.transpose())
                        # counts the frequency of each triple
                        unique_triples = np.unique(combine_pair_data_decimal)

                        for uni_value in unique_triples:
                            # add the indexes of each triple
                            rid_by_cat_pair.append(np.where(combine_pair_data_decimal == uni_value)[0])
                        self._rid_by_cat_cols_pair.append(rid_by_cat_pair)

                         ### calculate triple frequency ### 
                        unique_triples_freq, counts_triples_freq = np.unique(reduce(lambda a, b: 2*a+b, combine_pair_data.transpose()), return_counts=True) # counts_pair --> category_freq

                        pair_prob = counts_triples_freq / np.sum(counts_triples_freq)
                        
                        self._discrete_column_pair_prob[current_id_pair, :len(unique_triples_freq)] = (pair_prob)

                        self._discrete_pair_cond_st[current_id_pair, 0:len(unique_triples_freq)] = unique_triples_freq ## current_cond_st_pair
                        self._discrete_pair_n_category[current_id_pair] = len(unique_triples_freq)
                        current_cond_st_pair += (span_info_primary.dim * span_info_secondary.dim)

                        # Added 0 to always include the RD column
                        self.pair_id_dict[(0, index_primary, index_secondary)] = current_id_pair
                        current_id_pair += 1

                        st_secondary = ed_secondary

                    else:
                        st_secondary += sum([span_info.dim for span_info in column_info_secondary])

                st_primary = ed_primary

            else:
                st_primary += sum([span_info.dim for span_info in column_info_primary])

    ### modified by Chang ###
    def _random_choice_prob_pairs(self, converted_paired_discrete_column_id):
        pair_probs = self._discrete_column_pair_prob[converted_paired_discrete_column_id]
        r = np.expand_dims(np.random.rand(pair_probs.shape[0]), axis=1)
        return (pair_probs.cumsum(axis=1) > r).argmax(axis=1)

    ### modified by Chang ###
    def sample_condvec_pair(self, batch):
        """Generate the conditional vector for training.

        Args:
            batch (int): Size of the batch.
        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        paired_discrete_column_id = []
        for iter_gen in range(0, batch):
            # arange from 1 to exclude the RD column
            paired_discrete_column_id.append(np.concatenate(([0], np.random.choice(np.arange(1, self._n_discrete_columns), 2, replace=False))))

        # convert paired_discrete_column_id to current_id_pair type
        converted_paired_discrete_column_id = []

        for each in paired_discrete_column_id:
            converted_paired_discrete_column_id.append(self.pair_id_dict[tuple(np.sort(each))])#[tuple(self.get_position[np.sort(each)])])

        cond_pair = np.zeros((batch, self._n_categories), dtype='float32')
        mask_pair = np.zeros((batch, self._n_discrete_columns), dtype='int32')
        mask_pair[np.arange(batch), np.array(paired_discrete_column_id)[:, 0]] = 1
        mask_pair[np.arange(batch), np.array(paired_discrete_column_id)[:, 1]] = 1
        mask_pair[np.arange(batch), np.array(paired_discrete_column_id)[:, 2]] = 1

        pair_id_in_col = self._random_choice_prob_pairs(converted_paired_discrete_column_id)  # category_id_in_col: (0, max_num_categories), --> [0,2,1,0,2,1...]
        pair_id_decimal = (self._discrete_pair_cond_st[np.expand_dims(converted_paired_discrete_column_id, axis=1),
                                                       np.expand_dims(pair_id_in_col, axis=1)]).astype(int).flatten()
        pair_primary_secondary_cat = []
        pair_primary_secondary_col = []
        for itr_decimal in range(0, len(pair_id_decimal)):

            pair_categories = self._categories_each_column * mask_pair[itr_decimal]
            pair_id_binary = list(np.binary_repr(pair_id_decimal[itr_decimal], width=(pair_categories.sum())))

            cats = pair_categories[pair_categories!=0]
            pair_primary_position = np.argmax(pair_id_binary[:cats[0]])
            pair_secondary_position = np.argmax(pair_id_binary[cats[0]:cats[0]+cats[1]])
            pair_tertiary_position = np.argmax(pair_id_binary[cats[0]+cats[1]:])
            pair_primary_secondary_cat.append([pair_primary_position, pair_secondary_position, pair_tertiary_position])

            pair_primary_secondary_col.append(self._discrete_column_cond_st[np.where(mask_pair[itr_decimal]==1)[0]])

        pair_id_all_positions = np.add(np.array(pair_primary_secondary_col), np.array(pair_primary_secondary_cat))

        cond_pair[np.arange(batch), pair_id_all_positions[:, 0]] = 1
        cond_pair[np.arange(batch), pair_id_all_positions[:, 1]] = 1
        cond_pair[np.arange(batch), pair_id_all_positions[:, 2]] = 1

        return cond_pair, mask_pair, np.array(converted_paired_discrete_column_id), pair_id_in_col

    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')

        for i in range(batch):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.choice(np.arange(1, self._n_discrete_columns), replace=False, size=2)
            for ind in col_idx:
                mask[i, ind] = 1
                # _discrete_column_matrix_st is a matrix full of zeros
                matrix_st = self._discrete_column_matrix_st[ind]
                matrix_ed = matrix_st + self._discrete_column_n_category[ind]
                pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
                cond[i, pick + self._discrete_column_cond_st[ind]] = 1
            # adding the RD column
            mask[i, 0] = 1
            matrix_st = self._discrete_column_matrix_st[ind]
            matrix_ed = matrix_st + self._discrete_column_n_category[0]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[0]] = 1

        return cond, mask

    def sample_data_pair(self, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols_pair[c][o]))

        # Slice to exclude RD column for ZSL
        return self._data[idx][:, self._rd_col_dim:]

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id = self._discrete_column_matrix_st[condition_info["discrete_column_id"]
                                             ] + condition_info["value_id"]
        vec[:, id] = 1
        return vec

    def get_embeds_from_cat_ids(self, cat_ids, batch_size):
        """Get embeddings for a batch of data, from the category ids.

        Args:
            cat_ids (numpy.ndarray):
                Category ids, in the form of a binary matrix.
            batch_size (int):
                Size of the batch.
        Returns:
            cat_embeddings (numpy.ndarray):
                Matrix containing embeddings of rare_diseases.
        """
        embed_size = self._embedding.embed_size
        cat_embeddings = np.ndarray(shape=(batch_size, embed_size), dtype='float32')
        for r in range(batch_size):
            cat_inds = np.nonzero(cat_ids[r])[0]
            cat_embeddings[r, 0:embed_size] = self._embedding.get_embedding(self._rds[cat_inds[0]])

        return cat_embeddings

    def get_rds(self, cat_ids, batch_size):
        """Get list of rare_diseases from the category ids.

        Args:
            cat_ids (numpy.ndarray):
                Category ids, in the form of a binary matrix.
            batch_size (int):
                Size of the batch.
        Returns:
            rds (list[str]):
                List containing batch_size rare diseases labels.
        """
        rds = []
        for r in range(batch_size):
            cat_inds = np.nonzero(cat_ids[r])[0]
            rds.append(self._rds[cat_inds[0]])

        return rds

    def get_rd_embeds(self, rds):
        """Get rare diseases embeddings from a list of rare diseases (labels).

        Args:
            rds (list):
                List containing rare diseases labels.
        Returns:
            cat_embeddings (numpy.ndarray):
                Matrix containing embeddings of rare_diseases.
        """
        embed_size = self._embedding.embed_size
        cat_embeddings = np.ndarray(shape=(len(rds), embed_size), dtype='float32')
        for r, rd in enumerate(rds):
            cat_embeddings[r, :] = self._embedding.get_embedding(rd)

        return cat_embeddings