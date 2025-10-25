import numpy as np
### Chang ###
from functools import reduce


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def __init__(self, data, output_info, log_frequency):
        self._data = data

        def is_discrete_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == "softmax")

        n_discrete_columns = sum(
            [1 for column_info in output_info if is_discrete_column(column_info)])

        self._discrete_column_matrix_st = np.zeros(
            n_discrete_columns, dtype="int32")

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)

                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]


        self._rid_by_cat_cols_pair = []
        # Compute _rid_by_cat_cols_pair
        st_primary = 0
        for index_primary in range(0, len(output_info)):
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
                        combine_pair_data = np.append(data[:,st_primary:ed_primary], data[:,st_secondary:ed_secondary],axis=1)
                        combine_pair_data_decimal = reduce(lambda a,b: 2*a+b, combine_pair_data.transpose())
                        unique_pair, counts_pair = np.unique(combine_pair_data_decimal, return_counts=True) 

                        for uni_value in unique_pair:
                            rid_by_cat_pair.append(np.where(combine_pair_data_decimal == uni_value)[0])
                        self._rid_by_cat_cols_pair.append(rid_by_cat_pair)
                        
                        st_secondary = ed_secondary
                    else:
                        st_secondary += sum([span_info.dim for span_info in column_info_secondary])

                st_primary = ed_primary

            else:
                st_primary += sum([span_info.dim for span_info in column_info_primary])
        
        assert st_primary == data.shape[1]

                
                


        # Prepare an interval matrix for efficiently sample conditional vector
        # max_category = max(
        #     [column_info[0].dim for column_info in output_info
        #      if is_discrete_column(column_info)], default=0)
       
        ### Modified by Chang###        
        self.get_position=[]
        position_cnt = 0
        self._categories_each_column = []
        for column_info in output_info:
            if is_discrete_column(column_info):
                self._categories_each_column.append(column_info[0].dim)
                self.get_position.append(position_cnt)
            position_cnt += 1

        max_category = max(self._categories_each_column, default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(
            n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        # self._n_categories = sum(
        #     [column_info[0].dim for column_info in output_info
        #      if is_discrete_column(column_info)]) 
             
        ### Modified by Chang###        

        
        self.get_position = np.array(self.get_position)
        self._n_categories = sum(self._categories_each_column)

        self._categories_each_column = np.array(self._categories_each_column)
        if len(self._categories_each_column)>0: # if self._categories_each_column is not empty
            second_max_category = np.partition(self._categories_each_column.flatten(), -2)[-2]
        else:
            second_max_category = 0

        self._discrete_pair_cond_st = np.zeros((int(((n_discrete_columns)*(n_discrete_columns-1))/2),int((max_category+1) * (second_max_category+1))),dtype='int32')
        self._discrete_pair_n_category = np.zeros(int(((n_discrete_columns)*(n_discrete_columns-1))/2), dtype='int32')
        self._discrete_column_pair_prob = np.zeros((int(((n_discrete_columns)*(n_discrete_columns-1))/2), int((max_category+1) * (second_max_category+1))))


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
        st_primary = 0
        
        current_id_pair = 0
        current_cond_st_pair = 0
        self.pair_id_dict = {}

        for index_primary in range(0, len(output_info)): # [2, 3, 3, 2] (2,3),(2,3),(2,2),,,(3,3),(3,2),,,,,(3,2)
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
                        
                        ### calculate pair frequency ### 
                        combine_pair_data = np.append(data[:,st_primary:ed_primary], data[:,st_secondary:ed_secondary],axis=1)
                        unique_pair, counts_pair = np.unique(reduce(lambda a,b: 2*a+b, combine_pair_data.transpose()), return_counts=True) # counts_pair --> category_freq  

                        pair_prob = counts_pair / np.sum(counts_pair) ###???? Order in the array??
                        self._discrete_column_pair_prob[current_id_pair, :len(unique_pair)] = (pair_prob)

                        self._discrete_pair_cond_st[current_id_pair, 0:len(unique_pair)] = unique_pair ## current_cond_st_pair
                        self._discrete_pair_n_category[current_id_pair] = len(unique_pair)
                        current_cond_st_pair += (span_info_primary.dim * span_info_secondary.dim)

                        self.pair_id_dict[(index_primary,index_secondary)] = current_id_pair
                        current_id_pair += 1

                        st_secondary = ed_secondary
                    else:
                        st_secondary += sum([span_info.dim for span_info in column_info_secondary])

                st_primary = ed_primary

            else:
                st_primary += sum([span_info.dim for span_info in column_info_primary])
            



    # def _random_choice_prob_index(self, discrete_column_id):
    #     probs = self._discrete_column_category_prob[discrete_column_id]
    #     r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
    #     return (probs.cumsum(axis=1) > r).argmax(axis=1)

    ### modified by Chang ###
    def _random_choice_prob_pairs(self, converted_paired_discrete_column_id):
        pair_probs = self._discrete_column_pair_prob[converted_paired_discrete_column_id]
        r = np.expand_dims(np.random.rand(pair_probs.shape[0]), axis=1)
        return (pair_probs.cumsum(axis=1) > r).argmax(axis=1)


    # def sample_condvec(self, batch):
    #     """Generate the conditional vector for training.

    #     Returns:
    #         cond (batch x #categories):
    #             The conditional vector.
    #         mask (batch x #discrete columns):
    #             A one-hot vector indicating the selected discrete column.
    #         discrete column id (batch):
    #             Integer representation of mask.
    #         category_id_in_col (batch):
    #             Selected category in the selected discrete column.
    #     """
    #     if self._n_discrete_columns == 0:
    #         return None

    #     discrete_column_id = np.random.choice(
    #         np.arange(self._n_discrete_columns), batch)

    #     cond = np.zeros((batch, self._n_categories), dtype='float32')
    #     mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')
    #     mask[np.arange(batch), discrete_column_id] = 1
    #     category_id_in_col = self._random_choice_prob_index(discrete_column_id) # category_id_in_col: (0, max_num_cateogies), --> [0,2,1,0,2,1...]
    #     category_id = (self._discrete_column_cond_st[discrete_column_id] # _discrete_column_cond_st : adding up categories from each discrete var 
    #                    + category_id_in_col)
    #     cond[np.arange(batch), category_id] = 1

    #     return cond, mask, discrete_column_id, category_id_in_col

    ### modified by Chang ###
    def sample_condvec_pair(self, batch):
        """Generate the conditional vector for training.

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


        # discrete_column_id = np.random.choice(
        #     np.arange(self._n_discrete_columns), batch)

        paired_discrete_column_id = []
        for iter_gen in range(0, batch):
            paired_discrete_column_id.append(np.random.choice(np.arange(self._n_discrete_columns), 2, replace=False)) 
        
        # convert paired_discrete_column_id to current_id_pair type
        converted_paired_discrete_column_id = []
        for each in paired_discrete_column_id:
            converted_paired_discrete_column_id.append(self.pair_id_dict[tuple(self.get_position[np.sort(each)])])

        # full_possible_combi = 0
        # for basic_item in range(0, len(self._categories_each_column)-1):
        #     for mul_item in range(basic_item+1, len(self._categories_each_column)):
        #         full_possible_combi += self._categories_each_column[basic_item] * self._categories_each_column[mul_item]

        cond_pair = np.zeros((batch, self._n_categories), dtype='float32') ## 
        mask_pair = np.zeros((batch, self._n_discrete_columns), dtype='int32') ## 
        mask_pair[np.arange(batch), np.array(paired_discrete_column_id)[:,0]] = 1
        mask_pair[np.arange(batch), np.array(paired_discrete_column_id)[:,1]] = 1

        pair_id_in_col = self._random_choice_prob_pairs(converted_paired_discrete_column_id) # category_id_in_col: (0, max_num_cateogies), --> [0,2,1,0,2,1...]
        pair_id_decimal = (self._discrete_pair_cond_st[np.expand_dims(converted_paired_discrete_column_id,axis=1),
                                                                    np.expand_dims(pair_id_in_col,axis=1)]).astype(int).flatten()

        pair_primary_secondary_cat = []
        pair_primary_secondary_col = []
        for itr_decimal in range(0, len(pair_id_decimal)):
 
            pair_categories = self._categories_each_column * mask_pair[itr_decimal]
            pair_id_binary = list(np.binary_repr(pair_id_decimal[itr_decimal], width=(pair_categories.sum())))
            
            first_cat = pair_categories[pair_categories!=0][0]
            pair_primary_position = np.argmax(pair_id_binary[:first_cat])
            pair_secondary_position = np.argmax(pair_id_binary[first_cat:])
            pair_primary_secondary_cat.append([pair_primary_position, pair_secondary_position])

            pair_primary_secondary_col.append(self._discrete_column_cond_st[np.where(mask_pair[itr_decimal]==1)[0]])

        pair_id_all_positions = np.add(np.array(pair_primary_secondary_col), np.array(pair_primary_secondary_cat))
        
        cond_pair[np.arange(batch), pair_id_all_positions[:,0]] = 1
        cond_pair[np.arange(batch), pair_id_all_positions[:,1]] = 1

        return cond_pair, mask_pair, np.array(converted_paired_discrete_column_id), pair_id_in_col
        ### converted_paired_discrete_column_id [0,6]
        ### pair_id_in_col, 


    def sample_original_condvec(self, batch):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')

        for i in range(batch):
            row_idx = np.random.randint(0, len(self._data))
            col_idx = np.random.randint(0, self._n_discrete_columns)
            matrix_st = self._discrete_column_matrix_st[col_idx]
            matrix_ed = matrix_st + self._discrete_column_n_category[col_idx]
            pick = np.argmax(self._data[row_idx, matrix_st:matrix_ed])
            cond[i, pick + self._discrete_column_cond_st[col_idx]] = 1

        return cond

    # def sample_data_pair(self, n, col, opt):
    #     """Sample data from original training data satisfying the sampled conditional vector.

    #     Returns:
    #         n rows of matrix data.
    #     """
    #     if col is None:
    #         idx = np.random.randint(len(self._data), size=n)
    #         return self._data[idx]

    #     idx = []
    #     for c, o in zip(col, opt):
    #         print(c,o)
    #         idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

    #     return self._data[idx]
    
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

        return self._data[idx]

    def dim_cond_vec(self):
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id = self._discrete_column_matrix_st[condition_info["discrete_column_id"]
                                             ] + condition_info["value_id"]
        vec[:, id] = 1
        return vec
