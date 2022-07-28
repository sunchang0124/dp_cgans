# imports for the whole notebook
from xml.etree import ElementTree as ET
import csv
import pandas as pd
import re
import numpy as np
import math
import os
import argparse
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import nltk
from owl2vec_star import owl2vec_star


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Data pre-process',
        usage=''
    )
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--onto_dir', type=str, default='ontology')
    parser.add_argument('--syn_data_dir', type=str, default='syn_data')
    parser.add_argument('--onto_to_embed', type=str, default='hpObo_hoom_ordo.owl')
    parser.add_argument('--embedding_cfg_path', type=str, default='./embedding.cfg')

    return parser.parse_args(args)


# set the path to your data folders here
args = parse_args(args=['--data_dir', '../persistent/data'])
onto_dir_path = os.path.join(args.data_dir, args.onto_dir)
syn_data_dir_path = os.path.join(args.data_dir, args.syn_data_dir)

if not os.path.exists(onto_dir_path):
    raise ValueError(f'You need an existing ontology directory with the XML dataset inside it, please create \'{onto_dir_path}\'')

if not os.path.exists(syn_data_dir_path):
    os.makedirs(syn_data_dir_path)

frequency_dict = {  # frequency ids + associated probability
    28405: 1,  # Obligate (100%)
    28412: 0.895,  # Very frequent (99-80%)
    28419: 0.545,  # Frequent (79-30%)
    28426: 0.17,  # Occasional (29-5%)
    28433: 0.025,  # Very rare (<4-1%)
    28440: 0  # Excluded (0%)
}


def get_normalized_string(s):
    """Transforms a string to lowercase and replaces all whitespace runs with an underscore

    Args:
        s (str):
            String to normalize
    Returns:
        (str):
            The normalized string
    """
    return re.sub(r"\s+", '_', s.lower())


def gen_syn_data(patients_per_rd=10, unseen_pct=0.2, gen_small_file=True, print_every=0):
    """Generates synthetic seen and unseen data from the ontology into files

    Args:
        patients_per_rd (int):
            Number of generated patients per RD
        unseen_pct (float):
            The percentage of RDs to keep for unseen patients samples
        gen_small_file (bool):
            Whether to generate very small files to debug the model or the full ones
        print_every (int):
            Each number of patients to print progress during data generation (0 = no print)
    """
    df = pd.read_csv(os.path.join(onto_dir_path, 'en_product4.csv'))

    if gen_small_file:
        grouped = df.groupby('Name', sort=False)
        df = pd.concat([group for name, group in grouped][:10])
        grouped = df.groupby('Name', sort=False)
    else:
        # randomizing the order of the RDs for randomized seen/unseen RDs
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        grouped = df_shuffled.groupby('Name', sort=False)

    rd_count = grouped.ngroups
    unique_hps = df.HPOTerm.unique()
    total_hp_count = len(unique_hps)

    print(f'{rd_count} unique rare diseases, {total_hp_count} unique phenotypes')

    phenotypes = unique_hps.tolist()
    phenotypes_dict = {hp: i for i, hp in enumerate(phenotypes)}
    seen_unseen_th = rd_count - math.ceil(rd_count*unseen_pct)  # threshold of RDs to switch to unseen RDs

    seen_patients_data = [['patient_id', 'rare_disease'] + phenotypes]
    unseen_patients_data = [['patient_id'] + phenotypes]

    distribution_check = {  # key: count of patients with hp + maximum count of patients that could've had the hp
        28405: [0, 0],  # Obligate (100%)
        28412: [0, 0],  # Very frequent (99-80%)
        28419: [0, 0],  # Frequent (79-30%)
        28426: [0, 0],  # Occasional (29-5%)
        28433: [0, 0],  # Very rare (<4-1%)
        28440: [0, 0]  # Excluded (0%)
    }

    patients_count = 0
    lists_n_files = [
        (seen_patients_data, 'syn_patients_data_seen.csv'),
        (unseen_patients_data, 'syn_patients_data_unseen.csv')
    ]

    for group_nb, (name, group) in enumerate(grouped):  # for each RD
        hp_count = len(group)
        # generate patients_per_rd patients for each RD
        for patient_id in range(patients_count, patients_count+patients_per_rd):
            temp_hp = []
            proba_results = np.random.rand(hp_count)  # generating random floats for probabilities
            rd = ''
            for i, (rd_name, hp_name, frequency_id) in enumerate(zip(
                                                            group['Name'],
                                                            group['HPOTerm'],
                                                            group['HPOFrequency_id'])):
                distribution_check.get(frequency_id)[1] += 1
                if rd == '':
                    rd = rd_name
                if (proba_results[i] >= 1 - frequency_dict[frequency_id]):  # comparing generated float and proba
                    temp_hp.append(hp_name)
                    distribution_check.get(frequency_id)[0] += 1

            if len(temp_hp) > 0:
                row = np.zeros((total_hp_count,), dtype=int)
                if (group_nb <= seen_unseen_th):
                    for hp in temp_hp:
                        row[phenotypes_dict.get(hp)] = 1
                    seen_patients_data.append(np.concatenate([[patient_id], [rd], row]))
                else:
                    for hp in temp_hp:
                        row[phenotypes_dict.get(hp)] = 1
                    unseen_patients_data.append(np.concatenate([[patient_id - math.ceil(rd_count*unseen_pct)], row]))

                if print_every > 0:
                    if patients_count % print_every == 0:
                        print(f'{patients_count}/{patients_per_rd*rd_count} patients generated')

                patients_count += 1

    print(f'{len(seen_patients_data)-1} seen patients generated, {len(unseen_patients_data)-1} unseen patients generated, writing to files')

    # writing the 2 files
    for lst, fn in lists_n_files:
        print(f'Number of columns in {fn}: {len(lst[0])}')
        if gen_small_file:
            fn = 'small_' + fn
        pd.DataFrame(lst).to_csv(
            os.path.join(syn_data_dir_path, fn),
            encoding='utf-8', index=False, header=False
        )

    print(f'Total RDs: {rd_count} seen RDs: {seen_unseen_th}, unseen RDs: {rd_count-seen_unseen_th}')

    return distribution_check


distributions = gen_syn_data(patients_per_rd=10, unseen_pct=0, gen_small_file=False, print_every=1000)