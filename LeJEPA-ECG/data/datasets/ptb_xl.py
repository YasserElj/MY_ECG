from os import path
import pandas as pd


class PTB_XL:
    sampling_frequency = 500
    record_duration = 10
    channels = (
        'I', 'II', 'III', 'AVR', 'AVL', 'AVF',
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
    )
    mean = (
        -0.002, -0.002, 0.000, 0.002, -0.001, -0.001,
        0.000, -0.001, -0.002, -0.001, -0.001, -0.001
    )
    std = (
        0.191, 0.166, 0.173, 0.142, 0.149, 0.147,
        0.235, 0.338, 0.335, 0.299, 0.294, 0.242
    )

    @staticmethod
    def find_records(data_dir):
        record_list = pd.read_csv(path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
        record_names = [path.join(data_dir, filename) for filename in record_list.filename_hr.values]
        return record_names

