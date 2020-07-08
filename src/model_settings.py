import pandas as pd
from transformers import BertTokenizer

def initialize_model_settings():

    last_date = pd.Timestamp(2020,7,9)

    model_settings = {

    'root_path'             : '../',
    'input_path'            : ['database.db', 'database_next.db'],
    's3_data_path'          : 'PCI-Outbreak/data/',

    'keywords_old'          : ['非典', '肺炎', 'SARS', '疫情', '疫区', '抗疫'],
    'from_date_old'         : pd.Timestamp(2003,4,3),
    'to_date_old'           : pd.Timestamp(2003,7,5),
    'min_ratio_old'         : 0.003,
    'min_count_old'         : 3,

    'keywords_new'          : ['新型冠状病毒', '新冠', '肺炎', 'COVID', '疫情', '疫区', '抗疫'],
    'from_date_new'         : pd.Timestamp(2020,1,21),
    'to_date_new'           : last_date,
    'min_ratio_new'         : 0.003,
    'min_count_new'         : 3,
    
    'rand_state'            : 1,
    'pretrained_model_name' : "bert-base-chinese",
    'stage_1_num_labels'    : 2,
    'stage_2_num_labels'    : 1,
    
    'max_length'            : 100,
    'batch_size'            : 24,
    'predict_batch_size'    : 1280,
    'epochs'                : 20,
    'cuda_index'            : 0,
    'multi_gpu'             : True,
    'model_name'            : "Sentence_based",
    'text'                  : 'sentence',

    }

    model_settings['tokenizer'] = BertTokenizer.from_pretrained(model_settings['pretrained_model_name'])

    return model_settings