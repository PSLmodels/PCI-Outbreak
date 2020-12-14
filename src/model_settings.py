import pandas as pd
from transformers import BertTokenizer

def initialize_model_settings():

    last_date = pd.Timestamp(2020,9,15)

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

    'foreign_countries'     : ['美国', '巴西', '印度', '俄罗斯', '秘鲁', '南非', '哥伦比亚', '墨西哥', '西班牙', '智利', '阿根廷', '伊朗', '英国', '沙特', '孟加拉', '孟加拉国', '巴基斯坦', '法国', '土耳其', '意大利', '德国', '伊拉克', '菲律宾', '印尼', '印度尼西亚', '加拿大', '乌克兰', '卡塔尔', '玻利维亚', '以色列', '厄瓜多尔', '哈萨克斯坦', '埃及', '多米尼加', '巴拿马', '罗马尼亚', '阿曼', '比利时', '科威特', '瑞典', '危地马拉', '荷兰', '白俄罗斯', '阿联酋', '日本', '波兰', '摩洛哥', '洪都拉斯', '葡萄牙', '新加坡', '尼日利亚', '巴林', '埃塞俄比亚', '委内瑞拉', '加纳', '阿尔及利亚', '吉尔吉斯斯坦', '亚美尼亚', '瑞士', '乌兹别克斯坦', '哥斯达黎加', '尼泊尔', '阿富汗', '摩尔多瓦', '阿塞拜疆', '肯尼亚', '波多黎各', '塞尔维亚', '爱尔兰', '奥地利', '澳大利亚', '萨尔瓦多', '捷克', '韩国', '波黑', '喀麦隆', '科特迪瓦', '丹麦', '巴拉圭', '黎巴嫩', '保加利亚', '马达加斯加', '北马其顿', '塞内加尔', '利比亚', '苏丹', '科索沃', '赞比亚', '挪威', '希腊', '克罗地亚', '刚果（金）', '阿尔巴尼亚', '几内亚', '马来西亚', '塔吉克斯坦', '加蓬', '海地', '芬兰', '马尔代夫', '纳米比亚', '毛里塔尼亚', '卢森堡', '津巴布韦', '匈牙利', '马拉维', '吉布提', '赤道几内亚', '黑山', '中非', '中非共和国', '斯威士兰', '尼加拉瓜', '卢旺达', '苏里南', '刚果（布）', '古巴', '斯洛伐克', '佛得角', '莫桑比克', '突尼斯', '泰国', '索马里', '法属马约特', '斯里兰卡', '冈比亚', '乌干达', '立陶宛', '斯洛文尼亚', '马里', '叙利亚', '安哥拉', '南苏丹', '爱沙尼亚', '几内亚比绍', '贝宁', '巴哈马', '牙买加', '冰岛', '塞拉利昂', '约旦', '也门', '马耳他', '新西兰', '特立尼达和多巴哥', '博茨瓦纳', '乌拉圭', '塞浦路斯', '格鲁吉亚', '多哥', '拉脱维亚', '布基纳法索', '利比里亚', '圭亚那', '尼日尔', '安道尔', '越南', '乍得'],
    'foreign_min_count'     : 10

    }

    model_settings['tokenizer'] = BertTokenizer.from_pretrained(model_settings['pretrained_model_name'])

    return model_settings