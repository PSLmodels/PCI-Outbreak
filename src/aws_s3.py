import os, boto3
from botocore.errorfactory import ClientError


def gen_data_path(model_settings, x=""):
    return model_settings['root_path'] + "data/" + x

def gen_s3_data_path(model_settings, x=""):
    return model_settings['s3_data_path'] + x

def gen_raw_data_path(model_settings, x=""):
    return [gen_data_path(model_settings, i) for i in model_settings['input_path'] ] 


def gen_model_path(model_settings, x=""):
    return model_settings['root_path'] + "models/" + model_settings['model_name'] + "/" + x


def gen_s3_model_path(model_settings, x=""):
    return "PCI-Outbreak/models/" + model_settings['model_name'] + "/" + x


def connect_s3():
    cred = boto3.Session().get_credentials()
    return boto3.client(
        's3', 
        aws_access_key_id = cred.access_key, 
        aws_secret_access_key = cred.secret_key, 
        aws_session_token = cred.token
        )


def get_raw_data(model_settings, raw_data_loc):
    if raw_data_loc == "local":
        checklist = []
        for file in gen_raw_data_path(model_settings):
            checklist.append(os.path.isfile(file ))
        if all(checklist):
            print("Raw data already present locally.")
        else:
            print("Raw data incomplete locally. Consider setting raw_data_loc=s3.")

    elif raw_data_loc == "s3":
        s3 = connect_s3()
        if not os.path.exists(gen_data_path(model_settings)):
            os.makedirs(gen_data_path(model_settings))
        s3.download_file("policychangeindex", 'Data/database.db', gen_data_path(model_settings, model_settings['input_path'][0]) )
        s3.download_file("policychangeindex", 'Data/database_next.db', gen_data_path(model_settings, model_settings['input_path'][1]) )
        print("Raw data downloaded from s3.")
        
    else:
        print("Need to specify where to find raw data!")


def upload_processed_data_s3(model_settings):
    s3 = connect_s3()
    flist = ['SARS_articles', 'SARS_sentences', 'SARS_irrel_articles', 'SARS_irrel_sentences', \
    'COVID_articles', 'COVID_sentences', 'COVID_irrel_articles', 'COVID_irrel_sentences']

    for f in flist:
        s3.upload_file(
            gen_data_path(model_settings, f+'.pickle'),
            "policychangeindex",
            gen_s3_data_path(model_settings, f+'.pickle')
            )
    print('Uploaded processed data to s3.')


def download_processed_data_s3(model_settings):
    s3 = connect_s3()
    if not os.path.exists(gen_data_path(model_settings)):
        os.makedirs(gen_data_path(model_settings))
    flist = ['SARS_articles', 'SARS_sentences', 'SARS_irrel_articles', 'SARS_irrel_sentences', \
    'COVID_articles', 'COVID_sentences', 'COVID_irrel_articles', 'COVID_irrel_sentences']

    for f in flist:
        s3.download_file(
            "policychangeindex",
            gen_s3_data_path(model_settings, f+'.pickle'),
            gen_data_path(model_settings, f+'.pickle')
            )
    print('Downloaded processed data from s3.')


def upload_model_s3(model_settings, stage, epoch):
    s3 = connect_s3()
    s3.upload_file(
        gen_model_path(model_settings, 'stage_'+str(stage)+'_model_epoch_'+str(epoch)+'.tar'),
        "policychangeindex",
        gen_s3_model_path(model_settings, 'stage_'+str(stage)+'_model_epoch_'+str(epoch)+'.tar')
        )

def download_model_s3(model_settings, stage, epoch):
    s3 = connect_s3()
    if not os.path.exists(gen_model_path(model_settings)):
        os.makedirs(gen_model_path(model_settings))
    s3.download_file(
        "policychangeindex",
        gen_s3_model_path(model_settings, 'stage_'+str(stage)+'_model_epoch_'+str(epoch)+'.tar'),
        gen_model_path(model_settings, 'stage_'+str(stage)+'_model_epoch_'+str(epoch)+'.tar')
        )