import os

path_root = "D:\\CIAT\\Code\\USAID\\melisa_demeter"

path_data = os.path.join(path_root,"data")
path_data_preprocessing = os.path.join(path_data,"preprocessing")
path_data_inputs = os.path.join(path_data,"inputs")
path_data_workspace = os.path.join(path_data,"workspace")
path_data_models = os.path.join(path_data,"models")

path_key_openai = os.path.join(path_data,"key.txt")
path_model = os.path.join(path_data,"model.pkl")

encoding = 'iso-8859-1'
encoding_utf = 'utf-8-sig'
model_openai = 'gpt-3.5-turbo'
model_bert = 'bert-base-multilingual-cased'

limit_tokens = 4000
