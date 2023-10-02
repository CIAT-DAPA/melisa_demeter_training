import os
import glob
import pandas as pd
import openai
import pickle
import tiktoken

# Parameters
path_root = "D:\\CIAT\\Code\\USAID\\melisa_demeter"

path_data = os.path.join(path_root,"data")

path_raw = os.path.join(path_data,"raw.xlsx")

path_key_openai = os.path.join(path_data,"key.txt")
path_model = os.path.join(path_data,"model.pkl")

encoding = 'iso-8859-1'
limit_tokens = 4000
model_name = 'gpt-3.5-turbo'

# Loading data
print("Loading inputs")
xls_raw = pd.ExcelFile(path_raw)
df_raw = xls_raw.parse("raw")
print("DF loaded", df_raw.columns)

# Fixing the format of raw data
print("Fixing format")
columns_to_pivot = ['variante1', 'variante2', 'variante3', 'variante4', 'variante5', 'variante6', 'variante7']
df_inputs = pd.melt(df_raw, id_vars=['pregunta', 'clasificacion', 'guia'], value_vars=columns_to_pivot, var_name='variante', value_name='contenido')
df_inputs = df_inputs.dropna()
df_inputs['contenido'] = df_inputs['contenido'].str.replace('  ',' ')
df_inputs['contenido'] = df_inputs['contenido'].str.replace('?','')
df_inputs['contenido'] = df_inputs['contenido'].str.replace('¿','')
df_inputs['contenido'] = df_inputs['contenido'].str.replace('_','')
df_inputs = df_inputs.sample(n=100)
print(df_inputs.head(),df_inputs.shape)
df_inputs.to_csv(os.path.join(path_data,"inputs.csv"),encoding=encoding,index=False)

# Training CHAT GPT



# Defining questions and answers to train the model
key_openai = ""
with open(path_key_openai, 'r') as file:
    key_openai  = file.readline().strip()

openai.api_key = key_openai

# Cuenta tokens
def count_tokens(text):
    response = openai.TikToken.create(content=text)
    return response['usage']['total_tokens']

# Training the model
def train_and_save_model(training_dialogues, path_output):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=training_dialogues,
        temperature=0.7,
        max_tokens=150,
        n=1,
        stop=["\n"]
    )

    # Guarda el modelo y datos relacionados en un objeto
    model = {
        'training_dialogues': training_dialogues,
        'response': response
    }

    print(model)
    # Guarda el modelo en un archivo
    with open(path_output, 'wb') as file:
        pickle.dump(model, file)

    print('Model trained and saved in:', path_output)

# Creating array to train the model
print("Fixing inputs data")
training_data = []

for index, row in df_inputs.head(120).iterrows():
    training_data.append({ 
        'input': row['contenido'],
        'output': row['clasificacion']
    })

# Fitting data to the correct format to train the model
training_dialogues = []
total_inputs = 0
total_outputs = 0
for data in training_data:
    encoding = tiktoken.encoding_for_model(model_name)
    training_dialogues.append({'role': 'system', 'content': 'Instrucciones: Responde a las siguientes preguntas.'})
    tokens_input = len(encoding.encode(data['input']))
    tokens_output = len(encoding.encode(data['output']))
    total_inputs += tokens_input
    total_outputs += tokens_output
    if tokens_input < limit_tokens and tokens_output < limit_tokens:
        training_dialogues.append({'role': 'user', 'content': data['input']})
        training_dialogues.append({'role': 'assistant', 'content': data['output']})
    else:
        print(data['input'],tokens_input,data['output'],tokens_output)

print(total_inputs,total_outputs)
# Train model
print("Training")
train_and_save_model(training_dialogues, path_model)