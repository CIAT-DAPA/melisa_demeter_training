import os
import glob
import pandas as pd
import openai
import pickle

# Parameters
path_root = "D:\\CIAT\\Code\\Logistic\\apu"
path_data = os.path.join(path_root,"data")

path_raw = os.path.join(path_data,"raw.xlsx")

path_key_openai = os.path.join(path_data,"key.txt")
path_model = os.path.join(path_data,"model.pkl")

encoding = 'iso-8859-1'
limit_tokens = 4000
model_name = 'gpt-3.5-turbo'

key_openai = ""
with open(path_key_openai, 'r') as file:
    key_openai  = file.readline().strip()

openai.api_key = key_openai

def load_model(path):
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

# Función para detectar intención utilizando el modelo cargado
def detect_intent(question, model):
    conversation = model['training_dialogues']
    
    conversation.append({'role': 'user', 'content': question})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0,
        max_tokens=150,
        n=1,
        stop=["\n"]
    )
    print(response)
    assistant_response = response['choices'][0]['message']['content']
    return assistant_response

# Carga el modelo
model_trained = load_model(path_model)

# Pregunta para probar la detección de intenciones
#question = "Cuando puedo pasar por mis productos"
questions = ["los formatos actualizados del proceso de compras","hola, como estás?"]

# Detecta la intención y obtén la respuesta
for q in questions:
    answer = detect_intent(q, model_trained)
    print("Intención detectada:", answer)