import os
import openai
import pickle

class Model_OpenAI:

    def __init__(self, path_key):
        with open(path_key, 'r') as file:
            openai.api_key  = file.readline().strip()
        self.model = []

    def train(self,df,col_in,col_out,system_content,model_name,model_path,model_openai,temperature = 0.7,max_tokens=150,n=1,stop=["\n"]):
        print("\tCreating workspace")
        os.makedirs(model_path,exist_ok=True)

        training_dialogues = []
        print("\tAdding system role")
        training_dialogues.append({'role': 'system', 'content': system_content})
        print("\tAdding other roles")
        for index, row in df.iterrows():
            training_dialogues.append({'role': 'user', 'content': row[col_in]})
            training_dialogues.append({'role': 'assistant', 'content': row[col_out]})
        print("\tCalling OpenAI API")
        response = openai.ChatCompletion.create(
            model=model_openai,
            messages=training_dialogues,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop
        )

        # Create a model and data related and save it
        model = {
            'training_dialogues': training_dialogues,
            'response': response
        }
        print("\tSaving model")
        with open(os.path.join(model_path,model_name), 'wb') as file:
            pickle.dump(model, file)

        print('\tModel trained and saved in:', model_path,model_name)

    def run(self, input, model_name, model_path, model_openai,temperature = 0.7,max_tokens=150,n=1,stop=["\n"]):
        if self.model == []:
            print("\tOpening model",model_path,model_name)
            with open(os.path.join(model_path,model_name), 'rb') as file:
                self.model = pickle.load(file)

        conversation = self.model['training_dialogues']
        
        conversation.append({'role': 'user', 'content': input})
        print("\tCalling OpenAI API")
        print(input)
        response = openai.ChatCompletion.create(
            model=model_openai,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop
        )
        print(response)
        assistant_response = response['choices'][0]['message']['content']
        return assistant_response


