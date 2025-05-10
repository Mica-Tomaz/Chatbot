import torch
from transformers import pipeline
from huggingface_hub import login

login(token="SEU TOKEN") # login hugging face, necessário token de acesso

model_id = "meta-llama/Llama-3.2-3B-Instruct"

# Cria uma pipeline de geração de texto com o modelo Llama 3.2 3B
pipe = pipeline(
    "text-generation",          # Geração de texto
    model=model_id,             # Dita o modelo que será utilizado
    device="cuda",              # Usa GPU para acelerar a inferência, caso não tenha GPU basta apagar essa linha
    torch_dtype=torch.float16,
    max_new_tokens=200,         # Dita qual o máximo de novos tokens
    do_sample=False,            # Ativa amostragem para gerar respostas mais variadas
    temperature=0.5,            # Esse parâmetro dita o quão determinístico vai ser a resposta (valores menores são mais determinísticos)
    top_p=0.9,          
    repetition_penalty=1.2      # Penaliza repetições
)

# Lista que serve para armazenar a conversa entre o usuário e o assistente (o meu objetivo foi trazer um tipo de memória de curto prazo, para servir de contexto no prompt)
historico = []

# Mensagem de entrada do terminal
print("Assistente especializado em paleontologia iniciado. Digite 'sair' ou 'exit' para encerrar.")

# Loop principal do chat
while True:

    try:
        # Recebe a entrada do Usuário
        user_input = input('\nUsuário: ')

        # Caso a entrada seja 'sair' ou 'exit', o loop principal é interrompido
        if user_input in ["sair", "exit"]:
            break
        
        # Adciona a entrada do usuário na lista 'histórico' em forma de dicionário, de modo que seja possível identificar o autor (Usuário ou Assistente/Guia)
        historico.append({'role': 'Usuário', 'content': user_input})

        # Inicia o Prompt vazio e adciona ao prompt a conversa salva no histórico
        prompt = ''
        for linha in historico:
            if linha['role'] == 'Usuário':
                prompt += f'Usuário: {linha['content']}\n'
            else:
                prompt += f'Assistente: {linha['content']}\n'

        # Monta o prompt completo e estruturado para o modelo e com as instruções necessárias
        mensagem = [
            {"role": "system", "content": (
                "Você é um especialista em dinossauros e paleontologia. "
                "Responda apenas a última pergunta do Usuário de maneira clara, direta e levando em consideração o contexto das interações anteriores, mas caso julgue necessário puxe o próximo tópico da conversa."
                "Caso a pergunta fuja do tema paleontologia, responda: Desculpe, mas essa pergunta foge da minha especialidade."
                "Não use mais de dois parágrafos para a resposta."
            )},
            {"role": "user", "content": prompt},
        ]

        # Gera a resposta do modelo com base no prompt
        output = pipe(mensagem)

        # Extrai a resposta gerada 
        resposta = output[0]["generated_text"]
        resposta = resposta[-1]['content']

        # Mostra a resposta no terminal
        print(f"\nAssistente: {resposta}")

        # Adiciona a resposta ao histórico
        historico.append({"role": "assistente", "content": resposta})  

        # # Opcional: Printa o Histórico, estava usando para debugar
        # print("\nHistórico atual:")

        # for linha in historico:
        #     print(f'{linha['content']}\n')

        # Limita o tamanho do histórico para evitar prompts muito longos
        if len(historico) > 8: # Limita o histórico a 4 interações
            historico = historico[-8:]
    
    except Exception as e:
        print(f"\n Ocorreu um erro: {e}")
        continue  # continua o loop mesmo após erro