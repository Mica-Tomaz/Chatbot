from transformers import pipeline
from huggingface_hub import login
import streamlit as st
import torch

login(token="SEU TOKEN") # login hugging face, necessário token de acesso

# Defini o título e o ícone da página
st.set_page_config(page_title="Chat Paleontológico 🦖", page_icon="🦕")

# Usa cache para carregar o modelo apenas uma vez
@st.cache_resource

# Cria uma pipeline de geração de texto com o modelo Llama 3.2 3B
def carregar_modelo():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    return pipeline(
        "text-generation",          # Geração de texto
        model=model_id,             # Dita o modelo que será utilizado
        device="cuda",              # Usa GPU para acelerar a inferência, caso não tenha GPU basta apagar essa linha
        torch_dtype=torch.float16,
        max_new_tokens=200,         # Dita qual o máximo de novos tokens
        do_sample=False,            # Ativa ou Desativa amostragem para gerar respostas mais variadas
        temperature=0.5,            # Esse parâmetro dita o quão determinístico vai ser a resposta (valores menores são mais determinísticos)         
        top_p=0.9,
        repetition_penalty=1.2      # Penaliza repetições
    )

pipe = carregar_modelo()

# Cria uma chave de estado para o histórico se ainda não existir
if "historico" not in st.session_state:
    st.session_state["historico"] = []

# Título e descrição da interface
st.title("🦖 Assistente em Paleontologia (LLaMA-3.2)")
st.markdown("Especialista em dinossauros e paleontologia. Pergunte à vontade!")

# Formulário para enviar mensagem
with st.form(key="chat_form"):
    user_input = st.text_input("Você:", "")  # Entrada do usuário
    enviar = st.form_submit_button("Enviar") # Botão para clicar e enviar

if enviar and user_input:

    # Adciona a entrada do usuário na lista 'histórico' em forma de dicionário, de modo que seja possível identificar o autor (Usuário ou Assistente/Guia)
    st.session_state["historico"].append({"role": "Usuário", "content": user_input})

    # Inicia o Prompt vazio e adciona ao prompt a conversa salva no histórico
    prompt = ""
    for linha in st.session_state["historico"]:
        if linha['role'] == 'Usuário':
            prompt += f"Usuário: {linha['content']}\n"
        else:
            prompt += f"Assistente: {linha['content']}\n"

    # Monta o prompt completo e estruturado para o modelo e com as instruções necessárias
    mensagem = [
        {"role": "system", "content": (
            "Você é um especialista em dinossauros e paleontologia. "
            "Responda apenas a última pergunta do Usuário de maneira clara, direta e levando em considerando "
            "o contexto das interações anteriores, mas caso julgue necessário puxe o próximo tópico da conversa. "
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

    # Adiciona a resposta ao histórico
    st.session_state["historico"].append({"role": "Assistente", "content": resposta})

    # Limita o tamanho do histórico para evitar prompts muito longos
    if len(st.session_state["historico"]) > 8: # Limita o histórico a 4 interações
        st.session_state["historico"] = st.session_state["historico"][-8:]

# Mostra o histórico no formato de chat
for entrada in st.session_state["historico"]:
    if entrada["role"] == "Usuário":
        st.markdown(f"**🧑 Você:** {entrada['content']}")
    else:
        st.markdown(f"**🤖 Assistente:** {entrada['content']}")

