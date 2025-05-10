# Chatbot Paleontológico com Interface Terminal e Web

Este repositório contém um chatbot (assistente/guia utilizando o LLama 3.2 3B) funcional com duas formas de interação: via terminal e via interface web com Streamlit.

## Requisitos

As bibliotecas necessárias para executar os códigos estão listadas no arquivo [`requirements.txt`](./requirements.txt).

> ⚠️ **Importante sobre o PyTorch:**  
> Para instalar a biblioteca **PyTorch**, especialmente se você deseja utilizar a **GPU** durante o carregamento da pipeline do modelo, recomenda-se acessar o site oficial da [PyTorch](https://pytorch.org) e copiar o comando de instalação fornecido lá.  
> O site gera o comando ideal com base nas especificações da sua máquina e na versão dos drivers CUDA instalados, garantindo maior compatibilidade e desempenho.  
> EX: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  

![image](https://github.com/user-attachments/assets/c1f425c4-a504-4708-a53a-0acb8d498cfd)  

## Arquivos Principais

### `Chat.py`

Permite interações com o chatbot diretamente pelo terminal.

**Uso:**
python Chat.py

### `Chat_web.py`

Implementa a mesma lógica do Chat.py, mas com uma interface web utilizando a biblioteca Streamlit.
As interações acontecem via navegador, acessando o localhost.

**Uso:**
python -m streamlit run Chat_web.py
