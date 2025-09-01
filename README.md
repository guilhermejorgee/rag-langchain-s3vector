# RAG Com S3Vector e Langchain

Este projeto realiza indexação e recuperação de informações de um arquivo PDF utilizando LangChain, S3 Vector e OpenAI.

## Estrutura
- `index.py`: Script principal para indexação e consulta ao PDF.
- `content/`: Pasta com o PDF a ser indexado.
- `requirements.txt`: Dependências do projeto.

## Pré-requisitos
- Python 3.12+
- Chave de API da OpenAI
- Credenciais AWS

## Instalação
1. Clone o repositório ou copie os arquivos para sua máquina.
2. Crie e ative um ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Configuração
- Coloque o arquivo PDF desejado na pasta `content/`.
- Edite o arquivo `index.py` para ajustar o nome do PDF, se necessário.
- Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:
  ```env
  OPENAI_API_KEY=seu_token_openai
  AWS_ACCESS_KEY_ID=sua_access_key_aws
  AWS_SECRET_ACCESS_KEY=sua_secret_key_aws
  AWS_DEFAULT_REGION=sua_regiao_aws
  ```
- Certifique-se de que o script está lendo as variáveis do `.env` (pode ser necessário instalar e importar o pacote `python-dotenv`).

## Execução
Para rodar a aplicação:
```bash
python index.py
```

## Possíveis problemas
- **Erro de chave da OpenAI/AWS:**
  Verifique se a chave está correta e ativa.

## Referências
- [LangChain](https://python.langchain.com/)
- [LangChain AWS](https://pypi.org/project/langchain-aws/)
- [OpenAI](https://platform.openai.com/docs/)

## Licença
Este projeto é apenas para fins educacionais.
