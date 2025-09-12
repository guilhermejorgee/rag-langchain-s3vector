
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv()

## INDEXAÇÃO ##

# Carregando os documentos
loader = YoutubeLoader.from_youtube_url(
    "https://youtu.be/VaIeFjLMsF4",
    transcript_format = TranscriptFormat.CHUNKS,
    chunk_size_seconds = 30,
    add_video_info=True,
    language=["pt"],
)
docs = loader.load()

# Embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector Store

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

uuids = [str(uuid4()) for _ in range(len(docs))]
vector_store.add_documents(documents=docs, ids=uuids)

# Transformando o vector store em um objeto "pesquisável"
retriever = vector_store.as_retriever()

## RECUPERAÇÃO E GERAÇÃO ##

# Função que formata os documentos retornados
def formatDocuments(docs):
    return "\n\n".join([f"Document {k}:\n{doc.page_content}" for k, doc in enumerate(docs)])

# Fazendo uma pergunta ao documento
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
prompt = PromptTemplate.from_template("""
Com base no contexto, responda a pergunta do usuário. Se não souber, diga que não sabe.

Contexto:
{conteudo}

Pergunta:
{pergunta}

Sua resposta:
""")

rag = (
    {"conteudo": retriever | formatDocuments, "pergunta": RunnablePassthrough()} |
    prompt |
    llm |
    StrOutputParser()
)

resposta = rag.invoke("O que o sam altman falou sobre o risco de bolha da IA?")

print(resposta)