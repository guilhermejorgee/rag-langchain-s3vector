
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_aws.vectorstores.s3_vectors import AmazonS3Vectors
from dotenv import load_dotenv

load_dotenv()

## INDEXAÇÃO ##

# Carregando os documentos
loader = PDFPlumberLoader(
    file_path = "content/Eric Ries - The Lean Startup.pdf"
)

docs = loader.load()

# Dividindo o documento em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
splits = text_splitter.split_documents(docs)

# Embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector Store

vector_store = AmazonS3Vectors.from_documents(
            documents=docs,
            vector_bucket_name="guiaws-vectorstore",
            index_name="local",
            embedding=embeddings,
        )

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

resposta = rag.invoke("De onde vem o crescimento de uma empresa?")

print(resposta)