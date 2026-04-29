import os
import streamlit as st

try:
    from google.colab import userdata

    openai_key = userdata.get("OPENAI_API_KEY")

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

except Exception:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

PASTA_DOCUMENTOS = "documentos"
PASTA_CHROMA = "/tmp/chroma_rag_rh_db"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

QTD_CHUNKS_RECUPERADOS = 8
QTD_CHUNKS_APOS_RERANK = 4


# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Chat RAG - Políticas Internas",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat RAG - Políticas Internas")
st.write(
    "Faça perguntas sobre os documentos internos da empresa. "
    "O sistema irá recuperar os trechos mais relevantes, aplicar re-rank e gerar uma resposta baseada no contexto."
)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def identificar_categoria(nome_arquivo: str) -> str:
    nome = nome_arquivo.lower()

    if "ferias" in nome or "férias" in nome:
        return "férias"

    if "home" in nome or "office" in nome:
        return "home office"

    if "conduta" in nome or "codigo" in nome or "código" in nome:
        return "conduta"

    return "geral"


def carregar_documentos_pdf() -> list[Document]:
    documentos = []

    if not os.path.exists(PASTA_DOCUMENTOS):
        os.makedirs(PASTA_DOCUMENTOS)

    arquivos = [
        arquivo for arquivo in os.listdir(PASTA_DOCUMENTOS)
        if arquivo.lower().endswith(".pdf")
    ]

    for arquivo in arquivos:
        caminho = os.path.join(PASTA_DOCUMENTOS, arquivo)

        loader = PyPDFLoader(caminho)
        paginas = loader.load()

        categoria = identificar_categoria(arquivo)

        for pagina in paginas:
            pagina.metadata["origem"] = arquivo
            pagina.metadata["categoria"] = categoria
            pagina.metadata["tipo"] = "pdf"
            pagina.metadata["pagina"] = pagina.metadata.get("page")

        documentos.extend(paginas)

    return documentos


def criar_chunks(documentos: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documentos)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["estrategia_chunking"] = "recursive_character"
        chunk.metadata["chunk_size"] = CHUNK_SIZE
        chunk.metadata["chunk_overlap"] = CHUNK_OVERLAP

    return chunks


def obter_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def criar_ou_carregar_vectorstore():
    embeddings = obter_embeddings()

    # Se a base já existe, carrega do disco
    if os.path.exists(PASTA_CHROMA) and len(os.listdir(PASTA_CHROMA)) > 0:
        vectorstore = Chroma(
            persist_directory=PASTA_CHROMA,
            embedding_function=embeddings
        )
        return vectorstore

    # Caso não exista, cria a partir dos PDFs
    documentos = carregar_documentos_pdf()

    if not documentos:
        st.error(
            "Nenhum PDF encontrado na pasta 'documentos'. "
            "Adicione os arquivos e reinicie a aplicação."
        )
        st.stop()

    chunks = criar_chunks(documentos)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PASTA_CHROMA
    )

    return vectorstore


def formatar_chunks_para_prompt(chunks: list[Document]) -> str:
    textos = []

    for i, doc in enumerate(chunks, start=1):
        origem = doc.metadata.get("origem", "desconhecida")
        categoria = doc.metadata.get("categoria", "geral")
        pagina = doc.metadata.get("pagina", "não informada")
        chunk_id = doc.metadata.get("chunk_id", "sem id")

        texto = f"""
[CHUNK {i}]
Origem: {origem}
Categoria: {categoria}
Página: {pagina}
Chunk ID: {chunk_id}

Conteúdo:
{doc.page_content}
"""
        textos.append(texto)

    return "\n\n".join(textos)


def aplicar_rerank(pergunta: str, chunks: list[Document], llm: ChatOpenAI) -> list[Document]:
    """
    Re-rank simples usando o próprio LLM.
    O modelo recebe os chunks recuperados e devolve os índices dos mais relevantes.
    """

    contexto = ""

    for i, doc in enumerate(chunks, start=1):
        contexto += f"""
CHUNK {i}
Origem: {doc.metadata.get("origem")}
Categoria: {doc.metadata.get("categoria")}
Página: {doc.metadata.get("pagina")}
Conteúdo:
{doc.page_content}

---
"""

    prompt_rerank = ChatPromptTemplate.from_template("""
Você é um avaliador de relevância em uma pipeline RAG.

Sua tarefa é analisar a pergunta do usuário e os chunks recuperados.
Depois, escolha os {quantidade_final} chunks mais relevantes para responder à pergunta.

Responda somente com os números dos chunks, separados por vírgula.
Exemplo de resposta:
1, 3, 5, 7

Pergunta do usuário:
{pergunta}

Chunks recuperados:
{contexto}
""")

    chain_rerank = prompt_rerank | llm

    resposta = chain_rerank.invoke({
        "pergunta": pergunta,
        "contexto": contexto,
        "quantidade_final": QTD_CHUNKS_APOS_RERANK
    })

    conteudo = resposta.content.strip()

    indices = []

    for parte in conteudo.replace("\n", ",").split(","):
        parte = parte.strip()

        if parte.isdigit():
            indice = int(parte) - 1

            if 0 <= indice < len(chunks):
                indices.append(indice)

    # Remove duplicados mantendo ordem
    indices_unicos = []
    for indice in indices:
        if indice not in indices_unicos:
            indices_unicos.append(indice)

    chunks_rerankeados = [chunks[i] for i in indices_unicos]

    # Fallback caso o LLM responda em formato inesperado
    if not chunks_rerankeados:
        chunks_rerankeados = chunks[:QTD_CHUNKS_APOS_RERANK]

    return chunks_rerankeados[:QTD_CHUNKS_APOS_RERANK]


def gerar_resposta(pergunta: str, chunks: list[Document], llm: ChatOpenAI) -> str:
    contexto = formatar_chunks_para_prompt(chunks)

    prompt_resposta = ChatPromptTemplate.from_template("""
Você é um assistente especializado em políticas internas de RH.

Responda à pergunta do usuário usando somente as informações presentes no contexto.
Se a resposta não estiver no contexto, diga claramente que não encontrou essa informação nos documentos fornecidos.

Se possível, mencione a origem da informação, como o documento, categoria ou página.

Contexto:
{contexto}

Pergunta:
{pergunta}

Resposta:
""")

    chain_resposta = prompt_resposta | llm

    resposta = chain_resposta.invoke({
        "contexto": contexto,
        "pergunta": pergunta
    })

    return resposta.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Configurações")

    st.write("**Chunk size:**", CHUNK_SIZE)
    st.write("**Chunk overlap:**", CHUNK_OVERLAP)
    st.write("**Chunks recuperados:**", QTD_CHUNKS_RECUPERADOS)
    st.write("**Chunks após re-rank:**", QTD_CHUNKS_APOS_RERANK)

    if st.button("Recriar VectorStore"):
        if os.path.exists(PASTA_CHROMA):
            import shutil
            shutil.rmtree(PASTA_CHROMA)

        st.success("VectorStore removida. Reinicie a aplicação para recriar.")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

try:
    vectorstore = criar_ou_carregar_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": QTD_CHUNKS_RECUPERADOS}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY")
      )

    pergunta = st.text_input("Digite sua pergunta sobre as políticas internas:")
    botao_perguntar = st.button("Perguntar")

    if botao_perguntar and pergunta:
        with st.chat_message("user"):
            st.write(pergunta)

        with st.chat_message("assistant"):
            with st.spinner("Buscando documentos relevantes..."):
                chunks_recuperados = retriever.invoke(pergunta)

            with st.spinner("Aplicando re-rank nos chunks recuperados..."):
                chunks_rerankeados = aplicar_rerank(
                    pergunta=pergunta,
                    chunks=chunks_recuperados,
                    llm=llm
                )

            with st.spinner("Gerando resposta final..."):
                resposta_final = gerar_resposta(
                    pergunta=pergunta,
                    chunks=chunks_rerankeados,
                    llm=llm
                )

            st.subheader("Resposta")
            st.write(resposta_final)

            st.divider()

            st.subheader("Chunks utilizados na resposta")

            for i, doc in enumerate(chunks_rerankeados, start=1):
                origem = doc.metadata.get("origem", "desconhecida")
                categoria = doc.metadata.get("categoria", "geral")
                pagina = doc.metadata.get("pagina", "não informada")
                chunk_id = doc.metadata.get("chunk_id", "sem id")

                with st.expander(
                    f"Chunk {i} | {categoria} | {origem} | página {pagina} | ID {chunk_id}"
                ):
                    st.write(doc.page_content)
                    st.json(doc.metadata)

except Exception as erro:
    st.error("Ocorreu um erro ao executar a aplicação.")
    st.exception(erro)