from pathlib import Path
from uuid import uuid4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from qdrant_client import QdrantClient, models
from uuid import uuid4
from typing import List
from langchain.schema import Document
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointVectors
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

openai_client = wrap_openai(OpenAI())

QDRANT_URL = "https://49ca8618-c349-49c7-92a4-fefb6d84f392.us-east4-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "tIpw7Tp9g0aCD2-4AWGiySLOPRF6kxBg-en9wGDo-ZQFXnBAEPLzeQ"
COLLECTION_NAME = "CHATBOT-RAG-5"

DOCS_FOLDER = Path("documents")

def get_loaders():
    web_loader = WebBaseLoader(["https://www.telefonica.com/es/sala-comunicacion/blog/actualizamos-nuestros-principios-de-inteligencia-artificial/"])
    pdf_loaders = [PyPDFLoader(str(pdf_path)) for pdf_path in DOCS_FOLDER.glob("*.pdf")]
    return [web_loader] + pdf_loaders

def get_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def configure_qdrant():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    try:
        # Intentar obtener información de la colección existente
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"La colección {COLLECTION_NAME} ya existe.")
    except UnexpectedResponse as e:
        if e.status_code == 404:
            print(f"La colección {COLLECTION_NAME} no existe. Creándola...")
            # Crear la colección si no existe
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=rest.VectorParams(size=1536, distance=rest.Distance.COSINE),
                hnsw_config=rest.HnswConfigDiff(
                    payload_m=16,
                    m=0,
                ),
                optimizers_config=rest.OptimizersConfigDiff(
                    indexing_threshold=0,  # Deshabilitar indexación global
                ),
                on_disk_payload=True  # Almacenar payload en disco para ahorrar memoria
            )
        else:
            raise

    try:
        # Intentar crear el índice de payload
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="group_id",
            field_schema=rest.PayloadSchemaType.KEYWORD
        )
        print("Índice de payload 'group_id' creado con éxito.")
    except UnexpectedResponse as e:
        if e.status_code == 400:
            print("El índice de payload 'group_id' ya existe o no se pudo crear.")
            print(f"Detalles del error: {e.content}")
        else:
            raise

    # Crear shards para cada grupo de usuarios
    try:
        client.create_shard_key(COLLECTION_NAME, "user_group_1")
        client.create_shard_key(COLLECTION_NAME, "user_group_2")
        print("Shards creados con éxito.")
    except UnexpectedResponse as e:
        print(f"No se pudieron crear los shards: {e.content}")

    return client


def verify_documents(qdrant_client, collection_name):
    # Obtener los primeros 10 documentos de la colección
    response = qdrant_client.scroll(
        collection_name=collection_name,
        limit=10,
        with_payload=True,
        with_vectors=False
    )

    for point in response[0]:
        print(f"ID: {point.id}")
        print(f"Payload: {point.payload}")
        print("---")


def initialize_qdrant(documents: List[Document]):
    configure_qdrant()
    embeddings = OpenAIEmbeddings()

    # Preparar los documentos con metadatos adicionales
    prepared_docs = []
    for doc in documents:
        # Asignar un ID único, grupo y shard a cada documento
        doc_id = str(uuid4())
        group_id = "user_group_1"  # Puedes ajustar esto según tus necesidades

        # Actualizar los metadatos del documento
        doc.metadata.update({
            "id": doc_id,
            "group_id": group_id,
        })
        prepared_docs.append(doc)

    # Crear QdrantVectorStore con los documentos preparados
    qdrant = QdrantVectorStore.from_documents(
        prepared_docs,
        embeddings,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=False  # No recreamos la colección aquí
    )

    # Actualizar los vectores con toda la información
    points_to_upsert = []
    for doc in prepared_docs:
        vector = embeddings.embed_query(doc.page_content)  # Genera el vector de embeddings
        points_to_upsert.append(
            models.PointStruct(
                id=doc.metadata["id"],
                payload={
                    "group_id": doc.metadata["group_id"],
                    "metadata": doc.metadata,
                    "page_content": doc.page_content
                },
                vector=vector
            )
        )

    # Upsert points in batches
    batch_size = 100  # Adjust this based on your needs
    for i in range(0, len(points_to_upsert), batch_size):
        batch = points_to_upsert[i:i+batch_size]
        qdrant.client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

    return qdrant

@traceable(run_type="retriever")
def retriever(query: str, group_id: str, qdrant: QdrantVectorStore):
    results = qdrant.similarity_search(
        query,
        k=5,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="group_id",
                    match=models.MatchValue(value=group_id)
                )
            ]
        )
    )

    # Imprimir los resultados para depuración
    print(f"Resultados para group_id: {group_id}")
    for doc in results:
        print(f"Content: {doc.page_content[:100]}...")  # Primeros 100 caracteres
        print(f"Metadata: {doc.metadata}")
        print("---")

    return results

@traceable(metadata={"model": "gpt-4o-mini"})
def rag(question: str, group_id: str, qdrant: QdrantVectorStore):
    docs = retriever(question, group_id, qdrant)
    context = "\n".join(doc.page_content for doc in docs)
    system_message = f"""Responde a la pregunta del usuario utilizando solo la información proporcionada a continuación:

    {context}"""

    return openai_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ],
        model="gpt-4o-mini",
    )

def chatbot(question: str, group_id: str, qdrant: QdrantVectorStore):
    run_id = str(uuid4())
    response = rag(question, group_id, qdrant, langsmith_extra={"run_id": run_id})
    return response.choices[0].message.content, run_id

if __name__ == "__main__":
    print("Inicializando Qdrant y cargando documentos...")
    merged_loader = MergedDataLoader(loaders=get_loaders())
    all_docs = merged_loader.load()
    chunks = get_chunks(all_docs)

    qdrant = initialize_qdrant(chunks)

    print("Chatbot RAG con LangSmith y multi-tenancy inicializado. Escribe 'salir' para terminar.")
    group_id = input("Introduce el ID de grupo (ej. 'user_group_1'): ")
    while True:
        user_input = input("Tu pregunta: ")
        if user_input.lower() == 'salir':
            break
        response, run_id = chatbot(user_input, group_id, qdrant)
        print("Chatbot:", response)
        print(f"Run ID: {run_id}")

# Opcional: Recoger feedback
from langsmith import Client

ls_client = Client()
# ls_client.create_feedback(run_id, key="user-score", score=1.0)