import os
from pathlib import Path

from insight_bot.data_loader import df
from insight_bot.system_prompt import SYSTEM_PROMPT


ROOT_DIR = Path(__file__).resolve().parents[1]
CONTEXT_DIR = ROOT_DIR / "context"
STORAGE_DIR = ROOT_DIR / "storage"
GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
CHROMA_COLLECTION = "garmentpro_context_gemini_embedding_2"

# Disable optional telemetry; explicit model and embedding requests go to Gemini.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")

_vector_index = None


def _get_vector_index(embed_model):
    global _vector_index
    if _vector_index is not None:
        return _vector_index

    import chromadb
    from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
    from llama_index.vector_stores.chroma import ChromaVectorStore

    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    if collection.count() > 0:
        _vector_index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
    else:
        documents = SimpleDirectoryReader(
            str(CONTEXT_DIR), required_exts=[".md"]
        ).load_data()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        _vector_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
        )
    return _vector_index


def create_chat_engine():
    """Create a session-scoped conversational engine on the first chat message."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in the app environment."
        )

    from llama_index.core import PromptTemplate
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.base.response.schema import AsyncStreamingResponse, StreamingResponse
    from llama_index.core.schema import QueryBundle
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.selectors import LLMSingleSelector
    from llama_index.core.tools import QueryEngineTool
    from llama_index.experimental.query_engine import PandasQueryEngine
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.core.chat_engine import CondenseQuestionChatEngine

    llm = GoogleGenAI(
        model=GEMINI_MODEL,
        api_key=api_key,
        system_prompt=SYSTEM_PROMPT,
    )
    embed_model = GoogleGenAIEmbedding(
        model_name=GEMINI_EMBEDDING_MODEL,
        api_key=api_key,
    )
    vector_index = _get_vector_index(embed_model)

    # PandasQueryEngine executes generated Python locally; its dataframe preview is sent to Gemini.
    pandas_engine = PandasQueryEngine(
        df=df,
        llm=llm,
        instruction_str=(
            "Use only the provided dataframe. Do not import modules, access files, "
            "or use network resources. Return concise calculations with exact values."
        ),
        verbose=False,
    )
    context_engine = vector_index.as_query_engine(similarity_top_k=4, llm=llm)
    query_engine = RouterQueryEngine.from_defaults(
        query_engine_tools=[
            QueryEngineTool.from_defaults(
                query_engine=pandas_engine,
                description="Calculate exact statistics, comparisons, and trends from the productivity dataframe.",
            ),
            QueryEngineTool.from_defaults(
                query_engine=context_engine,
                description="Retrieve business definitions, glossary terms, and metric documentation.",
            ),
        ],
        selector=LLMSingleSelector.from_defaults(llm=llm),
        llm=llm,
    )
    answer_prompt = PromptTemplate(
        """Answer the question using only the dashboard context in the question and the retrieved evidence below.
Use exact values when present, do not claim correlation is causation, stay under 200 words, and end with 1-2 useful follow-up questions.

Question and dashboard context:
{question}

Retrieved evidence:
{evidence}"""
    )

    class StreamingRouterQueryEngine(BaseQueryEngine):
        def __init__(self):
            self._router = query_engine
            self._llm = llm
            super().__init__(callback_manager=query_engine.callback_manager)

        def _get_prompt_modules(self):
            return {"router": self._router}

        def _query(self, query_bundle: QueryBundle):
            result = self._router.query(query_bundle)
            tokens = self._llm.stream(
                answer_prompt,
                question=query_bundle.query_str,
                evidence=str(result),
            )
            return StreamingResponse(
                response_gen=tokens,
                source_nodes=getattr(result, "source_nodes", []),
            )

        async def _aquery(self, query_bundle: QueryBundle):
            result = await self._router.aquery(query_bundle)
            tokens = await self._llm.astream(
                answer_prompt,
                question=query_bundle.query_str,
                evidence=str(result),
            )
            return AsyncStreamingResponse(
                response_gen=tokens,
                source_nodes=getattr(result, "source_nodes", []),
            )

    condense_prompt = PromptTemplate(
        """Given the conversation history and a follow-up question, rewrite the question as a standalone question.
Keep the requested data context in the standalone question.

Conversation:
        {chat_history}
Follow-up: {question}
Standalone question:"""
    )
    return CondenseQuestionChatEngine.from_defaults(
        query_engine=StreamingRouterQueryEngine(),
        llm=llm,
        condense_question_prompt=condense_prompt,
        verbose=False,
    )