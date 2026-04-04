from pathlib import Path
import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import PROCESSED_DIR

INDEX_PATH      = PROCESSED_DIR / 'rag' / 'faiss_index'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def get_embeddings():
    """Load the embedding model."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings


def build_faiss_index(chunks: list, save: bool = True):
    """Build FAISS vector store from document chunks."""
    from langchain_community.vectorstores import FAISS
    embeddings  = get_embeddings()
    print(f'Building FAISS index for {len(chunks)} chunks...')
    print('This takes 2-5 minutes on CPU for 1000+ chunks...')
    batch_size  = 100
    vectorstore = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        pct = min(100, int((i + batch_size) / len(chunks) * 100))
        print(f'  Indexed {min(i+batch_size,len(chunks))}'
              f'/{len(chunks)} chunks ({pct}%)')
    if save:
        os.makedirs(INDEX_PATH, exist_ok=True)
        vectorstore.save_local(str(INDEX_PATH))
        print(f'\nFAISS index saved to {INDEX_PATH}')
    return vectorstore


def load_faiss_index():
    """Load existing FAISS index from disk."""
    from langchain_community.vectorstores import FAISS
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f'FAISS index not found at {INDEX_PATH}. '
            f'Run build_faiss_index() first.'
        )
    embeddings  = get_embeddings()
    vectorstore = FAISS.load_local(
        folder_path=str(INDEX_PATH),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print(f'FAISS index loaded — {vectorstore.index.ntotal} vectors')
    return vectorstore


if __name__ == '__main__':
    from src.rag.chunker import chunk_filings
    chunks      = chunk_filings()
    vectorstore = build_faiss_index(chunks)
    print(f'Index built: {vectorstore.index.ntotal} vectors')