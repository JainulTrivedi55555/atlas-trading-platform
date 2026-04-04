from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import CHUNK_SIZE, CHUNK_OVERLAP, RAW_DIR

def chunk_filings(filings_df: pd.DataFrame = None) -> list:
    """
    Split SEC filings into overlapping text chunks.
    Returns list of LangChain Document objects.
    """
    if filings_df is None:
        filings_path = RAW_DIR / 'sec_filings' / 'sec_filings.csv'
        if not filings_path.exists():
            raise FileNotFoundError(
                'sec_filings.csv not found. Run sec_collector.py first.'
            )
        filings_df = pd.read_csv(filings_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=['\n\n', '\n', '. ', ' ', ''],
        length_function=len
    ) 

    all_chunks = []
    for _, row in filings_df.iterrows():
        if not isinstance(row['text'], str) or len(row['text']) < 100:
            continue
        
        chunks = splitter.split_text(row['text'])
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'ticker':    row['ticker'],
                    'form_type': row['form_type'],
                    'date':      row['date'],
                    'chunk_id':  i,
                    'source':    f"{row['ticker']}_{row['form_type']}_{row['date']}"
                }
            )
            all_chunks.append(doc)

    if not all_chunks:
        print("Warning: No chunks were created.")
        return []

    print(f'Total chunks created: {len(all_chunks)}')
    print(f'Average chunk length: '
          f'{sum(len(d.page_content) for d in all_chunks)//len(all_chunks)} chars')
    
    # Show breakdown by ticker
    from collections import Counter
    ticker_counts = Counter(d.metadata['ticker'] for d in all_chunks)
    print('\nChunks per ticker:')
    for ticker, count in sorted(ticker_counts.items()):
        print(f'  {ticker}: {count} chunks')
        
    return all_chunks

if __name__ == '__main__':
    chunks = chunk_filings()
    if chunks:
        print(f'\nSample chunk:')
        print(f'Ticker: {chunks[0].metadata["ticker"]}')
        print(f'Form:   {chunks[0].metadata["form_type"]}')
        print(f'Text:   {chunks[0].page_content[:200]}...')