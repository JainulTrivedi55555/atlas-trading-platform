import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()   # Reads your .env file
ROOT_DIR      = Path(__file__).parent.parent.parent
DATA_DIR      = ROOT_DIR / 'data'
RAW_DIR       = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
FRED_API_KEY      = os.getenv('FRED_API_KEY', '')
NEWS_API_KEY      = os.getenv('NEWS_API_KEY', '')
TICKERS = ['AAPL','MSFT','GOOGL','AMZN','META','JPM','GS','BAC','NVDA','TSLA']
TRAIN_START = '2015-01-01'
TRAIN_END   = '2022-12-31'
VAL_START   = '2023-01-01'
VAL_END     = '2023-06-30'
TEST_START  = '2023-07-01'
TEST_END    = '2025-12-31'
LOOKBACK_WINDOW = 60   # Days of history fed into time-series models
PREDICTION_HORIZON = 5 # Days ahead to predict
CHUNK_SIZE = 500       # Tokens per RAG document chunk
CHUNK_OVERLAP = 50     
TOP_K_RETRIEVAL = 5    #Number of chunks returned per RAG query
