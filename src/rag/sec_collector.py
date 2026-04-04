import requests
import pandas as pd
import json
import time
import re
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.config import TICKERS, RAW_DIR

# SEC EDGAR requires a descriptive User-Agent with email
HEADERS = {
    'User-Agent': 'ATLAS-Research atlas@research.com',
    'Accept-Encoding': 'gzip, deflate',
}

# CIK numbers WITHOUT leading zeros (matches EDGAR data/ folder)
TICKER_CIK = {
    'AAPL':  '320193',
    'MSFT':  '789019',
    'GOOGL': '1652044',
    'AMZN':  '1018724',
    'META':  '1326801',
    'JPM':   '19617',
    'GS':    '886982',
    'BAC':   '70858',
    'NVDA':  '1045810',
    'TSLA':  '1318605',
}


def get_filings_list(cik: str, form_type: str = '10-K',
                     max_filings: int = 3) -> list:
    """Get list of recent filings using EDGAR submissions API."""
    cik_padded = str(cik).zfill(10)
    url = f'https://data.sec.gov/submissions/CIK{cik_padded}.json'
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        recent = data.get('filings', {}).get('recent', {})
        forms   = recent.get('form', [])
        dates   = recent.get('filingDate', [])
        accnums = recent.get('accessionNumber', [])
        result  = []
        for form, date, acc in zip(forms, dates, accnums):
            if form == form_type:
                result.append({
                    'form':      form,
                    'date':      date,
                    'accession': acc.replace('-', '')
                })
            if len(result) >= max_filings:
                break
        return result
    except Exception as e:
        print(f'  ERROR getting filings list: {e}')
        return []


def get_filing_documents(cik: str, accession: str) -> list:
    """
    Get list of documents in a filing using the JSON index.
    Returns list of dicts with name, type, size.
    """
    cik_int = str(int(cik))
    acc = accession.replace('-', '')
    acc_fmt = f'{acc[:10]}-{acc[10:12]}-{acc[12:]}'

    url = (f'https://www.sec.gov/Archives/edgar/data/'
           f'{cik_int}/{acc}/{acc_fmt}-index.json')
    try:
        resp = requests.get(
            url,
            headers={**HEADERS, 'Host': 'www.sec.gov'},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get('directory', {}).get('item', [])
    except Exception:
        pass
    return []


def get_filing_text(cik: str, accession: str,
                    ticker: str = '',
                    form_type: str = '',
                    date: str = '') -> str:
    """
    Get readable text from a SEC filing.
    Uses the filing's document index to find the main .htm file,
    then strips all HTML/XBRL tags to get clean English text.
    """
    from bs4 import BeautifulSoup

    cik_int = str(int(cik))
    acc = accession.replace('-', '')
    acc_fmt = f'{acc[:10]}-{acc[10:12]}-{acc[12:]}'
    base_url = f'https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/'
    web_headers = {**HEADERS, 'Host': 'www.sec.gov'}

    def extract_clean_text(url):
        """Download URL and extract clean English text."""
        try:
            time.sleep(0.4)
            r = requests.get(url, headers=web_headers, timeout=30)
            if r.status_code != 200:
                return ''
            soup = BeautifulSoup(r.content, 'lxml')
            # Remove all non-content tags
            for tag in soup(['script', 'style', 'meta', 'link',
                             'ix:header', 'ix:resources',
                             'ix:nonnumeric', 'ix:nonfraction',
                             'xbrli:xbrl', 'xbrl']):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception:
            return ''

    def is_good_text(text):
        """Check if text is readable English filing content."""
        if len(text) < 8000:
            return False
        english_words = ['revenue', 'risk', 'business', 'management',
                         'fiscal', 'results', 'operations', 'products',
                         'financial', 'income', 'company', 'market']
        hits = sum(1 for w in english_words if w in text.lower())
        return hits >= 4

    # ── Strategy 1: Use document index JSON ─────────────────────
    docs = get_filing_documents(cik, acc)
    if docs:
        # Sort by file size descending to try largest files first
        try:
            docs_sorted = sorted(
                docs,
                key=lambda x: int(str(x.get('size', '0'))
                                  .replace(',', '') or '0'),
                reverse=True
            )
        except Exception:
            docs_sorted = docs

        # Skip patterns — these are exhibits and XBRL instance docs
        skip = ['ex-', 'ex2', 'ex3', 'ex4', 'ex9', 'ex10',
                'ex21', 'ex23', 'ex31', 'ex32', 'exhibit',
                'xbrl', 'defm', 'defa', 'r1.htm', 'r2.htm']

        for doc in docs_sorted:
            name = str(doc.get('name', '')).lower()
            doc_type = str(doc.get('type', '')).upper()

            # Skip exhibits and XBRL viewers
            if any(p in name for p in skip):
                continue
            # Only process .htm files
            if not name.endswith('.htm'):
                continue
            # Prioritise files with 10k or 10q in name
            is_main = any(p in name for p in
                         ['10k', '10-k', '10q', '10-q',
                          'annual', 'quarterly', 'form10'])

            doc_url = base_url + doc.get('name', '')
            text = extract_clean_text(doc_url)
            if is_good_text(text):
                print(f'    ✓ Strategy 1 [{doc.get("name")}]: '
                      f'{len(text):,} chars')
                return text[:50000]

    # ── Strategy 2: Try the primary document URL directly ────────
    # Many filings have a primary doc named after the company+date
    primary_candidates = [
        f'{acc_fmt}.htm',
        f'{ticker.lower()}{date.replace("-","")[:8]}_'
        f'{form_type.lower().replace("-","")}.htm',
    ]
    for candidate in primary_candidates:
        url = base_url + candidate
        text = extract_clean_text(url)
        if is_good_text(text):
            print(f'    ✓ Strategy 2 [{candidate}]: {len(text):,} chars')
            return text[:50000]

    # ── Strategy 3: EDGAR full-text search API ───────────────────
    if date:
        try:
            search_url = (
                f'https://efts.sec.gov/LATEST/search-index?q='
                f'%22{ticker}%22&forms={form_type}'
                f'&dateRange=custom&startdt={date}&enddt={date}'
            )
            resp = requests.get(search_url, headers=HEADERS, timeout=10)
            if resp.status_code == 200:
                hits = (resp.json().get('hits', {})
                        .get('hits', []))
                for hit in hits:
                    text = hit.get('_source', {}).get('file_text', '')
                    if is_good_text(text):
                        print(f'    ✓ Strategy 3 (EFTS): '
                              f'{len(text):,} chars')
                        return text[:50000]
        except Exception:
            pass

    print(f'    ✗ All strategies failed for {ticker} {form_type} {date}')
    return ''


def collect_sec_filings(tickers=None, form_types=None,
                        max_per_ticker=3):
    """Download SEC filings for all tickers."""
    if tickers is None:
        tickers = list(TICKER_CIK.keys())
    if form_types is None:
        form_types = ['10-K', '10-Q']

    save_dir = RAW_DIR / 'sec_filings'
    save_dir.mkdir(parents=True, exist_ok=True)

    all_filings = []
    for ticker in tickers:
        if ticker not in TICKER_CIK:
            print(f'  SKIP: {ticker} not in CIK map')
            continue
        cik = TICKER_CIK[ticker]
        print(f'\nCollecting {ticker} (CIK: {cik})...')

        for form_type in form_types:
            filings = get_filings_list(cik, form_type, max_per_ticker)
            print(f'  Found {len(filings)} {form_type} filings')
            for filing in filings:
                print(f'  → {form_type} from {filing["date"]}...')
                text = get_filing_text(
                    cik,
                    filing['accession'],
                    ticker=ticker,
                    form_type=form_type,
                    date=filing['date']
                )
                if text:
                    all_filings.append({
                        'ticker':     ticker,
                        'form_type':  form_type,
                        'date':       filing['date'],
                        'accession':  filing['accession'],
                        'text':       text,
                        'char_count': len(text)
                    })
                    print(f'    Saved {len(text):,} chars ✓')
                else:
                    print(f'    Skipped — no readable text')
                time.sleep(1)

    if all_filings:
        df = pd.DataFrame(all_filings)
        df.to_csv(save_dir / 'sec_filings.csv', index=False)
        print(f'\n{"="*50}')
        print(f'Total filings saved: {len(df)}')
        print(df[['ticker', 'form_type', 'date', 'char_count']]
              .to_string(index=False))
        return df
    else:
        # Even if download fails, create synthetic data so
        # rest of pipeline works — see note below
        print('\nWARNING: No filings downloaded.')
        print('Creating synthetic filings from price data...')
        return create_synthetic_filings(tickers, save_dir)


def create_synthetic_filings(tickers, save_dir):
    """
    Fallback: create synthetic filing summaries from known
    company facts so the RAG pipeline can still run.
    This is used when SEC EDGAR download fails.
    """
    COMPANY_FACTS = {
        'AAPL': """Apple Inc. Annual Report. Apple designs, manufactures
        and markets smartphones, personal computers, tablets, wearables
        and accessories, and sells a variety of related services.
        iPhone revenue represents the largest segment. Risk factors include
        dependence on iPhone sales, competition from Samsung and Chinese
        manufacturers, supply chain concentration in Asia, regulatory
        scrutiny in Europe and China. Management discussion: revenue growth
        driven by services segment including App Store, Apple Music, iCloud.
        Operating income margins remain strong. China market faces headwinds
        from local competition and geopolitical risks. Capital allocation
        includes significant share buybacks. R&D investment in AI and
        augmented reality. Data privacy regulations present compliance costs.
        Macroeconomic conditions including consumer spending affect demand.""",

        'NVDA': """NVIDIA Corporation Annual Report. NVIDIA is a computing
        infrastructure company. Data Center segment revenue grew significantly
        driven by AI and machine learning workloads. Gaming GPU revenue
        faces cyclical demand. Products include H100 and A100 data center
        GPUs, GeForce gaming GPUs, automotive chips. Risk factors include
        export controls on advanced chips to China, customer concentration
        in hyperscale cloud providers, competition from AMD and custom silicon
        from Google and Amazon. Supply constraints from TSMC manufacturing.
        Management outlook: AI infrastructure spending expected to remain
        strong. Automotive segment growing with self-driving partnerships.
        Gross margins elevated due to data center product mix.""",

        'META': """Meta Platforms Inc Annual Report. Meta owns Facebook,
        Instagram, WhatsApp and Oculus. Advertising revenue represents
        substantially all revenue. Daily active users growing in Asia Pacific
        and Rest of World. Risk factors include regulatory scrutiny in EU and
        US, advertiser spending sensitivity to macroeconomic conditions,
        competition from TikTok for user attention, privacy regulations
        limiting ad targeting. Reality Labs segment generates operating losses
        from metaverse investments. Management discussion: AI investments
        improving ad targeting efficiency. Reels short video driving
        engagement growth. Cost reduction initiatives improving margins.""",

        'MSFT': """Microsoft Corporation Annual Report. Microsoft operates
        through Productivity and Business Processes, Intelligent Cloud, and
        More Personal Computing segments. Azure cloud revenue growing rapidly.
        Office 365 commercial seats expanding. Risk factors include
        competition from Amazon AWS and Google Cloud, cybersecurity threats,
        regulatory antitrust scrutiny of acquisitions, AI liability concerns.
        Management discussion: Copilot AI features driving Office premium
        upgrades. Azure market share gains continuing. Gaming segment
        includes Xbox and Activision Blizzard acquisition. Strong free
        cash flow supports dividend and buybacks.""",

        'GOOGL': """Alphabet Inc Annual Report. Google Search advertising
        remains dominant revenue source. YouTube advertising growing.
        Google Cloud gaining enterprise customers. Risk factors include
        search market share threat from AI assistants, regulatory antitrust
        actions in US and EU, privacy regulations affecting advertising,
        competition in cloud from AWS and Azure. Management discussion:
        AI integration into Search products with Gemini. Cost efficiency
        program reduced headcount. Other Bets segment investing in
        Waymo autonomous vehicles and life sciences.""",

        'AMZN': """Amazon.com Inc Annual Report. Amazon operates through
        North America, International, and AWS segments. AWS cloud services
        generate majority of operating income. E-commerce faces margin
        pressure from fulfillment costs. Risk factors include AWS competition
        from Microsoft Azure and Google Cloud, labor cost inflation in
        fulfillment network, regulatory scrutiny of marketplace practices,
        thin retail margins. Management discussion: AWS growth reaccelerating
        with AI workloads. Advertising business growing rapidly.
        Prime membership provides recurring revenue. Fulfillment
        network optimization improving retail margins.""",

        'JPM': """JPMorgan Chase Annual Report. JPMorgan is the largest
        US bank by assets. Consumer and Community Banking, Commercial Banking,
        Corporate and Investment Bank, Asset and Wealth Management segments.
        Net interest income benefits from higher interest rates. Risk factors
        include credit losses in consumer and commercial loan portfolios,
        regulatory capital requirements, interest rate sensitivity,
        cybersecurity threats. Management discussion: credit loss provisions
        reflect uncertain macroeconomic outlook. Investment banking fees
        recovering. Loan growth moderated as credit conditions tighten.
        Capital ratios well above regulatory minimums.""",

        'GS': """Goldman Sachs Annual Report. Goldman Sachs operates in
        Global Banking and Markets, Asset and Wealth Management, and
        Platform Solutions segments. Investment banking advisory and
        underwriting revenue cyclical. Risk factors include market volatility
        affecting trading revenues, credit losses in consumer lending,
        regulatory capital requirements, key personnel retention.
        Management discussion: refocused on core strengths after
        consumer banking exit. Asset management fees growing.
        Marcus consumer business wind-down reducing losses.""",

        'BAC': """Bank of America Annual Report. Bank of America serves
        consumer, commercial, and institutional clients. Consumer Banking,
        Global Wealth, Global Banking, Global Markets segments.
        Net interest income sensitive to Federal Reserve rate decisions.
        Risk factors include credit losses on consumer and commercial loans,
        interest rate risk on securities portfolio, regulatory requirements,
        mortgage origination slowdown. Management discussion: deposit
        repricing headwind diminishing. Net interest income expected to
        stabilize. Expense management improving operating leverage.""",

        'TSLA': """Tesla Inc Annual Report. Tesla designs and manufactures
        electric vehicles, energy generation and storage systems.
        Model 3, Model Y represent majority of vehicle deliveries.
        Risk factors include production ramp challenges at new factories,
        competition from traditional automakers and Chinese EV manufacturers,
        raw material cost volatility for lithium and cobalt, regulatory
        incentive changes affecting EV demand. Management discussion:
        vehicle delivery growth targets. Full self-driving software
        development timeline uncertain. Energy storage business growing.
        Price reductions maintaining demand but compressing margins.""",
    }

    all_filings = []
    for ticker in tickers:
        if ticker not in COMPANY_FACTS:
            continue
        text = COMPANY_FACTS[ticker]
        # Create both 10-K and 10-Q entries
        for form_type, date in [('10-K', '2024-01-01'),
                                 ('10-Q', '2024-04-01'),
                                 ('10-Q', '2024-07-01')]:
            all_filings.append({
                'ticker':     ticker,
                'form_type':  form_type,
                'date':       date,
                'accession':  f'synthetic_{ticker}_{form_type}',
                'text':       text * 3,  # repeat for more chunks
                'char_count': len(text) * 3
            })

    df = pd.DataFrame(all_filings)
    df.to_csv(save_dir / 'sec_filings.csv', index=False)
    print(f'Synthetic filings created: {len(df)} entries')
    print(df[['ticker', 'form_type', 'date', 'char_count']]
          .to_string(index=False))
    return df


if __name__ == '__main__':
    df = collect_sec_filings()
    print(f'\nDone: {len(df)} filings saved')