import sys
sys.path.insert(0, '.')
from src.data.live_cache import get_pipeline_status

statuses = get_pipeline_status()
for s in statuses:
    freshness = 'LIVE ' if s.get('is_fresh') else 'STALE'
    date      = s.get('as_of_date', 'NEVER FETCHED')
    age       = s.get('age_hours', '?')
    print(f"{s['ticker']:6} | {freshness} | as_of: {date} | age: {age}h")