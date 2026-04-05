## Deployment Details
- **Date:** April 2026
- **Platform:** Google Cloud Platform
- **Instance:** e2-standard-2 (2 vCPU, 8GB RAM)
- **OS:** Ubuntu 22.04 LTS
- **Region:** us-east1-b
- **Public IP:** 34.24.181.122
- **Domain:** atlas-trading-platform.me

## Services Running
- FastAPI (Uvicorn) — port 8000 — systemd auto-start
- Streamlit Dashboard — port 8501 — systemd auto-start
- Scheduler — daily 8:30 AM ET — systemd timer

## URLs
- Dashboard: http://atlas-trading-platform.me
- API Health: http://atlas-trading-platform.me/api/health
- API Docs: http://atlas-trading-platform.me/docs
- Direct IP: http://34.24.181.122

## CI/CD
- GitHub Actions auto-deploys on every push to main
- Workflow: .github/workflows/deploy.yml
