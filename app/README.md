# Web app

Type a variant (like `RHO:C187S`) and see predictions A to D, the time machine check, the AlphaFold structure with the variant and its 3D neighbors, the gene's uncertain variants ranked, and an agent that explains the result.

It's a plain HTML page plus one small Python function, so there's nothing to install or build.

```
app/
  index.html, static/      the page
  api/ask.py               the agent (keeps the API key on the server)
  data/                    made by build_data.py from predictions.json
  build_data.py            predictions.json -> data/
  dev_server.py            run it on your computer
  vercel.json              Vercel settings and security headers
```

## Update the data

The app shows only what's in the team's `predictions.json`. After it changes:

```bash
python app/build_data.py path/to/predictions.json
```

## Run it locally

```bash
python app/dev_server.py
```

Open http://localhost:8000. The agent works if `NEBIUS_API_KEY` is in `.env`.

## Deploy on Vercel (free)

1. Sign in at vercel.com with GitHub (the free Hobby plan is enough).
2. Add New, Project, import `Gene-ius`.
3. Set Root Directory to `app`. Leave the framework as Other.
4. Under Environment Variables add `NEBIUS_API_KEY`. Optional: `LLM_MODEL`, `NEBIUS_BASE_URL`.
5. Deploy. Every push to the branch redeploys.

## Security

- The API key lives only in Vercel's settings, never in the page or the repo.
- The agent reads evidence from the server's own data, not from the request, so it can't be fed fake evidence.
- Input is checked (gene and ID format, 400-character questions), and there's a rate limit per visitor.
- Strict security headers (CSP, no framing, HTTPS only). The only outside sources are the 3Dmol library and AlphaFold.
- Public data only. No patient data, no logins, nothing stored.
