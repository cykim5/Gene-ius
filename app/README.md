# Web app

Type a variant (like `RHO:C187S`) and see predictions A to D, the time machine check, the AlphaFold structure with the variant and its 3D neighbors, the gene's uncertain variants ranked, and an agent (NVIDIA Nemotron) that explains the result.

It's plain HTML, CSS and JavaScript reading JSON files, plus one small Python function for the agent. Nothing to install or build.

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

Open http://localhost:8000. For the agent, start it with `NEBIUS_API_KEY=... python app/dev_server.py`.

## Deploy on Vercel (free)

1. Sign in at vercel.com with GitHub (the free Hobby plan is enough).
2. Add New, Project, import `Gene-ius`.
3. Set Root Directory to `app`. Leave the framework as Other.
4. Under Environment Variables add `NEBIUS_API_KEY` (never put it in the code).
5. Deploy. Every push to the branch redeploys.

## Security

- The API key lives only in Vercel's settings, never in the page or the repo.
- The agent reads evidence from the server's own data, not from the request, so it can't be fed fake evidence.
- It only answers requests from our own page, limits each visitor to 6 questions a minute, caps total use, and limits question length.
- No logins and nothing stored.
- Strict security headers (CSP, no framing, HTTPS only). The only outside source is the 3Dmol viewer library. AlphaFold structures are stored with the site.
- Public data only. No patient data.
