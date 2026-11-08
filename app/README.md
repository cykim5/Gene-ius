# Web app

Type a variant (like `RHO:C187S`) and see predictions A to D, the time machine check, the AlphaFold structure with the variant and its 3D neighbors, and the gene's uncertain variants ranked.

It's a static site: plain HTML, CSS and JavaScript reading JSON files. No server code, nothing to install or build.

```
app/
  index.html, static/      the page
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

Open http://localhost:8000.

## Deploy on Vercel (free)

1. Sign in at vercel.com with GitHub (the free Hobby plan is enough).
2. Add New, Project, import `Gene-ius`.
3. Set Root Directory to `app`. Leave the framework as Other.
4. Deploy. Every push to the branch redeploys.

## Security

- Static site with no server code, no API keys, no logins, and nothing stored.
- Strict security headers (CSP, no framing, HTTPS only). The only outside sources are the 3Dmol library and AlphaFold.
- Public data only. No patient data.
- The agent from geneius/agent runs separately and isn't part of the public site.
