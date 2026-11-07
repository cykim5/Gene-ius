# Web app (Stage 10)

Three parts: an input box (variant or free-text symptoms), the 3D viewer, and the agent panel. The model and agent run behind an API on the server. The front end only displays results.

3D viewer (py3Dmol or Mol*):

- AlphaFold structure of the protein
- variants colored by predicted mechanism or disease
- the queried variant highlighted, with its 3D neighbors lit up when the agent cites them
- UniProt regions shaded, low-pLDDT regions faded

Agent panel:

- step-by-step tool calls (from `run_agent(...)["trace"]`)
- evidence paths and per-source contribution
- predictions A to D with confidence
- suggested assay and treatment strategy
- button to download the evidence report

Security (ArgosX): API keys on the server only, public data only, simple login, input validation, read-only tools, rate limiting, HTTPS.

Record a backup video of the demo in case the live version breaks.
