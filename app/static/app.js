"use strict";

const AA3 = {
  Ala: "A", Arg: "R", Asn: "N", Asp: "D", Cys: "C", Gln: "Q", Glu: "E", Gly: "G", His: "H", Ile: "I",
  Leu: "L", Lys: "K", Met: "M", Phe: "F", Pro: "P", Ser: "S", Thr: "T", Trp: "W", Tyr: "Y", Val: "V",
};
const AA1TO3 = Object.fromEntries(Object.entries(AA3).map(([k, v]) => [v, k.toUpperCase()]));
const CLINVAR = {
  P: ["Pathogenic", "p"], LP: ["Likely pathogenic", "p"], B: ["Benign", "b"], LB: ["Likely benign", "b"],
  VUS: ["Uncertain", "u"], conflicting: ["Conflicting", "u"], other: ["Other", ""],
};
const ROLE_2023 = {
  test: "Held-out test variant: the 2023 model never saw its answer.",
  test_1star: "Held-out test variant (1-star review): the 2023 model never saw its answer.",
  train: "Part of the 2023 training data, so this isn't a blind test.",
  unlabeled: "Uncertain in January 2023.",
  exclude: "Not used in the 2023 grading.",
};
const UNCERTAIN_TODAY = new Set(["VUS", "conflicting"]);

const $ = (id) => document.getElementById(id);
const state = { index: null, genes: {}, structures: {}, current: null, viewer: null };

function el(tag, attrs = {}, ...kids) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) node.append(kid);
  return node;
}

function setText(id, value, empty = "Not available") {
  const node = $(id);
  node.replaceChildren();
  if (value === undefined || value === null || value === "") {
    node.append(el("span", { class: "na" }, empty));
  } else if (value instanceof Node) {
    node.append(value);
  } else {
    node.textContent = String(value);
  }
}

function message(text) {
  const m = $("message");
  m.textContent = text || "";
  m.hidden = !text;
}

// "RHO:C187S", "rho c187s", "RHO p.Cys187Ser" -> {gene, ref, pos, alt}
function parseQuery(text) {
  const m = String(text).trim().match(/^([A-Za-z0-9-]+)\s*[:\s]\s*(?:p\.)?([A-Za-z]{1,3})(\d+)([A-Za-z]{1,3})$/);
  if (!m) return null;
  const aa = (s) => (s.length === 1 ? s.toUpperCase() : AA3[s[0].toUpperCase() + s.slice(1).toLowerCase()]);
  const ref = aa(m[2]), alt = aa(m[4]);
  if (!ref || !alt) return null;
  return { gene: m[1].toUpperCase(), ref, pos: Number(m[3]), alt };
}

const label = (r) => `${r.gene}:${r.ref}${r.pos}${r.alt}`;

function evidenceLabel(code) {
  if (!code) return "";
  if (code === "indeterminate") return "Indeterminate";
  const [c, strength] = code.split("_");
  return strength ? `${c} ${strength.toLowerCase()}` : c;
}

function clinvarTag(code) {
  if (!code) return null;
  const [text, cls] = CLINVAR[code] || [code, ""];
  return el("span", { class: `tag ${cls}` }, text);
}

async function getJSON(url) {
  // no-cache: always check for a newer file, so an update never shows stale data
  const r = await fetch(url, url.startsWith("/") ? { cache: "no-cache" } : {});
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function loadGene(gene) {
  if (!state.genes[gene]) state.genes[gene] = getJSON(`/data/genes/${encodeURIComponent(gene)}.json`);
  return state.genes[gene];
}

// ---------- Lookup ----------

async function lookup(text, pick = null) {
  message("");
  const q = parseQuery(text);
  if (!q) return message("Type a gene and protein change, like RHO:C187S or KCNQ1 p.Arg591Cys.");
  if (!state.index.genes[q.gene]) {
    return message(`${q.gene} isn't in our gene panels. Try one of: ${Object.keys(state.index.genes).join(", ")}.`);
  }
  const recs = (await loadGene(q.gene)).filter((r) => r.pos === q.pos && r.ref === q.ref && r.alt === q.alt);
  if (!recs.length) {
    const atPos = (await loadGene(q.gene)).filter((r) => r.pos === q.pos);
    const hint = atPos.length ? ` At residue ${q.pos} we have: ${atPos.map((r) => `${r.ref}${r.pos}${r.alt}`).join(", ")}.` : "";
    return message(`No prediction for ${label(q)}.${hint}`);
  }
  history.replaceState(null, "", `#${label(q)}`);
  $("q").value = label(q);
  show(recs, pick);
}

function show(recs, pick) {
  const r = recs.find((x) => x.variation_id === pick) || recs[0];
  state.current = r;
  const info = state.index.genes[r.gene];

  $("intro").hidden = true;
  $("result").hidden = false;
  document.querySelectorAll("#demo-list .chip").forEach((c) => c.classList.toggle("on", c.dataset.v === label(r)));

  $("r-panel").textContent = info.panel === "retina" ? "Retina panel" : info.panel === "cardiac" ? "Cardiac panel" : "Panel gene";
  $("r-title").textContent = `${r.gene} p.${r.ref}${r.pos}${r.alt}`;
  $("r-sub").textContent = `UniProt ${info.uniprot}, residue ${r.pos} of ${info.length}`;

  // Same protein change from different DNA changes
  const ids = $("r-ids");
  ids.replaceChildren();
  if (recs.length > 1) {
    ids.append(el("span", { class: "muted small" }, `${recs.length} DNA changes give this protein change:`));
    for (const x of recs) {
      ids.append(el("button", {
        class: `chip${x === r ? " on" : ""}`, type: "button",
        onclick: () => show(recs, x.variation_id),
      }, `ClinVar ${x.variation_id}`));
    }
  } else {
    ids.append(el("span", { class: "muted small" }, `ClinVar ${r.variation_id}`));
  }

  // A
  const h = Number(r.harmful);
  const bar = $("a-bar");
  bar.style.width = `${Math.round(h * 100)}%`;
  bar.className = h >= 0.8 ? "high" : h >= 0.5 ? "mid" : "";
  $("a-score").textContent = Number.isFinite(h) ? h.toFixed(2) : "?";
  setText("a-conf", r.confidence === "unsure" ? el("span", { class: "tag u" }, "Unsure") : r.confidence);
  setText("a-acmg", evidenceLabel(r.computational_evidence));
  setText("a-am", r.alphamissense != null ? Number(r.alphamissense).toFixed(2) : "", "Not in our data");
  setText("a-evidence", r.evidence);

  // Time machine
  const has2023 = r.harmful_2023_model !== undefined && r.harmful_2023_model !== null;
  $("time-card").hidden = !has2023;
  if (has2023) {
    $("t-2023").textContent = Number(r.harmful_2023_model).toFixed(2);
    $("t-2023-ev").textContent = r.evidence_2023 || "";
    setText("t-now", clinvarTag(r.clinvar_today));
    $("t-role").textContent = ROLE_2023[r.role_2023] || "";
  }

  // B
  const mech = r.mechanism || {};
  setText("b-label", mech.label ? mech.label.replace(/_/g, " ") : "",
    "Not modeled yet. FoldX ran on RHO, RPE65 and GLA.");
  setText("b-ddg", mech.ddg_kcal_mol != null ? `${mech.ddg_kcal_mol > 0 ? "+" : ""}${mech.ddg_kcal_mol} kcal/mol${mech.capped ? " (capped)" : ""}` : "", "Not modeled");
  setText("b-term", mech.dominant_term, "Not modeled");

  // C
  const groups = Object.entries(r.disease_groups || {}).sort((a, b) => b[1] - a[1]).slice(0, 3);
  const list = groups.length ? el("ol", { class: "diseases" }) : "";
  for (const [name, score] of groups) list.append(el("li", {}, name, el("span", { class: "muted small" }, ` ${score.toFixed(1)}`)));
  setText("c-disease", list, "No disease prediction");
  setText("c-near", r.nearest_known, "No known disease variant nearby");
  setText("c-para", r.paralog_evidence, "None at the matching residue");
  setText("c-mouse", r.mouse_evidence || info.mouse);

  // D
  setText("d-strategy", r.treatment, "No strategy in our data for this gene yet");
  setText("d-assay", r.suggested_assay);

  $("answer").replaceChildren();
  $("answer").className = "answer";
  renderRanked(r.gene, r);
  renderStructure(r, info).catch(() => setText("viewer-note", "Couldn't load the AlphaFold structure."));
}

async function renderRanked(gene, current) {
  const recs = (await loadGene(gene))
    .filter((x) => UNCERTAIN_TODAY.has(x.clinvar_today))
    .sort((a, b) => b.harmful - a.harmful)
    .slice(0, 15);
  $("rank-title").textContent = `${gene}: uncertain today, ranked`;
  const body = $("rank-body");
  body.replaceChildren();
  if (!recs.length) {
    body.append(el("tr", {}, el("td", { colspan: "5", class: "muted" }, "No uncertain variants for this gene in our data.")));
    return;
  }
  recs.forEach((x, i) => {
    body.append(el("tr", {
      class: x === current ? "on" : "",
      onclick: () => lookup(label(x), x.variation_id),
    },
    el("td", {}, String(i + 1)),
    el("td", {}, `${x.ref}${x.pos}${x.alt}`),
    el("td", {}, Number(x.harmful).toFixed(3)),
    el("td", {}, x.confidence || ""),
    el("td", {}, x.suggested_assay || "")));
  });
}

// ---------- Structure ----------

function plddtColor(b) {
  if (b >= 90) return "#0053D6";
  if (b >= 70) return "#65CBF3";
  if (b >= 50) return "#FFDB13";
  return "#FF7D45";
}

// AlphaFold models are stored with the site (see build_data.py), so the page never calls another server.
async function loadStructure(info) {
  if (!info.structure) return null;
  if (!state.structures[info.uniprot]) {
    state.structures[info.uniprot] = fetch(info.structure).then((r) => {
      if (!r.ok) throw new Error("pdb");
      return r.text();
    });
  }
  return state.structures[info.uniprot];
}

async function renderStructure(r, info) {
  const box = $("viewer");
  setText("viewer-note", "Loading AlphaFold structure...");
  if (!window.$3Dmol) return setText("viewer-note", "The 3D viewer didn't load.");
  const pdb = await loadStructure(info);
  if (state.current !== r) return;
  box.hidden = !pdb;
  document.querySelector(".legend").hidden = !pdb;
  if (!pdb) {
    if (state.viewer) state.viewer.clear();
    return setText("viewer-note", "AlphaFold has no single model for this whole protein, so there's no 3D view.");
  }

  if (!state.viewer) state.viewer = window.$3Dmol.createViewer(box, { backgroundColor: "#fbfbfd" });
  const v = state.viewer;
  v.clear();
  const model = v.addModel(pdb, "pdb");
  v.setStyle({}, { cartoon: { colorfunc: (a) => plddtColor(a.b) } });

  const ca = model.selectedAtoms({ atom: "CA" });
  const site = ca.find((a) => a.resi === r.pos);
  if (!site) {
    v.zoomTo();
    v.render();
    return setText("viewer-note", `Residue ${r.pos} isn't in the AlphaFold model.`);
  }
  const near = ca
    .filter((a) => a.resi !== r.pos && Math.hypot(a.x - site.x, a.y - site.y, a.z - site.z) <= 8)
    .map((a) => a.resi);

  if (near.length) v.addStyle({ resi: near }, { stick: { colorscheme: "orangeCarbon", radius: 0.18 } });
  v.addStyle({ resi: r.pos }, { stick: { colorscheme: "redCarbon", radius: 0.3 } });
  v.addStyle({ resi: r.pos, atom: "CA" }, { sphere: { color: "#d64545", radius: 1.4 } });
  v.zoomTo({ resi: [r.pos, ...near] });
  v.zoom(0.6);
  v.render();

  const match = site.resn === AA1TO3[r.ref];
  setText("viewer-note",
    `${site.resn} ${r.pos}, model confidence (pLDDT) ${Math.round(site.b)}. ${near.length} residues within 8 Å.` +
    (match ? "" : ` Note: the model has ${site.resn} here, not ${AA1TO3[r.ref]}, so numbering may differ.`));
}

// ---------- Agent ----------

async function ask(event) {
  event.preventDefault();
  const r = state.current;
  if (!r) return;
  const btn = $("ask").querySelector("button");
  const out = $("answer");
  btn.disabled = true;
  out.className = "answer muted";
  out.textContent = "Thinking... (about 10 seconds)";
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gene: r.gene, variation_id: r.variation_id, question: $("question").value }),
    });
    const body = await res.json().catch(() => ({}));
    if (state.current !== r) return;
    out.className = res.ok ? "answer" : "answer err";
    out.textContent = res.ok ? body.answer : body.error || "The agent didn't answer.";
  } catch {
    out.className = "answer err";
    out.textContent = "Couldn't reach the agent.";
  } finally {
    btn.disabled = false;
  }
}

// ---------- Start ----------

async function start() {
  try {
    state.index = await getJSON("/data/index.json");
  } catch {
    return message("No data yet. Run app/build_data.py with predictions.json.");
  }
  const list = $("gene-list");
  for (const g of Object.keys(state.index.genes)) list.append(el("option", { value: `${g}:` }));

  const demos = $("demo-list");
  for (const d of state.index.demos) {
    demos.append(el("button", {
      class: "chip", type: "button", "data-v": d.variant, onclick: () => lookup(d.variant),
    }, el("b", {}, d.label), d.variant));
  }

  $("search").addEventListener("submit", (e) => { e.preventDefault(); lookup($("q").value); });
  $("ask").addEventListener("submit", ask);
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask").requestSubmit(); }
  });

  const fromHash = () => {
    const hash = decodeURIComponent(location.hash.slice(1));
    if (hash && (!state.current || hash !== label(state.current))) lookup(hash);
  };
  window.addEventListener("hashchange", fromHash);
  fromHash();
}

start();
