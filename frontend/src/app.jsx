import { useState, useEffect, useRef, useCallback } from "react";
import { ArrowLeft, ArrowRight, Check, CheckCircle, FileArrowUp, Plus, Warning, WarningCircle, X } from "@phosphor-icons/react";

/*
 * DualR Clinical Risk Assessment Platform
 * License: Apache-2.0
 * Copyright (C) 2026 The Authors.
 * Su Lab · Biomedical Informatics,
 * Biostatistics & Health Data Science
 * Indiana University School of Medicine
 *
 * Fonts: Geist and Geist Mono (SIL OFL), self-hosted. Styles and tokens live in src/styles.css.
 */

// ══════════════════════════════════════════════
// REAL FEATURE DEFINITIONS (from ml.py)
// ══════════════════════════════════════════════

const ALL_CHARLSON = [
  { id: "HIV", label: "HIV/AIDS", q: "Have you been diagnosed with HIV?" },
  { id: "Cerebrovascular_Disease", label: "Cerebrovascular disease", q: "Any history of stroke or TIA?" },
  { id: "Congestive_Heart_Failure", label: "Congestive heart failure", q: "Have you been diagnosed with heart failure?" },
  { id: "Myocardial_Infarction", label: "History of heart attack", q: "Have you ever had a heart attack?" },
  { id: "Peripheral_Vascular_Disease", label: "Peripheral vascular disease", q: "Any history of peripheral artery disease?" },
  { id: "Chronic_Pulmonary_Disease", label: "Chronic lung disease", q: "Do you have COPD, asthma, or another chronic lung condition?" },
  { id: "Dementia", label: "Dementia", q: "Have you been diagnosed with dementia?" },
  { id: "Liver_Disease_Mild", label: "Liver disease (mild)", q: "Any history of mild liver disease (e.g., fatty liver)?" },
  { id: "Liver_Disease_Moderate_Severe", label: "Liver disease (moderate/severe)", q: "Any moderate-to-severe liver disease (e.g., cirrhosis)?" },
  { id: "Malignancy", label: "Cancer (non-metastatic)", q: "Have you been diagnosed with cancer (non-metastatic)?" },
  { id: "Metastatic_Solid_Tumor", label: "Metastatic cancer", q: "Do you have metastatic cancer?" },
  { id: "Peptic_Ulcer_Disease", label: "Peptic ulcer disease", q: "Have you been diagnosed with peptic ulcer disease?" },
  { id: "Renal_Disease_Mild_Moderate", label: "Kidney disease (mild/moderate)", q: "Any mild-to-moderate kidney disease?" },
  { id: "Renal_Disease_Severe", label: "Severe kidney disease", q: "Any severe kidney disease or dialysis?" },
  { id: "Rheumatic_Disease", label: "Rheumatic disease", q: "Any rheumatic disease (e.g., lupus, rheumatoid arthritis)?" },
  { id: "Hemiplegia_Paraplegia", label: "Hemiplegia or paraplegia", q: "Any history of hemiplegia or paraplegia?" },
  { id: "Diabetes_with_Chronic_Complications", label: "Diabetes with complications", q: "Do you have diabetes with chronic complications?" },
  { id: "Diabetes_without_Chronic_Complications", label: "Diabetes without complications", q: "Have you been diagnosed with diabetes (without complications)?" },
];

// Exact disease-feature mappings from DISEASE_FEATURE_MAP in ml.py
const DISEASE_COMO_MAP = {
  t2d: ALL_CHARLSON.filter(c => !c.id.startsWith("Diabetes")),
  htn: ALL_CHARLSON,
  aud: ALL_CHARLSON, // full Charlson per ml.py
};

const PHENOTYPES = {
  t2d: {
    id: "t2d", name: "Type 2 Diabetes Mellitus", abbr: "T2D",
    desc: "Metabolic disorder characterized by insulin resistance and hyperglycemia",
    prevalence: "10.9%",
    auc: { base: 0.766, pdrs: 0.819, dualr: 0.851 },
    n: "247,642",
  },
  htn: {
    id: "htn", name: "Hypertension", abbr: "HTN",
    desc: "Persistent elevation of systemic arterial blood pressure",
    prevalence: "33.0%",
    auc: { base: 0.846, pdrs: 0.875, dualr: 0.886 },
    n: "254,487",
  },
  aud: {
    id: "aud", name: "Alcohol Use Disorder", abbr: "AUD",
    desc: "Impaired control over alcohol use, often underdocumented in clinical records",
    prevalence: "7.8%",
    auc: { base: 0.798, pdrs: 0.834, dualr: 0.826 },
    n: "254,487",
  },
};

// Age is a separate numeric input (integer, 18-120); the three fields below are categorical selects.
const DEMO_FIELDS = [
  { id: "gender", label: "Biological Sex", options: ["Man", "Woman", "Other"] },
  { id: "race", label: "Race", options: ["White", "Black", "Others"] },
  { id: "ethnicity", label: "Ethnicity", options: ["Hispanic", "Others"] },
];

const SAMPLE_DRUGS = [
  "metformin hydrochloride 500 MG Oral Tablet",
  "lisinopril 10 MG Oral Tablet",
  "atorvastatin calcium 20 MG Oral Tablet",
  "amlodipine besylate 5 MG Oral Tablet",
  "omeprazole 20 MG Delayed Release Oral Capsule",
  "sertraline hydrochloride 50 MG Oral Tablet",
  "gabapentin 300 MG Oral Capsule",
];

// ── Risk band (thresholds unchanged: <20 Low, <40 Moderate, <65 Elevated, else High) ──
function riskBand(value) {
  const pct = Math.round(value * 100);
  if (pct < 20) return { pct, label: "Low", color: "var(--risk-low)" };
  if (pct < 40) return { pct, label: "Moderate", color: "var(--risk-moderate)" };
  if (pct < 65) return { pct, label: "Elevated", color: "var(--risk-elevated)" };
  return { pct, label: "High", color: "var(--risk-high)" };
}

const fmtSigned = (v, digits) => `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;

// ── Risk Gauge (270° arc) ──
function RiskGauge({ value, size = 140 }) {
  const { pct, label, color } = riskBand(value);
  const stroke = 8;
  const r = (size - stroke * 2) / 2;
  const circ = 2 * Math.PI * r;
  const dash = circ * 0.75;
  const offset = dash - dash * value;
  const h = size * 0.86;
  const arc = { strokeDasharray: `${dash} ${circ}` };

  return (
    <div className="gauge" style={{ width: size, height: h }} role="img" aria-label={`${pct}% risk, ${label}`}>
      <svg width={size} height={h} viewBox={`0 0 ${size} ${h}`}>
        <circle className="gauge-track" cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke}
          strokeLinecap="round" transform={`rotate(135 ${size / 2} ${size / 2})`} style={arc} />
        <circle className="gauge-fill" cx={size / 2} cy={size / 2} r={r} fill="none" strokeWidth={stroke}
          strokeLinecap="round" transform={`rotate(135 ${size / 2} ${size / 2})`}
          style={{ ...arc, stroke: color, strokeDashoffset: offset, "--dash": dash, animation: "gauge 1.1s var(--ease) both" }} />
      </svg>
      <div className="gauge-center">
        <div className="gauge-num" style={{ fontSize: size * 0.26 }}>{pct}<small>%</small></div>
        <div className="gauge-level" style={{ color }}>{label}</div>
      </div>
    </div>
  );
}

// ── Step progress ──
function Steps({ steps, current }) {
  return (
    <div className="steps-wrap">
      <ol className="steps" aria-label="Progress">
        {steps.map((s, i) => (
          <li key={s} className={`step ${i < current ? "done" : i === current ? "current" : ""}`}
            aria-current={i === current ? "step" : undefined}>
            <div className="step-line" />
            <div className="step-label"><span className="mono">{i + 1}</span>{s}</div>
          </li>
        ))}
      </ol>
      <div className="steps-compact">Step {current + 1} of {steps.length}: <strong>{steps[current]}</strong></div>
    </div>
  );
}

// ── Logo (clickable → home) ──
function Logo({ onClick }) {
  return (
    <button className="logo" onClick={onClick} aria-label="DualR home">
      <span className="logo-mark" aria-hidden="true">D</span>
      <span className="logo-word">Dual<span>R</span></span>
    </button>
  );
}

// ── Validated discrimination (real AUROC values from PHENOTYPES) ──
const AUC_MIN = 0.75;
const AUC_MAX = 0.9;
const aucPos = v => `${((v - AUC_MIN) / (AUC_MAX - AUC_MIN)) * 100}%`;

function ValidationChart() {
  const rows = Object.values(PHENOTYPES);
  return (
    <section className="panel evidence rise rise-2" aria-labelledby="evidence-title">
      <div className="evidence-head">
        <div>
          <h2 id="evidence-title" className="evidence-title">Validated discrimination</h2>
          <div className="evidence-sub">AUROC by condition, All of Us cohort</div>
        </div>
        <div className="legend" aria-hidden="true">
          <span><i className="base" />Baseline</span>
          <span><i className="dualr" />DualR</span>
        </div>
      </div>
      <div className="dumbbell">
        {rows.map(p => {
          const lo = Math.min(p.auc.base, p.auc.dualr);
          const hi = Math.max(p.auc.base, p.auc.dualr);
          return (
            <div className="db-row" key={p.id}>
              <div className="db-abbr" title={p.name}>{p.abbr}</div>
              <div className="db-track" role="img"
                aria-label={`${p.name}: baseline ${p.auc.base.toFixed(3)}, DualR ${p.auc.dualr.toFixed(3)}`}>
                <div className="db-bar" style={{ left: aucPos(lo), width: `calc(${aucPos(hi)} - ${aucPos(lo)})` }} />
                <div className="db-dot base" style={{ left: aucPos(p.auc.base) }} title={`Baseline ${p.auc.base.toFixed(3)}`} />
                <div className="db-dot dualr" style={{ left: aucPos(p.auc.dualr) }} title={`DualR ${p.auc.dualr.toFixed(3)}`} />
                <span className="db-val" style={{ left: aucPos(p.auc.base) }}>{p.auc.base.toFixed(3)}</span>
                <span className="db-val dualr" style={{ left: aucPos(p.auc.dualr) }}>{p.auc.dualr.toFixed(3)}</span>
              </div>
            </div>
          );
        })}
      </div>
      <div className="db-axis" aria-hidden="true">
        <div />
        <div className="db-ticks">
          {[0.75, 0.8, 0.85, 0.9].map(t => <span key={t} style={{ left: aucPos(t) }}>{t.toFixed(2)}</span>)}
        </div>
      </div>
    </section>
  );
}

// ── Computation progress (background job) ──
const fmtElapsed = (ms) => {
  const sec = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
};

function ComputeProgress({ progress, elapsedMs, nConditions, onCancel }) {
  const { total = 0, done = 0, stage = "queued", drugs = [] } = progress || {};
  const pct = total ? Math.round((done / total) * 100) : 0;
  const llmTotal = drugs.reduce((a, d) => a + d.total, 0);
  const llmDone = drugs.reduce((a, d) => a + d.done, 0);
  const at = ["queued", "estimating", "modeling", "done"].indexOf(stage);
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  const steps = [
    { key: "lookup", label: "Look up medications in the DualR table", state: at >= 1 ? "done" : "active" },
    {
      key: "llm",
      label: drugs.length
        ? `Estimate ${plural(drugs.length, "medication")} not in the table`
        : "No medications outside the table",
      detail: drugs.length ? `${llmDone} of ${llmTotal} estimates` : null,
      state: at > 1 ? "done" : at === 1 ? "active" : "pending",
    },
    { key: "model", label: `Run the risk model for ${plural(nConditions, "condition")}`, state: at >= 3 ? "done" : at === 2 ? "active" : "pending" },
  ];

  return (
    <section className="rise">
      <h1 className="step-title">Computing risk estimates</h1>
      <p className="step-desc">
        Medications in the DualR table are scored instantly. The others are estimated by a large
        language model, twice per condition, which can take a few minutes. Keep this tab open.
      </p>

      <div className="panel compute">
        <div className="compute-head">
          <span className="compute-pct">{pct}<small>%</small></span>
          <span className="compute-time">Elapsed <span className="mono">{fmtElapsed(elapsedMs)}</span></span>
        </div>
        <div className="compute-bar" role="progressbar" aria-label="Computation progress"
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}>
          <div className="compute-fill" style={{ width: `${Math.max(pct, 2)}%` }} />
        </div>

        <ol className="cstep-list" aria-live="polite">
          {steps.map(st => (
            <li key={st.key} className={`cstep ${st.state}`}>
              <span className="cstep-mark" aria-hidden="true">
                {st.state === "done" && <Check size={11} weight="bold" />}
              </span>
              <span className="cstep-label">{st.label}</span>
              {st.detail && <span className="cstep-detail mono">{st.detail}</span>}
            </li>
          ))}
        </ol>

        {drugs.length > 0 && (
          <ul className="cdrug-list" aria-label="Estimates per medication">
            {drugs.map(d => (
              <li key={d.name} className={d.done >= d.total ? "complete" : ""}>
                <span className="cdrug-name" title={d.name}>{d.name}</span>
                <span className="cdrug-bar"><span style={{ width: `${d.total ? (d.done / d.total) * 100 : 0}%` }} /></span>
                <span className="cdrug-count mono">{d.done}/{d.total}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="actions">
        <button className="btn btn-ghost" onClick={onCancel}>
          <X size={14} weight="bold" aria-hidden="true" />Cancel
        </button>
      </div>
    </section>
  );
}

// ═══════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════

export default function App() {
  const [view, setView] = useState("landing");
  const [step, setStep] = useState(0);
  const [selectedPhenos, setSelectedPhenos] = useState([]);
  const [demo, setDemo] = useState({});
  // Session comorbidity memory — persists across phenotype switches within session
  const [comoAnswers, setComoAnswers] = useState({});
  const [drugs, setDrugs] = useState([]);
  const [drugInput, setDrugInput] = useState("");
  const [results, setResults] = useState(null);
  const [comoStep, setComoStep] = useState(0);
  const [comoQueue, setComoQueue] = useState([]);
  const [chatMsgs, setChatMsgs] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Background prediction job: { id, progress, startedAt, showProgress }
  const [job, setJob] = useState(null);
  const [now, setNow] = useState(() => Date.now());
  const jobIdRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current) chatEndRef.current.scrollIntoView({ behavior: "smooth" });
  }, [chatMsgs]);

  const goHome = useCallback(() => {
    setView("landing");
    setStep(0);
    setSelectedPhenos([]);
    setDemo({});
    setComoAnswers({});
    setDrugs([]);
    setDrugInput("");
    setResults(null);
    setComoStep(0);
    setComoQueue([]);
    setChatMsgs([]);
    setLoading(false);
    setError(null);
    if (jobIdRef.current) {
      fetch(`/api/predict/jobs/${jobIdRef.current}`, { method: "DELETE" }).catch(() => {});
    }
    setJob(null);
  }, []);

  // Build comorbidity queue: union of required comos, minus already-answered ones
  function buildComoQueue(phenoIds, existingAnswers) {
    const seen = new Set();
    const queue = [];
    phenoIds.forEach(pid => {
      (DISEASE_COMO_MAP[pid] || []).forEach(c => {
        if (!seen.has(c.id) && !(c.id in existingAnswers)) {
          seen.add(c.id);
          // Tag which phenotype(s) need this
          const usedBy = phenoIds.filter(p => (DISEASE_COMO_MAP[p] || []).some(x => x.id === c.id)).map(p => PHENOTYPES[p].abbr);
          queue.push({ ...c, usedBy });
        }
      });
    });
    return queue;
  }

  function startComoPhase() {
    const queue = buildComoQueue(selectedPhenos, comoAnswers);
    if (queue.length === 0) {
      // All comos already answered — skip to drugs
      setChatMsgs([{ agent: true, text: "Your medical history from previous selections is still saved. Proceeding to medication entry." }]);
      setTimeout(() => setStep(3), 800);
      return;
    }
    setComoQueue(queue);
    setComoStep(0);
    setChatMsgs([
      { agent: true, text: `I'll ask ${queue.length} question${queue.length > 1 ? "s" : ""} about your medical history. Tap Yes or No for each.` },
      { agent: true, text: queue[0].q, tag: `Informs: ${queue[0].usedBy.join(", ")}`, isQ: true },
    ]);
  }

  function answerComo(answer) {
    const current = comoQueue[comoStep];
    const newAnswers = { ...comoAnswers, [current.id]: answer };
    setComoAnswers(newAnswers);
    const newMsgs = [...chatMsgs, { agent: false, text: answer ? "Yes" : "No" }];
    const nextIdx = comoStep + 1;
    if (nextIdx < comoQueue.length) {
      const next = comoQueue[nextIdx];
      newMsgs.push({ agent: true, text: next.q, tag: `Informs: ${next.usedBy.join(", ")}`, isQ: true });
      setChatMsgs(newMsgs);
      setComoStep(nextIdx);
    } else {
      newMsgs.push({ agent: true, text: "All set. Let's move to your medications." });
      setChatMsgs(newMsgs);
      setTimeout(() => setStep(3), 1000);
    }
  }

  function addDrug(d) {
    const trimmed = d.trim();
    if (trimmed && !drugs.includes(trimmed)) setDrugs(prev => [...prev, trimmed]);
    setDrugInput("");
  }

  function mapResults(data) {
    const mapped = {};
    for (const [pid, r] of Object.entries(data.results)) {
      mapped[pid] = {
        risk: r.risk,
        dualr_nocot: r.dualr_nocot,
        dualr_cot: r.dualr_cot,
        auc: PHENOTYPES[pid].auc,
        components: r.components || null,
        n_skipped_drugs: r.n_skipped_drugs || 0,
        topDrugs: r.top_drugs.map(d => ({
          label: d.name || d.short_name,
          contribution: d.contribution_combined,
          isSkipped: d.is_skipped,
        })),
      };
    }
    return mapped;
  }

  // Starts a background job; the polling effect below follows it to the result.
  async function fetchResults() {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch('/api/predict/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          diseases: selectedPhenos,
          demographics: {
            age: parseInt(demo.age, 10),
            gender: demo.gender,
            race: demo.race,
            ethnicity: demo.ethnicity,
          },
          comorbidities: Object.fromEntries(
            Object.entries(comoAnswers).map(([k, v]) => [k, v ? 1 : 0])
          ),
          drugs,
        }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error (HTTP ${resp.status})`);
      }
      const data = await resp.json();
      setJob({ id: data.job_id, progress: data.progress, startedAt: Date.now(), showProgress: false });
    } catch (e) {
      setError(e.message || "Prediction failed. Please try again.");
      setLoading(false);
    }
  }

  function cancelJob() {
    if (job) fetch(`/api/predict/jobs/${job.id}`, { method: "DELETE" }).catch(() => {});
    setJob(null);
    setLoading(false);
  }

  // Poll the running job. Table-only requests finish before the first poll, so the
  // progress view only appears when the job is still running at that point.
  const jobId = job?.id ?? null;
  useEffect(() => {
    jobIdRef.current = jobId;
    if (!jobId) return;
    let stopped = false;
    let failures = 0;
    let timer;
    const fail = (message) => {
      setError(message);
      setJob(null);
      setLoading(false);
    };
    const poll = async () => {
      try {
        const resp = await fetch(`/api/predict/jobs/${jobId}`, { cache: "no-store" });
        const data = await resp.json().catch(() => ({}));
        if (stopped) return;
        if (resp.status === 404) return fail(data.detail || "This computation is no longer available. Please run it again.");
        if (!resp.ok) throw new Error(data.detail || `Server error (HTTP ${resp.status})`);
        failures = 0;
        if (data.status === "done") {
          setResults(mapResults(data.result));
          setJob(null);
          setLoading(false);
          setView("results");
          return;
        }
        if (data.status === "error") return fail(data.error || "Prediction failed. Please try again.");
        setJob(j => (j && j.id === jobId ? { ...j, progress: data.progress, showProgress: true } : j));
      } catch (e) {
        if (stopped) return;
        failures += 1;
        if (failures >= 4) return fail("Lost contact with the server. Please try again.");
      }
      timer = setTimeout(poll, 1000);
    };
    timer = setTimeout(poll, 600);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [jobId]);

  // Elapsed-time clock for the progress view
  const showingProgress = !!job?.showProgress;
  useEffect(() => {
    if (!showingProgress) return;
    setNow(Date.now());
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [showingProgress]);

  // ── NAV ──
  function Nav({ landing }) {
    return (
      <nav className={`nav ${landing ? "" : "ruled"}`}>
        <Logo onClick={goHome} />
        {view !== "landing" && (
          <button className="btn btn-ghost btn-sm" onClick={goHome}>
            <ArrowLeft size={14} weight="bold" aria-hidden="true" />Home
          </button>
        )}
      </nav>
    );
  }

  // ── FOOTER ──
  function Footer() {
    return (
      <footer className="footer">
        <div>
          <div><strong>Su Lab</strong> · Biostatistics &amp; Health Data Science</div>
          <div>Indiana University School of Medicine</div>
          <div>Licensed under Apache 2.0</div>
        </div>
        <div className="footer-legal">
          This tool provides research-derived risk estimates and does not constitute clinical advice, diagnosis, or treatment recommendation. No personal data is collected, stored, or transmitted.
        </div>
      </footer>
    );
  }

  const nextIcon = <ArrowRight size={16} weight="bold" className="nudge" aria-hidden="true" />;
  const backBtn = (onClick) => (
    <button className="btn btn-ghost" onClick={onClick}>
      <ArrowLeft size={16} weight="bold" aria-hidden="true" />Back
    </button>
  );

  // ═══════════════════════════════════
  //  LANDING
  // ═══════════════════════════════════
  if (view === "landing") {
    return (
      <div className="app">
        <Nav landing />
        <main className="landing">
          <section className="hero">
            <div>
              <h1 className="rise">Phenotypic risk from <em>medication history</em></h1>
              <p className="rise rise-1">
                DualR turns medication records into disease risk estimates using knowledge from large language models, without sharing or storing patient data.
              </p>
              <button className="btn btn-primary btn-lg rise rise-2" onClick={() => { setView("flow"); setStep(0); }}>
                Start assessment{nextIcon}
              </button>
            </div>
            <ValidationChart />
          </section>

          <section className="facts rise rise-3" aria-label="About DualR">
            <div className="fact">
              <div className="fact-big">16,000+</div>
              <p>Drug associations pre-computed from large-scale cohorts: All of Us (N = 254K) and INPC (N = 1.13M).</p>
            </div>
            <div className="fact">
              <div className="fact-big">Three conditions</div>
              <p>Type 2 diabetes, hypertension and alcohol use disorder, assessed from a single medication list.</p>
            </div>
            <div className="fact">
              <div className="fact-big">Nothing stored</div>
              <p>No cookies, tracking, analytics or local storage. Inputs exist only in browser memory during this session.</p>
            </div>
          </section>
        </main>
        <Footer />
      </div>
    );
  }

  // ═══════════════════════════════════
  //  MAIN FLOW
  // ═══════════════════════════════════
  if (view === "flow") {
    const stepNames = ["Conditions", "Demographics", "Medical History", "Medications", "Review"];
    const ageVal = parseInt(demo.age, 10);
    const demoValid = !isNaN(ageVal) && ageVal >= 18 && ageVal <= 120 && demo.gender && demo.race && demo.ethnicity;
    const presentComos = Object.entries(comoAnswers).filter(([, v]) => v).map(([k]) => ALL_CHARLSON.find(c => c.id === k)?.label || k);

    return (
      <div className="app">
        <Nav />
        <main className="flow">
          <Steps steps={stepNames} current={step} />

          {/* ── STEP 0: Phenotype ── */}
          {step === 0 && (
            <section className="rise">
              <h1 className="step-title">Which conditions would you like to assess?</h1>
              <p className="step-desc">
                Select one or more. Comorbidity questions adapt automatically: circular inputs are excluded, and prior answers are remembered within this session.
              </p>
              <div className="options">
                {Object.values(PHENOTYPES).map(p => {
                  const sel = selectedPhenos.includes(p.id);
                  return (
                    <button key={p.id} className="option" aria-pressed={sel}
                      onClick={() => setSelectedPhenos(prev => prev.includes(p.id) ? prev.filter(x => x !== p.id) : [...prev, p.id])}>
                      <span className="tag">{p.abbr}</span>
                      <span>
                        <span className="option-name">{p.name}</span>
                        <span className="option-desc" style={{ display: "block" }}>{p.desc}</span>
                      </span>
                      <span className="option-meta mono">N = {p.n}<br />Prev. {p.prevalence}</span>
                      <span className="check" aria-hidden="true">{sel && <Check size={12} weight="bold" />}</span>
                    </button>
                  );
                })}
              </div>
              <div className="actions end">
                <button className="btn btn-primary" disabled={!selectedPhenos.length} onClick={() => setStep(1)}>
                  Continue{nextIcon}
                </button>
              </div>
            </section>
          )}

          {/* ── STEP 1: Demographics ── */}
          {step === 1 && (
            <section className="rise">
              <h1 className="step-title">Demographics</h1>
              <p className="step-desc">These form the baseline covariates in the prediction model.</p>

              <div className="fields">
                <div className="field">
                  <label htmlFor="age">Age (years)</label>
                  <input id="age" className="input input-num mono" type="number" min="18" max="120" inputMode="numeric"
                    value={demo.age || ""}
                    onChange={e => setDemo({ ...demo, age: e.target.value })}
                    placeholder="18-120" />
                </div>
                {DEMO_FIELDS.map(f => (
                  <div className="field" key={f.id}>
                    <label htmlFor={f.id}>{f.label}</label>
                    <select id={f.id} className={`select ${demo[f.id] ? "" : "empty"}`} value={demo[f.id] || ""}
                      onChange={e => setDemo({ ...demo, [f.id]: e.target.value })}>
                      <option value="">Select…</option>
                      {f.options.map(o => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </div>
                ))}
              </div>

              <div className="actions">
                {backBtn(() => setStep(0))}
                <button className="btn btn-primary" disabled={!demoValid} onClick={() => { setStep(2); startComoPhase(); }}>
                  Continue{nextIcon}
                </button>
              </div>
            </section>
          )}

          {/* ── STEP 2: Comorbidity Chat ── */}
          {step === 2 && (
            <section className="rise">
              <h1 className="step-title">Medical History</h1>
              <p className="step-desc">
                {comoQueue.length > 0
                  ? `${comoQueue.length} question${comoQueue.length > 1 ? "s" : ""} based on your selected conditions. Previously answered items are skipped.`
                  : "All comorbidity questions already answered from your previous selections."}
              </p>
              <div className="chat" aria-live="polite">
                {chatMsgs.map((m, i) => (
                  <div key={i} style={{ display: "grid", gap: 10 }}>
                    <div className={`msg ${m.agent ? "agent" : "user"}`}>
                      <div className="bubble">
                        {m.text}
                        {m.tag && <div className="bubble-tag">{m.tag}</div>}
                      </div>
                    </div>
                    {m.isQ && i === chatMsgs.length - 1 && (
                      <div className="answer">
                        <button className="btn btn-primary" onClick={() => answerComo(true)}>Yes</button>
                        <button className="btn btn-ghost" onClick={() => answerComo(false)}>No</button>
                      </div>
                    )}
                  </div>
                ))}
                <div ref={chatEndRef} />
              </div>
            </section>
          )}

          {/* ── STEP 3: Drug History ── */}
          {step === 3 && (
            <section className="rise">
              <h1 className="step-title">Medication History</h1>
              <p className="step-desc">
                Enter current and recent medications. Type drug names, paste a list, or upload a medication record. Names outside the DualR table are estimated by a language model and can take a few minutes.
              </p>

              <div className="dropzone">
                <span className="dropzone-icon"><FileArrowUp size={20} aria-hidden="true" /></span>
                <span>
                  <span className="dropzone-title" style={{ display: "block" }}>Drop medication record here</span>
                  <span className="dropzone-sub" style={{ display: "block" }}>PDF, PNG, JPG, or plain text</span>
                </span>
              </div>

              <div className="field">
                <label htmlFor="drug">Add a medication</label>
                <div className="add-row">
                  <input id="drug" className="input" value={drugInput} onChange={e => setDrugInput(e.target.value)}
                    onKeyDown={e => {
                      // Enter that confirms an IME composition (e.g. pinyin) must not add the
                      // entry; Safari reports it as keyCode 229 instead of isComposing.
                      if (e.key !== "Enter" || e.nativeEvent.isComposing || e.keyCode === 229) return;
                      addDrug(drugInput);
                    }}
                    placeholder="Type a medication name, press Enter" />
                  <button className="btn btn-ghost" onClick={() => addDrug(drugInput)}>
                    <Plus size={14} weight="bold" aria-hidden="true" />Add
                  </button>
                </div>
              </div>

              {SAMPLE_DRUGS.some(d => !drugs.includes(d)) && (
                <>
                  <div className="chips-label">Demo medications</div>
                  <div className="chips">
                    {SAMPLE_DRUGS.filter(d => !drugs.includes(d)).slice(0, 4).map(d => (
                      <button key={d} className="chip" title={d} onClick={() => addDrug(d)}>
                        <Plus size={11} weight="bold" aria-hidden="true" />{d.split(" ").slice(0, 2).join(" ")}
                      </button>
                    ))}
                  </div>
                </>
              )}

              {drugs.length > 0 && (
                <div className="med-list">
                  <div className="med-list-head">Added <span className="mono">({drugs.length})</span></div>
                  <ul>
                    {drugs.map((d, i) => (
                      <li key={d}>
                        <span>{d}</span>
                        <button className="icon-btn" aria-label={`Remove ${d}`} onClick={() => setDrugs(drugs.filter((_, j) => j !== i))}>
                          <X size={14} weight="bold" />
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {drugs.length === 0 && (
                <p className="field-hint" style={{ marginTop: 20 }}>
                  No medications? You can continue. The estimate then rests on demographics and medical history only.
                </p>
              )}

              <div className="actions">
                {backBtn(() => setStep(2))}
                <button className="btn btn-primary" onClick={() => { setError(null); setResults(null); setStep(4); }}>
                  Continue{nextIcon}
                </button>
              </div>
            </section>
          )}

          {/* ── STEP 4: Review ── */}
          {step === 4 && job?.showProgress && (
            <ComputeProgress
              progress={job.progress}
              elapsedMs={now - job.startedAt}
              nConditions={selectedPhenos.length}
              onCancel={cancelJob}
            />
          )}

          {step === 4 && !job?.showProgress && (
            <section className="rise">
              <h1 className="step-title">Review &amp; Compute</h1>
              <p className="step-desc">Verify your inputs before generating risk estimates.</p>

              <dl className="review">
                <div className="review-row">
                  <dt className="review-k">Conditions</dt>
                  <dd className="review-v">{selectedPhenos.map(pid => PHENOTYPES[pid].name).join(", ")}</dd>
                </div>
                <div className="review-row">
                  <dt className="review-k">Demographics</dt>
                  <dd className="review-v">
                    <ul>
                      <li>Age: {demo.age || "Not set"}</li>
                      {DEMO_FIELDS.map(f => <li key={f.id}>{f.label}: {demo[f.id] || "Not set"}</li>)}
                    </ul>
                  </dd>
                </div>
                <div className="review-row">
                  <dt className="review-k">Comorbidities <span className="mono">({presentComos.length} present)</span></dt>
                  <dd className="review-v">{presentComos.join(", ") || "None reported"}</dd>
                </div>
                <div className="review-row">
                  <dt className="review-k">Medications <span className="mono">({drugs.length})</span></dt>
                  <dd className={`review-v ${drugs.length ? "mono" : ""}`}>
                    {drugs.length ? <ul>{drugs.map(d => <li key={d}>{d}</li>)}</ul> : "None entered"}
                  </dd>
                </div>
              </dl>

              <div className="notice">
                <Warning size={16} weight="bold" aria-hidden="true" />
                <span>Risk estimates are derived from validated statistical models and do not constitute clinical advice, diagnosis, or treatment recommendation.</span>
              </div>

              {error && (
                <div className="error" role="alert">
                  <WarningCircle size={16} weight="bold" aria-hidden="true" />
                  <span>{error}</span>
                </div>
              )}

              <div className="actions">
                {backBtn(() => { setError(null); setStep(3); })}
                <button className="btn btn-primary btn-lg" onClick={fetchResults} disabled={loading} aria-busy={loading}>
                  {loading ? "Computing…" : <>Compute risk estimates{nextIcon}</>}
                </button>
              </div>
            </section>
          )}
        </main>
        <Footer />
      </div>
    );
  }

  // ═══════════════════════════════════
  //  RESULTS
  // ═══════════════════════════════════

  // ── Diverging bar chart for per-drug contributions ──
  function DrugWaterfall({ drugs: drugList }) {
    const scored = drugList.filter(d => !d.isSkipped);
    const skipped = drugList.filter(d => d.isSkipped);
    const maxAbs = scored.reduce((m, d) => Math.max(m, Math.abs(d.contribution)), 0.01);
    const half = (v) => (Math.abs(v) / maxAbs) * 50 * 0.7; // % of track width

    return (
      <div className="wf">
        {scored.map((d, i) => {
          const val = d.contribution;
          const w = half(val);
          const isPos = val >= 0;
          const side = isPos ? "left" : "right";
          return (
            <div className="wf-row" key={i}>
              <span className="wf-name" title={d.label}>{d.label}</span>
              <div className="wf-track" role="img" aria-label={`${d.label}: ${fmtSigned(val, 2)}`}>
                <div className="wf-bar" style={{
                  [side]: "50%", width: `max(${w}%, 2px)`,
                  background: isPos ? "var(--mark-up)" : "var(--mark-down)",
                  transformOrigin: `${side} center`,
                  animation: `grow 0.8s var(--ease) ${0.15 + i * 0.05}s both`,
                }} />
                <span className="wf-val" style={{ [side]: `calc(50% + ${w}% + 6px)` }}>{fmtSigned(val, 2)}</span>
              </div>
            </div>
          );
        })}
        {skipped.length > 0 && (
          <div className="wf-skipped">
            {skipped.map((d, i) => (
              <div className="wf-row skipped" key={i}>
                <span className="wf-name" title={d.label}>{d.label}</span>
                <div className="wf-track"><span className="wf-na">no estimate</span></div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  // ── Centered score bar ──
  function ScoreBar({ label, value, maxAbs }) {
    const pct = Math.min(Math.abs(value) / Math.max(maxAbs, 0.01), 1) * 50; // % of half
    const isPos = value >= 0;
    return (
      <div className="score">
        <span className="score-label">{label}</span>
        <div className="score-track">
          <div className="score-fill" style={{
            [isPos ? "left" : "right"]: "50%",
            width: `${pct}%`,
            background: isPos ? "var(--mark-up)" : "var(--mark-down)",
            transformOrigin: isPos ? "left center" : "right center",
            animation: "grow 0.9s var(--ease) 0.2s both",
          }} />
        </div>
        <span className="score-val">{fmtSigned(value, 3)}</span>
      </div>
    );
  }

  // ── Final risk probability bar ──
  function RiskBar({ value }) {
    const { pct, label, color } = riskBand(value);
    return (
      <div>
        <div className="riskbar-head">
          <span>Combined risk probability</span>
          <span><span className="mono">{pct}%</span> <span style={{ color, fontWeight: 600 }}>{label}</span></span>
        </div>
        <div className="riskbar" role="img" aria-label={`${pct}% combined risk, ${label}`}>
          <div className="riskbar-fill" style={{ width: `${pct}%`, background: color, animation: "grow 1.1s var(--ease) 0.2s both" }} />
        </div>
        <div className="riskbar-scale" aria-hidden="true"><span>0%</span><span>50%</span><span>100%</span></div>
      </div>
    );
  }

  if (view === "results" && results) {
    const nComos = Object.values(comoAnswers).filter(Boolean).length;

    return (
      <div className="app">
        <Nav />
        <main className="results">
          <div className="rise">
            <div className="status-line"><CheckCircle size={16} weight="fill" aria-hidden="true" />Analysis complete</div>
            <h1 className="results-title">Risk Assessment</h1>
            <p className="results-meta">
              {drugs.length} medication{drugs.length !== 1 ? "s" : ""}, {nComos} comorbidities, {selectedPhenos.map(pid => PHENOTYPES[pid].abbr).join(", ")}
            </p>
          </div>

          {/* One panel per phenotype */}
          {selectedPhenos.map((pid, phenoIdx) => {
            const p = PHENOTYPES[pid];
            const r = results[pid];
            const comp = r.components || {};
            const drugEffect = comp.drug_effect ?? null;
            const demoEffect = comp.demo_effect ?? null;
            const comoEffect = comp.como_effect ?? null;
            const maxDualR = Math.max(Math.abs(r.dualr_nocot), Math.abs(r.dualr_cot), 0.01);

            // Interpretation line
            const riskWord = riskBand(r.risk).label;
            const driverWord = drugEffect !== null
              ? (Math.abs(drugEffect) > Math.abs(demoEffect ?? 0) + Math.abs(comoEffect ?? 0)
                  ? "medication profile" : "clinical factors")
              : "medication profile";

            const adjustments = [
              demoEffect !== null && { name: "Demographic adjustment", ctx: `Age ${demo.age}, ${demo.gender}, ${demo.race}, ${demo.ethnicity}`, v: demoEffect },
              comoEffect !== null && { name: "Comorbidity adjustment", ctx: `${nComos} condition${nComos !== 1 ? "s" : ""} present`, v: comoEffect },
              drugEffect !== null && { name: "Drug signal adjustment", ctx: `${drugs.length - (r.n_skipped_drugs || 0)} drugs scored`, v: drugEffect },
            ].filter(Boolean);

            return (
              <article key={pid} className="panel result rise" style={{ animationDelay: `${0.08 + phenoIdx * 0.08}s` }}>
                <header className="result-head">
                  <span className="tag" style={{ background: "var(--accent)", color: "var(--on-accent)" }}>{p.abbr}</span>
                  <div>
                    <h2 className="result-name">{p.name}</h2>
                    <div className="result-sub">Population prevalence: <span className="mono">{p.prevalence}</span></div>
                  </div>
                  <div className="result-auc">
                    Validated AUC
                    <span className="mono">{r.auc.base.toFixed(3)} → <strong>{r.auc.dualr.toFixed(3)}</strong></span>
                  </div>
                </header>

                <div className="result-body">
                  {/* A. Summary */}
                  <div className="summary">
                    <RiskGauge value={r.risk} size={140} />
                    <div>
                      <div className="summary-lead">{riskWord} risk driven primarily by {driverWord}</div>
                      <p className="summary-text">
                        The model integrates drug associations, demographics, and comorbidities. The drug signal (DualR score) carries the most predictive weight.
                      </p>
                    </div>
                  </div>

                  {/* B. Drug signal */}
                  <section>
                    <div className="block-head">
                      <h3 className="block-title">Drug Signal</h3>
                      <span className="block-sub">DualR log₂ OR scores relative to prevalence</span>
                    </div>
                    <div className="subpanel tint">
                      <ScoreBar label="Fast reasoning (no CoT)" value={r.dualr_nocot} maxAbs={maxDualR} />
                      <ScoreBar label="Slow reasoning (CoT)" value={r.dualr_cot} maxAbs={maxDualR} />
                    </div>
                    <div className="subpanel">
                      <div className="block-head" style={{ marginBottom: 8 }}>
                        <h4 className="block-title" style={{ fontSize: 13.5 }}>Per-drug contributions</h4>
                      </div>
                      <div className="waterfall-legend">
                        <span><i style={{ background: "var(--mark-up)" }} />Risk-increasing (positive)</span>
                        <span><i style={{ background: "var(--mark-down)" }} />Protective (negative)</span>
                      </div>
                      {r.topDrugs.length > 0
                        ? <DrugWaterfall drugs={r.topDrugs} />
                        : <div className="muted" style={{ fontSize: 13, padding: "8px 0" }}>No drug contributions available.</div>
                      }
                    </div>
                  </section>

                  {/* C. Clinical adjustments */}
                  <section>
                    <div className="block-head">
                      <h3 className="block-title">Clinical Adjustments</h3>
                      <span className="block-sub">Change in predicted probability</span>
                    </div>
                    <div className="subpanel" style={{ paddingTop: 6, paddingBottom: 6 }}>
                      {adjustments.length > 0 ? (
                        <div className="adjust">
                          {adjustments.map(a => (
                            <div className="adjust-row" key={a.name}>
                              <span className="adjust-name">{a.name}<span className="adjust-ctx">{a.ctx}</span></span>
                              <span className="adjust-val">{fmtSigned(a.v, 4)}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="muted" style={{ fontSize: 13, padding: "8px 0" }}>Component breakdown not available.</div>
                      )}
                    </div>
                  </section>

                  {/* D. Final risk */}
                  <RiskBar value={r.risk} />
                </div>
              </article>
            );
          })}

          <div className="actions" style={{ justifyContent: "flex-start" }}>
            <button className="btn btn-primary" onClick={goHome}>New assessment</button>
            <button className="btn btn-ghost" onClick={() => { setStep(0); setView("flow"); }}>Modify inputs</button>
          </div>

          <p className="disclaimer">
            <strong>Important.</strong>{" "}
            Risk estimates are generated using the DualR method validated on All of Us (N=254,487) and Indiana Network for Patient Care (N=1.13M). Results reflect statistical associations and do not constitute clinical diagnosis or treatment recommendation. No information was stored during this session.
          </p>
        </main>
        <Footer />
      </div>
    );
  }

  return null;
}
