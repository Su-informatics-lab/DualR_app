import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { FileArrowUp, Plus, Warning, X } from "@phosphor-icons/react";
import { ACCEPT, candidateLines, coveredBy, fileKind, readFileText } from "./extract.js";

/*
 * Import medications from a photo, PDF, text file or a pasted list.
 *
 * Text is read in the browser (src/extract.js), the backend suggests which lines are
 * medications, and the user confirms every entry before it is added: nothing found in
 * a file reaches the medication list without being ticked here.
 */

const fmtElapsed = (ms) => {
  const sec = Math.max(0, Math.floor(ms / 1000));
  return `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, "0")}`;
};

const STAGE_TEXT = {
  reading: "Reading the file",
  loading: "Loading text recognition (the first time downloads about 20 MB)",
  ocr: "Recognising text",
  detecting: "Finding medication names",
};

let nextId = 0;
const row = (text, checked, origin) => ({ id: nextId++, text, checked, origin });

const MedicationImport = forwardRef(function MedicationImport({ existing, onAdd }, ref) {
  const [phase, setPhase] = useState("idle"); // idle | reading | review
  const [status, setStatus] = useState({ stage: "reading", progress: null });
  const [started, setStarted] = useState(0);
  const [now, setNow] = useState(0);
  const [source, setSource] = useState(null); // { name, kind, preview, text }
  const [rows, setRows] = useState([]);
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const runRef = useRef(0);
  const inputRef = useRef(null);
  const previewRef = useRef(null); // object URL of the photo shown in the review

  function setPreview(url) {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = url;
  }

  const existingSet = new Set(existing.map((d) => d.toLowerCase()));

  useEffect(() => {
    if (phase !== "reading") return;
    setNow(Date.now());
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [phase]);

  useEffect(() => () => setPreview(null), []);

  function reset() {
    runRef.current += 1; // results of a run still in flight are ignored
    setPhase("idle");
    setRows([]);
    setNote(null);
    setSource(null);
    setPreview(null);
  }

  function openReview(found, lines, src, message) {
    const detected = found.map((t) => row(t, !existingSet.has(t.toLowerCase()), "detected"));
    const others = lines
      .filter((l) => !coveredBy(l, found))
      .map((t) => row(t, false, "line"));
    setRows([...detected, ...others]);
    setSource(src);
    setNote(message);
    setPhase("review");
  }

  // A pasted multi-line list skips recognition and detection: each line is an entry.
  useImperativeHandle(ref, () => ({
    reviewPasted(items) {
      setError(null);
      openReview(items, [], { name: "Pasted list", kind: "pasted" }, null);
    },
  }));

  async function handleFile(file) {
    setError(null);
    if (!file) return;
    if (!fileKind(file)) {
      setError("This file type is not supported. Use a photo, a PDF or a text file.");
      return;
    }
    const run = ++runRef.current;
    const kind = fileKind(file);
    const preview = kind === "image" ? URL.createObjectURL(file) : null;
    setPreview(preview);
    setSource({ name: file.name, kind, preview });
    setStatus({ stage: "reading", progress: null });
    setStarted(Date.now());
    setPhase("reading");

    try {
      const { text, kind: readKind } = await readFileText(file, (s) => run === runRef.current && setStatus(s));
      if (run !== runRef.current) return;
      const src = { name: file.name, kind: readKind, preview, text };
      const lines = candidateLines(text);
      if (!lines.length) {
        openReview([], [], src, "No text could be read from this file. Try a sharper, well-lit photo, or type the medications in.");
        return;
      }

      setStatus({ stage: "detecting", progress: null });
      let found = null;
      try {
        const resp = await fetch("/api/extract-medications", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        if (resp.ok) found = (await resp.json()).medications;
      } catch {
        found = null;
      }
      if (run !== runRef.current) return;

      if (found === null) {
        openReview([], lines, src, "Automatic name detection was unavailable. Tick the lines that are medications.");
      } else if (!found.length) {
        openReview([], lines, src, "No medication names were detected. Tick any lines that are medications, or type them in.");
      } else {
        openReview(found, lines, src, null);
      }
    } catch (e) {
      if (run !== runRef.current) return;
      setPhase("idle");
      setSource(null);
      setPreview(null);
      setError(e?.message || "This file could not be read.");
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files?.[0]);
  }

  const update = (id, patch) => setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  const chosen = rows.filter((r) => r.checked && r.text.trim() && !existingSet.has(r.text.trim().toLowerCase()));
  const detected = rows.filter((r) => r.origin !== "line");
  const others = rows.filter((r) => r.origin === "line");

  function confirm() {
    onAdd(chosen.map((r) => r.text.trim()));
    reset();
  }

  const renderRow = (r) => {
    const already = existingSet.has(r.text.trim().toLowerCase());
    return (
      <li key={r.id} className={`import-row ${r.checked ? "on" : ""}`}>
        <input type="checkbox" checked={r.checked && !already} disabled={already}
          aria-label={`Include ${r.text}`} onChange={(e) => update(r.id, { checked: e.target.checked })} />
        <input className="import-text" value={r.text} aria-label="Medication name"
          onChange={(e) => update(r.id, { text: e.target.value, checked: true })} />
        {already && <span className="import-tag">Already added</span>}
      </li>
    );
  };

  if (phase === "reading") {
    const pct = status.progress == null ? null : Math.round(status.progress * 100);
    return (
      <div className="panel import" aria-live="polite">
        <div className="import-head">
          <div>
            <div className="import-title">Reading {source?.name}</div>
            <div className="import-sub">
              {STAGE_TEXT[status.stage]}{pct != null && status.stage !== "detecting" ? ` · ${pct}%` : ""}
            </div>
          </div>
          <span className="compute-time">Elapsed <span className="mono">{fmtElapsed(now - started)}</span></span>
        </div>
        <div className={`compute-bar ${pct == null ? "indeterminate" : ""}`} role="progressbar"
          aria-label="File reading progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct ?? undefined}>
          <div className="compute-fill" style={{ width: pct == null ? "35%" : `${Math.max(pct, 2)}%` }} />
        </div>
        <div className="import-actions">
          <button className="btn btn-ghost btn-sm" onClick={reset}>
            <X size={13} weight="bold" aria-hidden="true" />Cancel
          </button>
        </div>
      </div>
    );
  }

  if (phase === "review") {
    const what = source?.kind === "image" || source?.kind === "scan" ? "photo" : "document";
    return (
      <div className="panel import">
        <div className="import-head">
          <div>
            <div className="import-title">Check what was found</div>
            <div className="import-sub">
              {source?.kind === "pasted"
                ? `${detected.length} entries from the pasted list. Edit or untick any before adding.`
                : `${source?.name}: ${detected.length} medication name${detected.length === 1 ? "" : "s"} detected. Text recognition can miss or misread names, so compare the list with the ${what} before adding.`}
            </div>
          </div>
        </div>

        {note && (
          <div className="notice" style={{ marginTop: 16 }}>
            <Warning size={16} weight="bold" aria-hidden="true" />
            <span>{note}</span>
          </div>
        )}

        <div className={`import-body ${source?.preview || source?.text ? "with-source" : ""}`}>
          {(source?.preview || source?.text) && (
            <div className="import-source">
              {source.preview
                ? <a href={source.preview} target="_blank" rel="noreferrer" title="Open full size">
                    <img src={source.preview} alt={`Uploaded ${what}`} />
                  </a>
                : <pre>{source.text.trim()}</pre>}
            </div>
          )}

          <div className="import-lists">
            {detected.length > 0 && (
              <ul className="import-list" aria-label="Detected medications">{detected.map(renderRow)}</ul>
            )}
            {others.length > 0 && (
              <details className="import-others" open={detected.length === 0}>
                <summary>Other lines in the {what} <span className="mono">({others.length})</span></summary>
                <ul className="import-list">{others.map(renderRow)}</ul>
              </details>
            )}
          </div>
        </div>

        <div className="import-actions">
          <button className="btn btn-ghost" onClick={reset}>Cancel</button>
          <button className="btn btn-primary" disabled={!chosen.length} onClick={confirm}>
            <Plus size={14} weight="bold" aria-hidden="true" />
            Add {chosen.length} medication{chosen.length === 1 ? "" : "s"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <>
      <label
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input ref={inputRef} type="file" accept={ACCEPT} className="visually-hidden"
          onChange={(e) => { handleFile(e.target.files?.[0]); e.target.value = ""; }} />
        <span className="dropzone-icon"><FileArrowUp size={20} aria-hidden="true" /></span>
        <span>
          <span className="dropzone-title" style={{ display: "block" }}>Add from a photo or document</span>
          <span className="dropzone-sub" style={{ display: "block" }}>
            Drop a file here or choose one: photo, PDF or text. The file stays on this device; the text read from it is used to find medication names, and you confirm each one.
          </span>
        </span>
      </label>
      {error && (
        <div className="error" role="alert" style={{ marginTop: -12, marginBottom: 24 }}>
          <Warning size={16} weight="bold" aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}
    </>
  );
});

export default MedicationImport;
