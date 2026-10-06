/*
 * Reading medication lists out of files in the browser.
 *
 * Text files are read directly, PDFs through pdf.js, and photos (or scanned PDFs
 * without a text layer) through Tesseract.js. Recognition runs on the user's
 * device: the file itself is never uploaded. Tesseract.js downloads its engine and
 * language data from a CDN on first use. Only the recognised text is later sent to
 * the backend, which asks the language model which lines name medications.
 */

const MAX_SIDE = 2000;          // downscale large photos before recognition
const MAX_PDF_PAGES = 10;       // text pages read from a PDF
const MAX_SCANNED_PAGES = 3;    // scanned PDF pages sent to recognition
const OCR_LANGS = ["eng", "chi_sim"];

const TEXT_TYPES = /^text\/|\/(csv|json)$/;
// Tesseract's chi_sim model separates every CJK character with a space ("二 甲 双 胍").
const CJK_GAP = /(?<=[\u3400-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef])\s+(?=[\u3400-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef])/g;
const TEXT_EXT = /\.(txt|csv|tsv|md)$/i;

export const ACCEPT = "image/*,application/pdf,.pdf,text/plain,.txt,.csv";

export function fileKind(file) {
  if (file.type === "application/pdf" || /\.pdf$/i.test(file.name)) return "pdf";
  if (file.type.startsWith("image/")) return "image";
  if (TEXT_TYPES.test(file.type) || TEXT_EXT.test(file.name)) return "text";
  return null;
}

async function imageToCanvas(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
    return canvas;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function recognize(images, onStatus) {
  const { createWorker } = await import("tesseract.js");
  let page = 0;
  onStatus({ stage: "loading", progress: 0 });
  const worker = await createWorker(OCR_LANGS, 1, {
    logger: (m) => {
      if (m.status === "recognizing text") {
        onStatus({ stage: "ocr", progress: (page + m.progress) / images.length });
      } else if (typeof m.progress === "number" && /load|initializ/i.test(m.status)) {
        onStatus({ stage: "loading", progress: m.progress });
      }
    },
  });
  try {
    const texts = [];
    for (page = 0; page < images.length; page++) {
      const { data } = await worker.recognize(images[page]);
      texts.push(data.text.replace(CJK_GAP, ""));
    }
    return texts.join("\n");
  } finally {
    await worker.terminate();
  }
}

async function readPdf(file, onStatus) {
  const pdfjs = await import("pdfjs-dist");
  const { default: workerUrl } = await import("pdfjs-dist/build/pdf.worker.min.mjs?url");
  pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;
  onStatus({ stage: "reading", progress: null });
  const doc = await pdfjs.getDocument({ data: await file.arrayBuffer() }).promise;

  let text = "";
  for (let i = 1; i <= Math.min(doc.numPages, MAX_PDF_PAGES); i++) {
    const content = await (await doc.getPage(i)).getTextContent();
    text += content.items.map((it) => it.str + (it.hasEOL ? "\n" : "")).join("") + "\n";
  }
  if (text.replace(/\s/g, "").length >= 20) return { text, scanned: false };

  // No usable text layer: render the first pages and recognise them like photos.
  const canvases = [];
  for (let i = 1; i <= Math.min(doc.numPages, MAX_SCANNED_PAGES); i++) {
    const page = await doc.getPage(i);
    const viewport = page.getViewport({ scale: 2 });
    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
    canvases.push(canvas);
  }
  return { text: await recognize(canvases, onStatus), scanned: true };
}

/** Read a file into plain text. onStatus receives {stage, progress} updates. */
export async function readFileText(file, onStatus = () => {}) {
  const kind = fileKind(file);
  if (kind === "text") {
    onStatus({ stage: "reading", progress: null });
    return { text: await file.text(), kind };
  }
  if (kind === "pdf") {
    const { text, scanned } = await readPdf(file, onStatus);
    return { text, kind: scanned ? "scan" : "pdf" };
  }
  if (kind === "image") {
    const canvas = await imageToCanvas(file);
    return { text: await recognize([canvas], onStatus), kind };
  }
  throw new Error("This file type is not supported. Use a photo, a PDF or a text file.");
}

const LIST_MARKER = /^\s*(?:[-*•●▪]|\(?\d{1,2}[.)])\s*/;

function cleanLine(line) {
  return line.replace(LIST_MARKER, "").replace(/\s+/g, " ").trim();
}

/** Non-trivial lines of recognised text, for the user to tick what was missed. */
export function candidateLines(text) {
  const seen = new Set();
  const out = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = cleanLine(raw);
    if (line.length < 2 || line.length > 100 || !/\p{L}/u.test(line)) continue;
    const key = line.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(line);
  }
  return out;
}

/** Split pasted text into entries when it holds more than one (lines or semicolons). */
export function splitPastedList(text) {
  return candidateLines(text.replace(/;/g, "\n"));
}

/** True when a raw line is already represented by one of the detected names. */
export function coveredBy(line, names) {
  const l = line.toLowerCase();
  return names.some((n) => {
    const m = n.toLowerCase();
    if (l.includes(m) || m.includes(l)) return true;
    const word = m.split(/[\s,]+/)[0];
    return word.length >= 4 && l.includes(word);
  });
}
