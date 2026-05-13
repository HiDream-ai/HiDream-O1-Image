"""Minimal, elegant Flask app for HiDream-O1-Image.

Features
--------
- Text-to-image, image editing, and multi-reference subject-driven generation
- Standalone prompt refinement (local Gemma backend or OpenAI-compatible API)
- Server-Sent Events for real-time per-step progress
- Single-file: HTML / CSS / JS all embedded below

Run
---
    python app.py --model_path /path/to/HiDream-O1-Image --model_type full
"""

import argparse
import base64
import io
import json
import os
import queue
import tempfile
import threading
import uuid

import torch
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request
from PIL import Image
from transformers import AutoProcessor

load_dotenv()

from models.pipeline import DEFAULT_TIMESTEPS, generate_image
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from prompt_agent_v2 import DEFAULT_BASE_URL, DEFAULT_MODEL_ID, refine_prompt


# ── Globals ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
_GEN_LOCK = threading.Lock()
_STATE = {
    "model": None,
    "processor": None,
    "model_type": "full",
}
_JOBS = {}


def _add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def _get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def load_image_model(model_path):
    print(f"[app] Loading checkpoint from {model_path} ...")
    processor = AutoProcessor.from_pretrained(model_path)
    # NOTE: torch_dtype = torch.float32 will generate more detailed images but with more memory usage
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    _add_special_tokens(_get_tokenizer(processor))
    return processor, model


# ── HTML ─────────────────────────────────────────────────────────────────────

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>HiDream-O1-Image{% if model_type == 'dev' %}-Dev{% endif %}</title>
<style>
  :root {
    --bg: #fbfbfd;
    --panel: #ffffff;
    --panel-2: #f5f5f7;
    --border: rgba(0, 0, 0, 0.08);
    --border-strong: rgba(0, 0, 0, 0.14);
    --text: #1d1d1f;
    --muted: #86868b;
    --accent: #1d1d1f;
    --accent-soft: #f0f0f2;
    --blue: #0071e3;
    --blue-hover: #0077ed;
    --purple: #8a5cf6;
    --pink: #ff6ea1;
    --mint: #22c9a4;
    --gradient: linear-gradient(135deg, #8a5cf6 0%, #0071e3 45%, #22c9a4 100%);
    --gradient-warm: linear-gradient(135deg, #ff6ea1 0%, #8a5cf6 100%);
    --danger: #d70015;
    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.04);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.06);
    --radius: 12px;
    --radius-sm: 8px;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
      "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 32px;
    background: rgba(251, 251, 253, 0.85);
    backdrop-filter: saturate(180%) blur(20px);
    -webkit-backdrop-filter: saturate(180%) blur(20px);
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10;
  }
  .brand {
    font-weight: 600; font-size: 17px; letter-spacing: -0.01em;
    background: var(--gradient);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
  }
  .topbar-meta { color: var(--muted); font-size: 12px; }
  .layout {
    display: grid;
    grid-template-columns: 400px 1fr;
    gap: 24px;
    padding: 24px 32px 40px;
    max-width: 1600px; margin: 0 auto;
  }
  .sidebar, .canvas {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow-sm);
  }
  .sidebar {
    padding: 24px; max-height: calc(100vh - 130px);
    overflow-y: auto; align-self: start;
  }
  .canvas {
    min-height: calc(100vh - 130px);
    padding: 28px; display: flex; flex-direction: column;
  }
  h2 {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); margin: 0 0 12px;
  }
  .group { margin-bottom: 22px; }
  .group:last-of-type { margin-bottom: 0; }
  label {
    display: block; font-size: 12px; font-weight: 500;
    color: var(--text); margin-bottom: 8px;
  }
  input[type=text], input[type=number], input[type=password],
  textarea, select {
    width: 100%;
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
    font-size: 13px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s, box-shadow 0.15s;
    -webkit-appearance: none;
  }
  input:focus, textarea:focus, select:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.15);
  }
  textarea { resize: vertical; min-height: 96px; line-height: 1.5; }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .tabs {
    display: flex; gap: 0; margin-bottom: 22px;
    background: var(--panel-2);
    border-radius: 10px; padding: 3px;
  }
  .tab {
    flex: 1; text-align: center; padding: 8px 10px;
    font-size: 12.5px; font-weight: 500;
    color: var(--muted); border-radius: 7px; cursor: pointer;
    user-select: none; transition: all 0.2s;
  }
  .tab.active {
    background: var(--panel); color: var(--text);
    box-shadow: var(--shadow-sm);
  }
  .tab:hover:not(.active) { color: var(--text); }
  details {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 14px; margin-bottom: 16px;
    background: var(--panel-2);
    transition: background 0.15s;
  }
  details summary {
    cursor: pointer; color: var(--text);
    font-size: 12px; font-weight: 500;
    outline: none; list-style: none;
    display: flex; align-items: center; justify-content: space-between;
  }
  details summary::after {
    content: "⌄"; color: var(--muted); font-size: 14px;
    transition: transform 0.2s;
  }
  details[open] summary { margin-bottom: 14px; }
  details[open] summary::after { transform: rotate(180deg); }
  details summary::-webkit-details-marker { display: none; }
  .file-input {
    border: 1.5px dashed var(--border-strong);
    border-radius: var(--radius-sm);
    padding: 16px; text-align: center; color: var(--muted);
    cursor: pointer; transition: all 0.15s;
    font-size: 12.5px;
  }
  .file-input:hover {
    border-color: var(--blue); color: var(--blue);
    background: rgba(0, 113, 227, 0.03);
  }
  .file-input input { display: none; }
  .thumbs { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
  .thumb {
    width: 64px; height: 64px; border-radius: var(--radius-sm);
    overflow: hidden; border: 1px solid var(--border);
    position: relative; box-shadow: var(--shadow-sm);
  }
  .thumb img { width: 100%; height: 100%; object-fit: cover; }
  .thumb .x {
    position: absolute; top: 4px; right: 4px;
    width: 18px; height: 18px; border-radius: 50%;
    background: rgba(0,0,0,0.65); color: #fff;
    font-size: 11px; line-height: 18px; text-align: center;
    cursor: pointer; backdrop-filter: blur(4px);
  }
  button {
    font-family: inherit; cursor: pointer;
    transition: all 0.15s; -webkit-appearance: none;
  }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .btn-primary {
    width: 100%; background: var(--gradient); color: #fff;
    border: none; border-radius: 980px; padding: 12px 20px;
    font-size: 14px; font-weight: 600;
    letter-spacing: -0.01em;
    box-shadow: 0 4px 14px rgba(138, 92, 246, 0.25);
  }
  .btn-primary:hover:not(:disabled) {
    box-shadow: 0 6px 20px rgba(138, 92, 246, 0.35);
    transform: translateY(-1px);
  }
  .btn-secondary {
    width: 100%; background: var(--panel); color: var(--text);
    border: 1px solid var(--border-strong); border-radius: 980px;
    padding: 11px 20px; font-size: 13px; font-weight: 500;
    margin-bottom: 10px;
  }
  .btn-secondary:hover:not(:disabled) {
    background: var(--panel-2); border-color: var(--text);
  }
  .btn-link {
    background: none; border: none; color: var(--blue);
    padding: 4px 0; font-size: 12px; font-weight: 500;
  }
  .btn-link:hover:not(:disabled) { text-decoration: underline; }
  .canvas-empty {
    flex: 1; display: flex; align-items: center; justify-content: center;
    color: var(--muted); font-size: 14px;
  }
  .canvas-image {
    flex: 1; display: flex; align-items: center; justify-content: center;
    background: var(--panel-2);
    border-radius: var(--radius);
    overflow: hidden; padding: 16px; min-height: 400px;
  }
  .canvas-image img {
    max-width: 100%; max-height: calc(100vh - 220px);
    border-radius: var(--radius-sm);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
  }
  .meta {
    margin-top: 18px; padding: 16px 18px;
    background: var(--panel-2);
    border-radius: var(--radius-sm);
    font-size: 13px; color: var(--text);
  }
  .meta .label {
    color: var(--muted); font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-size: 10.5px; margin-bottom: 6px;
  }
  .meta pre {
    margin: 0 0 14px; white-space: pre-wrap; word-break: break-word;
    color: var(--text); font-family: inherit; line-height: 1.55;
  }
  .meta pre:last-child { margin-bottom: 0; }
  .progress {
    flex: 1; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    color: var(--muted); padding: 24px;
    position: relative;
  }
  .progress-preview {
    width: 100%; max-width: 520px; aspect-ratio: 1;
    border-radius: var(--radius);
    background: var(--panel-2);
    overflow: hidden; position: relative;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.08),
                0 0 0 1px var(--border);
  }
  .progress-preview::before {
    content: ""; position: absolute; inset: 0;
    background: var(--gradient);
    opacity: 0.08; pointer-events: none;
    mix-blend-mode: screen;
  }
  .progress-preview img {
    width: 100%; height: 100%; object-fit: cover;
    display: block; transition: filter 0.4s ease;
  }
  .progress-preview.empty { display: flex;
    align-items: center; justify-content: center; }
  .progress-preview.empty::after {
    content: ""; width: 80px; height: 80px;
    border-radius: 50%; background: var(--gradient);
    filter: blur(28px); opacity: 0.85;
    animation: pulse 1.4s ease-in-out infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  @keyframes pulse {
    0%, 100% { transform: scale(0.85); opacity: 0.65; }
    50% { transform: scale(1.2); opacity: 1; }
  }
  .progress-meta {
    display: flex; align-items: center; justify-content: space-between;
    width: 100%; max-width: 520px; margin-top: 20px;
    font-variant-numeric: tabular-nums;
  }
  .progress-label {
    font-size: 13px; font-weight: 600; color: var(--text);
    letter-spacing: -0.01em;
  }
  .progress-step {
    font-size: 12px; color: var(--muted);
    font-feature-settings: "tnum";
  }
  .progress-bar {
    width: 100%; max-width: 520px; height: 3px; border-radius: 999px;
    background: var(--panel-2); overflow: hidden; margin-top: 10px;
  }
  .progress-bar-fill {
    height: 100%; background: var(--gradient);
    border-radius: 999px; width: 0%;
    transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .refine-preview {
    margin-top: 12px; padding: 12px 14px;
    background: rgba(0, 113, 227, 0.06);
    border: 1px solid rgba(0, 113, 227, 0.2);
    border-radius: var(--radius-sm);
    font-size: 12.5px;
  }
  .refine-preview .label {
    color: var(--blue); font-weight: 600;
    font-size: 10.5px; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 6px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .refine-preview pre {
    margin: 0; white-space: pre-wrap; word-break: break-word;
    font-family: inherit; line-height: 1.5;
  }
  .refine-actions { display: flex; gap: 8px; margin-top: 10px; }
  .refine-actions button { flex: 1; padding: 7px 10px; font-size: 12px; }
  .err {
    color: var(--danger); margin-top: 12px; font-size: 12.5px;
    padding: 8px 12px; background: rgba(215, 0, 21, 0.06);
    border-radius: var(--radius-sm);
  }
  .toggle-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 0; margin-bottom: 12px;
  }
  .toggle-row label { margin: 0; font-weight: 500; }
  .toggle-row .hint { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .divider {
    height: 1px; background: var(--border);
    margin: 22px 0;
  }
  .spinner-inline {
    width: 14px; height: 14px; display: inline-block;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff; border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: -2px; margin-right: 8px;
  }
  .spinner-blue {
    border: 2px solid rgba(0, 113, 227, 0.2);
    border-top-color: var(--blue);
  }
  @media (max-width: 900px) {
    .layout { grid-template-columns: 1fr; }
    .sidebar { max-height: none; }
  }
</style>
</head>
<body>
  <div class="topbar">
    <div>
      <span class="brand">HiDream-O1-Image{% if model_type == 'dev' %}-Dev{% endif %}</span>
    </div>
    <span class="topbar-meta"></span>
  </div>

  <div class="layout">
    <aside class="sidebar">
      <div class="tabs" id="tabs">
        <div class="tab active" data-mode="t2i">Text → Image</div>
        <div class="tab" data-mode="edit">Edit</div>
        <div class="tab" data-mode="subject">Subject</div>
      </div>

      <div class="group">
        <label>Prompt</label>
        <textarea id="prompt" placeholder="Describe the image you want to create..."></textarea>
      </div>

      <div class="group" id="refs-group" style="display:none">
        <label id="refs-label">Reference image</label>
        <label class="file-input">
          <input id="refs" type="file" accept="image/*" multiple />
          <span>Click to upload images</span>
        </label>
        <div class="thumbs" id="thumbs"></div>
        <label id="keep-aspect-row" style="display:none; margin-top:10px; font-weight:400; text-transform:none; letter-spacing:0; cursor:pointer;">
          <input id="keep-aspect" type="checkbox" style="vertical-align:middle; margin-right:6px;" />
          Keep reference aspect (resize to 2048)
        </label>
        <div id="edit-scheduler-row" style="display:none; margin-top:12px;">
          <label>Scheduler</label>
          <select id="edit-scheduler">
            <option value="flow_match" selected>flow_match (default)</option>
            <option value="flash">flash</option>
          </select>
        </div>
      </div>

      <details id="refine-section">
        <summary>Prompt Refiner</summary>
        <div style="font-size:11.5px; color:var(--muted); margin-bottom:10px">
          Calls an OpenAI-compatible endpoint serving <code>HiDream-ai/gemma-4-31B-it-Prompt-Refine</code>.
          Start the server first with <code>bash start_vllm_server.sh</code>.
        </div>
        <label>Base URL</label>
        <input id="api-base" type="text" autocomplete="off" name="hd-base-url"
               placeholder="http://localhost:8000/v1" value="{{ env_base_url }}" />
        <label style="margin-top:10px">Model</label>
        <input id="api-model" type="text" autocomplete="off" name="hd-model"
               placeholder="HiDream-ai/gemma-4-31B-it-Prompt-Refine" value="{{ env_model }}" />
        <button class="btn-secondary" id="refine-btn" style="margin-top: 14px; margin-bottom: 0">
          Refine Prompt
        </button>
        <div id="refine-preview" class="refine-preview" style="display:none"></div>
      </details>

      <details>
        <summary>Generation Settings</summary>
        <div class="row" style="margin-bottom: 12px">
          <div>
            <label>Width</label>
            <input id="width" type="number" value="2048" step="64" min="512" />
          </div>
          <div>
            <label>Height</label>
            <input id="height" type="number" value="2048" step="64" min="512" />
          </div>
        </div>
        <label>Seed</label>
        <input id="seed" type="number" value="32" />
      </details>

      <button class="btn-primary" id="go">Generate</button>
      <div class="err" id="err" style="display:none"></div>
    </aside>

    <main class="canvas">
      <div class="canvas-empty" id="empty">
        <div style="text-align: center">
          <div style="font-size: 32px; opacity: 0.3; margin-bottom: 8px">◍</div>
          <div>Your generated image will appear here</div>
        </div>
      </div>
      <div class="progress" id="progress" style="display:none">
        <div class="progress-preview empty" id="progress-preview">
          <img id="progress-img" style="display:none" />
        </div>
        <div class="progress-meta">
          <span class="progress-label" id="progress-text">Preparing</span>
          <span class="progress-step" id="progress-sub">—</span>
        </div>
        <div class="progress-bar"><div class="progress-bar-fill" id="progress-fill"></div></div>
      </div>
      <div class="canvas-image" id="out" style="display:none">
        <img id="img" />
      </div>
      <div class="meta" id="meta" style="display:none"></div>
    </main>
  </div>

<script>
const $ = (id) => document.getElementById(id);
const MODEL_TYPE = "{{ model_type }}";
let mode = "t2i";
let refFiles = [];
let lastRefined = null;
let originalPrompt = null;


document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    mode = t.dataset.mode;
    const refsGroup = $("refs-group");
    const refineSection = $("refine-section");
    const keepAspectRow = $("keep-aspect-row");
    const editSchedRow = $("edit-scheduler-row");
    if (mode === "t2i") {
      refsGroup.style.display = "none";
      refFiles = []; renderThumbs();
      refineSection.style.display = "";
      keepAspectRow.style.display = "none";
      $("keep-aspect").checked = false;
      editSchedRow.style.display = "none";
    } else {
      refsGroup.style.display = "";
      $("refs-label").textContent = mode === "edit"
        ? "Source image (1)"
        : "Reference images (2+)";
      $("refs").multiple = mode !== "edit";
      // Refine only available for T2I per design spec.
      refineSection.style.display = "none";
      refineSection.removeAttribute("open");
      clearRefinePreview();
      // `keep_original_aspect` only applies when there is exactly one ref image.
      keepAspectRow.style.display = mode === "edit" ? "" : "none";
      if (mode !== "edit") $("keep-aspect").checked = false;
      // Editing scheduler selector only applies to the Dev model + Edit tab.
      editSchedRow.style.display = (mode === "edit" && MODEL_TYPE === "dev") ? "" : "none";
    }
  };
});

$("refs").onchange = (e) => {
  const files = Array.from(e.target.files);
  refFiles = mode === "edit" ? files.slice(0, 1) : refFiles.concat(files);
  renderThumbs();
  e.target.value = "";
};

function renderThumbs() {
  const c = $("thumbs"); c.innerHTML = "";
  refFiles.forEach((f, i) => {
    const url = URL.createObjectURL(f);
    const el = document.createElement("div");
    el.className = "thumb";
    el.innerHTML = `<img src="${url}" /><div class="x" data-i="${i}">×</div>`;
    el.querySelector(".x").onclick = () => {
      refFiles.splice(i, 1); renderThumbs();
    };
    c.appendChild(el);
  });
}

function fileToB64(f) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(",")[1]);
    r.onerror = rej;
    r.readAsDataURL(f);
  });
}

function showErr(msg) {
  const e = $("err"); e.textContent = msg; e.style.display = msg ? "" : "none";
}

function clearRefinePreview() {
  lastRefined = null; originalPrompt = null;
  $("refine-preview").style.display = "none";
  $("refine-preview").innerHTML = "";
}

function renderRefinePreview(refined) {
  lastRefined = refined;
  const html = `
    <div class="label">
      <span>Refined Prompt</span>
      <span style="font-weight: 400; color: var(--muted); text-transform: none; letter-spacing: 0">Preview</span>
    </div>
    <pre>${escapeHtml(refined.prompt)}</pre>
    <div class="refine-actions">
      <button class="btn-secondary" id="refine-apply" style="margin: 0">Use This Prompt</button>
      <button class="btn-secondary" id="refine-discard" style="margin: 0">Discard</button>
    </div>
  `;
  const box = $("refine-preview");
  box.innerHTML = html; box.style.display = "";
  $("refine-apply").onclick = () => {
    $("prompt").value = refined.prompt;
    clearRefinePreview();
  };
  $("refine-discard").onclick = () => clearRefinePreview();
}

$("refine-btn").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) { showErr("Please enter a prompt to refine."); return; }
  showErr("");
  const btn = $("refine-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner-inline spinner-blue"></span>Refining…';
  try {
    const body = {
      prompt,
      base_url: $("api-base").value.trim(),
      model: $("api-model").value.trim(),
    };
    const r = await fetch("/api/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Refine failed");
    renderRefinePreview(data);
  } catch (e) {
    showErr(e.message);
  } finally {
    btn.disabled = false; btn.textContent = "Refine Prompt";
  }
};

$("go").onclick = async () => {
  const prompt = $("prompt").value.trim();
  if (!prompt) { showErr("Please enter a prompt."); return; }
  if (mode === "edit" && refFiles.length !== 1) {
    showErr("Edit mode requires exactly one source image."); return;
  }
  if (mode === "subject" && refFiles.length < 2) {
    showErr("Subject mode requires at least two reference images."); return;
  }
  showErr("");
  const btn = $("go"); btn.disabled = true;
  btn.innerHTML = '<span class="spinner-inline"></span>Generating…';
  $("empty").style.display = "none";
  $("out").style.display = "none";
  $("meta").style.display = "none";
  $("progress").style.display = "";
  $("progress-text").textContent = "Preparing";
  $("progress-sub").textContent = "Encoding inputs";
  $("progress-fill").style.width = "0%";
  $("progress-img").style.display = "none";
  $("progress-img").removeAttribute("src");
  $("progress-preview").classList.add("empty");

  try {
    const refs_b64 = await Promise.all(refFiles.map(fileToB64));
    const keepAspect = mode === "edit" && $("keep-aspect").checked && refFiles.length === 1;
    const editingScheduler = (mode === "edit" && MODEL_TYPE === "dev")
      ? $("edit-scheduler").value : null;
    const startResp = await fetch("/api/generate/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        mode, prompt,
        width: parseInt($("width").value),
        height: parseInt($("height").value),
        seed: parseInt($("seed").value),
        refs_b64,
        keep_original_aspect: keepAspect,
        editing_scheduler: editingScheduler,
      }),
    });
    const startData = await startResp.json();
    if (!startResp.ok) throw new Error(startData.error || "Failed to start");
    const jobId = startData.job_id;

    await new Promise((resolve, reject) => {
      const es = new EventSource(`/api/generate/stream/${jobId}`);
      es.onmessage = (ev) => {
        const d = JSON.parse(ev.data);
        if (d.type === "progress") {
          const pct = Math.round((d.step / d.total) * 100);
          $("progress-text").textContent = `Generating · ${pct}%`;
          $("progress-sub").textContent = `Step ${d.step} / ${d.total}`;
          $("progress-fill").style.width = pct + "%";
          if (d.preview) {
            $("progress-img").src = "data:image/jpeg;base64," + d.preview;
            $("progress-img").style.display = "";
            $("progress-preview").classList.remove("empty");
          }
        } else if (d.type === "done") {
          $("img").src = "data:image/png;base64," + d.image;
          $("progress").style.display = "none";
          $("out").style.display = "";
          es.close(); resolve();
        } else if (d.type === "error") {
          es.close(); reject(new Error(d.message));
        }
      };
      es.onerror = () => { es.close(); reject(new Error("Stream connection lost")); };
    });
  } catch (e) {
    showErr(e.message);
    $("progress").style.display = "none";
    $("empty").style.display = "";
  } finally {
    btn.disabled = false;
    btn.textContent = "Generate";
  }
};

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"
  })[c]);
}
</script>
</body>
</html>
"""


# ── Routes ───────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template_string(
        INDEX_HTML,
        model_type=_STATE["model_type"],
        env_base_url=os.environ.get("HIDREAM_REFINE_BASE_URL", DEFAULT_BASE_URL),
        env_model=os.environ.get("HIDREAM_REFINE_MODEL", DEFAULT_MODEL_ID),
    )


@app.route("/api/refine", methods=["POST"])
def api_refine():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400
    base_url = (data.get("base_url") or DEFAULT_BASE_URL).strip()
    model_id = (data.get("model") or DEFAULT_MODEL_ID).strip()
    try:
        refined = refine_prompt(prompt, model_id=model_id, base_url=base_url)
        return jsonify({"prompt": refined})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate/start", methods=["POST"])
def api_generate_start():
    data = request.get_json(force=True)
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    mode = data.get("mode", "t2i")
    width = int(data.get("width", 2048))
    height = int(data.get("height", 2048))
    seed = int(data.get("seed", 32))
    refs_b64 = data.get("refs_b64") or []
    keep_original_aspect = bool(data.get("keep_original_aspect", False))
    editing_scheduler = data.get("editing_scheduler") or "flow_match"
    if editing_scheduler not in ("flow_match", "flash"):
        return jsonify({"error": f"Unknown editing_scheduler: {editing_scheduler}"}), 400

    if mode == "edit" and len(refs_b64) != 1:
        return jsonify({"error": "Edit mode requires exactly one reference image"}), 400
    if mode == "subject" and len(refs_b64) < 2:
        return jsonify({"error": "Subject mode requires at least two reference images"}), 400
    if keep_original_aspect and len(refs_b64) != 1:
        keep_original_aspect = False

    job_id = uuid.uuid4().hex
    q = queue.Queue()
    _JOBS[job_id] = q

    def worker():
        tmp_paths = []
        try:
            for b64 in refs_b64:
                raw = base64.b64decode(b64)
                path = os.path.join(tempfile.gettempdir(), f"hidream_{uuid.uuid4().hex}.png")
                with open(path, "wb") as f:
                    f.write(raw)
                tmp_paths.append(path)

            def cb(step, total, get_preview=None):
                msg = {"type": "progress", "step": step + 1, "total": total}
                # Only send a preview image at the 1/4, 1/2, and 3/4 milestones
                # to avoid flooding the SSE stream and keep UI progress in sync.
                milestones = {total // 4, total // 2, (3 * total) // 4}
                want_preview = (
                    get_preview is not None
                    and (step + 1) in milestones
                )
                if want_preview:
                    try:
                        img = get_preview()
                        # Downscale to keep payload tiny — full image is sent at the end.
                        max_side = 384
                        w, h = img.size
                        if max(w, h) > max_side:
                            scale = max_side / max(w, h)
                            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        buf2 = io.BytesIO()
                        img.save(buf2, format="JPEG", quality=72, optimize=True)
                        msg["preview"] = base64.b64encode(buf2.getvalue()).decode("ascii")
                    except Exception:
                        pass
                q.put(msg)

            with _GEN_LOCK:
                if _STATE["model_type"] == "full":
                    kwargs = dict(
                        num_inference_steps=50,
                        guidance_scale=5.0,
                        shift=3.0,
                        timesteps_list=None,
                        scheduler_name="default",
                    )
                elif mode == "edit" and editing_scheduler == "flow_match":
                    kwargs = dict(
                        num_inference_steps=28,
                        guidance_scale=0.0,
                        shift=1.0,
                        timesteps_list=DEFAULT_TIMESTEPS,
                        scheduler_name="flow_match",
                    )
                else:
                    kwargs = dict(
                        num_inference_steps=28,
                        guidance_scale=0.0,
                        shift=1.0,
                        timesteps_list=DEFAULT_TIMESTEPS,
                        scheduler_name="flash",
                        noise_scale_start=8.0,
                        noise_scale_end=8.0,
                        noise_clip_std=0.0,
                    )
                image = generate_image(
                    model=_STATE["model"],
                    processor=_STATE["processor"],
                    prompt=prompt,
                    ref_image_paths=tmp_paths if tmp_paths else None,
                    height=height,
                    width=width,
                    seed=seed,
                    keep_original_aspect=keep_original_aspect,
                    callback=cb,
                    **kwargs,
                )
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            q.put({"type": "done", "image": base64.b64encode(buf.getvalue()).decode("ascii")})
        except Exception as e:
            q.put({"type": "error", "message": str(e)})
        finally:
            for p in tmp_paths:
                try: os.remove(p)
                except OSError: pass
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/generate/stream/<job_id>")
def api_generate_stream(job_id):
    q = _JOBS.get(job_id)
    if q is None:
        return jsonify({"error": "Unknown job"}), 404

    def gen():
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in ("done", "error"):
                    break
        finally:
            _JOBS.pop(job_id, None)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Entrypoint ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser("HiDream-O1-Image Flask app")
    p.add_argument("--model_path", type=str,
                   default=os.environ.get("HIDREAM_MODEL_PATH"),
                   help="Path to HiDream-O1-Image checkpoint directory. "
                        "Defaults to $HIDREAM_MODEL_PATH from .env.")
    p.add_argument("--model_type", type=str,
                   default=os.environ.get("HIDREAM_MODEL_TYPE", "full"),
                   choices=["full", "dev"])
    p.add_argument("--host", type=str,
                   default=os.environ.get("HIDREAM_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int,
                   default=int(os.environ.get("HIDREAM_PORT", "7860")))
    args = p.parse_args()

    if not args.model_path:
        p.error("--model_path is required (or set HIDREAM_MODEL_PATH in .env)")

    assert torch.cuda.is_available(), "CUDA is required for inference."
    processor, model = load_image_model(args.model_path)
    _STATE["processor"] = processor
    _STATE["model"] = model
    _STATE["model_type"] = args.model_type

    print(f"[app] Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
