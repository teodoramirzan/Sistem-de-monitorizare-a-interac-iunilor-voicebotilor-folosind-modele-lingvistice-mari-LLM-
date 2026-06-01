const sessionId = `web-${Date.now()}`;
const chat = document.querySelector("#chat");
const form = document.querySelector("#messageForm");
const messageInput = document.querySelector("#message");
const result = document.querySelector("#result");
const player = document.querySelector("#player");
const apiKey = document.querySelector("#apiKey");
const voice = document.querySelector("#voice");
const voiceLabel = document.querySelector("#voiceLabel");
const cacheLabel = document.querySelector("#cacheLabel");
const micButton = document.querySelector("#micButton");
const micTitle = document.querySelector("#micTitle");
const micStatus = document.querySelector("#micStatus");
const kbStatus = document.querySelector("#kbStatus");
const kbExamples = document.querySelector("#kbExamples");
const recommendations = document.querySelector("#recommendations");
const batchConversation = document.querySelector("#batchConversation");
const batchTranscript = document.querySelector("#batchTranscript");
const executionMode = document.querySelector("#executionMode");
const envStatus = document.querySelector("#envStatus");
const runProgress = document.querySelector("#runProgress");
const progressTitle = document.querySelector("#progressTitle");
const progressMode = document.querySelector("#progressMode");
const progressSteps = document.querySelector("#progressSteps");
const analyzeButton = document.querySelector("#analyze");
const evaluateBatchButton = document.querySelector("#evaluateBatch");
const resultCards = document.querySelector("#resultCards");

let lastBotText = "";
let currentMode = "text";
let currentView = "live";
let recording = false;
let progressTimer = null;
let progressIndex = 0;
let evaluationOptions = { models: {}, recommendations: {}, tasks: [] };

async function api(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const raw = await response.text();
  let payload;
  try {
    payload = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(`Serverul nu a returnat JSON pentru ${path}. Repornește web_demo_server.py și reîncarcă pagina.`);
  }
  if (!response.ok) {
    throw new Error(payload.error || "Cererea a eșuat");
  }
  return payload;
}

function getModelConfig() {
  const config = { execution_mode: executionMode.value };
  document.querySelectorAll("[data-task-model]").forEach((select) => {
    config[select.dataset.taskModel] = select.value;
  });
  return config;
}

async function loadEvaluationOptions() {
  evaluationOptions = await api("/api/evaluation-options", {});
  populateModelSelectors();
  renderRecommendations();
}

function populateModelSelectors() {
  const models = evaluationOptions.models || {};
  document.querySelectorAll("[data-task-model]").forEach((select) => {
    const task = select.dataset.taskModel;
    const recommended = evaluationOptions.recommendations?.[task]?.model;
    select.innerHTML = Object.entries(models)
      .map(([key, model]) => {
        const recommendedSuffix = key === recommended ? " · recomandat" : "";
        const availabilitySuffix =
          executionMode.value === "real" && model.provider === "ollama" && model.installed === false
            ? " · lipsește în Ollama"
            : "";
        const suffix = `${recommendedSuffix}${availabilitySuffix}`;
        return `<option value="${escapeHtml(key)}">${escapeHtml(model.label)}${suffix}</option>`;
      })
      .join("");
    select.value = recommended || Object.keys(models)[0] || "";
  });
  updateModelHint();
  renderEnvStatus(evaluationOptions.env_status || {});
}

function renderRecommendations() {
  const recs = evaluationOptions.recommendations || {};
  const models = evaluationOptions.models || {};
  recommendations.innerHTML = Object.entries(recs)
    .map(([task, rec]) => {
      const model = models[rec.model]?.label || rec.model;
      return `
        <div class="rec-item">
          <strong>${taskLabel(task)}: ${escapeHtml(model)}</strong>
        </div>
      `;
    })
    .join("");
}

function updateModelHint() {
  document.querySelector("#modelHint").textContent =
    executionMode.value === "real"
      ? "Trimite prompturile către modelul ales. Ai nevoie de OpenAI/Gemini în .env sau Ollama pornit local."
      : "Rulează local fără chei API. Selectorul păstrează configurația de comparație.";
}

function renderEnvStatus(status) {
  const openai = status.openai_api_key_loaded ? "OpenAI ✓" : "OpenAI lipsă";
  const gemini = status.google_api_key_loaded ? "Gemini ✓" : "Gemini lipsă";
  const ollama = status.ollama_running
    ? `Ollama ✓ ${status.ollama_models?.length || 0} modele`
    : "Ollama oprit/lipsă";
  envStatus.textContent = `${openai} · ${gemini} · ${ollama}`;
  envStatus.classList.toggle("ok", Boolean(status.openai_api_key_loaded && status.google_api_key_loaded));
}

function progressLabels(config) {
  if (config.execution_mode === "real") {
    return [
      "Pregătesc transcriptul și contextul taskurilor",
      `Rulez intent cu ${modelLabel(config.intent)}`,
      `Rulez status final cu ${modelLabel(config.final_status)}`,
      `Rulez neconcordanțe cu ${modelLabel(config.incongruities)}`,
      "Parsez răspunsurile JSON și actualizez rezultatul",
    ];
  }
  return [
    "Pregătesc transcriptul",
    "Aplic regulile locale pentru intent",
    "Aplic regulile locale pentru status final",
    "Aplic regulile locale pentru neconcordanțe",
    "Actualizez rezultatul",
  ];
}

function startProgress(title, config) {
  stopProgressTimer();
  progressTitle.textContent = title;
  progressMode.textContent = config.execution_mode === "real" ? "model real/API/Ollama" : "local";
  progressIndex = 0;
  const labels = progressLabels(config);
  progressSteps.innerHTML = labels
    .map((label, index) => `<div class="progress-step" data-step="${index}"><span></span>${escapeHtml(label)}</div>`)
    .join("");
  runProgress.classList.remove("hidden");
  setBusy(true);
  markProgressStep();
  progressTimer = setInterval(() => {
    progressIndex = Math.min(progressIndex + 1, labels.length - 1);
    markProgressStep();
  }, config.execution_mode === "real" ? 4200 : 800);
}

function markProgressStep(finalState = null) {
  progressSteps.querySelectorAll(".progress-step").forEach((step, index) => {
    step.classList.toggle("done", index < progressIndex || finalState === "done");
    step.classList.toggle("active", index === progressIndex && !finalState);
    step.classList.toggle("error", finalState === "error" && index === progressIndex);
  });
}

function finishProgress(state, message = "") {
  stopProgressTimer();
  if (state === "done") {
    progressIndex = progressSteps.children.length - 1;
    markProgressStep("done");
  } else if (state === "error") {
    markProgressStep("error");
    if (message) showToast(message);
  }
  setBusy(false);
}

function stopProgressTimer() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

function setBusy(isBusy) {
  analyzeButton.disabled = isBusy;
  evaluateBatchButton.disabled = isBusy;
  document.body.classList.toggle("busy", isBusy);
}

function addBubble(role, text, target = chat) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  const label = role === "assistant" ? "Bănuțel" : "Tu";
  bubble.innerHTML = `<small>${label}</small>${escapeHtml(text)}`;
  target.appendChild(bubble);
  target.scrollTop = target.scrollHeight;
}

function renderTranscript(turns) {
  chat.innerHTML = "";
  turns.forEach((turn) => addBubble(turn.role, turn.text));
}

function renderMiniTranscript(turns) {
  batchTranscript.innerHTML = "";
  turns.forEach((turn) => addBubble(turn.role, turn.text, batchTranscript));
}

async function startSession() {
  result.textContent = "{}";
  clearResultCards();
  const payload = await api("/api/start", { session_id: sessionId });
  renderTranscript(payload.transcript);
  renderKnowledgeBase(payload.knowledge_base);
  lastBotText = payload.bot;
  speak(payload.bot, false);
}

async function sendMessage(text) {
  result.textContent = "{}";
  clearResultCards();
  const payload = await api("/api/message", { session_id: sessionId, message: text });
  renderTranscript(payload.transcript);
  renderKnowledgeBase(payload.knowledge_base);
  lastBotText = payload.bot;
  await speak(payload.bot, currentMode === "voice");
}

async function sendVoiceMessage(audioBase64) {
  result.textContent = "{}";
  clearResultCards();
  const payload = await api("/api/voice-message", {
    session_id: sessionId,
    audio_base64: audioBase64,
    key: apiKey.value.trim(),
  });
  renderTranscript(payload.transcript);
  renderKnowledgeBase(payload.knowledge_base);
  lastBotText = payload.bot;
  micStatus.textContent = `Ai spus: ${payload.user}`;
  await speak(payload.bot, true);
}

async function analyze() {
  const config = getModelConfig();
  startProgress("Analizez conversația live", config);
  try {
    const payload = await api("/api/analyze", {
      session_id: sessionId,
      model_config: config,
    });
    renderKnowledgeBase(payload.knowledge_base);
    renderAnalysisCards(payload.pipeline);
    result.textContent = JSON.stringify(payload.pipeline, null, 2);
    finishProgress("done");
  } catch (error) {
    finishProgress("error", error.message);
    throw error;
  }
}

async function evaluateBatch() {
  const config = getModelConfig();
  startProgress("Evaluez transcriptul complet", config);
  try {
    const payload = await api("/api/evaluate-conversation", {
      conversation_text: batchConversation.value,
      model_config: config,
    });
    renderMiniTranscript(payload.transcript);
    renderKnowledgeBase(payload.knowledge_base);
    renderAnalysisCards(payload.evaluation);
    result.textContent = JSON.stringify(payload.evaluation, null, 2);
    finishProgress("done");
  } catch (error) {
    finishProgress("error", error.message);
    throw error;
  }
}

function renderKnowledgeBase(kb) {
  if (!kb || !kb.available) {
    kbStatus.textContent = "Datasetul nu este disponibil.";
    kbExamples.innerHTML = "";
    return;
  }
  const examples = kb.examples || [];
  kbStatus.textContent = examples.length
    ? `Intenție sugerată: ${kb.suggested_intent || "necunoscut"}`
    : "Aștept mesajul utilizatorului pentru exemple similare.";
  kbExamples.innerHTML = examples
    .map(
      (example) => `
        <div class="kb-item">
          <strong>${escapeHtml(example.conversation_id)} · ${escapeHtml(example.mapped_intent)}</strong>
          <span>scor ${example.score} · status ${escapeHtml(example.final_status)}</span>
          <div>${escapeHtml(example.first_user_message || "")}</div>
        </div>
      `
    )
    .join("");
}

function renderAnalysisCards(pipeline) {
  const results = pipeline?.results || pipeline?.evaluation || {};
  const tasks = pipeline?.tasks || {};
  const intent = results.intent || {};
  const finalStatus = results.final_status || {};
  const incongruities = results.incongruities || {};
  const cards = [
    {
      title: "Intent",
      value: intent.intent || intent.error || "necunoscut",
      confidence: intent.confidence,
      reasoning: intent.reasoning || intent.message,
      meta: taskMeta(tasks.intent),
    },
    {
      title: "Status final",
      value: finalStatus.final_status || finalStatus.error || "necunoscut",
      confidence: finalStatus.confidence,
      reasoning: finalStatus.reasoning || finalStatus.message,
      meta: taskMeta(tasks.final_status),
    },
    {
      title: "Neconcordanțe",
      value: incongruityLabel(incongruities),
      confidence: incongruities.confidence,
      reasoning: incongruities.reasoning || incongruities.message,
      meta: taskMeta(tasks.incongruities),
      alert: Boolean(incongruities.has_incongruity),
    },
  ];
  resultCards.innerHTML = cards.map(renderResultCard).join("");
  resultCards.classList.remove("hidden");
}

function renderResultCard(card) {
  const confidence = card.confidence ? `<span class="pill">${escapeHtml(card.confidence)}</span>` : "";
  const alertClass = card.alert ? " alert" : "";
  return `
    <article class="result-card${alertClass}">
      <div class="result-main">
        <span>${escapeHtml(card.title)}</span>
        <strong>${escapeHtml(card.value)}</strong>
      </div>
      ${confidence}
      ${card.meta ? `<p class="result-meta">${escapeHtml(card.meta)}</p>` : ""}
      ${card.reasoning ? `<p>${escapeHtml(card.reasoning)}</p>` : ""}
    </article>
  `;
}

function taskMeta(task) {
  if (!task) return "";
  const recommended = task.is_recommended ? "recomandat" : "selectat";
  const warning = task.warning || task.error ? ` · ${task.warning || task.error}` : "";
  return `${task.model_label || task.model || "model"} · ${task.lang || ""} ${task.prompt_version || ""} · ${recommended}${warning}`;
}

function incongruityLabel(value) {
  if (value.error) return value.error;
  if (value.has_incongruity) {
    return value.incongruity_type ? `Da · ${value.incongruity_type}` : "Da";
  }
  if (value.has_incongruity === false) return "Nu";
  return "necunoscut";
}

function clearResultCards() {
  resultCards.innerHTML = "";
  resultCards.classList.add("hidden");
}

async function speak(text, autoplay) {
  if (!text) return;
  voiceLabel.textContent = voice.value;
  try {
    const payload = await api("/api/tts", {
      text,
      voice: voice.value,
      key: apiKey.value.trim(),
    });
    player.src = payload.audio_url;
    if (autoplay) {
      await player.play();
    }
  } catch (error) {
    showToast(error.message);
  }
}

async function clearCache() {
  const payload = await api("/api/cache/clear", {});
  cacheLabel.textContent = "golit";
  showToast(payload.message);
}

function startDictation() {
  if (recording) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Microfonul nu este disponibil în acest browser.");
    return;
  }
  recording = true;
  micButton.classList.add("listening");
  micTitle.textContent = "Ascult...";
  micStatus.textContent = "Vorbește acum. Înregistrarea se oprește automat după 4 secunde.";

  recordWav(4000)
    .then((audioBase64) => {
      micTitle.textContent = "Trimit către Zevo STT...";
      micStatus.textContent = "Aștept transcrierea.";
      return sendVoiceMessage(audioBase64);
    })
    .catch((error) => showToast(error.message))
    .finally(() => {
      recording = false;
      micButton.classList.remove("listening");
      if (currentMode === "voice") {
        micTitle.textContent = "Apasă microfonul și vorbește";
      }
    });
}

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });
  document.querySelectorAll(".mode-panel").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.panel !== mode);
  });
}

function setView(view) {
  currentView = view;
  document.querySelectorAll(".app-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.view === view);
  });
  document.querySelectorAll(".view-panel").forEach((panel) => {
    panel.classList.toggle("hidden", panel.dataset.viewPanel !== view);
  });
}

function loadSample() {
  batchConversation.value = `USER: Vreau să-mi blochez cardul de credit.
ASSISTANT: Pentru siguranță, spuneți ultimele 4 cifre.
USER: Ultimele cifre sunt 4321.
ASSISTANT: Cardul de debit terminat în 4321 a fost blocat.`;
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3600);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => {
    const entities = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
    return entities[char];
  });
}

function taskLabel(task) {
  return {
    intent: "Intent",
    final_status: "Status final",
    incongruities: "Neconcordanțe",
  }[task] || task;
}

function modelLabel(key) {
  return evaluationOptions.models?.[key]?.label || key || "modelul selectat";
}

async function recordWav(durationMs) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const chunks = [];

  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };

  source.connect(processor);
  processor.connect(audioContext.destination);
  await new Promise((resolve) => setTimeout(resolve, durationMs));
  processor.disconnect();
  source.disconnect();
  stream.getTracks().forEach((track) => track.stop());

  const input = mergeFloat32(chunks);
  const resampled = downsample(input, audioContext.sampleRate, 16000);
  const wav = encodeWav(resampled, 16000);
  await audioContext.close();
  return arrayBufferToBase64(wav);
}

function mergeFloat32(chunks) {
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(length);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function downsample(buffer, inputRate, outputRate) {
  if (outputRate === inputRate) return buffer;
  const ratio = inputRate / outputRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offset = 0;
  for (let i = 0; i < newLength; i += 1) {
    const nextOffset = Math.round((i + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let j = offset; j < nextOffset && j < buffer.length; j += 1) {
      accum += buffer[j];
      count += 1;
    }
    result[i] = accum / Math.max(count, 1);
    offset = nextOffset;
  }
  return result;
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }
  return buffer;
}

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

function arrayBufferToBase64(buffer) {
  let binary = "";
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  messageInput.value = "";
  try {
    await sendMessage(text);
  } catch (error) {
    showToast(error.message);
  }
});

analyzeButton.addEventListener("click", () => analyze().catch((error) => showToast(error.message)));
evaluateBatchButton.addEventListener("click", () => evaluateBatch().catch((error) => showToast(error.message)));
document.querySelector("#clearBatch").addEventListener("click", () => {
  batchConversation.value = "";
  batchTranscript.innerHTML = "";
});
document.querySelector("#loadSample").addEventListener("click", loadSample);
document.querySelector("#clearCache").addEventListener("click", () => clearCache().catch((error) => showToast(error.message)));
document.querySelector("#newSession").addEventListener("click", () => startSession().catch((error) => showToast(error.message)));
micButton.addEventListener("click", startDictation);
document.querySelector("#playLast").addEventListener("click", () => speak(lastBotText, true));
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode));
});
document.querySelectorAll(".app-tab").forEach((tab) => {
  tab.addEventListener("click", () => setView(tab.dataset.view));
});
voice.addEventListener("change", () => {
  voiceLabel.textContent = voice.value;
});
executionMode.addEventListener("change", () => {
  populateModelSelectors();
  updateModelHint();
});

loadEvaluationOptions().catch((error) => showToast(error.message));
startSession().catch((error) => showToast(error.message));
