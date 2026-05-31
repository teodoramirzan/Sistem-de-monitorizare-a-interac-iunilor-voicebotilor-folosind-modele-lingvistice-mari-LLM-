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
const phoneLabel = document.querySelector("#phoneLabel");
const phoneStatus = document.querySelector("#phoneStatus");
const callLink = document.querySelector("#callLink");
const phonePanelTitle = document.querySelector("#phonePanelTitle");
const phonePanelText = document.querySelector("#phonePanelText");
const micButton = document.querySelector("#micButton");
const micTitle = document.querySelector("#micTitle");
const micStatus = document.querySelector("#micStatus");
const kbStatus = document.querySelector("#kbStatus");
const kbExamples = document.querySelector("#kbExamples");
const recommendations = document.querySelector("#recommendations");
const batchConversation = document.querySelector("#batchConversation");
const batchTranscript = document.querySelector("#batchTranscript");

let lastBotText = "";
let currentMode = "text";
let currentView = "live";
let recording = false;
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
  const config = {};
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
        const suffix = key === recommended ? " · recomandat" : "";
        return `<option value="${escapeHtml(key)}">${escapeHtml(model.label)}${suffix}</option>`;
      })
      .join("");
    select.value = recommended || Object.keys(models)[0] || "";
  });
  document.querySelector("#modelHint").textContent =
    "În web demo, scorul se calculează local; selectorul arată modelul/promptul ales pentru comparație.";
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
          <span>${escapeHtml(rec.lang)} · ${escapeHtml(rec.prompt_version)} · ${escapeHtml(rec.metric)}</span>
        </div>
      `;
    })
    .join("");
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
  const payload = await api("/api/start", { session_id: sessionId });
  renderTranscript(payload.transcript);
  renderKnowledgeBase(payload.knowledge_base);
  lastBotText = payload.bot;
  speak(payload.bot, false);
}

async function sendMessage(text) {
  result.textContent = "{}";
  const payload = await api("/api/message", { session_id: sessionId, message: text });
  renderTranscript(payload.transcript);
  renderKnowledgeBase(payload.knowledge_base);
  lastBotText = payload.bot;
  await speak(payload.bot, currentMode === "voice");
}

async function sendVoiceMessage(audioBase64) {
  result.textContent = "{}";
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
  const payload = await api("/api/analyze", {
    session_id: sessionId,
    model_config: getModelConfig(),
  });
  renderKnowledgeBase(payload.knowledge_base);
  result.textContent = JSON.stringify(payload.pipeline, null, 2);
}

async function evaluateBatch() {
  const payload = await api("/api/evaluate-conversation", {
    conversation_text: batchConversation.value,
    model_config: getModelConfig(),
  });
  renderMiniTranscript(payload.transcript);
  renderKnowledgeBase(payload.knowledge_base);
  result.textContent = JSON.stringify(payload.evaluation, null, 2);
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

async function loadPhoneStatus() {
  const payload = await api("/api/telephony/status", {});
  phoneLabel.textContent = payload.configured ? "configurat" : "local";
  if (payload.configured) {
    phoneStatus.textContent = `Poți suna la ${payload.phone_number}.`;
    callLink.href = `tel:${payload.phone_number.replace(/\s+/g, "")}`;
    callLink.textContent = `Sună la ${payload.phone_number}`;
    callLink.classList.remove("disabled");
    phonePanelTitle.textContent = `Poți suna la ${payload.phone_number}.`;
    phonePanelText.textContent = payload.public_webhook_url
      ? `Webhook public: ${payload.public_webhook_url}`
      : "Numărul este afișat, dar pentru răspuns real trebuie configurat webhook-ul public în Zevo.";
  } else {
    phoneStatus.textContent = "Setează ZEVO_PHONE_NUMBER pentru a afișa numărul de apel.";
    callLink.href = "#";
    callLink.textContent = "Număr neconfigurat";
    callLink.classList.add("disabled");
    phonePanelTitle.textContent = "Pentru apel de pe telefon, configurează numărul Zevo.";
    phonePanelText.textContent = "Setează ZEVO_PHONE_NUMBER și configurează webhook-ul public către /api/telephony/inbound.";
  }
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
  batchConversation.value = `USER: Vreau să-mi verific soldul și să primesc extrasul pe luna trecută.
ASSISTANT: Pentru sold trebuie să confirmați identitatea. Puteți confirma codul primit prin SMS?
USER: Da, confirm.
ASSISTANT: Soldul disponibil este 2.450 de lei.`;
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

document.querySelector("#analyze").addEventListener("click", () => analyze().catch((error) => showToast(error.message)));
document.querySelector("#evaluateBatch").addEventListener("click", () => evaluateBatch().catch((error) => showToast(error.message)));
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

loadEvaluationOptions().catch((error) => showToast(error.message));
loadPhoneStatus().catch(() => {
  phoneLabel.textContent = "local";
});
startSession().catch((error) => showToast(error.message));
