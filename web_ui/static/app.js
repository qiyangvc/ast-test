const state = {
  models: {},
  metrics: [],
};

const $ = (id) => document.getElementById(id);
const fmt = (value) => (value === null || value === undefined ? "--" : Number(value).toFixed(4));
const pct = (value) => `${Math.round(Number(value || 0) * 100)}%`;

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2200);
}

async function api(path, body) {
  const options = body
    ? {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    : undefined;
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.message || data.error || response.statusText);
  }
  return data;
}

function currentPayload() {
  return {
    text: $("textInput").value.trim(),
    mode: $("modeSelect").value,
    model: $("modelSelect").value,
  };
}

function setBusy(button, busy) {
  button.disabled = busy;
  button.dataset.originalText ||= button.textContent;
  button.textContent = busy ? "运行中..." : button.dataset.originalText;
}

function renderModelSelectors() {
  const modeSelect = $("modeSelect");
  const modelSelect = $("modelSelect");
  modeSelect.innerHTML = "";
  for (const mode of Object.keys(state.models)) {
    const option = document.createElement("option");
    option.value = mode;
    option.textContent = mode;
    modeSelect.appendChild(option);
  }
  modeSelect.value = state.models.text_ast_fgm ? "text_ast_fgm" : modeSelect.value;

  const syncModels = () => {
    modelSelect.innerHTML = "";
    for (const model of state.models[modeSelect.value] || []) {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      modelSelect.appendChild(option);
    }
    if ((state.models[modeSelect.value] || []).includes("cnn")) {
      modelSelect.value = "cnn";
    }
  };
  modeSelect.addEventListener("change", syncModels);
  syncModels();
}

function renderMetrics() {
  $("metricsCount").textContent = `${state.metrics.length} 个模型`;
  $("metricsRows").innerHTML = state.metrics
    .map(
      (row) => `
        <tr>
          <td>${row.mode}</td>
          <td>${row.model}</td>
          <td>${fmt(row.clean_accuracy)}</td>
          <td>${fmt(row.ast_accuracy)}</td>
          <td>${fmt(row.robust_drop)}</td>
          <td>${fmt(row.uci_accuracy)}</td>
        </tr>
      `
    )
    .join("");
}

function renderPrediction(result) {
  const badge = $("resultBadge");
  badge.className = `badge ${result.label}`;
  badge.textContent = result.label.toUpperCase();
  $("spamProb").textContent = fmt(result.probabilities.spam);
  $("normalProb").textContent = fmt(result.probabilities.normal);
  $("spamBar").style.width = pct(result.probabilities.spam);
  $("normalBar").style.width = pct(result.probabilities.normal);
  $("confidence").textContent = fmt(result.confidence);
  $("tokenCount").textContent = `${result.used_token_count}/${result.token_count}`;
  $("unknownCount").textContent = result.unknown_count;
  $("tokens").innerHTML = result.tokens.map((token) => `<span>${escapeHtml(token)}</span>`).join("");
}

function renderAttack(result) {
  const rows = result.candidates || [];
  const profile = (result.strength || "mild") === "mild" ? "训练 AST" : result.strength;
  $("attackSummary").textContent = `${profile} · ${rows.length} 个候选`;
  const best = result.best;
  if (best) {
    $("bestAttack").classList.remove("empty");
    $("bestAttack").innerHTML = `
      <strong>最佳候选：</strong>${escapeHtml(best.text)}
      <br />
      <span>类型 ${best.attack_type || "--"}，预测 ${best.prediction.label.toUpperCase()}，Normal ${fmt(
        best.prediction.probabilities.normal
      )}</span>
    `;
  } else {
    $("bestAttack").classList.add("empty");
    $("bestAttack").textContent = "暂无候选";
  }
  $("attackRows").innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.attack_type || "--"}</td>
          <td class="text-cell">
            ${escapeHtml(row.text)}
            <div class="ops">${escapeHtml((row.operations || []).join("；") || "--")}</div>
          </td>
          <td><span class="badge ${row.prediction.label}">${row.prediction.label}</span></td>
          <td>${fmt(row.prediction.probabilities.normal)}</td>
        </tr>
      `
    )
    .join("");
}

function renderCompare(rows) {
  $("compareStatus").textContent = `${rows.length} 个模型`;
  $("compareRows").innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.mode}</td>
          <td>${row.model}</td>
          <td><span class="badge ${row.label}">${row.label}</span></td>
          <td>${fmt(row.spam_probability)}</td>
          <td>${fmt(row.normal_probability)}</td>
        </tr>
      `
    )
    .join("");
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadInitialState() {
  const data = await api("/api/models");
  state.models = data.models || {};
  state.metrics = data.metrics || [];
  renderModelSelectors();
  renderMetrics();
  $("artifactStatus").textContent = `已加载模型产物 ${data.output_dir || ""}`.trim();
}

async function runPredict() {
  const button = $("predictBtn");
  const payload = currentPayload();
  if (!payload.text) {
    showToast("请输入文本");
    return;
  }
  setBusy(button, true);
  try {
    const result = await api("/api/predict", payload);
    renderPrediction(result);
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(button, false);
  }
}

async function runAttack() {
  const button = $("attackBtn");
  const payload = { ...currentPayload(), label: "spam", strength: "mild" };
  if (!payload.text) {
    showToast("请输入文本");
    return;
  }
  setBusy(button, true);
  try {
    const result = await api("/api/attack", payload);
    renderPrediction(result.original);
    renderAttack(result);
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(button, false);
  }
}

async function runCompare() {
  const button = $("compareBtn");
  const payload = currentPayload();
  if (!payload.text) {
    showToast("请输入文本");
    return;
  }
  setBusy(button, true);
  try {
    const result = await api("/api/compare", payload);
    renderCompare(result.rows || []);
  } catch (error) {
    showToast(error.message);
  } finally {
    setBusy(button, false);
  }
}

function bindEvents() {
  $("predictBtn").addEventListener("click", runPredict);
  $("attackBtn").addEventListener("click", runAttack);
  $("compareBtn").addEventListener("click", runCompare);
  $("clearBtn").addEventListener("click", () => {
    $("textInput").value = "";
    $("textInput").focus();
  });
  for (const button of document.querySelectorAll(".sample")) {
    button.addEventListener("click", () => {
      $("textInput").value = button.dataset.text;
      runPredict();
    });
  }
}

loadInitialState()
  .then(() => {
    bindEvents();
    runPredict();
  })
  .catch((error) => {
    $("artifactStatus").textContent = "模型产物加载失败";
    showToast(error.message);
  });
