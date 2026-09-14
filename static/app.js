const FEATURES = [
  ["X1", "Current assets"],
  ["X2", "Cost of goods sold"],
  ["X3", "Depreciation and amortization"],
  ["X4", "EBITDA"],
  ["X5", "Inventory"],
  ["X6", "Net income"],
  ["X7", "Total receivables"],
  ["X8", "Market value"],
  ["X9", "Net sales"],
  ["X10", "Total assets"],
  ["X11", "Total long-term debt"],
  ["X12", "EBIT"],
  ["X13", "Gross profit"],
  ["X14", "Total current liabilities"],
  ["X15", "Retained earnings"],
  ["X17", "Total liabilities"],
  ["X18", "Total operating expenses"],
];

const FEATURE_GROUPS = [
  {
    name: "Balance sheet",
    description: "Liquidity, assets, and liability position",
    features: [
      ["X1", "Current assets"],
      ["X5", "Inventory"],
      ["X10", "Total assets"],
      ["X14", "Total current liabilities"],
      ["X17", "Total liabilities"],
    ],
  },
  {
    name: "Income & profitability",
    description: "Operating performance and earnings signals",
    features: [
      ["X2", "Cost of goods sold"],
      ["X3", "Depreciation and amortization"],
      ["X4", "EBITDA"],
      ["X6", "Net income"],
      ["X9", "Net sales"],
      ["X12", "EBIT"],
      ["X13", "Gross profit"],
      ["X18", "Total operating expenses"],
    ],
  },
  {
    name: "Market & capital",
    description: "Market value, debt, and retained capital",
    features: [
      ["X7", "Total receivables"],
      ["X8", "Market value"],
      ["X11", "Total long-term debt"],
      ["X15", "Retained earnings"],
    ],
  },
];

const fieldContainer = document.querySelector("#feature-fields");
const form = document.querySelector("#assessment-form");
const submitButton = document.querySelector("#submit-button");
const resetButton = document.querySelector("#reset-button");
const formError = document.querySelector("#form-error");
const emptyResults = document.querySelector("#empty-results");
const resultsContent = document.querySelector("#results-content");
const resultState = document.querySelector("#result-state");

const formatPercent = (value) => `${(value * 100).toFixed(2)}%`;
const formatShap = (value) => `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
const formatInputValue = (value) => Number(value).toLocaleString(undefined, {
  maximumFractionDigits: 4,
});
const escapeHtml = (value) =>
  String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[character]));

function renderFields() {
  fieldContainer.innerHTML = FEATURE_GROUPS.map((group) => {
    const groupId = `group-${group.name.replace(/[^a-z0-9]+/gi, "-").toLowerCase()}`;
    return `
      <section class="feature-group" aria-labelledby="${groupId}">
        <div class="feature-group-heading">
          <div>
            <h3 id="${groupId}">${group.name}</h3>
            <p>${group.description}</p>
          </div>
          <span class="group-rule"></span>
        </div>
        <div class="feature-group-grid">
          ${group.features.map(([code, label]) => `
            <div class="feature-field">
              <label for="input-${code}">
                <span>${label}</span>
                <span class="feature-code">${code}</span>
              </label>
              <input
                id="input-${code}"
                name="${code}"
                type="number"
                step="any"
                inputmode="decimal"
                placeholder="Enter value"
                required
                aria-label="${label} (${code})"
              />
            </div>
          `).join("")}
        </div>
      </section>
    `;
  }).join("");
}

function setLoading(loading) {
  submitButton.disabled = loading;
  submitButton.classList.toggle("is-loading", loading);
  submitButton.querySelector(".button-label").textContent = loading
    ? "Assessing profile…"
    : "Assess Bankruptcy Risk";
}

function setError(message = "") {
  formError.textContent = message;
  formError.hidden = !message;
}

function updateBoundaryMarker(id, value) {
  document.querySelector(`#${id}`).style.left = `${Math.min(100, Math.max(0, value * 100))}%`;
}

function renderContributions(contributions) {
  const chart = document.querySelector("#contribution-chart");
  const list = document.querySelector("#explanation-list");
  const maxAbsolute = Math.max(...contributions.map((item) => item.absolute_shap_value), 0.000001);

  chart.innerHTML = contributions.map((item) => {
    const width = Math.max(2, (item.absolute_shap_value / maxAbsolute) * 46);
    const polarity = item.shap_value >= 0 ? "positive" : "negative";
    const direction = item.shap_value >= 0 ? "Higher risk" : "Lower risk";
    return `
      <div class="contribution-row">
        <div class="contribution-meta">
          <strong>${escapeHtml(item.label)}</strong>
          <span>Input: ${formatInputValue(item.input_value)}</span>
        </div>
        <span class="contribution-visual">
          <span class="contribution-bar ${polarity}" style="width:${width}%"></span>
        </span>
        <div class="contribution-number ${polarity}">
          <strong>${formatShap(item.shap_value)}</strong>
          <span>${direction}</span>
        </div>
      </div>
    `;
  }).join("");

  list.innerHTML = contributions.slice(0, 5).map((item, index) => {
    const direction = item.shap_value >= 0 ? "pushes the model toward higher bankruptcy risk" : "pushes the model toward lower bankruptcy risk";
    return `
      <div class="explanation-item">
        <span class="explanation-item-index">${String(index + 1).padStart(2, "0")}</span>
        <span><strong>${escapeHtml(item.label)} (${escapeHtml(item.feature)})</strong> ${direction}. Input value: ${formatInputValue(item.input_value)}.</span>
      </div>
    `;
  }).join("");
}

function renderResults(result) {
  emptyResults.hidden = true;
  resultsContent.hidden = false;
  resultState.textContent = "Assessment complete";
  resultState.classList.add("complete");

  document.querySelector("#probability-value").textContent = formatPercent(result.bankruptcy_probability);
  document.querySelector("#meter-value").textContent = formatPercent(result.bankruptcy_probability);
  document.querySelector("#risk-category").textContent = result.risk_category;
  document.querySelector("#risk-category").className = `risk-badge ${result.risk_category === "LOW RISK" ? "low" : result.risk_category === "HIGH RISK" ? "high" : ""}`;
  document.querySelector("#decision-value").textContent = result.decision;
  document.querySelector("#suggested-action").textContent = result.suggested_action;

  updateBoundaryMarker("prediction-marker", result.bankruptcy_probability);
  updateBoundaryMarker("low-marker", result.low_risk_limit);
  updateBoundaryMarker("threshold-marker", result.decision_threshold);
  updateBoundaryMarker("high-marker", result.high_risk_limit);
  document.querySelector("#prediction-label").textContent = formatPercent(result.bankruptcy_probability);
  document.querySelector(".meter-segment.low").style.width = `${result.low_risk_limit * 100}%`;
  document.querySelector(".meter-segment.moderate").style.width = `${(result.high_risk_limit - result.low_risk_limit) * 100}%`;
  document.querySelector(".meter-segment.high").style.width = `${(1 - result.high_risk_limit) * 100}%`;
  document.querySelector("#low-label").textContent = formatPercent(result.low_risk_limit);
  document.querySelector("#threshold-label").textContent = formatPercent(result.decision_threshold);
  document.querySelector("#high-label").textContent = formatPercent(result.high_risk_limit);
  renderContributions(result.feature_contributions);
}

function collectValues() {
  const values = {};
  for (const [code] of FEATURES) {
    const raw = document.querySelector(`#input-${code}`).value.trim();
    if (!raw) throw new Error(`${code} is required.`);
    const value = Number(raw);
    if (!Number.isFinite(value)) throw new Error(`${code} must be a finite number.`);
    values[code] = value;
  }
  return values;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  let values;
  try {
    values = collectValues();
  } catch (error) {
    setError(error.message);
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(payload.detail)
        ? payload.detail.map((item) => item.msg).join(" ")
        : payload.detail || "The model could not assess this profile.";
      throw new Error(detail);
    }
    renderResults(payload);
    document.querySelector("#results-title").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setError(error.message || "Unable to reach the assessment service.");
  } finally {
    setLoading(false);
  }
});

resetButton.addEventListener("click", () => {
  form.reset();
  setError("");
  resultsContent.hidden = true;
  emptyResults.hidden = false;
  resultState.textContent = "Awaiting profile";
  resultState.classList.remove("complete");
});

renderFields();