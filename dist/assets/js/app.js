import { MaternalRiskModel } from "./model.js";

const model = new MaternalRiskModel();
const form = document.querySelector("[data-risk-form]");
const submitButton = document.querySelector("[data-submit]");
const resetButton = document.querySelector("[data-reset]");
const compareButton = document.querySelector("[data-compare]");
const status = document.querySelector("[data-model-status]");
const profileButtons = [...document.querySelectorAll("[data-profile]")];
const comparisonPanel = document.querySelector("[data-comparisons]");
const comparisonBody = document.querySelector("[data-comparison-body]");
const comparisonEmpty = document.querySelector("[data-comparison-empty]");
const clearComparisons = document.querySelector("[data-clear-comparisons]");

const profiles = {
  routine: { Age: 25, SystolicBP: 110, DiastolicBP: 70, BS: 7.1, BodyTemp: 98, HeartRate: 76 },
  glucose: { Age: 32, SystolicBP: 120, DiastolicBP: 80, BS: 12, BodyTemp: 98, HeartRate: 78 },
  pressure: { Age: 38, SystolicBP: 145, DiastolicBP: 95, BS: 7.5, BodyTemp: 98, HeartRate: 82 },
};

const labelCopy = {
  "low risk": "The model pattern aligns most strongly with the low-risk class in the research dataset.",
  "mid risk": "The model pattern aligns most strongly with the mid-risk class; this was the least reliable class in evaluation.",
  "high risk": "The model pattern aligns most strongly with the high-risk class and warrants prompt clinical review.",
};

const labelTitle = {
  "low risk": "Low-risk pattern",
  "mid risk": "Mid-risk pattern",
  "high risk": "High-risk pattern",
};

const labelColor = {
  "low risk": "#66d0c4",
  "mid risk": "#e9a928",
  "high risk": "#f67667",
};

let latestResult = null;
let latestInputs = null;

const valuesFromForm = () =>
  Object.fromEntries(new FormData(form).entries());

const numericValuesFromForm = () =>
  Object.fromEntries(Object.entries(valuesFromForm()).map(([key, value]) => [key, Number(value)]));

const applyProfile = (name) => {
  const profile = profiles[name];
  if (!profile) return;
  Object.entries(profile).forEach(([key, value]) => {
    const input = form.elements.namedItem(key);
    if (input) input.value = value;
  });
  profileButtons.forEach((button) => button.classList.toggle("active", button.dataset.profile === name));
};

const setStatus = (message, state = "loading") => {
  status.textContent = message;
  status.classList.toggle("ready", state === "ready");
  status.classList.toggle("error", state === "error");
};

const setBusy = (busy) => {
  submitButton.disabled = busy || !model.ready;
  submitButton.querySelector("span:last-child").textContent = busy ? "Assessing…" : "Assess risk profile";
  document.querySelector("[data-result-card]").classList.toggle("loading-shimmer", busy);
};

const percent = (value) => `${Math.round(value * 100)}%`;

const renderProbabilities = (probabilities) => {
  ["low", "mid", "high"].forEach((risk, index) => {
    const row = document.querySelector(`[data-probability="${risk}"]`);
    row.querySelector(".probability-fill").style.width = percent(probabilities[index]);
    row.querySelector(".probability-value").textContent = percent(probabilities[index]);
  });
};

const calculateInfluence = async (inputs, baseResult) => {
  const entries = [];
  for (const [feature, median] of Object.entries(model.medians)) {
    const altered = { ...inputs, [feature]: median };
    const result = await model.predict(altered);
    entries.push({
      feature,
      delta: Math.abs(baseResult.probabilities[baseResult.classIndex] - result.probabilities[baseResult.classIndex]),
    });
  }
  return entries.sort((a, b) => b.delta - a.delta).slice(0, 3);
};

const featureLabels = {
  Age: "Age",
  SystolicBP: "Systolic BP",
  DiastolicBP: "Diastolic BP",
  BS: "Blood glucose",
  BodyTemp: "Body temperature",
  HeartRate: "Heart rate",
};

const renderInfluence = (influences) => {
  const list = document.querySelector("[data-influence-list]");
  const max = Math.max(...influences.map((item) => item.delta), 0.001);
  list.innerHTML = influences
    .map(
      ({ feature, delta }) => `
        <li>
          <span>${featureLabels[feature]}</span>
          <span class="mini-track"><span class="mini-fill" style="width:${Math.max(8, (delta / max) * 100)}%"></span></span>
          <span>${(delta * 100).toFixed(1)} pts</span>
        </li>`,
    )
    .join("");
};

const renderResult = async (inputs, result) => {
  const ring = document.querySelector("[data-risk-ring]");
  ring.style.setProperty("--progress", `${result.confidence * 360}deg`);
  ring.style.setProperty("--ring-color", labelColor[result.label]);
  document.querySelector("[data-confidence]").textContent = percent(result.confidence);
  document.querySelector("[data-risk-title]").textContent = labelTitle[result.label];
  document.querySelector("[data-risk-copy]").textContent = labelCopy[result.label];
  renderProbabilities(result.probabilities);
  compareButton.disabled = false;
  compareButton.hidden = false;

  document.querySelector("[data-influence-list]").innerHTML = "<li><span>Calculating what-if influence…</span></li>";
  const influences = await calculateInfluence(inputs, result);
  renderInfluence(influences);
};

const loadComparisons = () => {
  try {
    return JSON.parse(localStorage.getItem("maternal-risk-comparisons") || "[]");
  } catch {
    return [];
  }
};

const saveComparisons = (comparisons) => {
  localStorage.setItem("maternal-risk-comparisons", JSON.stringify(comparisons));
};

const renderComparisons = () => {
  const comparisons = loadComparisons();
  comparisonEmpty.hidden = comparisons.length > 0;
  comparisonPanel.hidden = false;
  comparisonBody.parentElement.hidden = comparisons.length === 0;
  clearComparisons.hidden = comparisons.length === 0;
  comparisonBody.innerHTML = comparisons
    .map(
      (item, index) => `
        <tr>
          <td>Case ${index + 1}</td>
          <td>${item.Age}</td>
          <td>${item.SystolicBP}/${item.DiastolicBP}</td>
          <td>${item.BS}</td>
          <td>${item.BodyTemp}</td>
          <td>${item.HeartRate}</td>
          <td><span class="risk-pill ${item.label.split(" ")[0]}">${labelTitle[item.label]}</span></td>
          <td>${percent(item.confidence)}</td>
        </tr>`,
    )
    .join("");
};

profileButtons.forEach((button) => {
  button.addEventListener("click", () => applyProfile(button.dataset.profile));
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  const inputs = numericValuesFromForm();
  setBusy(true);
  try {
    const result = await model.predict(inputs);
    latestInputs = inputs;
    latestResult = result;
    await renderResult(inputs, result);
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    setBusy(false);
  }
});

resetButton.addEventListener("click", () => {
  applyProfile("routine");
  form.requestSubmit();
});

compareButton.addEventListener("click", () => {
  if (!latestInputs || !latestResult) return;
  const comparisons = loadComparisons();
  comparisons.push({ ...latestInputs, label: latestResult.label, confidence: latestResult.confidence });
  saveComparisons(comparisons.slice(-4));
  renderComparisons();
  comparisonPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
});

clearComparisons.addEventListener("click", () => {
  saveComparisons([]);
  renderComparisons();
});

const start = async () => {
  try {
    ort.env.wasm.wasmPaths = new URL(
      "assets/vendor/onnxruntime/",
      document.baseURI,
    ).href;
    ort.env.wasm.numThreads = 1;
    await model.load();
    setStatus("Model ready · runs on this device", "ready");
    setBusy(false);
    applyProfile("routine");
    renderComparisons();
    form.requestSubmit();
  } catch (error) {
    setStatus("Model unavailable · refresh to retry", "error");
    submitButton.disabled = true;
    document.querySelector("[data-risk-copy]").textContent = error.message;
  }
};

start();
