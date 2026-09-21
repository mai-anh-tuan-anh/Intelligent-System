const state = { metrics: [], charts: {} };

const titles = {
  dashboard: ["AI Model Demonstration", "Prediction, probability and evaluation for the Assignment 04 models."],
  mnist: ["MNIST Demo", "CNN classification with grayscale handwritten digits."],
  cifar: ["CIFAR-10 Demo", "CNN classification with RGB images."],
  diabetes: ["Diabetes Demo", "MLP classification for tabular health indicators."],
  comparison: ["Framework Comparison", "Same learning idea, different programming abstraction."]
};

function pct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "-";
}

function showView(id) {
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  document.querySelectorAll(".nav").forEach((nav) => nav.classList.remove("active"));

  document.getElementById(id).classList.add("active");
  document.querySelector(`.nav[data-view="${id}"]`).classList.add("active");
  document.getElementById("title").textContent = titles[id][0];
  document.getElementById("subtitle").textContent = titles[id][1];

  if (id === "mnist") {
    drawDatasetChart("MNIST", "mnist-chart");
    loadCM("MNIST", document.getElementById("mnist-cf").value, "mnist-cm");
  }

  if (id === "cifar") {
    drawDatasetChart("CIFAR-10", "cifar-chart");
    loadCM("CIFAR-10", document.getElementById("cifar-cf").value, "cifar-cm");
  }

  if (id === "diabetes") {
    drawDatasetChart("Diabetes", "diabetes-chart");
    loadCM("Diabetes", document.getElementById("diabetes-cf").value, "diabetes-cm");
  }

  if (id === "comparison") {
    drawComparison();
  }
}

document.querySelectorAll(".nav").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

document.querySelectorAll("[data-goto]").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.goto));
});

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    document.getElementById("status").textContent = data.models_ready
      ? "Models ready"
      : "Models missing";
  } catch {
    document.getElementById("status").textContent = "Backend unavailable";
  }
}

async function loadMetrics() {
  const response = await fetch("/api/metrics");
  const data = await response.json();
  state.metrics = data.rows || [];

  const body = document.querySelector("#overview tbody");
  body.innerHTML = state.metrics.map((row) => `
    <tr>
      <td>${row.dataset}</td>
      <td>${row.framework}</td>
      <td>${pct(row.accuracy)}</td>
      <td>${pct(row.f1_macro)}</td>
      <td>${row.parameters ? Number(row.parameters).toLocaleString() : "-"}</td>
      <td>${row.training_time_seconds !== undefined ? Number(row.training_time_seconds).toFixed(2) + " s" : "-"}</td>
    </tr>
  `).join("");

  drawDatasetChart("MNIST", "mnist-chart");
  drawDatasetChart("CIFAR-10", "cifar-chart");
  drawDatasetChart("Diabetes", "diabetes-chart");
  drawComparison();
}

function destroyChart(id) {
  if (state.charts[id]) {
    state.charts[id].destroy();
  }
}

function makeChart(id, labels, datasets, title, percentage = false) {
  destroyChart(id);
  const canvas = document.getElementById(id);
  if (!canvas) return;

  state.charts[id] = new Chart(canvas, {
    type: "bar",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "bottom" },
        title: { display: true, text: title }
      },
      scales: {
        y: {
          beginAtZero: true,
          max: percentage ? 1 : undefined,
          ticks: {
            callback: (value) => percentage ? `${Math.round(value * 100)}%` : value
          }
        }
      }
    }
  });
}

function rowsFor(dataset) {
  return state.metrics.filter((row) => row.dataset === dataset);
}

function drawDatasetChart(dataset, id) {
  const rows = rowsFor(dataset);
  makeChart(
    id,
    rows.map((row) => row.framework),
    [
      { label: "Accuracy", data: rows.map((row) => Number(row.accuracy || 0)) },
      { label: "Macro F1", data: rows.map((row) => Number(row.f1_macro || 0)) }
    ],
    `${dataset} model metrics`,
    true
  );
}

function drawComparison() {
  const labels = state.metrics.map((row) => `${row.dataset} / ${row.framework}`);

  makeChart(
    "accuracy-chart",
    labels,
    [{ label: "Accuracy", data: state.metrics.map((row) => Number(row.accuracy || 0)) }],
    "Accuracy comparison",
    true
  );

  makeChart(
    "f1-chart",
    labels,
    [{ label: "Macro F1", data: state.metrics.map((row) => Number(row.f1_macro || 0)) }],
    "Macro F1 comparison",
    true
  );

  makeChart(
    "parameter-chart",
    labels,
    [{ label: "Parameters", data: state.metrics.map((row) => Number(row.parameters || 0)) }],
    "Parameter count comparison"
  );

  makeChart(
    "time-chart",
    labels,
    [{ label: "Training time (s)", data: state.metrics.map((row) => Number(row.training_time_seconds || 0)) }],
    "Training time comparison"
  );
}

async function loadCM(dataset, framework, targetId) {
  try {
    const response = await fetch(
      `/api/confusion/${encodeURIComponent(dataset)}/${encodeURIComponent(framework)}`
    );
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "No confusion matrix available.");
    }

    let html = "<table class=\"matrix\"><tr><th>True / Pred</th>";
    html += data.labels.map((label) => `<th>${label}</th>`).join("");
    html += "</tr>";

    data.matrix.forEach((row, rowIndex) => {
      html += `<tr><th>${data.labels[rowIndex]}</th>`;
      html += row.map((value) => `<td>${value}</td>`).join("");
      html += "</tr>";
    });

    html += "</table>";
    document.getElementById(targetId).innerHTML = html;
  } catch (error) {
    document.getElementById(targetId).innerHTML =
      `<div style="padding:20px;color:#748094;font-size:12px">${error.message}</div>`;
  }
}

document.getElementById("mnist-cf").addEventListener("change", (event) => {
  loadCM("MNIST", event.target.value, "mnist-cm");
});

document.getElementById("cifar-cf").addEventListener("change", (event) => {
  loadCM("CIFAR-10", event.target.value, "cifar-cm");
});

document.getElementById("diabetes-cf").addEventListener("change", (event) => {
  loadCM("Diabetes", event.target.value, "diabetes-cm");
});

function renderResult(mainId, probabilityId, data) {
  document.getElementById(mainId).innerHTML = `
    <div>
      <div style="color:#748094;font-size:11px">${data.framework}</div>
      <div class="result-label">${data.prediction}</div>
      <div class="result-confidence">Confidence: ${pct(data.confidence)}</div>
    </div>
  `;

  document.getElementById(probabilityId).innerHTML = data.probabilities
    .slice()
    .sort((a, b) => b.probability - a.probability)
    .map((item) => `
      <div class="prob-row">
        <span>${item.class}</span>
        <div class="bar"><div class="fill" style="width:${Math.max(0, Math.min(100, item.probability * 100))}%"></div></div>
        <b>${pct(item.probability)}</b>
      </div>
    `)
    .join("");
}

function showError(id, message) {
  const box = document.getElementById(id);
  box.textContent = message || "";
  box.style.display = message ? "block" : "none";
}

function bindImageDemo(dataset, fileId, previewId, frameworkId, runId, mainId, probabilityId, errorId) {
  const fileInput = document.getElementById(fileId);
  const preview = document.getElementById(previewId);
  const button = document.getElementById(runId);

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    preview.src = URL.createObjectURL(file);
    preview.style.display = "block";
  });

  button.addEventListener("click", async () => {
    showError(errorId, "");

    const file = fileInput.files[0];
    if (!file) {
      showError(errorId, "Choose an image first.");
      return;
    }

    const form = new FormData();
    form.append("dataset", dataset);
    form.append("framework", document.getElementById(frameworkId).value);
    form.append("image", file);

    button.disabled = true;
    button.textContent = "Running...";

    try {
      const response = await fetch("/api/predict/image", {
        method: "POST",
        body: form
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      renderResult(mainId, probabilityId, data);
    } catch (error) {
      showError(errorId, error.message);
    } finally {
      button.disabled = false;
      button.textContent = "Run prediction";
    }
  });
}

bindImageDemo(
  "MNIST", "mnist-file", "mnist-preview", "mnist-fw", "mnist-run",
  "mnist-main", "mnist-prob", "mnist-error"
);

bindImageDemo(
  "CIFAR-10", "cifar-file", "cifar-preview", "cifar-fw", "cifar-run",
  "cifar-main", "cifar-prob", "cifar-error"
);

async function loadDiabetesSchema() {
  const response = await fetch("/api/diabetes/schema");
  const schema = await response.json();
  const container = document.getElementById("diabetes-fields");

  container.innerHTML = schema.features.map((feature) => `
    <div class="feature">
      <label>${feature}</label>
      <input
        id="df-${feature}"
        type="number"
        step="any"
        value="${schema.medians[feature]}"
        min="${schema.minimums[feature]}"
        max="${schema.maximums[feature]}"
      >
      <small>Range: ${schema.minimums[feature]} to ${schema.maximums[feature]}</small>
    </div>
  `).join("");
}

document.getElementById("diabetes-run").addEventListener("click", async () => {
  const button = document.getElementById("diabetes-run");
  showError("diabetes-error", "");

  try {
    const schemaResponse = await fetch("/api/diabetes/schema");
    const schema = await schemaResponse.json();
    const features = {};

    schema.features.forEach((feature) => {
      features[feature] = document.getElementById(`df-${feature}`).value;
    });

    button.disabled = true;
    button.textContent = "Running...";

    const response = await fetch("/api/predict/diabetes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        framework: document.getElementById("diabetes-fw").value,
        features
      })
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    renderResult("diabetes-main", "diabetes-prob", data);
  } catch (error) {
    showError("diabetes-error", error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run prediction";
  }
});

Promise.all([
  loadHealth(),
  loadMetrics(),
  loadDiabetesSchema()
]);
