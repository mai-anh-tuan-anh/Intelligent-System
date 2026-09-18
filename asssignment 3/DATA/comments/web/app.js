let meta = null;

const charts = {};

// ============================================================
// API HELPER
// ============================================================

async function api(path, options = {}) {
  const response = await fetch(path, options);

  if (!response.ok) {
    let message = await response.text();

    try {
      const parsed = JSON.parse(message);

      message = parsed.detail || message;
    } catch (_) {}

    throw new Error(message);
  }

  return response.json();
}

// ============================================================
// FORMATTERS
// ============================================================

function fmt(value, digits = 3) {
  if (value == null || Number.isNaN(Number(value))) {
    return "—";
  }

  return Number(value).toFixed(digits);
}

function pct(value) {
  return (Number(value) * 100).toFixed(1) + "%";
}

// ============================================================
// CHART CLEANUP
// ============================================================

function destroyChart(name) {
  if (charts[name]) {
    charts[name].destroy();

    charts[name] = null;
  }
}

// ============================================================
// TOP-5 ROWS
// ============================================================

function renderTop5(container, items, field) {
  container.innerHTML = items
    .map((item) => {
      const value = Number(item[field] ?? 0);

      return `

                        <div class="top5-row">

                            <div class="rank">
                                ${item.rank}
                            </div>

                            <div
                                class="name"
                                title="${item.category}"
                            >
                                ${item.category}
                            </div>

                            <div class="row-score">
                                ${pct(value)}
                            </div>

                            <div class="mini-track">

                                <span
                                    style="
                                        width:${Math.min(100, value * 100)}%
                                    "
                                ></span>

                            </div>

                        </div>

                    `;
    })
    .join("");
}

// ============================================================
// METRICS TABLE
// ============================================================

function renderMetricsTable(rows) {
  const keys = [
    "Accuracy",

    "Precision_macro",

    "Recall_macro",

    "F1_macro",

    "Balanced_Accuracy",

    "ROC_AUC_OVR_macro",
  ];

  let html =
    "<thead><tr>" +
    "<th>Model</th>" +
    keys.map((key) => `<th>${key.replaceAll("_", " ")}</th>`).join("") +
    "</tr></thead><tbody>";

  for (const row of rows) {
    html += `<tr>`;

    html += `<td><strong>${row.Model}</strong></td>`;

    html += keys.map((key) => `<td>${fmt(row[key])}</td>`).join("");

    html += `</tr>`;
  }

  html += "</tbody>";

  document.getElementById("metricsTable").innerHTML = html;
}

// ============================================================
// CHARTS
// ============================================================

function drawCharts(data) {
  const ml = data.best_ml.top5;

  const dl = data.deep_learning.top5;

  // --------------------------------------------------------
  // TOP 5
  // --------------------------------------------------------

  const labels = [
    ...new Set([
      ...ml.map((item) => item.category),

      ...dl.map((item) => item.category),
    ]),
  ].slice(0, 7);

  const mlMap = Object.fromEntries(
    ml.map((item) => [item.category, item.score]),
  );

  const dlMap = Object.fromEntries(
    dl.map((item) => [item.category, item.probability]),
  );

  destroyChart("top5");

  charts.top5 = new Chart(document.getElementById("top5Chart"), {
    type: "bar",

    data: {
      labels,

      datasets: [
        {
          label: "Best ML",

          data: labels.map((label) => mlMap[label] || 0),

          borderRadius: 8,
        },

        {
          label: "Deep Learning",

          data: labels.map((label) => dlMap[label] || 0),

          borderRadius: 8,
        },
      ],
    },

    options: {
      responsive: true,

      plugins: {
        legend: {
          position: "bottom",
        },
      },

      scales: {
        y: {
          beginAtZero: true,

          max: 1,

          ticks: {
            callback: (value) => Math.round(value * 100) + "%",
          },
        },
      },
    },
  });

  // --------------------------------------------------------
  // CONFIDENCE
  // --------------------------------------------------------

  destroyChart("confidence");

  charts.confidence = new Chart(document.getElementById("confidenceChart"), {
    type: "doughnut",

    data: {
      labels: [
        `ML • ${data.best_ml.prediction}`,

        `DL • ${data.deep_learning.prediction}`,
      ],

      datasets: [
        {
          data: [data.best_ml.top_score, data.deep_learning.top_score],
        },
      ],
    },

    options: {
      cutout: "62%",

      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });

  // --------------------------------------------------------
  // METRICS
  // --------------------------------------------------------

  const mlMetrics = data.model_metrics.best_ml;

  const dlMetrics = data.model_metrics.deep_learning;

  const metricKeys = [
    "Accuracy",

    "Precision_macro",

    "Recall_macro",

    "F1_macro",

    "Balanced_Accuracy",

    "ROC_AUC_OVR_macro",
  ];

  const metricLabels = [
    "Accuracy",

    "Precision",

    "Recall",

    "Macro F1",

    "Balanced Acc.",

    "ROC-AUC",
  ];

  destroyChart("metrics");

  charts.metrics = new Chart(document.getElementById("metricsChart"), {
    type: "bar",

    data: {
      labels: metricLabels,

      datasets: [
        {
          label: "Best ML",

          data: metricKeys.map((key) => mlMetrics[key] ?? 0),

          borderRadius: 8,
        },

        {
          label: "Deep Learning",

          data: metricKeys.map((key) => dlMetrics[key] ?? 0),

          borderRadius: 8,
        },
      ],
    },

    options: {
      responsive: true,

      plugins: {
        legend: {
          position: "bottom",
        },
      },

      scales: {
        y: {
          beginAtZero: true,

          max: 1,
        },
      },
    },
  });

  // --------------------------------------------------------
  // RADAR
  // --------------------------------------------------------

  destroyChart("radar");

  charts.radar = new Chart(document.getElementById("radarChart"), {
    type: "radar",

    data: {
      labels: metricLabels,

      datasets: [
        {
          label: "Best ML",

          data: metricKeys.map((key) => mlMetrics[key] ?? 0),

          fill: true,
        },

        {
          label: "Deep Learning",

          data: metricKeys.map((key) => dlMetrics[key] ?? 0),

          fill: true,
        },
      ],
    },

    options: {
      responsive: true,

      scales: {
        r: {
          beginAtZero: true,

          max: 1,

          ticks: {
            stepSize: 0.2,
          },
        },
      },

      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}

// ============================================================
// INITIALIZE DASHBOARD
// ============================================================

async function init() {
  try {
    const health = await api("/health");

    const healthElement = document.getElementById("health");

    healthElement.textContent = `● API online • ${health.classes} classes`;

    meta = await api("/metadata");

    document.getElementById("classCount").textContent = meta.classes.length;

    document.getElementById("bestMlName").textContent = meta.best_ml_model;

    document.getElementById("bestMlF1").textContent = fmt(
      meta.best_ml_metrics.F1_macro,
    );

    document.getElementById("dlF1").textContent = fmt(
      meta.deep_learning_metrics.F1_macro,
    );

    renderMetricsTable(meta.comparison);
  } catch (error) {
    const healthElement = document.getElementById("health");

    healthElement.textContent = "● API unavailable";

    healthElement.classList.add("offline");

    console.error(error);
  }
}

// ============================================================
// RUN PREDICTION
// ============================================================

async function runPrediction() {
  const title = document.getElementById("title").value.trim();

  const content = document.getElementById("content").value.trim();

  const rating = Number(document.getElementById("rating").value);

  if (!title && !content) {
    alert("Please enter a review title or review content.");

    return;
  }

  const button = document.getElementById("predictBtn");

  button.disabled = true;

  button.textContent = "Analyzing...";

  try {
    const data = await api("/predict", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        review_title: title,

        review_content: content,

        rating,
      }),
    });

    // ----------------------------------------------------
    // SHOW RESULTS
    // ----------------------------------------------------

    document.getElementById("results").classList.remove("hidden");

    // ----------------------------------------------------
    // ML
    // ----------------------------------------------------

    document.getElementById("mlModelName").textContent = data.best_ml.model;

    document.getElementById("mlPrediction").textContent =
      data.best_ml.prediction;

    document.getElementById("mlScoreNote").textContent =
      data.best_ml.score_note;

    document.getElementById("mlScore").textContent = pct(
      data.best_ml.top_score,
    );

    document.getElementById("mlBar").style.width = `${
      data.best_ml.top_score * 100
    }%`;

    // ----------------------------------------------------
    // DL
    // ----------------------------------------------------

    document.getElementById("dlPrediction").textContent =
      data.deep_learning.prediction;

    document.getElementById("dlScoreNote").textContent =
      data.deep_learning.score_note;

    document.getElementById("dlScore").textContent = pct(
      data.deep_learning.top_score,
    );

    document.getElementById("dlBar").style.width = `${
      data.deep_learning.top_score * 100
    }%`;

    // ----------------------------------------------------
    // TOP 5
    // ----------------------------------------------------

    renderTop5(
      document.getElementById("mlTop5"),

      data.best_ml.top5,

      "score",
    );

    renderTop5(
      document.getElementById("dlTop5"),

      data.deep_learning.top5,

      "probability",
    );

    // ----------------------------------------------------
    // AGREEMENT
    // ----------------------------------------------------

    const agreement = document.getElementById("agreement");

    agreement.textContent = data.agreement
      ? "✓ Models agree"
      : "⚠ Models disagree";

    agreement.classList.toggle("agree", data.agreement);

    agreement.classList.toggle("disagree", !data.agreement);

    // ----------------------------------------------------
    // DIAGNOSTICS
    // ----------------------------------------------------

    document.getElementById("agreementValue").textContent = data.agreement
      ? "YES"
      : "NO";

    document.getElementById("diagMl").textContent = pct(data.best_ml.top_score);

    document.getElementById("diagDl").textContent = pct(
      data.deep_learning.top_score,
    );

    document.getElementById("diagGap").textContent = pct(
      Math.abs(data.best_ml.top_score - data.deep_learning.top_score),
    );

    const diagnosticText = document.getElementById("diagnosticText");

    if (data.agreement) {
      diagnosticText.textContent =
        `Both models selected "${data.best_ml.prediction}". ` +
        `This is a useful consistency signal, ` +
        `although their score meanings are different.`;
    } else {
      diagnosticText.textContent =
        `The models disagree. ` +
        `Best ML selected "${data.best_ml.prediction}", ` +
        `while Deep Learning selected ` +
        `"${data.deep_learning.prediction}". ` +
        `Review the Top-5 candidates before drawing conclusions.`;
    }

    // ----------------------------------------------------
    // TABLE + CHARTS
    // ----------------------------------------------------

    renderMetricsTable(meta.comparison);

    drawCharts(data);

    document.getElementById("results").scrollIntoView({
      behavior: "smooth",
    });
  } catch (error) {
    alert(`Prediction failed: ${error.message}`);
  } finally {
    button.disabled = false;

    button.textContent = "Analyze interest";
  }
}

// ============================================================
// UI EVENTS
// ============================================================

document.getElementById("rating").addEventListener("input", (event) => {
  document.getElementById("ratingValue").textContent = event.target.value;
});

document.getElementById("predictBtn").addEventListener("click", runPrediction);

document.getElementById("sampleBtn").addEventListener("click", () => {
  document.getElementById("title").value =
    "Excellent battery and very good quality";

  document.getElementById("content").value =
    "I am really happy with this product. " +
    "The battery lasts a long time, " +
    "the quality feels premium, " +
    "and delivery was faster than expected.";

  document.getElementById("rating").value = "5";

  document.getElementById("ratingValue").textContent = "5";
});

// ============================================================
// START
// ============================================================

init();
