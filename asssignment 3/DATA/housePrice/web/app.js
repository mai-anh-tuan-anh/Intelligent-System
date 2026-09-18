"use strict";

/* ============================================================
   STATE
   ============================================================ */

let metadata = null;
let schema = null;
let evaluation = null;

let tableRows = [];

/* ============================================================
   HELPERS
   ============================================================ */

const $ = (selector) => document.querySelector(selector);

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }

  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: digits,
  });
}

function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }

  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: 0,
  });
}

function formatPercent(value, digits = 2) {
  if (value === null || value === undefined) {
    return "—";
  }

  return Number(value).toFixed(digits) + "%";
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")

    .replaceAll("<", "&lt;")

    .replaceAll(">", "&gt;")

    .replaceAll('"', "&quot;")

    .replaceAll("'", "&#039;");
}

/* ============================================================
   API
   ============================================================ */

async function getJSON(url, options = {}) {
  const response = await fetch(url, options);

  if (!response.ok) {
    let message = `HTTP ${response.status}`;

    try {
      const body = await response.json();

      if (body.error) {
        message = body.error;
      }
    } catch (_) {}

    throw new Error(message);
  }

  return response.json();
}

/* ============================================================
   TOAST
   ============================================================ */

function showToast(message) {
  const element = $("#toast");

  element.textContent = message;

  element.classList.add("show");

  clearTimeout(showToast.timer);

  showToast.timer = setTimeout(() => {
    element.classList.remove("show");
  }, 3000);
}

/* ============================================================
   NAVIGATION
   ============================================================ */

document.querySelectorAll(".nav-button").forEach((button) => {
  button.addEventListener("click", () => {
    const page = button.dataset.page;

    document.querySelectorAll(".nav-button").forEach((item) => {
      item.classList.toggle("active", item === button);
    });

    document.querySelectorAll(".page").forEach((section) => {
      section.classList.toggle("active", section.id === page);
    });
  });
});

/* ============================================================
   STATUS
   ============================================================ */

function updateStatus(online, text) {
  $("#status-text").textContent = text;

  $("#status-dot").classList.toggle("online", online);
}

/* ============================================================
   PLOTLY
   ============================================================ */

function baseLayout() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",

    plot_bgcolor: "rgba(0,0,0,0)",

    font: {
      family: "Inter, Arial, sans-serif",

      color: "#344054",
    },

    margin: {
      l: 65,

      r: 20,

      t: 20,

      b: 60,
    },

    xaxis: {
      gridcolor: "#eef2f6",
    },

    yaxis: {
      gridcolor: "#eef2f6",
    },

    legend: {
      orientation: "h",

      y: -0.18,
    },
  };
}

const plotConfig = {
  responsive: true,

  displaylogo: false,
};

/* ============================================================
   DASHBOARD
   ============================================================ */

function renderDashboard() {
  const ml = metadata.best_machine_learning;

  const dl = metadata.deep_learning;

  const mlMetrics = evaluation.metrics.find((item) => item.model === ml.model);

  const dlMetrics = evaluation.metrics.find((item) => item.model === dl.model);

  /* HERO */

  $("#hero-ml-name").textContent = ml.model;

  $("#hero-dl-name").textContent = dl.model;

  $("#hero-ml-rmse").textContent = formatNumber(mlMetrics.RMSE, 0);

  $("#hero-dl-rmse").textContent = formatNumber(dlMetrics.RMSE, 0);

  $("#hero-ml-r2").textContent = Number(mlMetrics.R2).toFixed(4);

  $("#hero-dl-r2").textContent = Number(dlMetrics.R2).toFixed(4);

  /* METRICS */

  const cards = [
    {
      name: "MAE",

      ml: mlMetrics.MAE,

      dl: dlMetrics.MAE,

      format: (value) => formatNumber(value, 0),
    },

    {
      name: "RMSE",

      ml: mlMetrics.RMSE,

      dl: dlMetrics.RMSE,

      format: (value) => formatNumber(value, 0),
    },

    {
      name: "R²",

      ml: mlMetrics.R2,

      dl: dlMetrics.R2,

      format: (value) => Number(value).toFixed(4),
    },

    {
      name: "MAPE",

      ml: mlMetrics["MAPE (%)"],

      dl: dlMetrics["MAPE (%)"],

      format: (value) => formatPercent(value, 2),
    },
  ];

  $("#metric-grid").innerHTML = cards
    .map(
      (card) => `

                    <div class="metric-card">

                        <div class="metric-name">
                            ${card.name}
                        </div>

                        <div class="metric-value">
                            ${card.format(card.ml)}
                        </div>

                        <div class="metric-model">
                            Best ML
                        </div>

                        <div class="metric-value">
                            ${card.format(card.dl)}
                        </div>

                        <div class="metric-model">
                            Deep Learning
                        </div>

                    </div>

                `,
    )
    .join("");

  renderMetricChart();

  renderScatterChart();

  renderQuartileChart();
}

/* ============================================================
   METRIC CHART
   ============================================================ */

function renderMetricChart() {
  const ml = evaluation.metrics[0];

  const dl = evaluation.metrics[1];

  const metricNames = ["MAE", "MSE", "RMSE", "MAPE (%)"];

  Plotly.newPlot(
    "metric-chart",

    [
      {
        x: metricNames,

        y: metricNames.map((name) => ml[name]),

        type: "bar",

        name: ml.model,
      },

      {
        x: metricNames,

        y: metricNames.map((name) => dl[name]),

        type: "bar",

        name: dl.model,
      },
    ],

    {
      ...baseLayout(),

      barmode: "group",

      yaxis: {
        title: "Metric value",

        gridcolor: "#eef2f6",
      },
    },

    plotConfig,
  );
}

/* ============================================================
   SCATTER
   ============================================================ */

function renderScatterChart() {
  const mlName = evaluation.metrics[0].model;

  const dlName = evaluation.metrics[1].model;

  const actual = evaluation.scatter.map((item) => item.actual);

  const ml = evaluation.scatter.map((item) => item.ml);

  const dl = evaluation.scatter.map((item) => item.dl);

  const allValues = [...actual, ...ml, ...dl];

  const min = Math.min(...allValues);

  const max = Math.max(...allValues);

  Plotly.newPlot(
    "scatter-chart",

    [
      {
        x: actual,

        y: ml,

        mode: "markers",

        type: "scattergl",

        name: mlName,

        marker: {
          size: 5,

          opacity: 0.45,
        },
      },

      {
        x: actual,

        y: dl,

        mode: "markers",

        type: "scattergl",

        name: dlName,

        marker: {
          size: 5,

          opacity: 0.4,
        },
      },

      {
        x: [min, max],

        y: [min, max],

        mode: "lines",

        name: "Perfect prediction",

        line: {
          dash: "dash",

          width: 2,
        },
      },
    ],

    {
      ...baseLayout(),

      xaxis: {
        title: "Actual resale price",

        gridcolor: "#eef2f6",
      },

      yaxis: {
        title: "Predicted resale price",

        gridcolor: "#eef2f6",
      },
    },

    plotConfig,
  );
}

/* ============================================================
   QUARTILE
   ============================================================ */

function renderQuartileChart() {
  const ml = evaluation.metrics[0].model;

  const dl = evaluation.metrics[1].model;

  const groups = evaluation.quartile_mae.map((row) => row.group);

  const mlValues = evaluation.quartile_mae.map((row) => row[ml]);

  const dlValues = evaluation.quartile_mae.map((row) => row[dl]);

  Plotly.newPlot(
    "quartile-chart",

    [
      {
        x: groups,

        y: mlValues,

        type: "bar",

        name: ml,
      },

      {
        x: groups,

        y: dlValues,

        type: "bar",

        name: dl,
      },
    ],

    {
      ...baseLayout(),

      barmode: "group",

      yaxis: {
        title: "Mean absolute error",

        gridcolor: "#eef2f6",
      },
    },

    plotConfig,
  );
}

/* ============================================================
   RESIDUAL
   ============================================================ */

function renderResidualChart() {
  Plotly.newPlot(
    "residual-chart",

    [
      {
        x: evaluation.residuals.ml,

        type: "histogram",

        nbinsx: 40,

        opacity: 0.6,

        name: metadata.best_machine_learning.model,
      },

      {
        x: evaluation.residuals.dl,

        type: "histogram",

        nbinsx: 40,

        opacity: 0.6,

        name: metadata.deep_learning.model,
      },
    ],

    {
      ...baseLayout(),

      barmode: "overlay",

      xaxis: {
        title: "Residual (Actual - Predicted)",
      },

      yaxis: {
        title: "Count",
      },
    },

    plotConfig,
  );
}

/* ============================================================
   FORM
   ============================================================ */

function prettyName(text) {
  return String(text)
    .replaceAll("_", " ")

    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function numericField(field) {
  return `

        <div class="field">

            <label for="${field.name}">
                ${escapeHTML(prettyName(field.name))}
            </label>

            <input
                id="${field.name}"
                name="${field.name}"
                type="number"
                step="any"
                placeholder="Median: ${formatNumber(field.median, 2)}"
            >

        </div>

    `;
}

function lowCategoryField(field) {
  const options = field.options
    .map(
      (option) => `

                    <option value="${escapeHTML(option)}">

                        ${escapeHTML(option)}

                    </option>

                `,
    )
    .join("");

  return `

        <div class="field">

            <label for="${field.name}">
                ${escapeHTML(prettyName(field.name))}
            </label>

            <select
                id="${field.name}"
                name="${field.name}"
            >

                <option value="">
                    Use training default
                </option>

                ${options}

            </select>

        </div>

    `;
}

function highCategoryField(field) {
  const listId = `${field.name}-list`;

  const options = field.options
    .map(
      (option) => `

                    <option value="${escapeHTML(option)}"></option>

                `,
    )
    .join("");

  return `

        <div class="field">

            <label for="${field.name}">
                ${escapeHTML(prettyName(field.name))}
            </label>

            <input
                id="${field.name}"
                name="${field.name}"
                type="text"
                list="${listId}"
                placeholder="Enter or select..."
            >

            <datalist id="${listId}">
                ${options}
            </datalist>

        </div>

    `;
}

function renderForm() {
  let html = "";

  schema.numeric.forEach((field) => {
    html += numericField(field);
  });

  schema.categorical_low.forEach((field) => {
    html += lowCategoryField(field);
  });

  schema.categorical_high.forEach((field) => {
    html += highCategoryField(field);
  });

  $("#form-grid").innerHTML = html;
}

/* ============================================================
   FORM DATA
   ============================================================ */

function collectFormData() {
  const payload = {};

  [
    ...schema.numeric,

    ...schema.categorical_low,

    ...schema.categorical_high,
  ].forEach((field) => {
    const element = document.getElementById(field.name);

    if (element) {
      payload[field.name] = element.value;
    }
  });

  return payload;
}

/* ============================================================
   EXAMPLE
   ============================================================ */

function loadExample() {
  const example = {
    floor_area_sqm: 70,

    lease_commence_date: 1985,

    transaction_year: 2020,

    transaction_month: 6,

    remaining_lease_years: 64,

    storey_mid: 8,
  };

  Object.entries(example).forEach(([key, value]) => {
    const element = document.getElementById(key);

    if (element) {
      element.value = value;
    }
  });

  showToast("Example values loaded.");
}

$("#example-button").addEventListener("click", loadExample);

/* ============================================================
   PREDICTION
   ============================================================ */

$("#predict-button").addEventListener("click", async () => {
  const button = $("#predict-button");

  button.disabled = true;

  button.textContent = "Predicting...";

  try {
    const result = await getJSON("/api/predict", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(collectFormData()),
    });

    $("#result-ml-name").textContent = result.best_machine_learning.model;

    $("#result-ml-price").textContent = formatMoney(
      result.best_machine_learning.prediction,
    );

    $("#result-dl-name").textContent = result.deep_learning.model;

    $("#result-dl-price").textContent = formatMoney(
      result.deep_learning.prediction,
    );

    $("#agreement-value").textContent = formatPercent(
      result.comparison.agreement,
      1,
    );

    $("#agreement-detail").textContent = `Difference: ${formatMoney(
      result.comparison.difference,
    )} | Average: ${formatMoney(result.comparison.average)}`;

    showToast("Prediction completed.");
  } catch (error) {
    console.error(error);

    showToast(error.message);
  } finally {
    button.disabled = false;

    button.textContent = "Predict with Both Models";
  }
});

/* ============================================================
   MODEL LAB
   ============================================================ */

function renderParameters(container, values) {
  container.innerHTML = Object.entries(values || {})

    .filter(([_, value]) => value !== null && value !== undefined)

    .map(
      ([key, value]) => `

                <div class="parameter">

                    <span>
                        ${escapeHTML(key)}
                    </span>

                    <strong>
                        ${escapeHTML(value)}
                    </strong>

                </div>

            `,
    )

    .join("");
}

function renderMiniMetrics(container, metrics) {
  const fields = [
    ["MAE", metrics.MAE],

    ["RMSE", metrics.RMSE],

    ["R²", metrics.R2],

    ["MAPE", metrics["MAPE (%)"]],
  ];

  container.innerHTML = fields
    .map(
      ([name, value]) => `

                    <div class="mini-metric">

                        <span>
                            ${name}
                        </span>

                        <strong>

                            ${
                              name === "R²"
                                ? Number(value).toFixed(4)
                                : name === "MAPE"
                                  ? formatPercent(value, 2)
                                  : formatNumber(value, 0)
                            }

                        </strong>

                    </div>

                `,
    )
    .join("");
}

function renderModelLab() {
  const ml = metadata.best_machine_learning;

  const dl = metadata.deep_learning;

  $("#lab-ml-name").textContent = ml.model;

  $("#lab-dl-name").textContent = dl.model;

  renderParameters(
    $("#lab-ml-parameters"),

    {
      Selection: ml.selection_criterion,

      ...ml.parameters,
    },
  );

  renderMiniMetrics(
    $("#lab-ml-metrics"),

    evaluation.metrics[0],
  );

  const architecture = dl.architecture || {};

  const training = dl.training || {};

  renderParameters(
    $("#lab-dl-parameters"),

    {
      Input: architecture.input_dim,

      "Hidden 1": architecture.hidden1,

      "Hidden 2": architecture.hidden2,

      Output: architecture.output_dim,

      Activation: architecture.activation_hidden,

      Loss: architecture.loss,

      "Learning Rate": training.learning_rate,

      Epochs: training.epochs,

      "Batch Size": training.batch_size,
    },
  );

  renderMiniMetrics(
    $("#lab-dl-metrics"),

    evaluation.metrics[1],
  );

  $("#feature-count").textContent = schema.feature_count;

  $("#low-count").textContent = schema.categorical_low.length;

  $("#high-count").textContent = schema.categorical_high.length;

  renderResidualChart();
}

/* ============================================================
   TABLE
   ============================================================ */

function renderTable(rows) {
  tableRows = rows;

  const query = ($("#table-search").value || "").trim().toLowerCase();

  const filtered = tableRows.filter((row) =>
    Object.values(row).some((value) =>
      String(value).toLowerCase().includes(query),
    ),
  );

  $("#table-body").innerHTML = filtered
    .map(
      (row) => `

                    <tr>

                        <td>
                            ${formatMoney(row.Actual)}
                        </td>

                        <td>
                            ${formatMoney(row.Best_ML_Predicted)}
                        </td>

                        <td>
                            ${formatMoney(row.Deep_Learning_Predicted)}
                        </td>

                        <td>
                            ${formatMoney(row.Best_ML_Absolute_Error)}
                        </td>

                        <td>
                            ${formatMoney(row.Deep_Learning_Absolute_Error)}
                        </td>

                    </tr>

                `,
    )
    .join("");
}

$("#table-search").addEventListener("input", () => renderTable(tableRows));

/* ============================================================
   REFRESH
   ============================================================ */

$("#refresh-button").addEventListener("click", async () => {
  await loadApplication();

  showToast("Dashboard refreshed.");
});

/* ============================================================
   INITIAL LOAD
   ============================================================ */

async function loadApplication() {
  try {
    updateStatus(false, "Loading...");

    const results = await Promise.all([
      getJSON("/api/metadata"),

      getJSON("/api/schema"),

      getJSON("/api/evaluation"),

      getJSON("/api/health"),
    ]);

    metadata = results[0];

    schema = results[1];

    evaluation = results[2];

    updateStatus(true, "API online");

    renderDashboard();

    renderForm();

    renderModelLab();

    renderTable(evaluation.preview);
  } catch (error) {
    console.error(error);

    updateStatus(false, "API error");

    showToast(error.message);
  }
}

loadApplication();
