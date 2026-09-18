let metricChart = null;
let probChart = null;
let radarChart = null;
let metadata = null;

// ============================================================
// API
// ============================================================

async function api(endpoint, options = {}) {
  const response = await fetch(endpoint, options);

  if (!response.ok) {
    const text = await response.text();

    try {
      const data = JSON.parse(text);

      throw new Error(
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail),
      );
    } catch (parseError) {
      if (parseError instanceof Error && parseError.message !== text) {
        throw parseError;
      }

      throw new Error(text);
    }
  }

  return response.json();
}

// ============================================================
// FORMAT
// ============================================================

function percent(value) {
  return (Number(value) * 100).toFixed(1) + "%";
}

function number(value, digits = 3) {
  return Number(value).toFixed(digits);
}

// ============================================================
// GET INPUT
// ============================================================

const FEATURE_IDS = [
  "Age",
  "BMI",
  "Waist_Circumference_cm",
  "Blood_Glucose",
  "HbA1c",
  "Fasting_Blood_Sugar",
  "Insulin_Level",
  "Blood_Pressure_Systolic",
  "Blood_Pressure_Diastolic",
  "Total_Cholesterol",
  "HDL",
  "Family_History_Diabetes",
  "Hypertension",
  "Heart_Disease",
  "Diet_Quality",
  "Physical_Activity_Level",
];

function getInput() {
  const data = {};

  FEATURE_IDS.forEach((id) => {
    const element = document.getElementById(id);

    if (element.tagName === "SELECT") {
      data[id] = element.value;
    } else {
      data[id] = Number(element.value);
    }
  });

  return data;
}

// ============================================================
// LIVE ENGINEERED FEATURES
// ============================================================

function updateDerivedFeatures() {
  const systolic = Number(
    document.getElementById("Blood_Pressure_Systolic").value,
  );

  const diastolic = Number(
    document.getElementById("Blood_Pressure_Diastolic").value,
  );

  const totalChol = Number(document.getElementById("Total_Cholesterol").value);

  const hdl = Number(document.getElementById("HDL").value);

  const pulse = systolic - diastolic;

  const ratio = hdl !== 0 ? totalChol / hdl : 0;

  document.getElementById("pulsePressure").textContent = Number.isFinite(pulse)
    ? pulse.toFixed(1)
    : "—";

  document.getElementById("cholRatio").textContent = Number.isFinite(ratio)
    ? ratio.toFixed(2)
    : "—";
}

// ============================================================
// PROBABILITY ROWS
// ============================================================

function renderProbabilities(containerId, probabilities) {
  const container = document.getElementById(containerId);

  const entries = Object.entries(probabilities);

  container.innerHTML = entries
    .map(([name, value]) => {
      return `

                        <div class="prob-row">

                            <div class="prob-name">
                                ${name}
                            </div>

                            <div class="prob-track">

                                <span
                                    style="
                                        width:${Number(value) * 100}%
                                    "
                                ></span>

                            </div>

                            <div class="prob-value">
                                ${percent(value)}
                            </div>

                        </div>

                    `;
    })
    .join("");
}

// ============================================================
// METRICS TABLE
// ============================================================

function renderMetrics(data) {
  const ml = data.model_metrics.best_ml;

  const dl = data.model_metrics.deep_learning;

  const rows = [
    ["Accuracy", ml.Accuracy, dl.Accuracy],

    ["Precision (Macro)", ml.Precision_macro, dl.Precision_macro],

    ["Recall (Macro)", ml.Recall_macro, dl.Recall_macro],

    ["F1 (Macro)", ml.F1_macro, dl.F1_macro],

    ["ROC-AUC OVR (Macro)", ml.ROC_AUC_OVR_macro, dl.ROC_AUC_OVR_macro],
  ];

  document.getElementById("metricsBody").innerHTML = rows
    .map(
      (row) => `

                    <tr>

                        <td>
                            <strong>
                                ${row[0]}
                            </strong>
                        </td>

                        <td>
                            ${number(row[1])}
                        </td>

                        <td>
                            ${number(row[2])}
                        </td>

                    </tr>

                `,
    )
    .join("");
}

// ============================================================
// MODEL METRICS CHART
// ============================================================

function drawMetricChart(data) {
  const ml = data.model_metrics.best_ml;

  const dl = data.model_metrics.deep_learning;

  if (metricChart) {
    metricChart.destroy();
  }

  metricChart = new Chart(
    document.getElementById("metricChart"),

    {
      type: "bar",

      data: {
        labels: ["Accuracy", "Precision", "Recall", "Macro F1", "ROC-AUC"],

        datasets: [
          {
            label: "Best ML",

            data: [
              ml.Accuracy,
              ml.Precision_macro,
              ml.Recall_macro,
              ml.F1_macro,
              ml.ROC_AUC_OVR_macro,
            ],
          },

          {
            label: "Deep Learning",

            data: [
              dl.Accuracy,
              dl.Precision_macro,
              dl.Recall_macro,
              dl.F1_macro,
              dl.ROC_AUC_OVR_macro,
            ],
          },
        ],
      },

      options: {
        responsive: true,

        scales: {
          y: {
            beginAtZero: true,

            max: 1,
          },
        },

        plugins: {
          legend: {
            position: "bottom",
          },
        },
      },
    },
  );
}

// ============================================================
// PATIENT PROBABILITY CHART
// ============================================================

function drawProbabilityChart(data) {
  const ml = data.best_ml.probabilities;

  const dl = data.deep_learning.probabilities;

  if (probChart) {
    probChart.destroy();
  }

  probChart = new Chart(
    document.getElementById("probChart"),

    {
      type: "bar",

      data: {
        labels: Object.keys(ml),

        datasets: [
          {
            label: "Best ML",

            data: Object.values(ml),
          },

          {
            label: "Deep Learning",

            data: Object.values(dl),
          },
        ],
      },

      options: {
        responsive: true,

        scales: {
          y: {
            beginAtZero: true,

            max: 1,

            ticks: {
              callback: (value) => Math.round(value * 100) + "%",
            },
          },
        },

        plugins: {
          legend: {
            position: "bottom",
          },
        },
      },
    },
  );
}

// ============================================================
// RADAR CHART
// ============================================================

function drawRadarChart(data) {
  const ml = data.model_metrics.best_ml;

  const dl = data.model_metrics.deep_learning;

  if (radarChart) {
    radarChart.destroy();
  }

  radarChart = new Chart(
    document.getElementById("radarChart"),

    {
      type: "radar",

      data: {
        labels: ["Accuracy", "Precision", "Recall", "Macro F1", "ROC-AUC"],

        datasets: [
          {
            label: "Best ML",

            data: [
              ml.Accuracy,
              ml.Precision_macro,
              ml.Recall_macro,
              ml.F1_macro,
              ml.ROC_AUC_OVR_macro,
            ],
          },

          {
            label: "Deep Learning",

            data: [
              dl.Accuracy,
              dl.Precision_macro,
              dl.Recall_macro,
              dl.F1_macro,
              dl.ROC_AUC_OVR_macro,
            ],
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
    },
  );
}

// ============================================================
// SUMMARY
// ============================================================

function renderSummary(data) {
  const input = data.input;

  const items = [
    ["Age", input.Age],

    ["BMI", input.BMI],

    ["Blood glucose", input.Blood_Glucose],

    ["HbA1c", input.HbA1c],

    [
      "Blood pressure",
      `${input.Blood_Pressure_Systolic}/${input.Blood_Pressure_Diastolic}`,
    ],

    ["Diet quality", input.Diet_Quality],

    ["Physical activity", input.Physical_Activity_Level],

    ["Pulse pressure", number(data.engineered_features.Pulse_Pressure, 1)],

    ["Chol/HDL ratio", number(data.engineered_features.Chol_HDL_Ratio, 2)],
  ];

  document.getElementById("summary").innerHTML = items
    .map(
      ([label, value]) => `

                    <div class="summary-item">

                        <span>
                            ${label}
                        </span>

                        <strong>
                            ${value}
                        </strong>

                    </div>

                `,
    )
    .join("");
}

// ============================================================
// RUN PREDICTION
// ============================================================

async function predict() {
  const payload = getInput();

  // basic validation
  const numericFields = [
    "Age",
    "BMI",
    "Waist_Circumference_cm",
    "Blood_Glucose",
    "HbA1c",
    "Fasting_Blood_Sugar",
    "Insulin_Level",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Total_Cholesterol",
    "HDL",
  ];

  const invalid = numericFields.some((key) => !Number.isFinite(payload[key]));

  if (invalid) {
    alert("Please enter valid numeric values.");

    return;
  }

  const btn = document.getElementById("predictBtn");

  btn.disabled = true;

  btn.textContent = "Running both models...";

  try {
    const data = await api("/predict", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(payload),
    });

    // ==================================================
    // SHOW RESULT SECTION
    // ==================================================

    document.getElementById("results").classList.remove("hidden");

    // ==================================================
    // BEST ML
    // ==================================================

    document.getElementById("mlName").textContent = data.best_ml.model;

    document.getElementById("mlPrediction").textContent =
      data.best_ml.prediction;

    document.getElementById("mlConfidence").textContent = percent(
      data.best_ml.confidence,
    );

    document.getElementById("mlProgress").style.width = `${
      data.best_ml.confidence * 100
    }%`;

    renderProbabilities("mlProbabilities", data.best_ml.probabilities);

    // ==================================================
    // DEEP LEARNING
    // ==================================================

    document.getElementById("dlPrediction").textContent =
      data.deep_learning.prediction;

    document.getElementById("dlConfidence").textContent = percent(
      data.deep_learning.confidence,
    );

    document.getElementById("dlProgress").style.width = `${
      data.deep_learning.confidence * 100
    }%`;

    renderProbabilities("dlProbabilities", data.deep_learning.probabilities);

    // ==================================================
    // AGREEMENT
    // ==================================================

    const agreement = document.getElementById("agreement");

    agreement.textContent = data.agreement
      ? "✓ Models agree"
      : "⚠ Models disagree";

    agreement.style.color = data.agreement ? "var(--green)" : "var(--yellow)";

    // ==================================================
    // CHARTS
    // ==================================================

    renderMetrics(data);

    drawProbabilityChart(data);

    drawMetricChart(data);

    drawRadarChart(data);

    renderSummary(data);

    // scroll
    document.getElementById("results").scrollIntoView({
      behavior: "smooth",
    });
  } catch (error) {
    alert("Prediction failed:\n" + error.message);

    console.error(error);
  } finally {
    btn.disabled = false;

    btn.textContent = "Predict diabetes risk";
  }
}

// ============================================================
// SAMPLE
// ============================================================

document.getElementById("sampleBtn").addEventListener("click", () => {
  document.getElementById("Age").value = 42;

  document.getElementById("BMI").value = 28.5;

  document.getElementById("Waist_Circumference_cm").value = 94;

  document.getElementById("Blood_Glucose").value = 145;

  document.getElementById("HbA1c").value = 7.2;

  document.getElementById("Fasting_Blood_Sugar").value = 135;

  document.getElementById("Insulin_Level").value = 110;

  document.getElementById("Blood_Pressure_Systolic").value = 140;

  document.getElementById("Blood_Pressure_Diastolic").value = 90;

  document.getElementById("Total_Cholesterol").value = 220;

  document.getElementById("HDL").value = 50;

  document.getElementById("Family_History_Diabetes").value = "Yes";

  document.getElementById("Hypertension").value = "Yes";

  document.getElementById("Heart_Disease").value = "No";

  document.getElementById("Diet_Quality").value = "Average";

  document.getElementById("Physical_Activity_Level").value = "Moderate";

  updateDerivedFeatures();
});

// ============================================================
// RESET
// ============================================================

document.getElementById("resetBtn").addEventListener("click", () => {
  document.querySelectorAll("input").forEach((element) => {
    if (element.type === "number") {
      element.value = "";
    }
  });

  document.getElementById("Family_History_Diabetes").value = "Yes";

  document.getElementById("Hypertension").value = "No";

  document.getElementById("Heart_Disease").value = "No";

  document.getElementById("Diet_Quality").value = "Healthy";

  document.getElementById("Physical_Activity_Level").value = "Moderate";

  document.getElementById("results").classList.add("hidden");

  updateDerivedFeatures();
});

// ============================================================
// LIVE EVENTS
// ============================================================

[
  "Blood_Pressure_Systolic",
  "Blood_Pressure_Diastolic",
  "Total_Cholesterol",
  "HDL",
].forEach((id) => {
  document.getElementById(id).addEventListener("input", updateDerivedFeatures);
});

document.getElementById("predictBtn").addEventListener("click", predict);

// ============================================================
// INIT
// ============================================================

async function init() {
  try {
    const health = await api("/health");

    const status = document.getElementById("apiStatus");

    status.textContent = `● API online • ${health.best_ml} + ${
      health.deep_learning
    }`;

    metadata = await api("/metadata");

    // startup derived values
    updateDerivedFeatures();
  } catch (error) {
    const status = document.getElementById("apiStatus");

    status.textContent = "● API unavailable";

    status.classList.add("offline");

    console.error(error);
  }
}

init();
