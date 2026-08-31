// ============================================================
// script.js
// Olist Customer Intelligence Web Client
// ============================================================

const API_BASE = window.location.origin;

// ============================================================
// DOM
// ============================================================

const form = document.getElementById("predictionForm");

const requiredFieldsContainer = document.getElementById("requiredFields");

const otherFieldsContainer = document.getElementById("otherFields");

const predictButton = document.getElementById("predictButton");

const sampleButton = document.getElementById("sampleButton");

const clearButton = document.getElementById("clearButton");

const apiStatus = document.getElementById("apiStatus");

const modelName = document.getElementById("modelName");

const representation = document.getElementById("representation");

const totalFeatureCount = document.getElementById("totalFeatureCount");

const userInputCount = document.getElementById("userInputCount");

const resultSection = document.getElementById("resultSection");

const errorSection = document.getElementById("errorSection");

const errorMessage = document.getElementById("errorMessage");

const behaviorResult = document.getElementById("behaviorResult");

const behaviorConfidence = document.getElementById("behaviorConfidence");

const behaviorConfidenceBar = document.getElementById("behaviorConfidenceBar");

const interestResult = document.getElementById("interestResult");

const interestConfidence = document.getElementById("interestConfidence");

const interestConfidenceBar = document.getElementById("interestConfidenceBar");

const probabilityChart = document.getElementById("probabilityChart");

const profileGrid = document.getElementById("profileGrid");

const productGrid = document.getElementById("productGrid");

const recommendationSubtitle = document.getElementById(
  "recommendationSubtitle",
);

// ============================================================
// STATE
// ============================================================

let metadata = null;

// ============================================================
// FORMATTERS
// ============================================================

function prettyName(name) {
  return String(name)
    .replaceAll("__", " ")

    .replaceAll("_", " ")

    .replace(/\s+/g, " ")

    .trim()

    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatPercentage(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }

  return (Number(value) * 100).toFixed(1) + "%";
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "N/A";
  }

  return Number(value).toLocaleString("en-US", {
    maximumFractionDigits: 2,
  });
}

// ============================================================
// ERROR
// ============================================================

function showError(message) {
  errorMessage.textContent = message;

  errorSection.classList.remove("hidden");
}

function clearError() {
  errorMessage.textContent = "";

  errorSection.classList.add("hidden");
}

// ============================================================
// CREATE FIELD
// ============================================================

function createField(feature, type, required = false) {
  const wrapper = document.createElement("div");

  wrapper.className = "field";

  if (type === "text") {
    wrapper.classList.add("full");
  }

  const labelRow = document.createElement("div");

  labelRow.className = "label-row";

  const label = document.createElement("label");

  label.textContent = prettyName(feature);

  if (required) {
    const requiredMark = document.createElement("span");

    requiredMark.className = "required-mark";

    requiredMark.textContent = "*";

    label.appendChild(requiredMark);
  }

  labelRow.appendChild(label);

  let input;

  if (type === "text") {
    input = document.createElement("textarea");

    input.placeholder = "Enter review text...";
  } else {
    input = document.createElement("input");

    input.type = type === "categorical" ? "text" : "number";

    if (type !== "categorical") {
      input.step = "any";
    }

    input.placeholder = "Enter value";
  }

  input.name = feature;

  input.id = `field-${feature}`;

  if (required) {
    input.required = true;
  }

  wrapper.appendChild(labelRow);

  wrapper.appendChild(input);

  return wrapper;
}

// ============================================================
// BUILD FORM
// ============================================================

function buildForm(data) {
  requiredFieldsContainer.innerHTML = "";

  otherFieldsContainer.innerHTML = "";

  const required = data.required_inputs;

  const other = data.other_inputs;

  if (!required || !other) {
    showError("Input configuration is unavailable.");

    return;
  }

  // --------------------------------------------------------
  // Required numerical
  // --------------------------------------------------------

  required.numerical.forEach((feature) => {
    requiredFieldsContainer.appendChild(
      createField(feature, "numerical", true),
    );
  });

  // --------------------------------------------------------
  // Required categorical
  // --------------------------------------------------------

  required.categorical.forEach((feature) => {
    requiredFieldsContainer.appendChild(
      createField(feature, "categorical", true),
    );
  });

  // --------------------------------------------------------
  // Required text
  // --------------------------------------------------------

  required.text.forEach((feature) => {
    requiredFieldsContainer.appendChild(createField(feature, "text", true));
  });

  // --------------------------------------------------------
  // Other numerical
  // --------------------------------------------------------

  other.numerical.forEach((feature) => {
    otherFieldsContainer.appendChild(createField(feature, "numerical", false));
  });

  // --------------------------------------------------------
  // Other categorical
  // --------------------------------------------------------

  other.categorical.forEach((feature) => {
    otherFieldsContainer.appendChild(
      createField(feature, "categorical", false),
    );
  });

  // --------------------------------------------------------
  // Other text
  // --------------------------------------------------------

  other.text.forEach((feature) => {
    otherFieldsContainer.appendChild(createField(feature, "text", false));
  });

  userInputCount.textContent =
    required.numerical.length +
    required.categorical.length +
    required.text.length +
    other.numerical.length +
    other.categorical.length +
    other.text.length;
}

// ============================================================
// LOAD METADATA
// ============================================================

async function loadMetadata() {
  try {
    const response = await fetch(`${API_BASE}/metadata`);

    if (!response.ok) {
      throw new Error("Cannot load model metadata.");
    }

    metadata = await response.json();

    apiStatus.textContent = "API Online";

    apiStatus.className = "status online";

    modelName.textContent = metadata.model || "N/A";

    representation.textContent = metadata.representation || "N/A";

    const totalFeatures =
      metadata.numerical_features.length +
      metadata.categorical_features.length +
      metadata.text_features.length;

    totalFeatureCount.textContent = totalFeatures;

    buildForm(metadata);
  } catch (error) {
    apiStatus.textContent = "API Offline";

    apiStatus.className = "status offline";

    showError(error.message);
  }
}

// ============================================================
// LOAD SAMPLE
// ============================================================

async function loadSample() {
  clearError();

  try {
    sampleButton.disabled = true;

    sampleButton.textContent = "Loading Sample...";

    const response = await fetch(`${API_BASE}/sample`);

    if (!response.ok) {
      throw new Error("Cannot load sample customer.");
    }

    const result = await response.json();

    const sample = result.features || {};

    Object.entries(sample).forEach(([feature, value]) => {
      const input = form.querySelector(`[name="${feature}"]`);

      if (!input) {
        return;
      }

      input.value = value === null || value === undefined ? "" : value;
    });
  } catch (error) {
    showError(error.message);
  } finally {
    sampleButton.disabled = false;

    sampleButton.textContent = "Load Sample Customer";
  }
}

// ============================================================
// CLEAR
// ============================================================

function clearForm() {
  form.reset();

  resultSection.classList.add("hidden");

  clearError();
}

// ============================================================
// COLLECT INPUT
// ============================================================

function collectFormData() {
  const fields = form.querySelectorAll("input, textarea");

  const payload = {};

  fields.forEach((field) => {
    const name = field.name;

    const value = field.value.trim();

    if (field.type === "number") {
      payload[name] = value === "" ? null : Number(value);
    } else {
      payload[name] = value === "" ? null : value;
    }
  });

  return payload;
}

// ============================================================
// CONFIDENCE
// ============================================================

function renderConfidence(labelElement, barElement, value) {
  if (value === null || value === undefined) {
    labelElement.textContent = "Confidence unavailable";

    barElement.style.width = "0%";

    return;
  }

  const percentage = Number(value) * 100;

  labelElement.textContent = `Confidence: ${percentage.toFixed(1)}%`;

  barElement.style.width = `${percentage}%`;
}

// ============================================================
// PROBABILITY DISTRIBUTION
// ============================================================

function renderProbabilities(probabilities) {
  probabilityChart.innerHTML = "";

  if (!probabilities || Object.keys(probabilities).length === 0) {
    probabilityChart.innerHTML = `
            <p>
                Probability information is unavailable.
            </p>
            `;

    return;
  }

  const entries = Object.entries(probabilities).sort(
    (first, second) => second[1] - first[1],
  );

  entries.forEach(([label, value]) => {
    const row = document.createElement("div");

    row.className = "probability-row";

    const labelElement = document.createElement("div");

    labelElement.className = "probability-label";

    labelElement.textContent = label;

    const track = document.createElement("div");

    track.className = "probability-track";

    const fill = document.createElement("div");

    fill.className = "probability-fill";

    fill.style.width = `${Number(value) * 100}%`;

    const valueElement = document.createElement("div");

    valueElement.className = "probability-value";

    valueElement.textContent = formatPercentage(value);

    track.appendChild(fill);

    row.appendChild(labelElement);

    row.appendChild(track);

    row.appendChild(valueElement);

    probabilityChart.appendChild(row);
  });
}

// ============================================================
// CUSTOMER PROFILE
// ============================================================

function renderProfile(profile) {
  profileGrid.innerHTML = "";

  if (!profile || Object.keys(profile).length === 0) {
    profileGrid.innerHTML = `
            <p>
                No customer profile information available.
            </p>
            `;

    return;
  }

  const order = [
    "prior_order_count",

    "recency_days",

    "total_spending",

    "average_order_value",

    "total_items",

    "average_review_score",

    "customer_state",

    "dominant_payment_type",

    "review_text",
  ];

  order.forEach((key) => {
    if (!Object.prototype.hasOwnProperty.call(profile, key)) {
      return;
    }

    const item = document.createElement("div");

    item.className = "profile-item";

    const label = document.createElement("span");

    label.textContent = prettyName(key);

    const value = document.createElement("strong");

    if (key === "review_text") {
      value.textContent = profile[key];
    } else {
      value.textContent = formatNumber(profile[key]);
    }

    item.appendChild(label);

    item.appendChild(value);

    profileGrid.appendChild(item);
  });
}

// ============================================================
// PRODUCTS
// ============================================================

function renderProducts(products, predictedInterest) {
  productGrid.innerHTML = "";

  if (predictedInterest) {
    recommendationSubtitle.textContent = `Top products for ${predictedInterest}.`;
  } else {
    recommendationSubtitle.textContent =
      "Products related to the predicted customer interest.";
  }

  if (!Array.isArray(products) || products.length === 0) {
    productGrid.innerHTML = `
            <p>
                No products were found in the predicted category.
            </p>
            `;

    return;
  }

  products.forEach((product, index) => {
    const card = document.createElement("div");

    card.className = "product-card";

    card.innerHTML = `
                <div class="product-rank">
                    ${index + 1}
                </div>

                <h3>
                    ${product.product_id}
                </h3>

                <p>
                    Category:
                    ${product.category}
                </p>

                <p>
                    Average price:
                    ${formatNumber(product.price)}
                </p>

                <p>
                    Rating:
                    ${formatNumber(product.rating)}
                </p>

                <p>
                    Sold units:
                    ${formatNumber(product.sold_units)}
                </p>

                <div class="product-score">
                    Recommendation score:
                    ${Number(product.recommendation_score ?? 0).toFixed(3)}
                </div>
            `;

    productGrid.appendChild(card);
  });
}

// ============================================================
// COMPLETE RESULT
// ============================================================

function renderResult(result) {
  behaviorResult.textContent = prettyName(result.predicted_behavior);

  interestResult.textContent = result.predicted_interest;

  renderConfidence(
    behaviorConfidence,
    behaviorConfidenceBar,
    result.behavior_confidence,
  );

  renderConfidence(
    interestConfidence,
    interestConfidenceBar,
    result.interest_confidence,
  );

  renderProbabilities(result.interest_probabilities);

  renderProfile(result.customer_profile);

  renderProducts(result.recommended_products, result.predicted_interest);

  resultSection.classList.remove("hidden");

  resultSection.scrollIntoView({
    behavior: "smooth",
  });
}

// ============================================================
// SUBMIT
// ============================================================

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  clearError();

  predictButton.disabled = true;

  predictButton.textContent = "Predicting...";

  try {
    const payload = collectFormData();

    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(payload),
    });

    let result;

    try {
      result = await response.json();
    } catch (error) {
      throw new Error("Server returned invalid JSON.");
    }

    if (!response.ok) {
      if (
        result.detail &&
        typeof result.detail === "object" &&
        result.detail.missing_fields
      ) {
        throw new Error(
          "Missing required inputs: " +
            result.detail.missing_fields.map(prettyName).join(", "),
        );
      }

      throw new Error(result.detail || "Prediction failed.");
    }

    renderResult(result);
  } catch (error) {
    showError(error.message);
  } finally {
    predictButton.disabled = false;

    predictButton.textContent = "Predict Customer";
  }
});

// ============================================================
// BUTTON EVENTS
// ============================================================

sampleButton.addEventListener("click", loadSample);

clearButton.addEventListener("click", clearForm);

// ============================================================
// INITIALIZE
// ============================================================

loadMetadata();
