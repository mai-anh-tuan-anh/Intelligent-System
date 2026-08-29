// ============================================================
// app.js
//
// Dynamic Web Client for Diabetes Prediction
//
// Responsibilities:
//   1. Load feature schema from Flask
//   2. Build input form dynamically
//   3. Fill sample patient data
//   4. Collect user input
//   5. Send POST request
//   6. Display prediction result
// ============================================================

// ============================================================
// GLOBAL VARIABLES
// ============================================================

let featureSchema = [];

let predictionForm;
let formGrid;
let errorBox;
let resultCard;
let predictButton;
let sampleButton;

// ============================================================
// 1. INITIALIZE APPLICATION
// ============================================================

document.addEventListener("DOMContentLoaded", async function () {
  // ----------------------------------------------------
  // Find HTML elements
  // ----------------------------------------------------

  predictionForm = document.getElementById("predictionForm");

  formGrid = document.getElementById("formGrid");

  errorBox = document.getElementById("errorBox");

  resultCard = document.getElementById("resultCard");

  predictButton = document.getElementById("predictButton");

  sampleButton = document.getElementById("sampleButton");

  // ----------------------------------------------------
  // Load the exact feature schema used by the model
  // ----------------------------------------------------

  await loadFeatureSchema();

  // ----------------------------------------------------
  // Register event handlers
  // ----------------------------------------------------

  predictionForm.addEventListener("submit", handlePrediction);

  sampleButton.addEventListener("click", fillSampleData);
});

// ============================================================
// 2. LOAD MODEL FEATURE SCHEMA
// ============================================================

async function loadFeatureSchema() {
  try {
    const response = await fetch("/diabetes/v1/schema");

    if (!response.ok) {
      throw new Error("Unable to load model schema.");
    }

    const data = await response.json();

    featureSchema = data.features;

    // Build form using the actual model schema
    buildForm(featureSchema);

    // Enable buttons after schema loading
    predictButton.disabled = false;

    sampleButton.disabled = false;
  } catch (error) {
    showError(error.message);
  }
}

// ============================================================
// 3. BUILD FORM DYNAMICALLY
// ============================================================

function buildForm(schema) {
  // Remove loading message
  formGrid.innerHTML = "";

  schema.forEach(function (feature) {
    // ------------------------------------------------
    // Create field container
    // ------------------------------------------------

    const field = document.createElement("div");

    field.className = "field";

    // ------------------------------------------------
    // Create label
    // ------------------------------------------------

    const label = document.createElement("label");

    label.textContent = formatFeatureName(feature.name);

    label.htmlFor = `feature-${feature.name}`;

    field.appendChild(label);

    // ------------------------------------------------
    // Categorical feature
    // ------------------------------------------------

    if (feature.type === "category") {
      const select = document.createElement("select");

      select.id = `feature-${feature.name}`;

      select.name = feature.name;

      // Placeholder
      const placeholder = document.createElement("option");

      placeholder.value = "";

      placeholder.textContent = `Select ${formatFeatureName(feature.name)}`;

      placeholder.disabled = true;

      placeholder.selected = true;

      select.appendChild(placeholder);

      // Add actual learned categories
      feature.categories.forEach(function (category) {
        const option = document.createElement("option");

        option.value = category;

        option.textContent = category;

        select.appendChild(option);
      });

      field.appendChild(select);
    }

    // ------------------------------------------------
    // Numerical feature
    // ------------------------------------------------
    else {
      const input = document.createElement("input");

      input.type = "number";

      input.step = "any";

      input.id = `feature-${feature.name}`;

      input.name = feature.name;

      input.required = true;

      field.appendChild(input);
    }

    formGrid.appendChild(field);
  });
}

// ============================================================
// 4. SAMPLE PATIENT DATA
// ============================================================
//
// These values are provided only for demonstration/testing.
// They are not a clinical recommendation.
//
// Values are assigned by feature name so the function works
// with the dynamic schema returned by the API.
// ============================================================

const SAMPLE_VALUES = {
  // Demographic
  age: 50,

  gender: "Male",

  ethnicity: "White",

  education_level: "Bachelor's Degree",

  income_level: "Middle",

  employment_status: "Employed",

  marital_status: "Married",

  // Lifestyle
  smoking_status: "Never",

  alcohol_consumption_per_week: 2,

  physical_activity_minutes_per_week: 150,

  diet_score: 7,

  sleep_hours_per_day: 7,

  screen_time_hours_per_day: 4,

  // Anthropometric
  bmi: 28.5,

  waist_to_hip_ratio: 0.9,

  // Vital signs
  systolic_bp: 125,

  diastolic_bp: 80,

  heart_rate: 72,

  // Lipid profile
  cholesterol_total: 190,

  hdl_cholesterol: 55,

  ldl_cholesterol: 110,

  triglycerides: 140,

  // Glucose / metabolic
  glucose_fasting: 105,

  glucose_postprandial: 145,

  insulin_level: 12,

  hba1c: 5.8,
};

// ============================================================
// 5. FILL SAMPLE DATA
// ============================================================

function fillSampleData() {
  clearError();

  featureSchema.forEach(function (feature) {
    const element = document.getElementById(`feature-${feature.name}`);

    if (!element) {
      return;
    }

    // ------------------------------------------------
    // Check whether we have a predefined sample
    // ------------------------------------------------

    if (Object.prototype.hasOwnProperty.call(SAMPLE_VALUES, feature.name)) {
      const sampleValue = SAMPLE_VALUES[feature.name];

      // ============================================
      // Numerical field
      // ============================================

      if (feature.type === "number") {
        element.value = sampleValue;
      }

      // ============================================
      // Categorical field
      // ============================================
      else {
        // Only use the sample category if it
        // actually exists in the learned schema.

        const matchingOption = Array.from(element.options).find(
          function (option) {
            return option.value === String(sampleValue);
          },
        );

        if (matchingOption) {
          element.value = String(sampleValue);
        }
      }
    } else {
      // ------------------------------------------------
      // No predefined sample:
      // use the first available category for testing.
      // ------------------------------------------------

      if (feature.type === "category") {
        if (feature.categories && feature.categories.length > 0) {
          element.value = feature.categories[0];
        }
      } else {
        // For an unknown numerical feature,
        // use a neutral placeholder value.

        element.value = 0;
      }
    }
  });

  // --------------------------------------------------------
  // Notify user
  // --------------------------------------------------------

  sampleButton.textContent = "Sample Data Loaded";

  setTimeout(function () {
    sampleButton.textContent = "Fill Sample Data";
  }, 1500);
}

// ============================================================
// 6. HANDLE PREDICTION
// ============================================================

async function handlePrediction(event) {
  event.preventDefault();

  clearError();

  predictButton.disabled = true;

  predictButton.textContent = "Predicting...";

  try {
    const data = {};

    // ----------------------------------------------------
    // Collect every model feature
    // ----------------------------------------------------

    featureSchema.forEach(function (feature) {
      const element = document.getElementById(`feature-${feature.name}`);

      if (!element) {
        throw new Error(`Input field not found: ${feature.name}`);
      }

      let value = element.value;

      // ==========================================
      // Numerical feature
      // ==========================================

      if (feature.type === "number") {
        if (value === "") {
          throw new Error(`${formatFeatureName(feature.name)} is required.`);
        }

        value = Number(value);

        if (Number.isNaN(value)) {
          throw new Error(
            `${formatFeatureName(feature.name)} must be a valid number.`,
          );
        }
      }

      // ==========================================
      // Categorical feature
      // ==========================================
      else {
        if (value === "") {
          throw new Error(`${formatFeatureName(feature.name)} is required.`);
        }
      }

      data[feature.name] = value;
    });

    // ----------------------------------------------------
    // Send request to Flask
    // ----------------------------------------------------

    const response = await fetch("/diabetes/v1/predict", {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify(data),
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "Prediction failed.");
    }

    // ----------------------------------------------------
    // Display the prediction
    // ----------------------------------------------------

    displayPrediction(result);
  } catch (error) {
    showError(error.message);
  } finally {
    predictButton.disabled = false;

    predictButton.textContent = "Predict Diabetes";
  }
}

// ============================================================
// 7. DISPLAY PREDICTION RESULT
// ============================================================

function displayPrediction(result) {
  // --------------------------------------------------------
  // Diagnosis
  // --------------------------------------------------------

  document.getElementById("diagnosis").textContent = result.diagnosis;

  // --------------------------------------------------------
  // Confidence
  // --------------------------------------------------------

  document.getElementById("confidence").textContent = formatPercentage(
    result.confidence,
  );

  // --------------------------------------------------------
  // Diabetes probability
  // --------------------------------------------------------

  document.getElementById("diabetesProbability").textContent = formatPercentage(
    result.probability_diabetes,
  );

  // --------------------------------------------------------
  // No diabetes probability
  // --------------------------------------------------------

  document.getElementById("noDiabetesProbability").textContent =
    formatPercentage(result.probability_no_diabetes);

  // --------------------------------------------------------
  // Numerical prediction
  // --------------------------------------------------------

  document.getElementById("predictionClass").textContent = result.prediction;

  // --------------------------------------------------------
  // Model name
  // --------------------------------------------------------

  document.getElementById("modelName").textContent = result.model;

  // --------------------------------------------------------
  // Diabetes probability progress bar
  // --------------------------------------------------------

  const probabilityBar = document.getElementById("probabilityBar");

  probabilityBar.style.width = `${result.probability_diabetes || 0}%`;

  // --------------------------------------------------------
  // Show result card
  // --------------------------------------------------------

  resultCard.classList.remove("hidden");

  // Smooth scrolling
  resultCard.scrollIntoView({
    behavior: "smooth",
  });
}

// ============================================================
// 8. FORMAT PERCENTAGE
// ============================================================

function formatPercentage(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return Number(value).toFixed(2) + "%";
}

// ============================================================
// 9. FORMAT FEATURE NAME
// ============================================================

function formatFeatureName(name) {
  return name

    .replaceAll("_", " ")

    .replace(/\b\w/g, function (character) {
      return character.toUpperCase();
    });
}

// ============================================================
// 10. SHOW ERROR
// ============================================================

function showError(message) {
  errorBox.textContent = message;

  errorBox.style.display = "block";
}

// ============================================================
// 11. CLEAR ERROR
// ============================================================

function clearError() {
  errorBox.textContent = "";

  errorBox.style.display = "none";
}
