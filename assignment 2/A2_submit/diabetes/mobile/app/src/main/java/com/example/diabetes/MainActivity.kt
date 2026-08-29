package com.example.diabetes

// ============================================================
// IMPORTS
// ============================================================

import android.content.Context
import android.graphics.Color
import android.os.Bundle
import android.text.InputType
import android.view.View
import android.widget.Toast

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding

import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll

import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Divider
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue

import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import org.json.JSONObject

import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL


// ============================================================
// MAIN ACTIVITY
// ============================================================

class MainActivity : ComponentActivity() {

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(
            savedInstanceState
        )

        // ----------------------------------------------------
        // Start Jetpack Compose UI
        // ----------------------------------------------------

        setContent {
            MaterialTheme {
                DiabetesApp()
            }
        }
    }
}


// ============================================================
// API CONFIGURATION
// ============================================================

object ApiConfig {

    /*
     * Android Emulator:
     *
     * 10.0.2.2
     *
     * represents the host computer.
     *
     * Flask is running on:
     *
     * http://127.0.0.1:5000
     *
     * Therefore the emulator uses:
     *
     * http://10.0.2.2:5000
     */

    const val BASE_URL =
        "http://192.168.1.20:5000"

    const val SCHEMA_URL =
        "$BASE_URL/diabetes/v1/schema"

    const val PREDICT_URL =
        "$BASE_URL/diabetes/v1/predict"
}


// ============================================================
// FEATURE DEFINITION
// ============================================================

data class FeatureDefinition(

    val name: String,

    val type: String,

    val categories: List<String>

)


// ============================================================
// DIABETES APPLICATION
// ============================================================

@Composable
fun DiabetesApp() {

    // --------------------------------------------------------
    // Android context
    // --------------------------------------------------------

    val context =
        LocalContext.current


    // --------------------------------------------------------
    // Coroutine scope
    // --------------------------------------------------------

    val scope =
        rememberCoroutineScope()


    // --------------------------------------------------------
    // Feature schema returned from Flask
    // --------------------------------------------------------

    var featureSchema by remember {

        mutableStateOf(
            emptyList<FeatureDefinition>()
        )

    }


    // --------------------------------------------------------
    // User input values
    //
    // Key   = feature name
    // Value = input text
    // --------------------------------------------------------

    val inputValues =
        remember {

            mutableStateMapOf<
                    String,
                    String
                    >()

        }


    // --------------------------------------------------------
    // Application status
    // --------------------------------------------------------

    var isLoading by remember {

        mutableStateOf(
            true
        )

    }


    var isPredicting by remember {

        mutableStateOf(
            false
        )

    }


    var statusMessage by remember {

        mutableStateOf(
            "Connecting to prediction server..."
        )

    }


    var errorMessage by remember {

        mutableStateOf(
            ""
        )

    }


    // --------------------------------------------------------
    // Prediction result
    // --------------------------------------------------------

    var prediction by remember {

        mutableStateOf<Int?>(
            null
        )

    }


    var diagnosis by remember {

        mutableStateOf(
            ""
        )

    }


    var confidence by remember {

        mutableStateOf<Double?>(
            null
        )

    }


    var diabetesProbability by remember {

        mutableStateOf<Double?>(
            null
        )

    }


    var noDiabetesProbability by remember {

        mutableStateOf<Double?>(
            null
        )

    }


    var modelName by remember {

        mutableStateOf(
            ""
        )

    }


    // ========================================================
    // LOAD MODEL SCHEMA WHEN APP STARTS
    // ========================================================

    LaunchedEffect(Unit) {

        try {

            statusMessage =
                "Loading model features..."


            val response =
                withContext(
                    Dispatchers.IO
                ) {

                    httpGet(
                        ApiConfig.SCHEMA_URL
                    )

                }


            // ------------------------------------------------
            // Parse JSON response
            // ------------------------------------------------

            val root =
                JSONObject(
                    response
                )


            val featuresJson =
                root.getJSONArray(
                    "features"
                )


            val parsedFeatures =
                mutableListOf<FeatureDefinition>()


            // ------------------------------------------------
            // Parse every feature
            // ------------------------------------------------

            for (
            index in
            0 until featuresJson.length()
            ) {

                val featureObject =
                    featuresJson.getJSONObject(
                        index
                    )


                val name =
                    featureObject.getString(
                        "name"
                    )


                val type =
                    featureObject.getString(
                        "type"
                    )


                val categories =
                    mutableListOf<String>()


                val categoriesJson =
                    featureObject.optJSONArray(
                        "categories"
                    )


                if (
                    categoriesJson != null
                ) {

                    for (
                    categoryIndex in
                    0 until categoriesJson.length()
                    ) {

                        categories.add(
                            categoriesJson.getString(
                                categoryIndex
                            )
                        )

                    }

                }


                parsedFeatures.add(

                    FeatureDefinition(

                        name =
                            name,

                        type =
                            type,

                        categories =
                            categories

                    )

                )

            }


            // ------------------------------------------------
            // Store schema
            // ------------------------------------------------

            featureSchema =
                parsedFeatures


            // ------------------------------------------------
            // Automatically load sample data
            // ------------------------------------------------

            fillSampleValues(
                schema =
                    parsedFeatures,
                values =
                    inputValues
            )


            statusMessage =
                "Model connected successfully."

            errorMessage =
                ""

            isLoading =
                false

        }

        catch (
            exception: Exception
        ) {

            statusMessage =
                "Unable to connect to prediction server."

            errorMessage =
                exception.message
                    ?: "Unknown connection error."

            isLoading =
                false

        }

    }


    // ========================================================
    // MAIN UI
    // ========================================================

    Column(

        modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(
                    rememberScrollState()
                )
                .padding(
                    20.dp
                ),

        verticalArrangement =
            Arrangement.spacedBy(
                12.dp
            )

    ) {


        // ====================================================
        // HEADER
        // ====================================================

        Text(

            text =
                "Diabetes Prediction",

            style =
                MaterialTheme
                    .typography
                    .headlineMedium,

            fontWeight =
                FontWeight.Bold

        )


        Text(

            text =
                "Machine Learning Based Health Assessment",

            style =
                MaterialTheme
                    .typography
                    .bodyMedium

        )


        Text(

            text =
                statusMessage,

            style =
                MaterialTheme
                    .typography
                    .bodySmall

        )


        // ====================================================
        // LOADING INDICATOR
        // ====================================================

        if (isLoading) {

            LinearProgressIndicator(

                modifier =
                    Modifier
                        .fillMaxWidth()

            )

        }


        // ====================================================
        // ERROR
        // ====================================================

        if (
            errorMessage.isNotBlank()
        ) {

            Card(

                modifier =
                    Modifier
                        .fillMaxWidth()

            ) {

                Text(

                    text =
                        errorMessage,

                    modifier =
                        Modifier
                            .padding(
                                16.dp
                            ),

                    color =
                        MaterialTheme
                            .colorScheme
                            .error

                )

            }

        }


        // ====================================================
        // PATIENT INPUT CARD
        // ====================================================

        if (
            featureSchema.isNotEmpty()
        ) {

            Card(

                modifier =
                    Modifier
                        .fillMaxWidth()

            ) {

                Column(

                    modifier =
                        Modifier
                            .padding(
                                16.dp
                            ),

                    verticalArrangement =
                        Arrangement.spacedBy(
                            10.dp
                        )

                ) {


                    // ----------------------------------------
                    // Section title
                    // ----------------------------------------

                    Text(

                        text =
                            "Patient Information",

                        style =
                            MaterialTheme
                                .typography
                                .titleLarge,

                        fontWeight =
                            FontWeight.Bold

                    )


                    Text(

                        text =
                            "Enter the patient's health information.",

                        style =
                            MaterialTheme
                                .typography
                                .bodyMedium

                    )


                    Divider()


                    // ----------------------------------------
                    // Dynamic feature fields
                    // ----------------------------------------

                    featureSchema.forEach {

                            feature ->


                        if (
                            feature.type ==
                            "category"
                        ) {

                            CategoryInput(

                                feature =
                                    feature,

                                value =
                                    inputValues[
                                        feature.name
                                    ]
                                        ?: "",

                                onValueChange = {

                                        value ->

                                    inputValues[
                                        feature.name
                                    ] =
                                        value

                                }

                            )

                        }

                        else {

                            NumericInput(

                                featureName =
                                    feature.name,

                                value =
                                    inputValues[
                                        feature.name
                                    ]
                                        ?: "",

                                onValueChange = {

                                        value ->

                                    inputValues[
                                        feature.name
                                    ] =
                                        value

                                }

                            )

                        }

                    }


                    Spacer(
                        modifier =
                            Modifier.height(
                                8.dp
                            )
                    )


                    // =================================================
                    // BUTTONS
                    // =================================================

                    Row(

                        modifier =
                            Modifier
                                .fillMaxWidth(),

                        horizontalArrangement =
                            Arrangement.spacedBy(
                                10.dp
                            )

                    ) {


                        // ------------------------------------------------
                        // SAMPLE BUTTON
                        // ------------------------------------------------

                        Button(

                            modifier =
                                Modifier
                                    .weight(
                                        1f
                                    ),

                            enabled =
                                !isPredicting,

                            onClick = {

                                fillSampleValues(

                                    schema =
                                        featureSchema,

                                    values =
                                        inputValues

                                )


                                Toast
                                    .makeText(

                                        context,

                                        "Sample data loaded.",

                                        Toast.LENGTH_SHORT

                                    )
                                    .show()

                            }

                        ) {

                            Text(
                                "Sample Data"
                            )

                        }


                        // ------------------------------------------------
                        // PREDICT BUTTON
                        // ------------------------------------------------

                        Button(

                            modifier =
                                Modifier
                                    .weight(
                                        1.5f
                                    ),

                            enabled =
                                !isLoading &&
                                        !isPredicting,

                            onClick = {

                                // Clear old result/error
                                errorMessage =
                                    ""

                                prediction =
                                    null

                                diagnosis =
                                    ""

                                confidence =
                                    null

                                diabetesProbability =
                                    null

                                noDiabetesProbability =
                                    null

                                modelName =
                                    ""


                                // Start prediction
                                isPredicting =
                                    true

                                statusMessage =
                                    "Predicting..."


                                scope.launch {

                                    try {


                                        // --------------------------------
                                        // Build JSON
                                        // --------------------------------

                                        val requestJson =
                                            buildPredictionJson(

                                                schema =
                                                    featureSchema,

                                                values =
                                                    inputValues

                                            )


                                        // --------------------------------
                                        // Send HTTP POST
                                        // --------------------------------

                                        val response =
                                            withContext(
                                                Dispatchers.IO
                                            ) {

                                                httpPostJson(

                                                    endpoint =
                                                        ApiConfig
                                                            .PREDICT_URL,

                                                    jsonBody =
                                                        requestJson
                                                            .toString()

                                                )

                                            }


                                        // --------------------------------
                                        // Parse prediction response
                                        // --------------------------------

                                        val result =
                                            parsePredictionResult(
                                                response
                                            )


                                        // --------------------------------
                                        // Update UI
                                        // --------------------------------

                                        prediction =
                                            result.prediction

                                        diagnosis =
                                            result.diagnosis

                                        confidence =
                                            result.confidence

                                        diabetesProbability =
                                            result.diabetesProbability

                                        noDiabetesProbability =
                                            result.noDiabetesProbability

                                        modelName =
                                            result.model


                                        statusMessage =
                                            "Prediction completed."


                                    }

                                    catch (
                                        exception: Exception
                                    ) {

                                        errorMessage =
                                            exception.message
                                                ?: "Prediction failed."

                                        statusMessage =
                                            "Prediction failed."

                                    }

                                    finally {

                                        isPredicting =
                                            false

                                    }

                                }

                            }

                        ) {

                            Text(

                                if (
                                    isPredicting
                                ) {

                                    "Predicting..."

                                }

                                else {

                                    "Predict Diabetes"

                                }

                            )

                        }

                    }

                }

            }

        }


        // ====================================================
        // RESULT CARD
        // ====================================================

        if (
            prediction != null
        ) {

            Card(

                modifier =
                    Modifier
                        .fillMaxWidth()

            ) {

                Column(

                    modifier =
                        Modifier
                            .padding(
                                16.dp
                            ),

                    verticalArrangement =
                        Arrangement.spacedBy(
                            10.dp
                        )

                ) {


                    // ----------------------------------------
                    // Result title
                    // ----------------------------------------

                    Text(

                        text =
                            "Prediction Result",

                        style =
                            MaterialTheme
                                .typography
                                .titleLarge,

                        fontWeight =
                            FontWeight.Bold

                    )


                    Divider()


                    // ----------------------------------------
                    // Diagnosis
                    // ----------------------------------------

                    Text(

                        text =
                            "Diagnosis: $diagnosis",

                        style =
                            MaterialTheme
                                .typography
                                .headlineSmall,

                        fontWeight =
                            FontWeight.Bold

                    )


                    // ----------------------------------------
                    // Confidence
                    // ----------------------------------------

                    Text(

                        text =
                            "Confidence: " +
                                    formatPercentage(
                                        confidence
                                    ),

                        style =
                            MaterialTheme
                                .typography
                                .bodyLarge

                    )


                    // ----------------------------------------
                    // Diabetes probability
                    // ----------------------------------------

                    Text(

                        text =
                            "Diabetes Probability: " +
                                    formatPercentage(
                                        diabetesProbability
                                    ),

                        style =
                            MaterialTheme
                                .typography
                                .bodyLarge

                    )


                    // ----------------------------------------
                    // No diabetes probability
                    // ----------------------------------------

                    Text(

                        text =
                            "No Diabetes Probability: " +
                                    formatPercentage(
                                        noDiabetesProbability
                                    ),

                        style =
                            MaterialTheme
                                .typography
                                .bodyLarge

                    )


                    // ----------------------------------------
                    // Prediction class
                    // ----------------------------------------

                    Text(

                        text =
                            "Prediction Class: $prediction",

                        style =
                            MaterialTheme
                                .typography
                                .bodyLarge

                    )


                    // ----------------------------------------
                    // Model
                    // ----------------------------------------

                    Text(

                        text =
                            "Model: $modelName",

                        style =
                            MaterialTheme
                                .typography
                                .bodyLarge

                    )


                    Spacer(
                        modifier =
                            Modifier.height(
                                8.dp
                            )
                    )


                    // =================================================
                    // PROBABILITY VISUALIZATION
                    // =================================================

                    Text(

                        text =
                            "Diabetes Probability",

                        style =
                            MaterialTheme
                                .typography
                                .titleMedium,

                        fontWeight =
                            FontWeight.Bold

                    )


                    if (
                        diabetesProbability != null
                    ) {

                        LinearProgressIndicator(

                            progress = {

                                (
                                        diabetesProbability!!
                                            .toFloat()
                                                /
                                                100f
                                        )
                                    .coerceIn(
                                        0f,
                                        1f
                                    )

                            },

                            modifier =
                                Modifier
                                    .fillMaxWidth()

                        )

                    }


                    // ----------------------------------------
                    // Disclaimer
                    // ----------------------------------------

                    Spacer(
                        modifier =
                            Modifier.height(
                                8.dp
                            )
                    )


                    Text(

                        text =
                            "Educational use only. " +
                                    "This prediction is not a medical diagnosis.",

                        style =
                            MaterialTheme
                                .typography
                                .bodySmall

                    )

                }

            }

        }


        Spacer(
            modifier =
                Modifier.height(
                    20.dp
                )
        )

    }

}


// ============================================================
// NUMERICAL INPUT
// ============================================================

@Composable
fun NumericInput(

    featureName: String,

    value: String,

    onValueChange:
        (String) -> Unit

) {

    OutlinedTextField(

        value =
            value,

        onValueChange =
            onValueChange,

        modifier =
            Modifier
                .fillMaxWidth(),

        label = {

            Text(
                formatFeatureName(
                    featureName
                )
            )

        },

        singleLine =
            true

    )

}


// ============================================================
// CATEGORICAL INPUT
// ============================================================

@OptIn(
    ExperimentalMaterial3Api::class
)
@Composable
fun CategoryInput(

    feature: FeatureDefinition,

    value: String,

    onValueChange:
        (String) -> Unit

) {

    var expanded by remember {

        mutableStateOf(
            false
        )

    }


    ExposedDropdownMenuBox(

        expanded =
            expanded,

        onExpandedChange = {

            expanded =
                !expanded

        }

    ) {


        OutlinedTextField(

            value =
                value,

            onValueChange = {},

            readOnly =
                true,

            modifier =
                Modifier
                    .fillMaxWidth()
                    .menuAnchor(),

            label = {

                Text(

                    formatFeatureName(
                        feature.name
                    )

                )

            },

            trailingIcon = {

                ExposedDropdownMenuDefaults
                    .TrailingIcon(

                        expanded =
                            expanded

                    )

            }

        )


        ExposedDropdownMenu(

            expanded =
                expanded,

            onDismissRequest = {

                expanded =
                    false

            }

        ) {

            feature.categories.forEach {

                    category ->


                DropdownMenuItem(

                    text = {

                        Text(
                            category
                        )

                    },

                    onClick = {

                        onValueChange(
                            category
                        )

                        expanded =
                            false

                    }

                )

            }

        }

    }

}


// ============================================================
// SAMPLE VALUES
// ============================================================

private val SAMPLE_VALUES =
    mapOf<String, Any>(

        // ----------------------------------------------------
        // Demographic
        // ----------------------------------------------------

        "age" to 50,

        "gender" to "Male",

        "ethnicity" to "White",

        "education_level" to
                "Bachelor's Degree",

        "income_level" to
                "Middle",

        "employment_status" to
                "Employed",

        "marital_status" to
                "Married",


        // ----------------------------------------------------
        // Lifestyle
        // ----------------------------------------------------

        "smoking_status" to
                "Never",

        "alcohol_consumption_per_week" to
                2,

        "physical_activity_minutes_per_week" to
                150,

        "diet_score" to
                7,

        "sleep_hours_per_day" to
                7,

        "screen_time_hours_per_day" to
                4,


        // ----------------------------------------------------
        // Medical history
        // ----------------------------------------------------

        "family_history_diabetes" to
                0,

        "hypertension_history" to
                0,

        "cardiovascular_history" to
                0,


        // ----------------------------------------------------
        // Anthropometric
        // ----------------------------------------------------

        "bmi" to
                28.5,

        "waist_to_hip_ratio" to
                0.9,


        // ----------------------------------------------------
        // Vital signs
        // ----------------------------------------------------

        "systolic_bp" to
                125,

        "diastolic_bp" to
                80,

        "heart_rate" to
                72,


        // ----------------------------------------------------
        // Lipid profile
        // ----------------------------------------------------

        "cholesterol_total" to
                190,

        "hdl_cholesterol" to
                55,

        "ldl_cholesterol" to
                110,

        "triglycerides" to
                140,


        // ----------------------------------------------------
        // Glucose / metabolic indicators
        // ----------------------------------------------------

        "glucose_fasting" to
                105,

        "glucose_postprandial" to
                145,

        "insulin_level" to
                12,

        "hba1c" to
                5.8

    )


// ============================================================
// FILL SAMPLE VALUES
// ============================================================

private fun fillSampleValues(

    schema:
    List<FeatureDefinition>,

    values:
    MutableMap<String, String>

) {

    schema.forEach {

            feature ->


        // ----------------------------------------------------
        // Explicit sample value
        // ----------------------------------------------------

        val explicitValue =
            SAMPLE_VALUES[
                feature.name
            ]


        if (
            explicitValue != null
        ) {

            // For numerical values
            if (
                feature.type ==
                "number"
            ) {

                values[
                    feature.name
                ] =
                    explicitValue.toString()

            }

            // For categorical values, verify that the
            // category exists in the model schema.
            else {

                val matchingCategory =
                    feature.categories
                        .firstOrNull {

                                category ->

                            category.equals(
                                explicitValue.toString(),
                                ignoreCase = true
                            )

                        }


                if (
                    matchingCategory != null
                ) {

                    values[
                        feature.name
                    ] =
                        matchingCategory

                }

                else if (
                    feature.categories.isNotEmpty()
                ) {

                    values[
                        feature.name
                    ] =
                        feature.categories.first()

                }

            }

            return@forEach

        }


        // ----------------------------------------------------
        // Numerical fallback
        // ----------------------------------------------------

        if (
            feature.type ==
            "number"
        ) {

            values[
                feature.name
            ] =
                inferNumericSample(
                    feature.name
                )

        }


        // ----------------------------------------------------
        // Categorical fallback
        // ----------------------------------------------------

        else {

            if (
                feature.categories.isNotEmpty()
            ) {

                values[
                    feature.name
                ] =
                    feature.categories.first()

            }

        }

    }

}


// ============================================================
// INFER NUMERICAL SAMPLE
// ============================================================

private fun inferNumericSample(
    featureName: String
): String {

    val name =
        featureName.lowercase()


    return when {

        name.contains("age") ->
            "50"


        name.contains("bmi") ->
            "28.5"


        name.contains("glucose") ->
            "105"


        name.contains("hba1c") ->
            "5.8"


        name.contains("insulin") ->
            "12"


        name.contains("systolic") ->
            "125"


        name.contains("diastolic") ->
            "80"


        name.contains("heart_rate") ||
                name.contains("heart") ->
            "72"


        name.contains("hdl") ->
            "55"


        name.contains("ldl") ->
            "110"


        name.contains(
            "cholesterol_total"
        ) ->
            "190"


        name.contains(
            "cholesterol"
        ) ->
            "190"


        name.contains("triglyceride") ->
            "140"


        name.contains(
            "physical_activity"
        ) ->
            "150"


        name.contains("alcohol") ->
            "2"


        name.contains("diet") ->
            "7"


        name.contains("sleep") ->
            "7"


        name.contains("screen") ->
            "4"


        name.contains(
            "waist_to_hip"
        ) ->
            "0.9"


        name.contains("history") ->
            "0"


        else ->
            "0"

    }

}


// ============================================================
// BUILD JSON REQUEST
// ============================================================

private fun buildPredictionJson(

    schema:
    List<FeatureDefinition>,

    values:
    Map<String, String>

): JSONObject {

    val json =
        JSONObject()


    schema.forEach {

            feature ->


        val value =
            values[
                feature.name
            ]
                ?.trim()
                ?: ""


        // ----------------------------------------------------
        // Empty-value validation
        // ----------------------------------------------------

        if (
            value.isEmpty()
        ) {

            throw Exception(
                "${formatFeatureName(
                    feature.name
                )} is required."
            )

        }


        // ----------------------------------------------------
        // Numerical values
        // ----------------------------------------------------

        if (
            feature.type ==
            "number"
        ) {

            val number =
                value.toDoubleOrNull()


            if (
                number == null
            ) {

                throw Exception(
                    "${formatFeatureName(
                        feature.name
                    )} must be a valid number."
                )

            }


            json.put(
                feature.name,
                number
            )

        }


        // ----------------------------------------------------
        // Categorical values
        // ----------------------------------------------------

        else {

            json.put(
                feature.name,
                value
            )

        }

    }


    return json

}


// ============================================================
// PREDICTION RESULT
// ============================================================

data class PredictionResult(

    val prediction: Int,

    val diagnosis: String,

    val confidence: Double?,

    val diabetesProbability: Double?,

    val noDiabetesProbability: Double?,

    val model: String

)


// ============================================================
// PARSE PREDICTION RESPONSE
// ============================================================

private fun parsePredictionResult(

    response:
    String

): PredictionResult {

    val json =
        JSONObject(
            response
        )


    // --------------------------------------------------------
    // API error
    // --------------------------------------------------------

    if (
        json.has("error")
    ) {

        throw Exception(
            json.getString(
                "error"
            )
        )

    }


    return PredictionResult(

        prediction =
            json.getInt(
                "prediction"
            ),

        diagnosis =
            json.getString(
                "diagnosis"
            ),

        confidence =
            getNullableDouble(
                json,
                "confidence"
            ),

        diabetesProbability =
            getNullableDouble(
                json,
                "probability_diabetes"
            ),

        noDiabetesProbability =
            getNullableDouble(
                json,
                "probability_no_diabetes"
            ),

        model =
            json.optString(
                "model",
                "Unknown"
            )

    )

}


// ============================================================
// READ NULLABLE DOUBLE
// ============================================================

private fun getNullableDouble(

    json:
    JSONObject,

    key:
    String

): Double? {

    if (
        !json.has(key)
    ) {

        return null

    }


    if (
        json.isNull(key)
    ) {

        return null

    }


    return json.getDouble(
        key
    )

}


// ============================================================
// HTTP GET
// ============================================================

private fun httpGet(
    endpoint: String
): String {

    val connection =
        URL(endpoint)
            .openConnection()
                as HttpURLConnection


    return try {

        // ----------------------------------------------------
        // Request configuration
        // ----------------------------------------------------

        connection.requestMethod =
            "GET"

        connection.connectTimeout =
            10_000

        connection.readTimeout =
            10_000


        // ----------------------------------------------------
        // Read HTTP response
        // ----------------------------------------------------

        val responseCode =
            connection.responseCode


        val stream =
            if (
                responseCode in 200..299
            ) {

                connection.inputStream

            }

            else {

                connection.errorStream

            }


        val response =
            BufferedReader(
                InputStreamReader(
                    stream
                )
            ).use {

                it.readText()

            }


        // ----------------------------------------------------
        // Handle HTTP error
        // ----------------------------------------------------

        if (
            responseCode !in 200..299
        ) {

            throw Exception(
                "HTTP $responseCode: $response"
            )

        }


        response

    }

    finally {

        connection.disconnect()

    }

}


// ============================================================
// HTTP POST JSON
// ============================================================

private fun httpPostJson(

    endpoint:
    String,

    jsonBody:
    String

): String {

    val connection =
        URL(endpoint)
            .openConnection()
                as HttpURLConnection


    return try {

        // ----------------------------------------------------
        // Request configuration
        // ----------------------------------------------------

        connection.requestMethod =
            "POST"

        connection.connectTimeout =
            10_000

        connection.readTimeout =
            10_000

        connection.doOutput =
            true


        connection.setRequestProperty(
            "Content-Type",
            "application/json"
        )

        connection.setRequestProperty(
            "Accept",
            "application/json"
        )


        // ----------------------------------------------------
        // Send JSON body
        // ----------------------------------------------------

        OutputStreamWriter(

            connection.outputStream,

            Charsets.UTF_8

        ).use {

                writer ->

            writer.write(
                jsonBody
            )

            writer.flush()

        }


        // ----------------------------------------------------
        // Read response
        // ----------------------------------------------------

        val responseCode =
            connection.responseCode


        val stream =
            if (
                responseCode in 200..299
            ) {

                connection.inputStream

            }

            else {

                connection.errorStream

            }


        val response =
            BufferedReader(
                InputStreamReader(
                    stream
                )
            ).use {

                it.readText()

            }


        // ----------------------------------------------------
        // Handle HTTP errors
        // ----------------------------------------------------

        if (
            responseCode !in 200..299
        ) {

            try {

                val errorJson =
                    JSONObject(
                        response
                    )


                throw Exception(
                    errorJson.optString(
                        "error",
                        "HTTP $responseCode"
                    )
                )

            }

            catch (
                exception:
                org.json.JSONException
            ) {

                throw Exception(
                    "HTTP $responseCode: $response"
                )

            }

        }


        response

    }

    finally {

        connection.disconnect()

    }

}


// ============================================================
// FORMAT FEATURE NAME
// ============================================================

private fun formatFeatureName(
    name: String
): String {

    return name

        .replace(
            "_",
            " "
        )

        .split(" ")

        .joinToString(" ") {

                word ->

            if (
                word.isEmpty()
            ) {

                word

            }

            else {

                word.replaceFirstChar {

                        character ->

                    character.uppercase()

                }

            }

        }

}


// ============================================================
// FORMAT PERCENTAGE
// ============================================================

private fun formatPercentage(
    value: Double?
): String {

    return if (
        value == null
    ) {

        "N/A"

    }

    else {

        String.format(
            "%.2f%%",
            value
        )

    }

}