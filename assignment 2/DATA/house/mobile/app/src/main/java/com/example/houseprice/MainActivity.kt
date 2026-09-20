package com.example.houseprice

import android.os.Bundle

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
import androidx.compose.foundation.layout.width

import androidx.compose.foundation.lazy.LazyColumn

import androidx.compose.foundation.text.KeyboardOptions

import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue

import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp

import org.json.JSONObject

import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter

import java.net.HttpURLConnection
import java.net.URL

import java.text.NumberFormat

import java.util.Locale

import kotlin.concurrent.thread


class MainActivity : ComponentActivity() {

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {

        super.onCreate(
            savedInstanceState
        )

        setContent {

            MaterialTheme {

                HousePriceApp()

            }

        }

    }


    // =========================================================
    // Main application
    // =========================================================

    @Composable
    private fun HousePriceApp() {

        // -----------------------------------------------------
        // API configuration
        //
        // Android Emulator:
        // http://10.0.2.2:5000
        //
        // Physical phone:
        // replace with the IP address of the PC running Flask.
        // -----------------------------------------------------

        var apiUrl by remember {

            mutableStateOf(
                "http://192.168.1.20:5000"
            )

        }


        // =====================================================
        // User input state
        // =====================================================

        var latitude by remember {
            mutableStateOf("51.50")
        }


        var longitude by remember {
            mutableStateOf("-0.12")
        }


        var outcode by remember {
            mutableStateOf("SW1A")
        }


        var floorArea by remember {
            mutableStateOf("80")
        }


        var bedrooms by remember {
            mutableStateOf("2")
        }


        var bathrooms by remember {
            mutableStateOf("1")
        }


        var livingRooms by remember {
            mutableStateOf("1")
        }


        var propertyType by remember {
            mutableStateOf("Bungalow Property")
        }


        var tenure by remember {
            mutableStateOf("Freehold")
        }


        var energyRating by remember {
            mutableStateOf("A")
        }


        // =====================================================
        // Prediction state
        // =====================================================

        var isPredicting by remember {

            mutableStateOf(
                false
            )

        }


        var predictedPrice by remember {

            mutableStateOf<String?>(
                null
            )

        }


        var confidence by remember {

            mutableStateOf<String?>(
                null
            )

        }


        var validationR2 by remember {

            mutableStateOf<String?>(
                null
            )

        }


        var validationMAE by remember {

            mutableStateOf<String?>(
                null
            )

        }


        var validationRMSE by remember {

            mutableStateOf<String?>(
                null
            )

        }


        var selectedModel by remember {

            mutableStateOf<String?>(
                null
            )

        }


        var errorMessage by remember {

            mutableStateOf<String?>(
                null
            )

        }


        // =====================================================
        // Main layout
        // =====================================================

        LazyColumn(

            modifier =
                Modifier
                    .fillMaxSize()
                    .padding(16.dp),

            verticalArrangement =
                Arrangement.spacedBy(
                    12.dp
                )

        ) {


            // =================================================
            // Header
            // =================================================

            item {

                Column {

                    Text(

                        text =
                            "London House Price",

                        style =
                            MaterialTheme
                                .typography
                                .headlineMedium,

                        fontWeight =
                            FontWeight.Bold

                    )


                    Spacer(
                        modifier =
                            Modifier.height(4.dp)
                    )


                    Text(

                        text =
                            "Machine Learning Property Estimator",

                        style =
                            MaterialTheme
                                .typography
                                .bodyMedium

                    )

                }

            }


            // =================================================
            // API server
            // =================================================

            item {

                OutlinedTextField(

                    value =
                        apiUrl,

                    onValueChange = {

                        apiUrl =
                            it.trimEnd('/')

                    },

                    modifier =
                        Modifier.fillMaxWidth(),

                    label = {

                        Text(
                            "API URL"
                        )

                    },

                    singleLine =
                        true

                )

            }


            // =================================================
            // Location section
            // =================================================

            item {

                SectionTitle(
                    "Location"
                )

            }


            item {

                NumberInput(

                    label =
                        "Latitude",

                    value =
                        latitude,

                    onValueChange = {

                        latitude =
                            it

                    }

                )

            }


            item {

                NumberInput(

                    label =
                        "Longitude",

                    value =
                        longitude,

                    onValueChange = {

                        longitude =
                            it

                    }

                )

            }


            item {

                TextInput(

                    label =
                        "Outcode",

                    value =
                        outcode,

                    onValueChange = {

                        outcode =
                            it

                    }

                )

            }


            // =================================================
            // Property section
            // =================================================

            item {

                SectionTitle(
                    "Property Characteristics"
                )

            }


            item {

                NumberInput(

                    label =
                        "Floor Area (m²)",

                    value =
                        floorArea,

                    onValueChange = {

                        floorArea =
                            it

                    }

                )

            }


            item {

                NumberInput(

                    label =
                        "Bedrooms",

                    value =
                        bedrooms,

                    onValueChange = {

                        bedrooms =
                            it

                    },

                    integer =
                        true

                )

            }


            item {

                NumberInput(

                    label =
                        "Bathrooms",

                    value =
                        bathrooms,

                    onValueChange = {

                        bathrooms =
                            it

                    },

                    integer =
                        true

                )

            }


            item {

                NumberInput(

                    label =
                        "Living Rooms",

                    value =
                        livingRooms,

                    onValueChange = {

                        livingRooms =
                            it

                    },

                    integer =
                        true

                )

            }


            item {

                TextInput(

                    label =
                        "Property Type",

                    value =
                        propertyType,

                    onValueChange = {

                        propertyType =
                            it

                    }

                )

            }


            item {

                TextInput(

                    label =
                        "Tenure",

                    value =
                        tenure,

                    onValueChange = {

                        tenure =
                            it

                    }

                )

            }


            item {

                TextInput(

                    label =
                        "Current Energy Rating",

                    value =
                        energyRating,

                    onValueChange = {

                        energyRating =
                            it

                    }

                )

            }


            // =================================================
            // Action buttons
            // =================================================

            item {

                Row(

                    modifier =
                        Modifier.fillMaxWidth(),

                    horizontalArrangement =
                        Arrangement.spacedBy(
                            10.dp
                        )

                ) {


                    OutlinedButton(

                        onClick = {

                            latitude =
                                "51.50"

                            longitude =
                                "-0.12"

                            outcode =
                                "SW1A"

                            floorArea =
                                "80"

                            bedrooms =
                                "2"

                            bathrooms =
                                "1"

                            livingRooms =
                                "1"

                            propertyType =
                                "Bungalow Property"

                            tenure =
                                "Freehold"

                            energyRating =
                                "A"


                            predictedPrice =
                                null

                            confidence =
                                null

                            validationR2 =
                                null

                            validationMAE =
                                null

                            validationRMSE =
                                null

                            selectedModel =
                                null

                            errorMessage =
                                null

                        },

                        modifier =
                            Modifier.weight(
                                1f
                            )

                    ) {

                        Text(
                            "Load Sample"
                        )

                    }


                    OutlinedButton(

                        onClick = {

                            latitude =
                                ""

                            longitude =
                                ""

                            outcode =
                                ""

                            floorArea =
                                ""

                            bedrooms =
                                ""

                            bathrooms =
                                ""

                            livingRooms =
                                ""

                            propertyType =
                                ""

                            tenure =
                                ""

                            energyRating =
                                ""


                            predictedPrice =
                                null

                            confidence =
                                null

                            validationR2 =
                                null

                            validationMAE =
                                null

                            validationRMSE =
                                null

                            selectedModel =
                                null

                            errorMessage =
                                null

                        },

                        modifier =
                            Modifier.weight(
                                1f
                            )

                    ) {

                        Text(
                            "Clear"
                        )

                    }

                }

            }


            // =================================================
            // Prediction button
            // =================================================

            item {

                Button(

                    onClick = {

                        isPredicting =
                            true

                        predictedPrice =
                            null

                        errorMessage =
                            null


                        predictHousePrice(

                            apiUrl =
                                apiUrl,

                            latitude =
                                latitude,

                            longitude =
                                longitude,

                            outcode =
                                outcode,

                            floorArea =
                                floorArea,

                            bedrooms =
                                bedrooms,

                            bathrooms =
                                bathrooms,

                            livingRooms =
                                livingRooms,

                            propertyType =
                                propertyType,

                            tenure =
                                tenure,

                            energyRating =
                                energyRating,

                            onSuccess = {
                                    result ->

                                isPredicting =
                                    false

                                predictedPrice =
                                    result.predictedPrice

                                confidence =
                                    result.confidence

                                validationR2 =
                                    result.r2

                                validationMAE =
                                    result.mae

                                validationRMSE =
                                    result.rmse

                                selectedModel =
                                    result.model

                            },

                            onError = {
                                    message ->

                                isPredicting =
                                    false

                                errorMessage =
                                    message

                            }

                        )

                    },

                    enabled =
                        !isPredicting,

                    modifier =
                        Modifier.fillMaxWidth()

                ) {

                    if (
                        isPredicting
                    ) {

                        CircularProgressIndicator(

                            modifier =
                                Modifier
                                    .width(
                                        20.dp
                                    )
                                    .height(
                                        20.dp
                                    ),

                            strokeWidth =
                                2.dp

                        )


                        Spacer(
                            modifier =
                                Modifier.width(
                                    10.dp
                                )
                        )


                        Text(
                            "Estimating..."
                        )

                    } else {

                        Text(
                            "Estimate House Price"
                        )

                    }

                }

            }


            // =================================================
            // Prediction result
            // =================================================

            if (
                predictedPrice != null
            ) {

                item {

                    PredictionResultCard(

                        predictedPrice =
                            predictedPrice!!,

                        confidence =
                            confidence
                                ?: "-",

                        r2 =
                            validationR2
                                ?: "-",

                        mae =
                            validationMAE
                                ?: "-",

                        rmse =
                            validationRMSE
                                ?: "-",

                        model =
                            selectedModel
                                ?: "-"

                    )

                }

            }


            // =================================================
            // Error message
            // =================================================

            if (
                errorMessage != null
            ) {

                item {

                    Card(

                        modifier =
                            Modifier.fillMaxWidth()

                    ) {

                        Text(

                            text =
                                errorMessage!!,

                            modifier =
                                Modifier.padding(
                                    16.dp
                                ),

                            color =
                                MaterialTheme
                                    .colorScheme
                                    .error

                        )

                    }

                }

            }


            // =================================================
            // Disclaimer
            // =================================================

            item {

                Spacer(
                    modifier =
                        Modifier.height(
                            8.dp
                        )
                )


                Text(

                    text =
                        "The estimate is generated by a machine-learning "
                                +
                                "model and should not be treated as a professional valuation.",

                    style =
                        MaterialTheme
                            .typography
                            .bodySmall

                )

            }

        }

    }


    // =========================================================
    // Section title
    // =========================================================

    @Composable
    private fun SectionTitle(
        title: String
    ) {

        Text(

            text =
                title,

            style =
                MaterialTheme
                    .typography
                    .titleMedium,

            fontWeight =
                FontWeight.Bold

        )

    }


    // =========================================================
    // Numeric input
    // =========================================================

    @Composable
    private fun NumberInput(

        label: String,

        value: String,

        onValueChange:
            (String) -> Unit,

        integer: Boolean =
            false

    ) {

        OutlinedTextField(

            value =
                value,

            onValueChange =
                onValueChange,

            modifier =
                Modifier.fillMaxWidth(),

            label = {

                Text(
                    label
                )

            },

            singleLine =
                true,

            keyboardOptions =

                KeyboardOptions(

                    keyboardType =

                        if (
                            integer
                        ) {

                            KeyboardType.Number

                        } else {

                            KeyboardType.Decimal

                        }

                )

        )

    }


    // =========================================================
    // Text input
    // =========================================================

    @Composable
    private fun TextInput(

        label: String,

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
                Modifier.fillMaxWidth(),

            label = {

                Text(
                    label
                )

            },

            singleLine =
                true

        )

    }


    // =========================================================
    // Prediction result
    // =========================================================

    @Composable
    private fun PredictionResultCard(

        predictedPrice: String,

        confidence: String,

        r2: String,

        mae: String,

        rmse: String,

        model: String

    ) {

        Card(

            modifier =
                Modifier.fillMaxWidth()

        ) {

            Column(

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .padding(
                            20.dp
                        )

            ) {


                Text(

                    text =
                        "Estimated Historical Price",

                    style =
                        MaterialTheme
                            .typography
                            .labelLarge

                )


                Spacer(
                    modifier =
                        Modifier.height(
                            8.dp
                        )
                )


                Text(

                    text =
                        predictedPrice,

                    style =
                        MaterialTheme
                            .typography
                            .headlineLarge,

                    fontWeight =
                        FontWeight.Bold

                )


                Spacer(
                    modifier =
                        Modifier.height(
                            18.dp
                        )
                )


                MetricRow(

                    label =
                        "Model Confidence Indicator",

                    value =
                        "$confidence%"

                )


                MetricRow(

                    label =
                        "Validation R²",

                    value =
                        r2

                )


                MetricRow(

                    label =
                        "Validation MAE",

                    value =
                        formatCurrency(
                            mae
                        )

                )


                MetricRow(

                    label =
                        "Validation RMSE",

                    value =
                        formatCurrency(
                            rmse
                        )

                )


                MetricRow(

                    label =
                        "Selected Model",

                    value =
                        model

                )


                Spacer(
                    modifier =
                        Modifier.height(
                            12.dp
                        )
                )


                Text(

                    text =
                        "The confidence indicator is based on validation R². "
                                +
                                "It is not a probability for this individual prediction.",

                    style =
                        MaterialTheme
                            .typography
                            .bodySmall

                )

            }

        }

    }


    // =========================================================
    // Metric row
    // =========================================================

    @Composable
    private fun MetricRow(

        label: String,

        value: String

    ) {

        Row(

            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(
                        vertical = 5.dp
                    ),

            horizontalArrangement =
                Arrangement.SpaceBetween

        ) {

            Text(
                text =
                    label
            )


            Text(

                text =
                    value,

                fontWeight =
                    FontWeight.SemiBold

            )

        }

    }


    // =========================================================
    // Prediction response
    // =========================================================

    private data class PredictionResponse(

        val predictedPrice:
        String,

        val confidence:
        String,

        val r2:
        String,

        val mae:
        String,

        val rmse:
        String,

        val model:
        String

    )


    // =========================================================
    // Call Flask prediction API
    // =========================================================

    private fun predictHousePrice(

        apiUrl: String,

        latitude: String,

        longitude: String,

        outcode: String,

        floorArea: String,

        bedrooms: String,

        bathrooms: String,

        livingRooms: String,

        propertyType: String,

        tenure: String,

        energyRating: String,

        onSuccess:
            (PredictionResponse) -> Unit,

        onError:
            (String) -> Unit

    ) {

        thread {

            var connection:
                    HttpURLConnection? =
                null


            try {

                // ------------------------------------------------
                // URL
                // ------------------------------------------------

                val url =
                    URL(

                        apiUrl.trimEnd('/')
                                +
                                "/house/v1/predict"

                    )


                // ------------------------------------------------
                // Connection
                // ------------------------------------------------

                connection =
                    url.openConnection()
                            as HttpURLConnection


                connection.requestMethod =
                    "POST"


                connection.connectTimeout =
                    10000


                connection.readTimeout =
                    15000


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


                // ------------------------------------------------
                // JSON payload
                // ------------------------------------------------

                val json =
                    JSONObject()


                json.put(
                    "latitude",
                    latitude
                )


                json.put(
                    "longitude",
                    longitude
                )


                json.put(
                    "outcode",
                    outcode
                )


                json.put(
                    "floorAreaSqM",
                    floorArea
                )


                json.put(
                    "bedrooms",
                    bedrooms
                )


                json.put(
                    "bathrooms",
                    bathrooms
                )


                json.put(
                    "livingRooms",
                    livingRooms
                )


                json.put(
                    "propertyType",
                    propertyType
                )


                json.put(
                    "tenure",
                    tenure
                )


                json.put(
                    "currentEnergyRating",
                    energyRating
                )


                // ------------------------------------------------
                // Send request
                // ------------------------------------------------

                OutputStreamWriter(

                    connection.outputStream

                ).use {

                        writer ->

                    writer.write(
                        json.toString()
                    )

                    writer.flush()

                }


                // ------------------------------------------------
                // Response
                // ------------------------------------------------

                val responseCode =
                    connection.responseCode


                val inputStream =

                    if (
                        responseCode in 200..299
                    ) {

                        connection.inputStream

                    } else {

                        connection.errorStream

                    }


                if (
                    inputStream == null
                ) {

                    throw Exception(
                        "Server returned no response."
                    )

                }


                val response =
                    BufferedReader(

                        InputStreamReader(
                            inputStream
                        )

                    ).use {

                        it.readText()

                    }


                // ------------------------------------------------
                // Parse response
                // ------------------------------------------------

                val result =
                    JSONObject(
                        response
                    )


                if (
                    responseCode !in 200..299
                ) {

                    throw Exception(

                        result.optString(

                            "error",

                            "Prediction failed."

                        )

                    )

                }


                val price =
                    result.getDouble(
                        "predicted_price"
                    )


                val confidenceValue =
                    result.getDouble(
                        "confidence_indicator"
                    )


                val metrics =
                    result.getJSONObject(
                        "validation_metrics"
                    )


                val r2 =
                    metrics.getDouble(
                        "R2"
                    )


                val mae =
                    metrics.getDouble(
                        "MAE"
                    )


                val rmse =
                    metrics.getDouble(
                        "RMSE"
                    )


                val model =
                    result.optString(

                        "model",

                        "Gradient Boosting Tuned"

                    )


                val predictionResponse =
                    PredictionResponse(

                        predictedPrice =
                            formatCurrency(
                                price
                            ),

                        confidence =
                            String.format(

                                Locale.US,

                                "%.2f",

                                confidenceValue

                            ),

                        r2 =
                            String.format(

                                Locale.US,

                                "%.3f",

                                r2

                            ),

                        mae =
                            mae.toString(),

                        rmse =
                            rmse.toString(),

                        model =
                            model

                    )


                runOnUiThread {

                    onSuccess(
                        predictionResponse
                    )

                }


            } catch (
                exception: Exception
            ) {

                runOnUiThread {

                    onError(

                        exception.message
                            ?: "Unable to connect to the prediction server."

                    )

                }

            } finally {

                connection?.disconnect()

            }

        }

    }


    // =========================================================
    // Currency formatting
    // =========================================================

    private fun formatCurrency(
        value: Double
    ): String {

        return NumberFormat
            .getCurrencyInstance(
                Locale.UK
            )
            .format(
                value
            )

    }


    private fun formatCurrency(
        value: String
    ): String {

        return try {

            formatCurrency(
                value.toDouble()
            )

        } catch (
            exception: Exception
        ) {

            value

        }

    }

}