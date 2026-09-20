package com.example.e_commerce

import android.os.Bundle
import android.widget.Toast

import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.lifecycleScope

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder


// ============================================================
// SERVER
// ============================================================
//
// Android Emulator:
//     http://10.0.2.2:8000/
//
// Physical phone:
//     change this to your PC LAN IP.
//     Example:
//     http://192.168.1.5:8000/
//

private const val BASE_URL = "http://192.168.1.20:8000/"


// ============================================================
// DATA CLASSES
// ============================================================

data class ProductResult(
    val productId: String,
    val category: String,
    val price: Double,
    val rating: Double,
    val soldUnits: Int,
    val recommendationScore: Double
)

data class PredictionResult(
    val behavior: String,
    val behaviorConfidence: Double?,
    val interest: String,
    val interestConfidence: Double?,
    val probabilities: List<Pair<String, Double>>,
    val products: List<ProductResult>,
    val model: String,
    val representation: String
)


// ============================================================
// MAIN ACTIVITY
// ============================================================

class MainActivity : ComponentActivity() {

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        setContent {
            OlistApp()
        }
    }


    // ========================================================
    // API: SAMPLE
    // ========================================================

    private suspend fun loadSample(): Map<String, Any?> {

        val json =
            httpGet(
                "${BASE_URL}sample"
            )

        val root =
            JSONObject(json)

        val features =
            root.getJSONObject(
                "features"
            )

        val result =
            mutableMapOf<String, Any?>()

        val keys =
            features.keys()

        while (keys.hasNext()) {

            val key =
                keys.next()

            val value =
                features.get(key)

            result[key] =
                if (
                    value == JSONObject.NULL
                ) {
                    null
                } else {
                    value
                }
        }

        return result
    }


    // ========================================================
    // API: PREDICT
    // ========================================================

    private suspend fun predict(
        payload: Map<String, Any?>
    ): PredictionResult {

        val jsonBody =
            JSONObject()

        payload.forEach {
                (key, value) ->

            if (value == null) {
                jsonBody.put(
                    key,
                    JSONObject.NULL
                )
            } else {
                jsonBody.put(
                    key,
                    value
                )
            }
        }

        val json =
            httpPost(
                "${BASE_URL}predict",
                jsonBody.toString()
            )

        val root =
            JSONObject(json)


        // ----------------------------------------------------
        // Behavior
        // ----------------------------------------------------

        val behavior =
            root.optString(
                "predicted_behavior",
                "Unknown"
            )

        val behaviorConfidence =
            if (
                root.isNull(
                    "behavior_confidence"
                )
            ) {
                null
            } else {
                root.optDouble(
                    "behavior_confidence",
                    Double.NaN
                ).takeUnless {
                    it.isNaN()
                }
            }


        // ----------------------------------------------------
        // Interest
        // ----------------------------------------------------

        val interest =
            root.optString(
                "predicted_interest",
                "Unknown"
            )

        val interestConfidence =
            if (
                root.isNull(
                    "interest_confidence"
                )
            ) {
                null
            } else {
                root.optDouble(
                    "interest_confidence",
                    Double.NaN
                ).takeUnless {
                    it.isNaN()
                }
            }


        // ----------------------------------------------------
        // Probabilities
        // ----------------------------------------------------

        val probabilityList =
            mutableListOf<Pair<String, Double>>()

        val probabilityObject =
            root.optJSONObject(
                "interest_probabilities"
            )

        if (
            probabilityObject != null
        ) {

            val names =
                probabilityObject.keys()

            while (names.hasNext()) {

                val name =
                    names.next()

                val probability =
                    probabilityObject.optDouble(
                        name,
                        0.0
                    )

                probabilityList.add(
                    name to probability
                )
            }
        }

        probabilityList.sortByDescending {
            it.second
        }


        // ----------------------------------------------------
        // Products
        // ----------------------------------------------------

        val products =
            mutableListOf<ProductResult>()

        val productArray =
            root.optJSONArray(
                "recommended_products"
            )

        if (
            productArray != null
        ) {

            for (
            index in
            0 until productArray.length()
            ) {

                val product =
                    productArray.getJSONObject(
                        index
                    )

                products.add(
                    ProductResult(
                        productId =
                            product.optString(
                                "product_id",
                                ""
                            ),

                        category =
                            product.optString(
                                "category",
                                ""
                            ),

                        price =
                            product.optDouble(
                                "price",
                                0.0
                            ),

                        rating =
                            product.optDouble(
                                "rating",
                                0.0
                            ),

                        soldUnits =
                            product.optInt(
                                "sold_units",
                                0
                            ),

                        recommendationScore =
                            product.optDouble(
                                "recommendation_score",
                                0.0
                            )
                    )
                )
            }
        }


        // ----------------------------------------------------
        // Model information
        // ----------------------------------------------------

        val modelInfo =
            root.optJSONObject(
                "model_info"
            )

        val model =
            modelInfo?.optString(
                "model",
                "Unknown"
            ) ?: "Unknown"

        val representation =
            modelInfo?.optString(
                "representation",
                "Unknown"
            ) ?: "Unknown"


        return PredictionResult(
            behavior =
                behavior,

            behaviorConfidence =
                behaviorConfidence,

            interest =
                interest,

            interestConfidence =
                interestConfidence,

            probabilities =
                probabilityList,

            products =
                products,

            model =
                model,

            representation =
                representation
        )
    }


    // ========================================================
    // HTTP GET
    // ========================================================

    private suspend fun httpGet(
        urlString: String
    ): String {

        return withContext(
            Dispatchers.IO
        ) {

            val connection =
                URL(urlString)
                    .openConnection()
                        as HttpURLConnection

            try {

                connection.requestMethod =
                    "GET"

                connection.connectTimeout =
                    10000

                connection.readTimeout =
                    30000

                connection.doInput =
                    true


                val code =
                    connection.responseCode


                val stream =
                    if (
                        code in 200..299
                    ) {
                        connection.inputStream
                    } else {
                        connection.errorStream
                    }


                val response =
                    stream
                        ?.bufferedReader()
                        ?.use {
                            it.readText()
                        }
                        ?: ""


                if (
                    code !in 200..299
                ) {

                    throw Exception(
                        "HTTP $code\n$response"
                    )
                }


                response

            } finally {

                connection.disconnect()
            }
        }
    }


    // ========================================================
    // HTTP POST
    // ========================================================

    private suspend fun httpPost(
        urlString: String,
        body: String
    ): String {

        return withContext(
            Dispatchers.IO
        ) {

            val connection =
                URL(urlString)
                    .openConnection()
                        as HttpURLConnection

            try {

                connection.requestMethod =
                    "POST"

                connection.connectTimeout =
                    10000

                connection.readTimeout =
                    30000

                connection.doInput =
                    true

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


                connection.outputStream
                    .bufferedWriter()
                    .use {
                        it.write(body)
                    }


                val code =
                    connection.responseCode


                val stream =
                    if (
                        code in 200..299
                    ) {
                        connection.inputStream
                    } else {
                        connection.errorStream
                    }


                val response =
                    stream
                        ?.bufferedReader()
                        ?.use {
                            it.readText()
                        }
                        ?: ""


                if (
                    code !in 200..299
                ) {

                    throw Exception(
                        "HTTP $code\n$response"
                    )
                }


                response

            } finally {

                connection.disconnect()
            }
        }
    }


    // ========================================================
    // MAIN UI
    // ========================================================

    @Composable
    private fun OlistApp() {

        var priorOrderCount by remember {
            mutableStateOf("")
        }

        var recencyDays by remember {
            mutableStateOf("")
        }

        var totalSpending by remember {
            mutableStateOf("")
        }

        var averageOrderValue by remember {
            mutableStateOf("")
        }

        var totalItems by remember {
            mutableStateOf("")
        }

        var averageReviewScore by remember {
            mutableStateOf("")
        }

        var customerState by remember {
            mutableStateOf("")
        }

        var paymentType by remember {
            mutableStateOf("")
        }

        var reviewText by remember {
            mutableStateOf("")
        }


        var result by remember {
            mutableStateOf(
                null as PredictionResult?
            )
        }


        var loading by remember {
            mutableStateOf(false)
        }


        var errorMessage by remember {
            mutableStateOf("")
        }


        Column(

            modifier =
                Modifier
                    .fillMaxSize()
                    .background(
                        Color(
                            0xFFF5F7FB
                        )
                    )
                    .verticalScroll(
                        rememberScrollState()
                    )
                    .padding(
                        18.dp
                    ),

            verticalArrangement =
                Arrangement.spacedBy(
                    14.dp
                )
        ) {


            // =================================================
            // HEADER
            // =================================================

            Text(
                text =
                    "OLIST",

                fontSize =
                    28.sp,

                fontWeight =
                    FontWeight.Bold,

                color =
                    Color(
                        0xFF111827
                    )
            )


            Text(
                text =
                    "Customer Intelligence",

                fontSize =
                    14.sp,

                color =
                    Color(
                        0xFF64748B
                    )
            )


            Text(
                text =
                    "Predict customer behavior and interests.",

                fontSize =
                    13.sp,

                color =
                    Color(
                        0xFF64748B
                    )
            )


            // =================================================
            // INPUT CARD
            // =================================================

            Card(

                modifier =
                    Modifier.fillMaxWidth(),

                shape =
                    RoundedCornerShape(
                        18.dp
                    ),

                colors =
                    CardDefaults.cardColors(
                        containerColor =
                            Color.White
                    )
            ) {

                Column(

                    modifier =
                        Modifier.padding(
                            18.dp
                        ),

                    verticalArrangement =
                        Arrangement.spacedBy(
                            10.dp
                        )
                ) {

                    Text(
                        text =
                            "Customer Information",

                        fontSize =
                            20.sp,

                        fontWeight =
                            FontWeight.Bold
                    )


                    Text(
                        text =
                            "Required Inputs",

                        fontSize =
                            13.sp,

                        color =
                            Color(
                                0xFF64748B
                            )
                    )


                    SimpleInput(
                        value =
                            priorOrderCount,

                        onValueChange = {
                            priorOrderCount =
                                it
                        },

                        label =
                            "Prior Order Count"
                    )


                    SimpleInput(
                        value =
                            recencyDays,

                        onValueChange = {
                            recencyDays =
                                it
                        },

                        label =
                            "Recency Days"
                    )


                    SimpleInput(
                        value =
                            totalSpending,

                        onValueChange = {
                            totalSpending =
                                it
                        },

                        label =
                            "Total Spending"
                    )


                    SimpleInput(
                        value =
                            averageOrderValue,

                        onValueChange = {
                            averageOrderValue =
                                it
                        },

                        label =
                            "Average Order Value"
                    )


                    SimpleInput(
                        value =
                            totalItems,

                        onValueChange = {
                            totalItems =
                                it
                        },

                        label =
                            "Total Items"
                    )


                    SimpleInput(
                        value =
                            averageReviewScore,

                        onValueChange = {
                            averageReviewScore =
                                it
                        },

                        label =
                            "Average Review Score"
                    )


                    SimpleInput(
                        value =
                            customerState,

                        onValueChange = {
                            customerState =
                                it
                        },

                        label =
                            "Customer State"
                    )


                    SimpleInput(
                        value =
                            paymentType,

                        onValueChange = {
                            paymentType =
                                it
                        },

                        label =
                            "Dominant Payment Type"
                    )


                    OutlinedTextField(

                        value =
                            reviewText,

                        onValueChange = {
                            reviewText =
                                it
                        },

                        modifier =
                            Modifier.fillMaxWidth(),

                        label = {
                            Text(
                                "Review Text"
                            )
                        },

                        placeholder = {
                            Text(
                                "Example: Great product and fast delivery"
                            )
                        },

                        minLines =
                            4,

                        maxLines =
                            6,

                        shape =
                            RoundedCornerShape(
                                12.dp
                            )
                    )
                }
            }


            // =================================================
            // BUTTONS
            // =================================================

            Row(

                modifier =
                    Modifier.fillMaxWidth(),

                horizontalArrangement =
                    Arrangement.spacedBy(
                        10.dp
                    )
            ) {

                Button(

                    modifier =
                        Modifier.weight(
                            1f
                        ),

                    enabled =
                        !loading,

                    onClick = {

                        loading =
                            true

                        errorMessage =
                            ""


                        lifecycleScope.launch {

                            try {

                                val sample =
                                    loadSample()


                                priorOrderCount =
                                    sample[
                                        "prior_order_count"
                                    ]?.toString()
                                        .orEmpty()


                                recencyDays =
                                    sample[
                                        "recency_days"
                                    ]?.toString()
                                        .orEmpty()


                                totalSpending =
                                    sample[
                                        "total_spending"
                                    ]?.toString()
                                        .orEmpty()


                                averageOrderValue =
                                    sample[
                                        "average_order_value"
                                    ]?.toString()
                                        .orEmpty()


                                totalItems =
                                    sample[
                                        "total_items"
                                    ]?.toString()
                                        .orEmpty()


                                averageReviewScore =
                                    sample[
                                        "average_review_score"
                                    ]?.toString()
                                        .orEmpty()


                                customerState =
                                    sample[
                                        "customer_state"
                                    ]?.toString()
                                        .orEmpty()


                                paymentType =
                                    sample[
                                        "dominant_payment_type"
                                    ]?.toString()
                                        .orEmpty()


                                reviewText =
                                    sample[
                                        "review_text"
                                    ]?.toString()
                                        .orEmpty()


                                result =
                                    null

                                Toast.makeText(
                                    this@MainActivity,
                                    "Sample loaded",
                                    Toast.LENGTH_SHORT
                                ).show()

                            }

                            catch (
                                e: Exception
                            ) {

                                errorMessage =
                                    "Cannot load sample:\n${
                                        e.message
                                    }"

                            }

                            finally {

                                loading =
                                    false
                            }
                        }
                    }
                ) {

                    Text(
                        "Load Sample"
                    )
                }


                Button(

                    modifier =
                        Modifier.weight(
                            1f
                        ),

                    enabled =
                        !loading,

                    colors =
                        ButtonDefaults.buttonColors(
                            containerColor =
                                Color(
                                    0xFF64748B
                                )
                        ),

                    onClick = {

                        priorOrderCount =
                            ""

                        recencyDays =
                            ""

                        totalSpending =
                            ""

                        averageOrderValue =
                            ""

                        totalItems =
                            ""

                        averageReviewScore =
                            ""

                        customerState =
                            ""

                        paymentType =
                            ""

                        reviewText =
                            ""

                        result =
                            null

                        errorMessage =
                            ""
                    }
                ) {

                    Text(
                        "Clear"
                    )
                }
            }


            // =================================================
            // PREDICT
            // =================================================

            Button(

                modifier =
                    Modifier
                        .fillMaxWidth()
                        .height(
                            52.dp
                        ),

                enabled =
                    !loading,

                onClick = {

                    val missing =
                        mutableListOf<String>()


                    if (
                        priorOrderCount.isBlank()
                    ) {
                        missing.add(
                            "Prior Order Count"
                        )
                    }

                    if (
                        recencyDays.isBlank()
                    ) {
                        missing.add(
                            "Recency Days"
                        )
                    }

                    if (
                        totalSpending.isBlank()
                    ) {
                        missing.add(
                            "Total Spending"
                        )
                    }

                    if (
                        averageOrderValue.isBlank()
                    ) {
                        missing.add(
                            "Average Order Value"
                        )
                    }

                    if (
                        totalItems.isBlank()
                    ) {
                        missing.add(
                            "Total Items"
                        )
                    }

                    if (
                        averageReviewScore.isBlank()
                    ) {
                        missing.add(
                            "Average Review Score"
                        )
                    }

                    if (
                        customerState.isBlank()
                    ) {
                        missing.add(
                            "Customer State"
                        )
                    }

                    if (
                        paymentType.isBlank()
                    ) {
                        missing.add(
                            "Dominant Payment Type"
                        )
                    }

                    if (
                        reviewText.isBlank()
                    ) {
                        missing.add(
                            "Review Text"
                        )
                    }


                    if (
                        missing.isNotEmpty()
                    ) {

                        errorMessage =
                            "Please fill in:\n" +
                                    missing.joinToString(
                                        "\n"
                                    )

                        return@Button
                    }


                    val payload =
                        mutableMapOf<
                                String,
                                Any?
                                >()


                    payload[
                        "prior_order_count"
                    ] =
                        priorOrderCount.toDoubleOrNull()


                    payload[
                        "recency_days"
                    ] =
                        recencyDays.toDoubleOrNull()


                    payload[
                        "total_spending"
                    ] =
                        totalSpending.toDoubleOrNull()


                    payload[
                        "average_order_value"
                    ] =
                        averageOrderValue.toDoubleOrNull()


                    payload[
                        "total_items"
                    ] =
                        totalItems.toDoubleOrNull()


                    payload[
                        "average_review_score"
                    ] =
                        averageReviewScore.toDoubleOrNull()


                    payload[
                        "customer_state"
                    ] =
                        customerState


                    payload[
                        "dominant_payment_type"
                    ] =
                        paymentType


                    payload[
                        "review_text"
                    ] =
                        reviewText


                    loading =
                        true

                    errorMessage =
                        ""

                    result =
                        null


                    lifecycleScope.launch {

                        try {

                            result =
                                predict(
                                    payload
                                )

                        }

                        catch (
                            e: Exception
                        ) {

                            errorMessage =
                                "Prediction failed:\n${
                                    e.message
                                }"

                        }

                        finally {

                            loading =
                                false
                        }
                    }
                }
            ) {

                if (
                    loading
                ) {

                    CircularProgressIndicator(
                        modifier =
                            Modifier
                                .height(
                                    20.dp
                                ),

                        color =
                            Color.White
                    )

                    Spacer(
                        modifier =
                            Modifier.width(
                                8.dp
                            )
                    )
                }


                Text(
                    "Predict Customer",
                    fontWeight =
                        FontWeight.Bold
                )
            }


            // =================================================
            // ERROR
            // =================================================

            if (
                errorMessage.isNotBlank()
            ) {

                Card(

                    modifier =
                        Modifier.fillMaxWidth(),

                    colors =
                        CardDefaults.cardColors(
                            containerColor =
                                Color(
                                    0xFFFEE2E2
                                )
                        )
                ) {

                    Text(

                        text =
                            errorMessage,

                        modifier =
                            Modifier.padding(
                                14.dp
                            ),

                        color =
                            Color(
                                0xFF991B1B
                            ),

                        fontSize =
                            13.sp
                    )
                }
            }


            // =================================================
            // RESULT
            // =================================================

            result?.let {

                PredictionResultView(
                    result =
                        it
                )
            }


            Spacer(
                modifier =
                    Modifier.height(
                        20.dp
                    )
            )
        }
    }
}


// ============================================================
// INPUT
// ============================================================

@Composable
private fun SimpleInput(
    value: String,
    onValueChange: (String) -> Unit,
    label: String
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

        shape =
            RoundedCornerShape(
                12.dp
            )
    )
}


// ============================================================
// RESULT VIEW
// ============================================================

@Composable
private fun PredictionResultView(
    result:
    PredictionResult
) {

    Column(

        verticalArrangement =
            Arrangement.spacedBy(
                14.dp
            )
    ) {


        Text(

            text =
                "Prediction Result",

            fontSize =
                24.sp,

            fontWeight =
                FontWeight.Bold,

            color =
                Color(
                    0xFF111827
                )
        )


        // ----------------------------------------------------
        // Behavior
        // ----------------------------------------------------

        ResultCard(

            title =
                "Customer Behavior",

            value =
                result.behavior,

            confidence =
                result.behaviorConfidence
        )


        // ----------------------------------------------------
        // Interest
        // ----------------------------------------------------

        ResultCard(

            title =
                "Customer Interest",

            value =
                result.interest,

            confidence =
                result.interestConfidence
        )


        // ----------------------------------------------------
        // Probability
        // ----------------------------------------------------

        Card(

            modifier =
                Modifier.fillMaxWidth(),

            shape =
                RoundedCornerShape(
                    18.dp
                ),

            colors =
                CardDefaults.cardColors(
                    containerColor =
                        Color.White
                )
        ) {

            Column(

                modifier =
                    Modifier.padding(
                        18.dp
                    )
            ) {

                Text(

                    text =
                        "Interest Probability",

                    fontSize =
                        19.sp,

                    fontWeight =
                        FontWeight.Bold
                )


                Spacer(
                    modifier =
                        Modifier.height(
                            12.dp
                        )
                )


                result.probabilities.forEach {

                        (
                            category,
                            probability
                        ) ->


                    Text(

                        text =
                            "$category  ${
                                percent(
                                    probability
                                )
                            }",

                        fontSize =
                            13.sp
                    )


                    Spacer(
                        modifier =
                            Modifier.height(
                                4.dp
                            )
                    )


                    LinearProgressIndicator(

                        progress = {
                            probability
                                .toFloat()
                                .coerceIn(
                                    0f,
                                    1f
                                )
                        },

                        modifier =
                            Modifier.fillMaxWidth()
                    )


                    Spacer(
                        modifier =
                            Modifier.height(
                                8.dp
                            )
                    )
                }
            }
        }


        // ----------------------------------------------------
        // Model
        // ----------------------------------------------------

        Card(

            modifier =
                Modifier.fillMaxWidth(),

            shape =
                RoundedCornerShape(
                    18.dp
                ),

            colors =
                CardDefaults.cardColors(
                    containerColor =
                        Color.White
                )
        ) {

            Column(

                modifier =
                    Modifier.padding(
                        18.dp
                    )
            ) {

                Text(

                    text =
                        "Model Information",

                    fontSize =
                        19.sp,

                    fontWeight =
                        FontWeight.Bold
                )


                Spacer(
                    modifier =
                        Modifier.height(
                            8.dp
                        )
                )


                Text(
                    "Model: ${result.model}"
                )


                Spacer(
                    modifier =
                        Modifier.height(
                            4.dp
                        )
                )


                Text(

                    text =
                        "Representation: ${result.representation}",

                    fontSize =
                        13.sp,

                    color =
                        Color(
                            0xFF64748B
                        )
                )
            }
        }


        // ----------------------------------------------------
        // Top 5
        // ----------------------------------------------------

        Card(

            modifier =
                Modifier.fillMaxWidth(),

            shape =
                RoundedCornerShape(
                    18.dp
                ),

            colors =
                CardDefaults.cardColors(
                    containerColor =
                        Color.White
                )
        ) {

            Column(

                modifier =
                    Modifier.padding(
                        18.dp
                    )
            ) {

                Text(

                    text =
                        "Top 5 Recommended Products",

                    fontSize =
                        19.sp,

                    fontWeight =
                        FontWeight.Bold
                )


                Spacer(
                    modifier =
                        Modifier.height(
                            5.dp
                        )
                )


                Text(

                    text =
                        "Products related to ${result.interest}",

                    fontSize =
                        13.sp,

                    color =
                        Color(
                            0xFF64748B
                        )
                )


                Spacer(
                    modifier =
                        Modifier.height(
                            14.dp
                        )
                )


                if (
                    result.products.isEmpty()
                ) {

                    Text(
                        "No products found for this category."
                    )

                } else {

                    result.products
                        .take(5)
                        .forEachIndexed {

                                index,
                                product ->


                            ProductRow(

                                rank =
                                    index + 1,

                                product =
                                    product
                            )


                            if (
                                index <
                                result.products
                                    .take(5)
                                    .lastIndex
                            ) {

                                Spacer(
                                    modifier =
                                        Modifier.height(
                                            12.dp
                                        )
                                )
                            }
                        }
                }
            }
        }
    }
}


// ============================================================
// RESULT CARD
// ============================================================

@Composable
private fun ResultCard(
    title: String,
    value: String,
    confidence: Double?
) {

    Card(

        modifier =
            Modifier.fillMaxWidth(),

        shape =
            RoundedCornerShape(
                18.dp
            ),

        colors =
            CardDefaults.cardColors(
                containerColor =
                    Color.White
            )
    ) {

        Column(

            modifier =
                Modifier.padding(
                    18.dp
                )
        ) {

            Text(

                text =
                    title,

                fontSize =
                    13.sp,

                color =
                    Color(
                        0xFF64748B
                    )
            )


            Spacer(
                modifier =
                    Modifier.height(
                        6.dp
                    )
            )


            Text(

                text =
                    value,

                fontSize =
                    24.sp,

                fontWeight =
                    FontWeight.Bold,

                color =
                    Color(
                        0xFF312E81
                    ),

                maxLines =
                    2,

                overflow =
                    TextOverflow.Ellipsis
            )


            Spacer(
                modifier =
                    Modifier.height(
                        6.dp
                    )
            )


            Text(

                text =
                    "Confidence: ${
                        percent(
                            confidence
                        )
                    }",

                fontSize =
                    13.sp,

                color =
                    Color(
                        0xFF64748B
                    )
            )
        }
    }
}


// ============================================================
// PRODUCT
// ============================================================

@Composable
private fun ProductRow(
    rank: Int,
    product:
    ProductResult
) {

    Column {

        Text(

            text =
                "#$rank  ${product.productId}",

            fontWeight =
                FontWeight.Bold,

            fontSize =
                14.sp
        )


        Spacer(
            modifier =
                Modifier.height(
                    4.dp
                )
        )


        Text(
            text =
                "Category: ${product.category}",

            fontSize =
                12.sp,

            color =
                Color(
                    0xFF64748B
                )
        )


        Text(
            text =
                "Price: ${
                    String.format(
                        "%.2f",
                        product.price
                    )
                }",

            fontSize =
                12.sp
        )


        Text(
            text =
                "Rating: ${
                    String.format(
                        "%.2f",
                        product.rating
                    )
                }",

            fontSize =
                12.sp
        )


        Text(
            text =
                "Sold units: ${product.soldUnits}",

            fontSize =
                12.sp
        )


        Text(
            text =
                "Recommendation score: ${
                    String.format(
                        "%.3f",
                        product.recommendationScore
                    )
                }",

            fontSize =
                12.sp,

            color =
                Color(
                    0xFF4F46E5
                )
        )
    }
}


// ============================================================
// FORMAT
// ============================================================

private fun percent(
    value: Double?
): String {

    if (
        value == null
    ) {
        return "N/A"
    }

    return String.format(
        "%.1f%%",
        value * 100
    )
}