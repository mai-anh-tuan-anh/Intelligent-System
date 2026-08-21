package com.example.bankmarketing

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.launch


class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                BankMarketingApp()
            }
        }
    }
}


@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BankMarketingApp() {

    // =========================================================
    // COROUTINE SCOPE
    // =========================================================

    val scope = rememberCoroutineScope()

    // =========================================================
    // NUMERICAL VALUES
    // =========================================================

    var age by remember {
        mutableStateOf("30")
    }

    var campaign by remember {
        mutableStateOf("1")
    }

    var pdays by remember {
        mutableStateOf("999")
    }

    var previous by remember {
        mutableStateOf("0")
    }

    var empVarRate by remember {
        mutableStateOf("-3.0")
    }

    var consPriceIdx by remember {
        mutableStateOf("92.0")
    }

    var consConfIdx by remember {
        mutableStateOf("-40.0")
    }

    var euribor3m by remember {
        mutableStateOf("0.8")
    }

    var nrEmployed by remember {
        mutableStateOf("5000.0")
    }

    // =========================================================
    // CATEGORICAL VALUES
    // =========================================================

    var job by remember {
        mutableStateOf("student")
    }

    var marital by remember {
        mutableStateOf("single")
    }

    var education by remember {
        mutableStateOf("high.school")
    }

    var defaultStatus by remember {
        mutableStateOf("no")
    }

    var housing by remember {
        mutableStateOf("no")
    }

    var loan by remember {
        mutableStateOf("no")
    }

    var contact by remember {
        mutableStateOf("cellular")
    }

    var month by remember {
        mutableStateOf("mar")
    }

    var dayOfWeek by remember {
        mutableStateOf("thu")
    }

    var poutcome by remember {
        mutableStateOf("success")
    }

    // =========================================================
    // RESULT STATES
    // =========================================================

    var prediction by remember {
        mutableStateOf<String?>(null)
    }

    var errorMessage by remember {
        mutableStateOf<String?>(null)
    }

    var isLoading by remember {
        mutableStateOf(false)
    }

    Scaffold(
        topBar = {

            TopAppBar(
                title = {
                    Text(
                        text = "Bank Marketing Prediction",
                        fontWeight = FontWeight.Bold
                    )
                }
            )
        }

    ) { paddingValues ->

        Column(

            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(horizontal = 16.dp)
                .verticalScroll(
                    rememberScrollState()
                )

        ) {

            Spacer(
                modifier = Modifier.height(16.dp)
            )

            Text(
                text = "Predict whether a client is likely to subscribe to a term deposit.",
                style = MaterialTheme.typography.bodyMedium
            )

            Spacer(
                modifier = Modifier.height(20.dp)
            )

            // =================================================
            // CUSTOMER INFORMATION
            // =================================================

            SectionTitle(
                text = "Customer Information"
            )

            Spacer(
                modifier = Modifier.height(10.dp)
            )

            NumberInput(
                label = "Age",
                value = age,
                onValueChange = {
                    age = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            DropdownField(
                label = "Job",
                value = job,
                options = listOf(
                    "admin.",
                    "blue-collar",
                    "entrepreneur",
                    "housemaid",
                    "management",
                    "retired",
                    "self-employed",
                    "services",
                    "student",
                    "technician",
                    "unemployed",
                    "unknown"
                ),
                onValueChange = {
                    job = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            DropdownField(
                label = "Marital Status",
                value = marital,
                options = listOf(
                    "married",
                    "single",
                    "divorced",
                    "unknown"
                ),
                onValueChange = {
                    marital = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            DropdownField(
                label = "Education",
                value = education,
                options = listOf(
                    "basic.4y",
                    "basic.6y",
                    "basic.9y",
                    "high.school",
                    "illiterate",
                    "professional.course",
                    "university.degree",
                    "unknown"
                ),
                onValueChange = {
                    education = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            DropdownField(
                label = "Credit Default",
                value = defaultStatus,
                options = listOf(
                    "no",
                    "yes",
                    "unknown"
                ),
                onValueChange = {
                    defaultStatus = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            DropdownField(
                label = "Housing Loan",
                value = housing,
                options = listOf(
                    "no",
                    "yes",
                    "unknown"
                ),
                onValueChange = {
                    housing = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            DropdownField(
                label = "Personal Loan",
                value = loan,
                options = listOf(
                    "no",
                    "yes",
                    "unknown"
                ),
                onValueChange = {
                    loan = it
                }
            )

            Spacer(
                modifier = Modifier.height(20.dp)
            )

            // =================================================
            // CAMPAIGN INFORMATION
            // =================================================

            SectionTitle(
                text = "Campaign Information"
            )

            Spacer(
                modifier = Modifier.height(10.dp)
            )

            DropdownField(
                label = "Contact Type",
                value = contact,
                options = listOf(
                    "cellular",
                    "telephone"
                ),
                onValueChange = {
                    contact = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            DropdownField(
                label = "Month",
                value = month,
                options = listOf(
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec"
                ),
                onValueChange = {
                    month = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            DropdownField(
                label = "Day of Week",
                value = dayOfWeek,
                options = listOf(
                    "mon",
                    "tue",
                    "wed",
                    "thu",
                    "fri"
                ),
                onValueChange = {
                    dayOfWeek = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            NumberInput(
                label = "Campaign Contacts",
                value = campaign,
                onValueChange = {
                    campaign = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            NumberInput(
                label = "Days Since Previous Contact",
                value = pdays,
                onValueChange = {
                    pdays = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            NumberInput(
                label = "Previous Contacts",
                value = previous,
                onValueChange = {
                    previous = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            DropdownField(
                label = "Previous Campaign Outcome",
                value = poutcome,
                options = listOf(
                    "success",
                    "failure",
                    "nonexistent"
                ),
                onValueChange = {
                    poutcome = it
                }
            )

            Spacer(
                modifier = Modifier.height(20.dp)
            )

            // =================================================
            // ECONOMIC INFORMATION
            // =================================================

            SectionTitle(
                text = "Economic Information"
            )

            Spacer(
                modifier = Modifier.height(10.dp)
            )

            NumberInput(
                label = "Employment Variation Rate",
                value = empVarRate,
                onValueChange = {
                    empVarRate = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            NumberInput(
                label = "Consumer Price Index",
                value = consPriceIdx,
                onValueChange = {
                    consPriceIdx = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            NumberInput(
                label = "Consumer Confidence Index",
                value = consConfIdx,
                onValueChange = {
                    consConfIdx = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            NumberInput(
                label = "Euribor 3 Month",
                value = euribor3m,
                onValueChange = {
                    euribor3m = it
                }
            )

            Spacer(
                modifier = Modifier.height(8.dp)
            )

            NumberInput(
                label = "Number of Employees",
                value = nrEmployed,
                onValueChange = {
                    nrEmployed = it
                }
            )

            Spacer(
                modifier = Modifier.height(24.dp)
            )

            // =================================================
            // PREDICT BUTTON
            // =================================================

            Button(

                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp),

                onClick = {

                    prediction = null
                    errorMessage = null

                    // -----------------------------------------
                    // Validate numerical fields
                    // -----------------------------------------

                    val ageValue =
                        age.toIntOrNull()

                    val campaignValue =
                        campaign.toIntOrNull()

                    val pdaysValue =
                        pdays.toIntOrNull()

                    val previousValue =
                        previous.toIntOrNull()

                    val empVarRateValue =
                        empVarRate.toDoubleOrNull()

                    val consPriceIdxValue =
                        consPriceIdx.toDoubleOrNull()

                    val consConfIdxValue =
                        consConfIdx.toDoubleOrNull()

                    val euribor3mValue =
                        euribor3m.toDoubleOrNull()

                    val nrEmployedValue =
                        nrEmployed.toDoubleOrNull()

                    if (
                        ageValue == null ||
                        campaignValue == null ||
                        pdaysValue == null ||
                        previousValue == null ||
                        empVarRateValue == null ||
                        consPriceIdxValue == null ||
                        consConfIdxValue == null ||
                        euribor3mValue == null ||
                        nrEmployedValue == null
                    ) {

                        errorMessage =
                            "Please check the numeric input values."

                        return@Button
                    }

                    // -----------------------------------------
                    // Build request
                    // -----------------------------------------

                    val request = BankRequest(

                        age = ageValue,

                        job = job,

                        marital = marital,

                        education = education,

                        default = defaultStatus,

                        housing = housing,

                        loan = loan,

                        contact = contact,

                        month = month,

                        day_of_week = dayOfWeek,

                        campaign = campaignValue,

                        pdays = pdaysValue,

                        previous = previousValue,

                        poutcome = poutcome,

                        empVarRate =
                            empVarRateValue,

                        consPriceIdx =
                            consPriceIdxValue,

                        consConfIdx =
                            consConfIdxValue,

                        euribor3m =
                            euribor3mValue,

                        nrEmployed =
                            nrEmployedValue
                    )

                    // -----------------------------------------
                    // Start API request
                    // -----------------------------------------

                    isLoading = true

                    scope.launch {

                        try {

                            val response =
                                RetrofitClient.api.predict(
                                    request
                                )

                            prediction =
                                response.label

                        } catch (e: Exception) {

                            errorMessage =
                                "Unable to connect to prediction server."

                        } finally {

                            isLoading = false
                        }
                    }
                }

            ) {

                Text(
                    text = "Predict"
                )
            }

            Spacer(
                modifier = Modifier.height(16.dp)
            )

            // =================================================
            // LOADING
            // =================================================

            if (isLoading) {

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement =
                        Arrangement.Center
                ) {

                    CircularProgressIndicator()
                }

                Spacer(
                    modifier = Modifier.height(16.dp)
                )
            }

            // =================================================
            // PREDICTION RESULT
            // =================================================

            prediction?.let { result ->

                Card(

                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp),

                    colors = CardDefaults.cardColors(
                        containerColor =
                            MaterialTheme.colorScheme.primaryContainer
                    )

                ) {

                    Column(
                        modifier = Modifier.padding(20.dp)
                    ) {

                        Text(
                            text = "Prediction Result",
                            style =
                                MaterialTheme.typography.titleMedium,
                            fontWeight =
                                FontWeight.Bold
                        )

                        Spacer(
                            modifier = Modifier.height(8.dp)
                        )

                        Text(
                            text = result,
                            style =
                                MaterialTheme.typography.headlineSmall,
                            fontWeight =
                                FontWeight.Bold
                        )
                    }
                }
            }

            // =================================================
            // ERROR MESSAGE
            // =================================================

            errorMessage?.let { error ->

                Card(

                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 16.dp),

                    colors = CardDefaults.cardColors(
                        containerColor =
                            MaterialTheme.colorScheme.errorContainer
                    )

                ) {

                    Text(
                        text = error,
                        modifier = Modifier.padding(16.dp),
                        color =
                            MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            }

            Spacer(
                modifier = Modifier.height(24.dp)
            )
        }
    }
}


// =============================================================
// SECTION TITLE
// =============================================================

@Composable
fun SectionTitle(
    text: String
) {

    Text(
        text = text,
        style =
            MaterialTheme.typography.titleLarge,
        fontWeight =
            FontWeight.Bold
    )
}


// =============================================================
// NUMERICAL INPUT
// =============================================================

@Composable
fun NumberInput(
    label: String,
    value: String,
    onValueChange: (String) -> Unit
) {

    OutlinedTextField(

        value = value,

        onValueChange = onValueChange,

        label = {
            Text(label)
        },

        modifier =
            Modifier.fillMaxWidth(),

        singleLine = true
    )
}


// =============================================================
// DROPDOWN INPUT
// =============================================================

@Composable
fun DropdownField(
    label: String,
    value: String,
    options: List<String>,
    onValueChange: (String) -> Unit
) {

    var expanded by remember {
        mutableStateOf(false)
    }

    Column(
        modifier =
            Modifier.fillMaxWidth()
    ) {

        Text(
            text = label,
            style =
                MaterialTheme.typography.bodyMedium
        )

        Spacer(
            modifier = Modifier.height(4.dp)
        )

        Box(
            modifier =
                Modifier.fillMaxWidth()
        ) {

            OutlinedButton(

                modifier =
                    Modifier.fillMaxWidth(),

                onClick = {
                    expanded = true
                }

            ) {

                Row(
                    modifier =
                        Modifier.fillMaxWidth(),

                    horizontalArrangement =
                        Arrangement.SpaceBetween
                ) {

                    Text(
                        text = value
                    )

                    Text(
                        text = "▼"
                    )
                }
            }

            DropdownMenu(

                expanded =
                    expanded,

                onDismissRequest = {
                    expanded = false
                }

            ) {

                options.forEach { option ->

                    DropdownMenuItem(

                        text = {
                            Text(option)
                        },

                        onClick = {

                            onValueChange(
                                option
                            )

                            expanded = false
                        }
                    )
                }
            }
        }
    }
}