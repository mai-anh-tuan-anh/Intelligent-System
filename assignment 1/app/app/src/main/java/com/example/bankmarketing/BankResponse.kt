package com.example.bankmarketing

data class BankResponse(

    val prediction: Int,

    val label: String,

    val probability_no: Double,

    val probability_yes: Double
)