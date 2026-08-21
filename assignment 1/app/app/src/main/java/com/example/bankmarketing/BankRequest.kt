package com.example.bankmarketing

import com.google.gson.annotations.SerializedName

data class BankRequest(

    val age: Int,

    val job: String,

    val marital: String,

    val education: String,

    val default: String,

    val housing: String,

    val loan: String,

    val contact: String,

    val month: String,

    val day_of_week: String,

    val campaign: Int,

    val pdays: Int,

    val previous: Int,

    val poutcome: String,

    @SerializedName("emp.var.rate")
    val empVarRate: Double,

    @SerializedName("cons.price.idx")
    val consPriceIdx: Double,

    @SerializedName("cons.conf.idx")
    val consConfIdx: Double,

    val euribor3m: Double,

    @SerializedName("nr.employed")
    val nrEmployed: Double
)