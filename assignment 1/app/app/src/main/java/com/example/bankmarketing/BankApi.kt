package com.example.bankmarketing

import retrofit2.http.Body
import retrofit2.http.POST

interface BankApi {

    @POST("predict")
    suspend fun predict(
        @Body request: BankRequest
    ): BankResponse
}