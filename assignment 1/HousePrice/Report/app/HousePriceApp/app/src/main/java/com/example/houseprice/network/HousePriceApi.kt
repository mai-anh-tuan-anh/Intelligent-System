package com.example.houseprice.network

import com.example.houseprice.data.HousePriceRequest
import com.example.houseprice.data.HousePriceResponse
import retrofit2.Call
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST

interface HousePriceApi {

    @GET("house/v1/options")
    fun getOptions(): Call<Map<String, List<String>>>

    @POST("house/v1/predict")
    fun predict(
        @Body request: HousePriceRequest
    ): Call<HousePriceResponse>
}