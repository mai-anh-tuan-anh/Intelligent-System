package com.example.houseprice.network

import java.util.concurrent.TimeUnit

import okhttp3.OkHttpClient

import com.google.gson.GsonBuilder

import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory


object RetrofitClient {

    private const val BASE_URL =
        "http://192.168.33.149:5000/"


    private val client =
        OkHttpClient.Builder()
            .connectTimeout(
                10,
                TimeUnit.SECONDS
            )
            .readTimeout(
                10,
                TimeUnit.SECONDS
            )
            .writeTimeout(
                10,
                TimeUnit.SECONDS
            )
            .build()


    private val gson =
        GsonBuilder()
            .serializeNulls()
            .create()


    val api: HousePriceApi by lazy {

        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(
                GsonConverterFactory.create(gson)
            )
            .build()
            .create(
                HousePriceApi::class.java
            )
    }
}