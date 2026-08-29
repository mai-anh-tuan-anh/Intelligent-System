package com.example.houseprice.data

import com.google.gson.annotations.SerializedName

data class HousePriceRequest(

    @SerializedName("Address")
    val address: String?,

    @SerializedName("Area")
    val area: Float?,

    @SerializedName("Frontage")
    val frontage: Float?,

    @SerializedName("Access Road")
    val accessRoad: Float?,

    @SerializedName("House direction")
    val houseDirection: String?,

    @SerializedName("Balcony direction")
    val balconyDirection: String?,

    @SerializedName("Floors")
    val floors: Float?,

    @SerializedName("Bedrooms")
    val bedrooms: Float?,

    @SerializedName("Bathrooms")
    val bathrooms: Float?,

    @SerializedName("Legal status")
    val legalStatus: String?,

    @SerializedName("Furniture state")
    val furnitureState: String?,

    @SerializedName("Address_Copy")
    val addressCopy: String?,

    @SerializedName("City_Province")
    val cityProvince: String?,

    @SerializedName("District")
    val district: String?
)