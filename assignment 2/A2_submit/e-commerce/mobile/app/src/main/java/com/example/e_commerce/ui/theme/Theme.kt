package com.example.ecommerce.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val AppColors = lightColorScheme()

@Composable
fun EcommerceTheme(
    content: @Composable () -> Unit
) {

    MaterialTheme(
        colorScheme = AppColors,
        content = content
    )
}