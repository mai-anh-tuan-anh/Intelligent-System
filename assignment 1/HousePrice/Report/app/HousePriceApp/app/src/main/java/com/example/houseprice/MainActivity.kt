package com.example.houseprice

import android.os.Bundle
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast

import androidx.appcompat.app.AppCompatActivity

import com.example.houseprice.data.HousePriceRequest
import com.example.houseprice.data.HousePriceResponse
import com.example.houseprice.network.RetrofitClient

import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response


class MainActivity : AppCompatActivity() {

    private lateinit var progressBar: ProgressBar
    private lateinit var resultText: TextView

    private lateinit var houseDirectionSpinner: Spinner
    private lateinit var balconyDirectionSpinner: Spinner
    private lateinit var legalSpinner: Spinner
    private lateinit var furnitureSpinner: Spinner


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(
            R.layout.activity_main
        )

        initViews()
        setupButtons()
        loadOptions()
    }


    // =====================================================
    // INIT VIEWS
    // =====================================================

    private fun initViews() {

        progressBar = findViewById(
            R.id.progressBar
        )

        resultText = findViewById(
            R.id.txtResult
        )

        houseDirectionSpinner = findViewById(
            R.id.spinnerHouseDirection
        )

        balconyDirectionSpinner = findViewById(
            R.id.spinnerBalconyDirection
        )

        legalSpinner = findViewById(
            R.id.spinnerLegal
        )

        furnitureSpinner = findViewById(
            R.id.spinnerFurniture
        )
    }


    // =====================================================
    // BUTTONS
    // =====================================================

    private fun setupButtons() {

        findViewById<Button>(
            R.id.btnPredict
        ).setOnClickListener {

            predictHousePrice()
        }


        findViewById<Button>(
            R.id.btnClear
        ).setOnClickListener {

            clearForm()
        }
    }


    // =====================================================
    // LOAD OPTIONS
    // =====================================================

    private fun loadOptions() {

        RetrofitClient.api
            .getOptions()
            .enqueue(
                object :
                    Callback<Map<String, List<String>>> {

                    override fun onResponse(
                        call: Call<Map<String, List<String>>>,
                        response: Response<Map<String, List<String>>>
                    ) {

                        if (!response.isSuccessful) {

                            Toast.makeText(
                                this@MainActivity,
                                "Không tải được danh sách lựa chọn",
                                Toast.LENGTH_LONG
                            ).show()

                            return
                        }


                        val options =
                            response.body()


                        if (options == null) {

                            Toast.makeText(
                                this@MainActivity,
                                "Server không trả dữ liệu",
                                Toast.LENGTH_LONG
                            ).show()

                            return
                        }


                        setupSpinner(
                            houseDirectionSpinner,
                            options["House direction"]
                        )


                        setupSpinner(
                            balconyDirectionSpinner,
                            options["Balcony direction"]
                        )


                        setupSpinner(
                            legalSpinner,
                            options["Legal status"]
                        )


                        setupSpinner(
                            furnitureSpinner,
                            options["Furniture state"]
                        )
                    }


                    override fun onFailure(
                        call: Call<Map<String, List<String>>>,
                        t: Throwable
                    ) {

                        Toast.makeText(
                            this@MainActivity,
                            "Không thể kết nối server",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            )
    }


    // =====================================================
    // SPINNER
    // =====================================================

    private fun setupSpinner(
        spinner: Spinner,
        values: List<String>?
    ) {

        val items =
            mutableListOf<String>()

        items.add("Không chọn")


        if (values != null) {
            items.addAll(values)
        }


        val adapter =
            ArrayAdapter(
                this,
                android.R.layout.simple_spinner_item,
                items
            )


        adapter.setDropDownViewResource(
            android.R.layout.simple_spinner_dropdown_item
        )


        spinner.adapter =
            adapter
    }


    // =====================================================
    // GET TEXT
    // =====================================================

    private fun getTextValue(
        id: Int
    ): String? {

        val value =
            findViewById<EditText>(id)
                .text
                .toString()
                .trim()


        return if (value.isEmpty()) {
            null
        } else {
            value
        }
    }


    // =====================================================
    // GET NUMBER
    // =====================================================

    private fun getNumberValue(
        id: Int
    ): Float? {

        val value =
            findViewById<EditText>(id)
                .text
                .toString()
                .trim()


        if (value.isEmpty()) {
            return null
        }


        return value.toFloatOrNull()
    }


    // =====================================================
    // GET SPINNER
    // =====================================================

    private fun getSpinnerValue(
        spinner: Spinner
    ): String? {

        val value =
            spinner.selectedItem
                ?.toString()


        if (
            value == null ||
            value == "Không chọn"
        ) {
            return null
        }


        return value
    }


    // =====================================================
    // PREDICT
    // =====================================================

    private fun predictHousePrice() {

        val area =
            getNumberValue(
                R.id.edtArea
            )


        // -------------------------------
        // Validate Area
        // -------------------------------

        if (area == null) {

            Toast.makeText(
                this,
                "Vui lòng nhập diện tích",
                Toast.LENGTH_SHORT
            ).show()

            return
        }


        if (area <= 0f) {

            Toast.makeText(
                this,
                "Diện tích phải lớn hơn 0",
                Toast.LENGTH_SHORT
            ).show()

            return
        }


        // -------------------------------
        // Floors
        // -------------------------------

        val floors =
            getNumberValue(
                R.id.edtFloors
            )


        if (
            floors != null &&
            floors < 0f
        ) {

            Toast.makeText(
                this,
                "Số tầng không hợp lệ",
                Toast.LENGTH_SHORT
            ).show()

            return
        }


        // -------------------------------
        // Address
        // -------------------------------

        val address =
            getTextValue(
                R.id.edtAddress
            )


        // -------------------------------
        // CREATE REQUEST
        // -------------------------------

        val request =
            HousePriceRequest(

                address = address,

                area = area,

                frontage =
                    getNumberValue(
                        R.id.edtFrontage
                    ),

                accessRoad =
                    getNumberValue(
                        R.id.edtAccessRoad
                    ),

                houseDirection =
                    getSpinnerValue(
                        houseDirectionSpinner
                    ),

                balconyDirection =
                    getSpinnerValue(
                        balconyDirectionSpinner
                    ),

                floors = floors,

                bedrooms =
                    getNumberValue(
                        R.id.edtBedrooms
                    ),

                bathrooms =
                    getNumberValue(
                        R.id.edtBathrooms
                    ),

                legalStatus =
                    getSpinnerValue(
                        legalSpinner
                    ),

                furnitureState =
                    getSpinnerValue(
                        furnitureSpinner
                    ),

                addressCopy =
                    address,

                cityProvince =
                    getTextValue(
                        R.id.edtCity
                    ),

                district =
                    getTextValue(
                        R.id.edtDistrict
                    )
            )


        // -------------------------------
        // LOADING
        // -------------------------------

        progressBar.visibility =
            View.VISIBLE


        resultText.text =
            "⏳ Đang dự đoán..."


        findViewById<Button>(
            R.id.btnPredict
        ).isEnabled = false


        // -------------------------------
        // API
        // -------------------------------

        RetrofitClient.api
            .predict(request)
            .enqueue(
                object :
                    Callback<HousePriceResponse> {

                    override fun onResponse(
                        call: Call<HousePriceResponse>,
                        response: Response<HousePriceResponse>
                    ) {

                        progressBar.visibility =
                            View.GONE


                        findViewById<Button>(
                            R.id.btnPredict
                        ).isEnabled = true


                        if (
                            response.isSuccessful &&
                            response.body() != null
                        ) {

                            val price =
                                response.body()!!
                                    .prediction


                            resultText.text =
                                String.format(
                                    "💰 Giá dự đoán\n%.2f tỷ VNĐ",
                                    price
                                )

                        } else {

                            val error =
                                response
                                    .errorBody()
                                    ?.string()


                            resultText.text =
                                "❌ Không thể dự đoán"


                            Toast.makeText(
                                this@MainActivity,
                                "Server lỗi: $error",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }


                    override fun onFailure(
                        call: Call<HousePriceResponse>,
                        t: Throwable
                    ) {

                        progressBar.visibility =
                            View.GONE


                        findViewById<Button>(
                            R.id.btnPredict
                        ).isEnabled = true


                        resultText.text =
                            "❌ Không thể kết nối server"


                        Toast.makeText(
                            this@MainActivity,
                            t.message
                                ?: "Connection error",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            )
    }


    // =====================================================
    // CLEAR FORM
    // =====================================================

    private fun clearForm() {

        findViewById<EditText>(
            R.id.edtAddress
        ).text.clear()


        findViewById<EditText>(
            R.id.edtArea
        ).text.clear()


        findViewById<EditText>(
            R.id.edtFrontage
        ).text.clear()


        findViewById<EditText>(
            R.id.edtAccessRoad
        ).text.clear()


        findViewById<EditText>(
            R.id.edtFloors
        ).text.clear()


        findViewById<EditText>(
            R.id.edtBedrooms
        ).text.clear()


        findViewById<EditText>(
            R.id.edtBathrooms
        ).text.clear()


        findViewById<EditText>(
            R.id.edtCity
        ).text.clear()


        findViewById<EditText>(
            R.id.edtDistrict
        ).text.clear()


        houseDirectionSpinner.setSelection(0)

        balconyDirectionSpinner.setSelection(0)

        legalSpinner.setSelection(0)

        furnitureSpinner.setSelection(0)


        resultText.text =
            "Giá dự đoán sẽ hiển thị ở đây"
    }
}