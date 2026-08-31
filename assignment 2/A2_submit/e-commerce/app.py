# ============================================================
# app.py
# Olist E-Commerce Customer Intelligence API
# ============================================================

from pathlib import Path
from typing import Any

import json
import math
import socket

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, ConfigDict


# ============================================================
# 1. JOBLIB COMPATIBILITY FUNCTION
# ============================================================

def text_to_1d(X):
    """
    Compatibility function required by the persisted sklearn
    pipeline created in the Jupyter Notebook.
    """
    return np.asarray(X).ravel()


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

CURRENT_DIR = Path(__file__).resolve()

ECOMMERCE_DIR = CURRENT_DIR.parent

MODEL_DIR = ECOMMERCE_DIR / "model"

WEB_DIR = ECOMMERCE_DIR / "web"


# ============================================================
# 3. MODEL ARTIFACTS
# ============================================================

INTEREST_MODEL_PATH = (
    MODEL_DIR / "interest_model_pipeline.joblib"
)

BEHAVIOR_MODEL_PATH = (
    MODEL_DIR / "behavior_model_pipeline.joblib"
)

RECOMMENDATION_CATALOG_PATH = (
    MODEL_DIR / "recommendation_catalog.joblib"
)

METADATA_PATH = (
    MODEL_DIR / "metadata.json"
)


# ============================================================
# 4. VERIFY ARTIFACTS
# ============================================================

required_files = [
    INTEREST_MODEL_PATH,
    BEHAVIOR_MODEL_PATH,
    RECOMMENDATION_CATALOG_PATH,
    METADATA_PATH,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.exists()
]

if missing_files:
    raise FileNotFoundError(
        "Missing required files:\n"
        + "\n".join(missing_files)
    )


# ============================================================
# 5. LOAD ARTIFACTS
# ============================================================

print("=" * 80)
print("OLIST CUSTOMER INTELLIGENCE API")
print("=" * 80)

print()
print("Model directory:")
print(MODEL_DIR.resolve())

print()
print("Loading saved artifacts...")

interest_model = joblib.load(
    INTEREST_MODEL_PATH
)

behavior_model = joblib.load(
    BEHAVIOR_MODEL_PATH
)

recommendation_catalog = joblib.load(
    RECOMMENDATION_CATALOG_PATH
)

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)


# ============================================================
# 6. METADATA
# ============================================================

NUMERICAL_FEATURES = list(
    metadata.get(
        "numerical_features",
        []
    )
)

CATEGORICAL_FEATURES = list(
    metadata.get(
        "categorical_features",
        []
    )
)

TEXT_FEATURES = list(
    metadata.get(
        "text_features",
        ["review_text"]
    )
)

TARGET_CATEGORIES = list(
    metadata.get(
        "interest_target_categories",
        []
    )
)

FINAL_MODEL_NAME = str(
    metadata.get(
        "final_model",
        "Unknown"
    )
)

FINAL_REPRESENTATION = str(
    metadata.get(
        "final_representation",
        "Unknown"
    )
)

MODEL_METRICS = metadata.get(
    "model_metrics",
    {}
)

VALIDATION_METRICS = metadata.get(
    "validation_metrics",
    {}
)

EXPERIMENTS = metadata.get(
    "experiments",
    {}
)


# ============================================================
# 7. USER INPUT GROUPING
# ============================================================

REQUIRED_NUMERICAL_FEATURES = [
    "prior_order_count",
    "recency_days",
    "total_spending",
    "average_order_value",
    "total_items",
    "average_review_score",
]

REQUIRED_CATEGORICAL_FEATURES = [
    "customer_state",
    "dominant_payment_type",
]

REQUIRED_TEXT_FEATURES = [
    "review_text",
]


REQUIRED_NUMERICAL_FEATURES = [
    feature
    for feature in REQUIRED_NUMERICAL_FEATURES
    if feature in NUMERICAL_FEATURES
]

REQUIRED_CATEGORICAL_FEATURES = [
    feature
    for feature in REQUIRED_CATEGORICAL_FEATURES
    if feature in CATEGORICAL_FEATURES
]

REQUIRED_TEXT_FEATURES = [
    feature
    for feature in REQUIRED_TEXT_FEATURES
    if feature in TEXT_FEATURES
]

REQUIRED_FEATURES = (
    REQUIRED_NUMERICAL_FEATURES
    +
    REQUIRED_CATEGORICAL_FEATURES
    +
    REQUIRED_TEXT_FEATURES
)


OTHER_NUMERICAL_FEATURES = [
    feature
    for feature in NUMERICAL_FEATURES
    if feature not in REQUIRED_NUMERICAL_FEATURES
]

OTHER_CATEGORICAL_FEATURES = [
    feature
    for feature in CATEGORICAL_FEATURES
    if feature not in REQUIRED_CATEGORICAL_FEATURES
]

OTHER_TEXT_FEATURES = [
    feature
    for feature in TEXT_FEATURES
    if feature not in REQUIRED_TEXT_FEATURES
]

OTHER_FEATURES = (
    OTHER_NUMERICAL_FEATURES
    +
    OTHER_CATEGORICAL_FEATURES
    +
    OTHER_TEXT_FEATURES
)


# ============================================================
# 8. CATEGORY TRANSLATION
# ============================================================

CATEGORY_DISPLAY_NAMES = {
    "esporte_lazer": "Sports & Leisure",
    "utilidades_domesticas": "Home Utilities",
    "beleza_saude": "Health & Beauty",
    "cama_mesa_banho": "Bed, Bath & Table",
    "brinquedos": "Toys",
    "fashion_bolsas_e_acessorios":
        "Fashion, Bags & Accessories",
    "moveis_decoracao": "Home & Decoration",
    "relogios_presentes": "Watches & Gifts",
    "informatica_acessorios":
        "Computers & Accessories",
    "telefonia": "Telephony",
    "automotivo": "Automotive",
    "bebes": "Baby",
    "ferramentas_jardim": "Garden Tools",
    "pet_shop": "Pet Shop",
    "papelaria": "Stationery",
    "instrumentos_musicais":
        "Musical Instruments",
    "cool_stuff": "Cool Stuff",
    "casa_conforto": "Home Comfort",
    "casa_construcao": "Home Construction",
    "eletrodomesticos": "Home Appliances",
    "eletrodomesticos_2":
        "Home Appliances",
    "market_place": "Marketplace",
    "alimentos": "Food",
    "bebidas": "Drinks",
    "perfumaria": "Perfume & Fragrance",
    "audio": "Audio",
    "livros_interesse_geral": "Books",
    "livros_tecnicos": "Technical Books",
    "artigos_de_festas": "Party Supplies",
    "artes": "Arts",
    "artes_e_artesanato": "Arts & Crafts",
    "fashion_calcados": "Fashion Footwear",
    "fashion_roupa_masculina": "Men's Fashion",
    "fashion_roupa_feminina": "Women's Fashion",
    "fashion_roupa_infanto_juvenil":
        "Children's Fashion",
    "fashion_underwear_e_moda_praia":
        "Underwear & Beachwear",
    "flores": "Flowers",
    "fraldas_higiene": "Diapers & Hygiene",
    "musica": "Music",
    "portateis_cozinha_e_preparadores_de_alimentos":
        "Kitchen Appliances",
    "sinalizacao_e_seguranca":
        "Signs & Security",
    "tablets_impressao_imagem":
        "Tablets & Printing",
    "telefonia_fixa": "Fixed Telephony",
    "consoles_games": "Consoles & Games",
    "cine_foto": "Cameras & Photography",
    "pcs": "PCs",
    "pc_gamer": "Gaming PCs",
    "seguros_e_servicos":
        "Insurance & Services",
    "industria_comercio_e_negocios":
        "Industry, Commerce & Business",
    "artigos_de_natal":
        "Christmas Supplies",
    "other": "Other",
}


def normalize_category_key(
    value: Any
) -> str:
    if value is None:
        return ""

    return " ".join(
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("&", " and ")
        .split()
    )


def get_english_category(
    category: Any
) -> str:

    if category is None:
        return "Other"

    original = str(
        category
    ).strip()

    if not original:
        return "Other"

    normalized = normalize_category_key(
        original
    )

    for key, display_name in (
        CATEGORY_DISPLAY_NAMES.items()
    ):

        if (
            normalize_category_key(key)
            ==
            normalized
        ):
            return display_name

    return original


# ============================================================
# 9. FASTAPI
# ============================================================

app = FastAPI(
    title="Olist Customer Intelligence API",
    description=(
        "Customer behavior prediction, customer interest "
        "prediction and Top 5 product recommendation."
    ),
    version="1.0.0",
)


# ============================================================
# 10. CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 11. REQUEST MODEL
# ============================================================

class PredictRequest(BaseModel):

    model_config = ConfigDict(
        extra="allow"
    )


# ============================================================
# 12. PREPARE INPUT
# ============================================================

def prepare_input(
    payload: dict[str, Any]
) -> pd.DataFrame:
    """
    Build the complete model feature vector.

    Missing optional numerical/categorical values are
    represented as NaN.

    They are NOT replaced with zero.
    """

    data: dict[str, Any] = {}


    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    for feature in NUMERICAL_FEATURES:

        value = payload.get(
            feature
        )

        if (
            value is None
            or
            value == ""
        ):

            data[feature] = np.nan

        else:

            try:

                data[feature] = float(
                    value
                )

            except (
                TypeError,
                ValueError
            ):

                data[feature] = np.nan


    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    for feature in CATEGORICAL_FEATURES:

        value = payload.get(
            feature
        )

        if (
            value is None
            or
            str(
                value
            ).strip()
            == ""
        ):

            data[feature] = np.nan

        else:

            data[feature] = str(
                value
            ).strip()


    # --------------------------------------------------------
    # Text features
    # --------------------------------------------------------

    for feature in TEXT_FEATURES:

        value = payload.get(
            feature
        )

        if (
            value is None
            or
            str(
                value
            ).strip()
            == ""
        ):

            data[feature] = ""

        else:

            data[feature] = str(
                value
            )


    # --------------------------------------------------------
    # Exact feature ordering
    # --------------------------------------------------------

    all_features = (
        NUMERICAL_FEATURES
        +
        CATEGORICAL_FEATURES
        +
        TEXT_FEATURES
    )

    return pd.DataFrame(
        [data],
        columns=all_features
    )


# ============================================================
# 13. PROBABILITY DISTRIBUTION
# ============================================================

def get_probability_distribution(
    model,
    X: pd.DataFrame
) -> dict[str, float]:

    if not hasattr(
        model,
        "predict_proba"
    ):

        return {}


    probabilities = (
        model
        .predict_proba(
            X
        )[0]
    )


    try:

        classifier = (
            model
            .named_steps[
                "classifier"
            ]
        )

        classes = (
            classifier.classes_
        )

    except Exception:

        classes = getattr(
            model,
            "classes_",
            range(
                len(
                    probabilities
                )
            )
        )


    result: dict[str, float] = {}


    for class_name, probability in zip(
        classes,
        probabilities
    ):

        display_name = (
            get_english_category(
                class_name
            )
        )

        result[
            display_name
        ] = round(
            float(
                probability
            ),
            6
        )


    return result


# ============================================================
# 14. RECOMMENDATION
# ============================================================

def recommend_products(
    predicted_interest: str,
    top_n: int = 5
) -> list[dict[str, Any]]:

    if (
        recommendation_catalog is None
        or
        recommendation_catalog.empty
        or
        "category_en"
        not in
        recommendation_catalog.columns
    ):

        return []


    catalog = (
        recommendation_catalog
        .copy()
    )


    catalog[
        "_display_category"
    ] = (
        catalog[
            "category_en"
        ]
        .apply(
            get_english_category
        )
    )


    catalog[
        "_normalized_category"
    ] = (
        catalog[
            "_display_category"
        ]
        .apply(
            normalize_category_key
        )
    )


    target = normalize_category_key(
        predicted_interest
    )


    matched = catalog[
        catalog[
            "_normalized_category"
        ]
        ==
        target
    ].copy()


    if matched.empty:
        return []


    sort_columns = []


    if (
        "recommendation_score"
        in
        matched.columns
    ):

        sort_columns.append(
            "recommendation_score"
        )


    if (
        "sold_units"
        in
        matched.columns
    ):

        sort_columns.append(
            "sold_units"
        )


    if sort_columns:

        matched = matched.sort_values(
            sort_columns,
            ascending=[
                False
                for _ in sort_columns
            ]
        )


    matched = matched.head(
        top_n
    )


    products = []


    for _, row in matched.iterrows():

        try:

            price = float(
                row.get(
                    "average_price",
                    0
                )
            )

        except Exception:

            price = 0.0


        try:

            rating = float(
                row.get(
                    "average_review_score",
                    0
                )
            )

        except Exception:

            rating = 0.0


        try:

            sold_units = int(
                row.get(
                    "sold_units",
                    0
                )
            )

        except Exception:

            sold_units = 0


        try:

            score = float(
                row.get(
                    "recommendation_score",
                    0
                )
            )

        except Exception:

            score = 0.0


        products.append({

            "product_id":
                str(
                    row.get(
                        "product_id",
                        ""
                    )
                ),

            "category":
                get_english_category(
                    row.get(
                        "category_en",
                        predicted_interest
                    )
                ),

            "price":
                round(
                    price,
                    2
                ),

            "rating":
                round(
                    rating,
                    2
                ),

            "sold_units":
                sold_units,

            "recommendation_score":
                round(
                    score,
                    4
                ),
        })


    return products


# ============================================================
# 15. CUSTOMER PROFILE
# ============================================================

def extract_customer_profile(
    X: pd.DataFrame
) -> dict[str, Any]:

    display_features = [

        "prior_order_count",

        "recency_days",

        "total_spending",

        "average_order_value",

        "total_items",

        "average_review_score",

        "customer_state",

        "dominant_payment_type",

        "review_text",
    ]


    profile: dict[str, Any] = {}


    for feature in display_features:

        if feature not in X.columns:
            continue


        value = X.iloc[0][
            feature
        ]


        if (
            value is None
            or
            pd.isna(value)
        ):

            continue


        if isinstance(
            value,
            (
                np.integer,
                int
            )
        ):

            profile[
                feature
            ] = int(
                value
            )


        elif isinstance(
            value,
            (
                np.floating,
                float
            )
        ):

            number = float(
                value
            )


            if math.isnan(
                number
            ):
                continue


            profile[
                feature
            ] = round(
                number,
                2
            )


        else:

            profile[
                feature
            ] = str(
                value
            )


    return profile


# ============================================================
# 16. COMPLETE SAMPLE CUSTOMER
# ============================================================

def build_complete_sample_customer() -> dict[str, Any]:
    """
    Build a complete sample for demonstration.

    All model features are filled so that clicking
    'Load Sample Customer' populates both Required and
    Other Inputs sections.
    """

    sample: dict[str, Any] = {}


    # --------------------------------------------------------
    # Meaningful numerical values
    # --------------------------------------------------------

    known_values = {

        "prior_order_count":
            5.0,

        "recency_days":
            18.0,

        "total_spending":
            620.50,

        "average_order_value":
            124.10,

        "total_items":
            9.0,

        "average_review_score":
            4.5,

        "max_order_value":
            245.0,

        "history_duration_days":
            142.0,

        "average_days_between_orders":
            35.5,

        "min_days_between_orders":
            18.0,

        "max_days_between_orders":
            55.0,

        "average_item_price":
            68.50,

        "average_freight_value":
            15.20,

        "average_items_per_order":
            1.8,

        "average_installments":
            2.5,

        "average_payment_value":
            124.10,

        "review_count":
            4.0,

        "text_review_count":
            3.0,
    }


    for feature in NUMERICAL_FEATURES:

        if feature in known_values:

            sample[
                feature
            ] = known_values[
                feature
            ]

            continue


        name = feature.lower()


        # ----------------------------------------------------
        # Category counts
        # ----------------------------------------------------

        if (
            "count"
            in
            name
        ):

            sample[
                feature
            ] = 2.0


        # ----------------------------------------------------
        # Category share
        # ----------------------------------------------------

        elif (
            "share"
            in
            name
        ):

            sample[
                feature
            ] = 0.20


        # ----------------------------------------------------
        # Ratio
        # ----------------------------------------------------

        elif (
            "ratio"
            in
            name
        ):

            sample[
                feature
            ] = 0.25


        # ----------------------------------------------------
        # Review / score
        # ----------------------------------------------------

        elif (
            "review"
            in
            name
            or
            "score"
            in
            name
        ):

            sample[
                feature
            ] = 4.0


        # ----------------------------------------------------
        # Time
        # ----------------------------------------------------

        elif (
            "day"
            in
            name
            or
            "duration"
            in
            name
        ):

            sample[
                feature
            ] = 30.0


        # ----------------------------------------------------
        # Money
        # ----------------------------------------------------

        elif (
            "price"
            in
            name
            or
            "value"
            in
            name
            or
            "amount"
            in
            name
            or
            "spending"
            in
            name
        ):

            sample[
                feature
            ] = 50.0


        # ----------------------------------------------------
        # Items / quantity
        # ----------------------------------------------------

        elif (
            "item"
            in
            name
            or
            "quantity"
            in
            name
        ):

            sample[
                feature
            ] = 2.0


        # ----------------------------------------------------
        # Installments
        # ----------------------------------------------------

        elif (
            "install"
            in
            name
        ):

            sample[
                feature
            ] = 2.0


        # ----------------------------------------------------
        # Generic numerical field
        # ----------------------------------------------------

        else:

            sample[
                feature
            ] = 1.0


    # ========================================================
    # CATEGORICAL
    # ========================================================

    for feature in CATEGORICAL_FEATURES:

        if feature == "customer_state":

            sample[
                feature
            ] = "SP"


        elif feature == "dominant_payment_type":

            sample[
                feature
            ] = "credit_card"


        else:

            sample[
                feature
            ] = "unknown"


    # ========================================================
    # TEXT
    # ========================================================

    for feature in TEXT_FEATURES:

        sample[
            feature
        ] = (
            "Good product, excellent quality, "
            "fast delivery and very satisfied "
            "with the purchase."
        )


    # ========================================================
    # Guarantee all columns exist
    # ========================================================

    all_features = (
        NUMERICAL_FEATURES
        +
        CATEGORICAL_FEATURES
        +
        TEXT_FEATURES
    )


    for feature in all_features:

        if feature in sample:
            continue


        if feature in NUMERICAL_FEATURES:

            sample[
                feature
            ] = 1.0


        elif feature in CATEGORICAL_FEATURES:

            sample[
                feature
            ] = "unknown"


        else:

            sample[
                feature
            ] = ""


    return sample


# ============================================================
# 17. HEALTH
# ============================================================

@app.get(
    "/health"
)
def health():

    total_features = (
        len(NUMERICAL_FEATURES)
        +
        len(CATEGORICAL_FEATURES)
        +
        len(TEXT_FEATURES)
    )


    return {

        "status":
            "ok",

        "service":
            "Olist Customer Intelligence API",

        "interest_model":
            FINAL_MODEL_NAME,

        "representation":
            FINAL_REPRESENTATION,

        "total_model_features":
            total_features,

        "required_input_count":
            len(REQUIRED_FEATURES),

        "optional_input_count":
            len(OTHER_FEATURES),

        "recommendation_catalog_size":
            int(
                len(
                    recommendation_catalog
                )
            ),
    }


# ============================================================
# 18. METADATA
# ============================================================

@app.get(
    "/metadata"
)
def get_metadata():

    return {

        "model":
            FINAL_MODEL_NAME,

        "representation":
            FINAL_REPRESENTATION,


        "numerical_features":
            NUMERICAL_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "text_features":
            TEXT_FEATURES,


        "required_inputs": {

            "numerical":
                REQUIRED_NUMERICAL_FEATURES,

            "categorical":
                REQUIRED_CATEGORICAL_FEATURES,

            "text":
                REQUIRED_TEXT_FEATURES,
        },


        "other_inputs": {

            "numerical":
                OTHER_NUMERICAL_FEATURES,

            "categorical":
                OTHER_CATEGORICAL_FEATURES,

            "text":
                OTHER_TEXT_FEATURES,
        },


        "target_categories":
            [
                get_english_category(
                    category
                )
                for category
                in TARGET_CATEGORIES
            ],


        "model_metrics":
            MODEL_METRICS,

        "validation_metrics":
            VALIDATION_METRICS,

        "experiments":
            EXPERIMENTS,
    }


# ============================================================
# 19. SAMPLE
# ============================================================

@app.get(
    "/sample"
)
def sample_customer():

    sample = (
        build_complete_sample_customer()
    )


    total_features = (
        len(NUMERICAL_FEATURES)
        +
        len(CATEGORICAL_FEATURES)
        +
        len(TEXT_FEATURES)
    )


    populated_features = sum(

        1

        for value in sample.values()

        if (
            value is not None
            and
            value != ""
        )
    )


    return {

        "success":
            True,

        "sample_name":
            "Complete Demo Customer",

        "description":
            (
                "Complete demonstration profile. "
                "Both Required and Other model inputs "
                "are populated."
            ),

        "total_model_features":
            total_features,

        "populated_features":
            populated_features,

        "features":
            sample,
    }


# ============================================================
# 20. PREDICT
# ============================================================

@app.post(
    "/predict"
)
def predict_customer(
    request: PredictRequest
):

    try:

        payload = (
            request.model_dump()
        )


        # ----------------------------------------------------
        # Validate required fields
        # ----------------------------------------------------

        missing_required = []


        for feature in (
            REQUIRED_FEATURES
        ):

            value = payload.get(
                feature
            )


            if (
                value is None
                or
                str(
                    value
                ).strip()
                == ""
            ):

                missing_required.append(
                    feature
                )


        if missing_required:

            raise HTTPException(

                status_code=400,

                detail={

                    "message":
                        "Required inputs are missing.",

                    "missing_fields":
                        missing_required,
                }
            )


        # ----------------------------------------------------
        # Prepare full feature vector
        # ----------------------------------------------------

        X = prepare_input(
            payload
        )


        # ----------------------------------------------------
        # Interest prediction
        # ----------------------------------------------------

        raw_interest = (
            interest_model
            .predict(
                X
            )[0]
        )


        predicted_interest = (
            get_english_category(
                raw_interest
            )
        )


        interest_probabilities = (
            get_probability_distribution(
                interest_model,
                X
            )
        )


        interest_confidence = (
            interest_probabilities.get(
                predicted_interest
            )
        )


        # ----------------------------------------------------
        # Behavior prediction
        # ----------------------------------------------------

        raw_behavior = (
            behavior_model
            .predict(
                X
            )[0]
        )


        predicted_behavior = str(
            raw_behavior
        )


        behavior_probabilities = (
            get_probability_distribution(
                behavior_model,
                X
            )
        )


        behavior_confidence = (
            behavior_probabilities.get(
                predicted_behavior
            )
        )


        # ----------------------------------------------------
        # Recommendations
        # ----------------------------------------------------

        recommendations = (
            recommend_products(
                predicted_interest,
                top_n=5
            )
        )


        # ----------------------------------------------------
        # Profile
        # ----------------------------------------------------

        customer_profile = (
            extract_customer_profile(
                X
            )
        )


        # ----------------------------------------------------
        # Optional-input statistics
        # ----------------------------------------------------

        optional_filled = 0

        optional_missing = 0


        for feature in OTHER_FEATURES:

            value = payload.get(
                feature
            )


            if (
                value is None
                or
                str(
                    value
                ).strip()
                == ""
            ):

                optional_missing += 1

            else:

                optional_filled += 1


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "success":
                True,


            "predicted_behavior":
                predicted_behavior,

            "behavior_confidence":
                (
                    round(
                        behavior_confidence,
                        6
                    )
                    if behavior_confidence
                    is not None
                    else None
                ),

            "behavior_probabilities":
                behavior_probabilities,


            "predicted_interest":
                predicted_interest,

            "interest_confidence":
                (
                    round(
                        interest_confidence,
                        6
                    )
                    if interest_confidence
                    is not None
                    else None
                ),

            "interest_probabilities":
                interest_probabilities,


            "customer_profile":
                customer_profile,


            "recommended_products":
                recommendations,


            "model_info": {

                "model":
                    FINAL_MODEL_NAME,

                "representation":
                    FINAL_REPRESENTATION,
            },


            "input_summary": {

                "required_features":
                    len(
                        REQUIRED_FEATURES
                    ),

                "optional_features_filled":
                    optional_filled,

                "optional_features_missing":
                    optional_missing,
            },


            "metrics":
                MODEL_METRICS,

            "validation_metrics":
                VALIDATION_METRICS,
        }


    except HTTPException:

        raise


    except Exception as error:

        print()
        print(
            "Prediction error:",
            repr(error)
        )


        raise HTTPException(

            status_code=500,

            detail=(
                "Prediction failed: "
                +
                str(error)
            )
        )


# ============================================================
# 21. WEB
# ============================================================

if WEB_DIR.exists():

    app.mount(

        "/web",

        StaticFiles(
            directory=str(
                WEB_DIR
            ),
            html=True
        ),

        name="web"
    )


# ============================================================
# 22. ROOT
# ============================================================

@app.get(
    "/"
)
def root():

    index_file = (
        WEB_DIR /
        "index.html"
    )


    if index_file.exists():

        return FileResponse(
            index_file
        )


    return {

        "message":
            "Olist Customer Intelligence API is running.",

        "docs":
            "/docs",

        "health":
            "/health",

        "metadata":
            "/metadata",

        "sample":
            "/sample",

        "predict":
            "/predict",
    }


# ============================================================
# 23. RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn


    HOST = "0.0.0.0"

    PORT = 8000


    # --------------------------------------------------------
    # Detect LAN IP
    # --------------------------------------------------------

    try:

        local_ip = socket.gethostbyname(
            socket.gethostname()
        )


        if local_ip.startswith(
            "127."
        ):

            sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM
            )


            try:

                sock.connect(
                    (
                        "8.8.8.8",
                        80
                    )
                )


                local_ip = (
                    sock.getsockname()[0]
                )


            finally:

                sock.close()


    except Exception:

        local_ip = "YOUR_PC_IP"


    # --------------------------------------------------------
    # Server URLs
    # --------------------------------------------------------

    print()

    print("=" * 80)

    print(
        "🚀 OLIST CUSTOMER INTELLIGENCE API"
    )

    print("=" * 80)

    print()

    print(
        "Localhost:"
    )

    print(
        f"  http://127.0.0.1:{PORT}/"
    )

    print()

    print(
        "PC LAN:"
    )

    print(
        f"  http://{local_ip}:{PORT}/"
    )

    print()

    print(
        "Swagger:"
    )

    print(
        f"  http://127.0.0.1:{PORT}/docs"
    )

    print(
        f"  http://{local_ip}:{PORT}/docs"
    )

    print()

    print(
        "Sample:"
    )

    print(
        f"  http://127.0.0.1:{PORT}/sample"
    )

    print(
        f"  http://{local_ip}:{PORT}/sample"
    )

    print()

    print(
        "Android Emulator:"
    )

    print(
        f"  http://10.0.2.2:{PORT}/"
    )

    print()

    print(
        "Android Physical Phone:"
    )

    print(
        f"  http://{local_ip}:{PORT}/"
    )

    print()

    print("=" * 80)

    print(
        "Press CTRL+C to stop."
    )

    print("=" * 80)

    print()


    # --------------------------------------------------------
    # IMPORTANT:
    # reload=False
    # --------------------------------------------------------

    uvicorn.run(

        app,

        host=HOST,

        port=PORT,

        reload=False
    )