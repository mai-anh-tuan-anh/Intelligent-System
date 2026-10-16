# E-commerce service-process dataset

## Source

Brazilian E-Commerce Public Dataset by Olist:

https://www.kaggle.com/olistbr/brazilian-ecommerce

The notebook uses the `olist_orders_dataset.csv` table because it contains the order lifecycle timestamps required to construct a service-performance time series.

Relevant timestamps include:

- `order_purchase_timestamp`
- `order_approved_at`
- `order_delivered_carrier_date`
- `order_delivered_customer_date`
- `order_estimated_delivery_date`

## Assignment transformation

The raw records are filtered to completed deliveries and aggregated by customer-delivery date. The engineered time series contains delivery volume, approval delay, carrier delay, promised delivery time, late-delivery rate, and average delivery time.

The RNN predicts the **next observed service period's average delivery time** from a fixed 14-step history.

This is an assignment-oriented forecasting transformation rather than a dedicated public RNN benchmark dataset.
