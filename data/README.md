# Data

The project uses the public **Brazilian E-Commerce Public Dataset by Olist**:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

It's ~100k orders from 2016-2018 on a Brazilian marketplace. I don't commit the
raw files (they're Olist's, and ~120 MB). Download the zip from Kaggle and put
these files in `data/raw/`:

| file | used for |
|---|---|
| `olist_orders_dataset.csv` | purchase date, order status, customer |
| `olist_order_items_dataset.csv` | price, freight, product, seller (one row per item) |
| `olist_products_dataset.csv` | product category |
| `olist_customers_dataset.csv` | customer zip prefix / city (for the geo test) |
| `product_category_name_translation.csv` | Portuguese -> English category names |

The other files in the zip (payments, reviews, sellers, geolocation) aren't needed.

Cleaning rules (all in `pricing/data.py`):
- canceled and unavailable orders are dropped
- only orders from 2017-01-01 to 2018-08-22: before 2017 there are almost no
  orders, and after 22 Aug 2018 volume drops to near zero because the export
  ends, not because demand did
- a "listing" is a product sold by a specific seller (the same product can be
  sold by several sellers at different prices)
