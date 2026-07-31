# 🍽️ TasteMatch — Apriori Food Recommendation System

A focused Streamlit app that recommends food items using the **Apriori
market-basket algorithm** (mlxtend), trained on order-transaction history.
No generic catalog browsing — just the recommendation engine and full
transparency into how it works.

## ✨ Features

- **🎯 Recommendation System (home page)** — pick an item a customer
  ordered and instantly see recommended items as image cards, each
  showing **Support**, **Confidence**, and **Lift** clearly, plus a
  plain-English explanation and a confidence/lift comparison chart.
  Apriori thresholds (min support / min confidence) are adjustable live
  from the sidebar.
- **📈 Rule Explorer** — full frequent-itemsets table, the complete
  association-rules table (support/confidence/lift, color-graded), and an
  interactive item-association network graph (edge thickness = lift,
  edge color intensity = confidence).

## 🗂️ Project Structure

```
food_recommender/
├── app.py                     # Main Streamlit app (2 pages)
├── requirements.txt           # Python dependencies
├── data/
│   ├── transactions.csv       # Order-transaction history for Apriori mining
│   └── item_images.csv        # Image lookup for menu items
├── utils/
│   └── apriori_recommender.py # Market-basket engine (Apriori, mlxtend)
├── .streamlit/
│   └── config.toml            # App theme
└── README.md
```

## 🚀 Setup & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 🧠 How it works

`utils/apriori_recommender.py`:
1. One-hot encodes each order (`TransactionEncoder`).
2. Mines frequent itemsets with `apriori()` at a chosen min support.
3. Builds association rules with `association_rules()` at a chosen min
   confidence, giving `support`, `confidence`, and `lift` per rule.
4. `recommend(item)` returns every rule whose antecedent contains that
   item, ranked by confidence.

**Support** — how often the itemset appears across all orders.
**Confidence** — P(consequent | antecedent): given someone ordered the
antecedent, how often they also ordered the consequent.
**Lift** — how much more likely the consequent is, given the antecedent,
compared to random chance (lift > 1 = positive association).

## 🖼️ About the images

Item images are loaded live from Unsplash URLs in `data/item_images.csv`.
To use your own images, replace those URLs with local paths (e.g.
`images/pizza.jpg`) — `st.image()` works the same way either way.

## ➕ Extending

- **Add more orders**: append rows to `data/transactions.csv`
  (`order_id,items` with comma-separated item names inside the `items`
  cell — quote the cell since it contains commas).
- **Add a new menu item**: add its image URL to `data/item_images.csv`
  so it renders correctly once it appears in enough transactions.
- **Tune the engine**: adjust min support / min confidence live from the
  sidebar, or change the defaults in `load_apriori_engine(...)` in
  `app.py`.

Enjoy building on top of TasteMatch! 🍜🍕🍣
