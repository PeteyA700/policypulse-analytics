# policypulse-analytics
# Insurance Customer Retention & Pricing Analytics

## Project Overview
This project analyzes insurance customer behavior to answer a business question:

Should the company raise prices, or should it focus on retaining high-risk, high-value customers?

## Skills Demonstrated
- Python data analysis
- SQL querying with SQLite
- KPI analysis
- Customer segmentation
- Churn prediction
- Feature importance
- Business recommendations
- Executive summary writing

## Business Problem
Insurance companies often want to increase premiums to grow revenue, but higher prices can increase churn. This project estimates the tradeoff between a price increase and a targeted retention strategy.

## Files
- `main.py`: Full project code
- `data/insurance_customers.csv`: Generated customer dataset
- `data/insurance.db`: SQLite database
- `outputs/policy_summary.csv`: KPI summary by policy type
- `outputs/segment_summary.csv`: Customer segment analysis
- `outputs/customer_risk_scores.csv`: Predicted churn risk by customer
- `outputs/executive_summary.md`: Final business recommendation
- `outputs/*.png`: Charts

## How to Run
```bash
pip install -r requirements.txt
python main.py
