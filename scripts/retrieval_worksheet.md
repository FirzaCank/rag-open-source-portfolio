# Retrieval set worksheet, 66 queries

Fill every `Q1:` `Q2:` `Q3:` line, then run:

```bash
python scripts/worksheet_to_json.py
```

That converts this file into `scripts/retrieval_set.json` and runs the checker.

## Rules

- Write like a visitor who has never read the portfolio. That is the real query population.
- Never reuse the source's own phrasing. A query that quotes its chunk measures string
  matching, and it wins under every retrieval strategy, so it separates nothing.
- Three different angles per source. One direct, one indirect, one about a number or a skill.
- `(ID)` slots are Bahasa Indonesia. 12 of them, one in each of 12 sources.

Real example of why the phrasing rule matters: a visitor writes "right now", the corpus
writes "current", and that single difference pushed the Hypefast chunks to rank 17.

---

## 1. About Firza

`about-firza` | 1 doc, 4 chunk, 3682 chars

> I’m a Data & AI Engineer with ~4 years of experience building end-to-end data pipelines and scalable MLOps frameworks. I turn raw data infrastructure into automated systems: recommendation models in production, LLM-powered document processing, and pipelines that run unattended. I’ve spent the last several years inside Telco and consumer brand environments, where the systems I designed had to run reliably at scale, with proven impact: processing 1B+ daily records, driving hundreds of millions of IDR per month in GMV via recommendation systems, achieving a 200× average monthly return on voucher budget per brand, and cutting OCR extraction time from 8 hours to 5 minutes (96× faster). Beyond my  ...

```
Q1: What university did he graduate from?
Q2: Does he know any Japanese?
Q3 (ID): Dia pernah memimpin organisasi mahasiswa?
```

## 2. Career Timeline

`career-timeline` | 1 doc, 1 chunk, 554 chars

> Firza's complete career history in chronological order (earliest to latest): 1. Business Development Internship at Banopolis Inovasi Kendara, June 2020 - August 2020 (internship). 2. Production Systems and Business Analyst Internship at PT Arindo Cipta Perkasa, November 2020 - April 2021 (internship). 3. Data Scientist Internship at Telkom Indonesia, February 2021 - August 2021 (internship). 4. Data Engineer and Analytics / ETL Developer at IDstar Cipta Teknologi, July 2022 - March 2025. 5. Data Engineer at Hypefast, March 2025 - Present (current).

```
Q1: Which company did he join first?
Q2: How many employers has he had so far?
Q3 (ID): Urutan tempat kerjanya dari yang paling awal?
```

## 3. Experience: Data Engineer at Hypefast

`hypefast` | 7 doc, 7 chunk, 5461 chars

> Data Engineer at Hypefast, March 2025 - Present (current role). Location: Jakarta, Indonesia. Building data and Machine Learning (ML) infrastructure for a multi-brand consumer aggregator. Pipelines, recommendation systems, AI-driven automation across 11+ brands. Designed L0–L2 ETL pipelines ingesting order and customer data from 4 marketplace APIs and Google Sheets to BigQuery for 11+ brands, with Flask-based token management on Cloud Run and PostgreSQL for API authentication, applying SCD handling on dimension tables, enabling unified analytics and reporting. Stack: Google Cloud Platform (GCP), Composer (Airflow), BigQuery, Vertex AI, Agent Platform, Cloud Run, Compute Engine, Pub/Sub, Data ...

Other documents under this source: 6, one per highlight plus an overview.

```
Q1: Who is he working for these days?
Q2: Has he shipped a recommendation engine that real users hit?
Q3 (ID): Sekarang dia menangani berapa merek?
```

## 4. Experience: Data Engineer and Analytics / ETL Developer at IDstar Cipta Teknologi

`idstar-cipta-teknologi` | 4 doc, 4 chunk, 2276 chars

> Data Engineer and Analytics / ETL Developer at IDstar Cipta Teknologi (Placement in PT XL Axiata Tbk.), July 2022 - March 2025. Location: Jakarta, Indonesia. Built telco-scale data platforms processing 1B+ daily records with Medallion architecture, plus partner-facing analytics products. Developed an end-to-end telco user behavior & credit scoring platform using a Medallion Architecture (Bronze/Silver/Gold) on Snowflake; built PySpark ETL pipelines on AWS (S3, EC2, Lambda) processing 1B+ daily records with schema evolution, partition pruning, and data quality gates enforced at each layer, plus slowly changing dimension (SCD) management across Types 0-6. Stack: AWS, Snowflake, PySpark, Hadoop ...

Other documents under this source: 3, one per highlight plus an overview.

```
Q1: Where did he handle really large daily volumes?
Q2: Has he used Snowflake professionally?
Q3 (ID): Dia pernah mengerjakan proyek telekomunikasi?
```

## 5. Experience: Data Scientist Internship at Telkom Indonesia

`telkom-indonesia` | 3 doc, 3 chunk, 1140 chars

> Data Scientist Internship at Telkom Indonesia, February 2021 - August 2021. Location: Jakarta, Indonesia. Supported exploratory data analysis and product modeling for digital products including T-Money and KALISA blockchain waqf platform. Performed data cleansing and exploratory data analysis on T-Money data within 1 month, reporting and interpreting results to support data-driven decisions. Stack: Python, pandas, Business Modeling.

Other documents under this source: 2, one per highlight plus an overview.

```
Q1: Did he ever intern at a state owned company?
Q2: Has he touched anything blockchain related?
Q3 (ID): Magang data science-nya di mana?
```

## 6. Experience: Production Systems and Business Analyst Internship at PT Arindo Cipta Perkasa

`pt-arindo-cipta-perkasa` | 3 doc, 3 chunk, 1208 chars

> Production Systems and Business Analyst Internship at PT Arindo Cipta Perkasa, November 2020 - April 2021. Location: Bogor Regency, Indonesia. Production operations analysis and scheduling optimization for manufacturing operations. Analyzed production data to identify root causes of operational problems and recommend improvements to management. Stack: Production Scheduling, Heuristic Algorithm, Manufacturing, Data Analysis.

Other documents under this source: 2, one per highlight plus an overview.

```
Q1: Any factory or manufacturing background?
Q2: Has he done scheduling optimisation work?
Q3 (ID): Dia pernah magang di bidang produksi?
```

## 7. Experience: Business Development Internship at Banopolis Inovasi Kendara

`banopolis-inovasi-kendara` | 5 doc, 5 chunk, 2163 chars

> Business Development Internship at Banopolis Inovasi Kendara, June 2020 - August 2020. Location: Bandung, Indonesia. Business development for GETBIKE ride sharing product. Proposed B2B and B2B2G models, designed partnership scheme with PT Telkom Indonesia. Proposed a new business model of GETBIKE B2B and B2B2G ride sharing using 3 tools (Business Model Canvas, Value Proposition Canvas, Lean Canvas). Management implemented the ideas for the product within 1 month. Stack: Business Model Canvas, Value Proposition Canvas, Lean Canvas, Agile.

Other documents under this source: 4, one per highlight plus an overview.

```
Q1: Did he ever work on a ride hailing product?
Q2: Any experience on the business side rather than engineering?
Q3 (ID): Dia pernah magang business development?
```

## 8. Project: Video Platform Data Analysis

`video-platform-data-analysis` | 1 doc, 3 chunk, 2665 chars

> Video Platform Data Analysis (2025) for AI gaming clip platform. Behavior and growth deep dive for an AI gaming clip platform: activation levers that move retention and clip output.  Categories: Data Analyst, Dashboard. Stack: SQL, Python, Tableau, Pitch Deck.  Context  An AI assistant for gaming creators that auto-turns long gameplay streams into highlight clips. It sits between gameplay capture and social distribution (TikTok, YouTube Shorts, Reels), so every onboarded creator either becomes a recurring clip source or churns silently. Growth had plateaued, and the founding team wanted to know whether the bottleneck was acquisition, activation, or content output per creator.   Problem  Inte ...

```
Q1: Has he analysed user retention for a video product?
Q2: Any work involving gaming content?
Q3 (ID): Dia pernah menganalisis platform video?
```

## 9. Project: Affiliate Commission Performance Review & Recovery Plan

`affiliate-commission-perform` | 1 doc, 3 chunk, 3184 chars

> Affiliate Commission Performance Review & Recovery Plan (2026) for E-commerce affiliate platform. Root-cause analysis of a 34% commission revenue decline: funnel diagnosis, cohort retention, and a two-track recovery playbook for leadership.  Categories: Data Analyst, Pitch Deck. Stack: Python, Excel, PowerPoint, Funnel Analysis, Cohort Analysis.  Context  An e-commerce affiliate operation earning commission through video content that drives marketplace transactions. Commission revenue had fallen four months straight - a ~34% cumulative drop. Leadership needed a clear diagnosis and recovery plan before the trend turned structural.   Problem  Three questions before scoping any fix:  - Where in ...

```
Q1: Has he investigated why revenue dropped somewhere?
Q2: Any work on affiliate programmes?
Q3: Did he propose how to win back what was lost?
```

## 10. Project: Content Strategy & Revenue Projection Pitch Deck

`content-strategy-revenue-pro` | 1 doc, 3 chunk, 2996 chars

> Content Strategy & Revenue Projection Pitch Deck (2026) for Agency X. Market gap analysis, podcast-to-workshop funnel strategy, and 3-month financial projection for a social media influencer's relationship content platform.  Categories: Pitch Deck, Data Analyst, Financial Analyst. Stack: PowerPoint, Excel, Python, Market Analysis, Financial Modeling.  Context  Agency X wanted to build a relationship content platform around a social media influencer (established credibility in the personal-growth and relationship space) lined up as host. The influencer had strong engagement but content was limited to short-form snippets with no structured monetization. The opportunity: turn disposable content ...

```
Q1: Has he made a deck for an influencer business?
Q2: Any revenue forecasting for a small venture?
Q3: Did he study where competitors were thin?
```

## 11. Project: E-commerce Strategic Valuation & Growth Outlook

`ecommerce-strategic-valuatio` | 1 doc, 2 chunk, 1945 chars

> E-commerce Strategic Valuation & Growth Outlook (2025) for Indonesian e-commerce platform. Financial-analyst-led 5-year forecast, DCF, and regional comparable valuation framing the margin re-rating thesis.  Categories: Financial Analyst, Pitch Deck. Stack: Excel, PowerPoint, DCF, Comparable Analysis, Financial Modeling.  Context  A financial-analyst brief on a publicly listed Indonesian e-commerce platform. The thesis: margin expansion, not revenue growth, was the next leg of the story. The deck had to defend that with a credible bottom-up forecast, a DCF, and a regional-peer comparable read - in a format that holds up in an investment committee.   Problem  The narrative existed but the numb ...

```
Q1: Has he ever valued a company?
Q2: Any DCF modelling experience?
Q3: Did he project how a business would look in half a decade?
```

## 12. Project: Nuclear Energy Policy Discourse Network Analysis

`nuclear-energy-policy-discou` | 1 doc, 3 chunk, 2582 chars

> Nuclear Energy Policy Discourse Network Analysis (2025) for Policy Research Initiative. Scraping news + social media on Indonesia's nuclear policy, then Gemini-powered DNA + SNA + sentiment analysis dashboard.  Categories: AI Engineer, Data Scientist, Data Engineer. Stack: Gemini API, LLM Extraction, Prompt Engineering, Sentiment Analysis, MongoDB, NoSQL, pyvis, GCP.  Context  A policy research initiative on Indonesia's nuclear and EBET (new and renewable energy) debate. The team had a research design but no production-grade pipeline to collect, parse, and visualize the discourse at scale. The deliverable: a proof of concept combining Discourse Network Analysis (DNA), Social Network Analysis ...

```
Q1: Any work on energy policy?
Q2: Has he mapped how public debate connects across actors?
Q3: Did he store anything in MongoDB?
```

## 13. Project: Regional Writing Ecosystem Index (PCA)

`regional-writing-ecosystem-i` | 1 doc, 2 chunk, 1843 chars

> Regional Writing Ecosystem Index (PCA) (2024) for Cultural Policy Think Tank. PCA-based composite model (68.2% variance) analyzing regional writing ecosystem strength across Indonesia.  Categories: Data Scientist, Data Analyst. Stack: scikit-learn, PCA, Looker Studio, Statistical Analysis.  Context  A cultural policy think tank needed a defensible way to compare regional writing ecosystems across Indonesia. The think tank supplied its own internal dataset of regional indicators - certified writers, publishing activity, and literacy proxies - but had no single index that could rank provinces in a way policymakers would accept.   Problem  Single indicators told contradictory stories. Three iss ...

```
Q1: Has he built a composite index?
Q2: Any dimensionality reduction work?
Q3: Did he work with a think tank?
```

## 14. Project: Social Media Scraping Integration

`social-media-scraping-integr` | 1 doc, 3 chunk, 2269 chars

> Social Media Scraping Integration (2025) for Marketing research agency. Unified scraper for 5 platforms (YouTube, TikTok, Instagram, Facebook, X) on Cloud Run with resumable batches.  Categories: Data Engineer. Stack: Python, Cloud Run, GCS, APIFY, Flask, Docker.  Context  A marketing research team needed continuous profile and content data from five social platforms to feed campaign benchmarking and influencer scoping. The existing setup was a patchwork of one-off notebooks per platform - each rerun needed manual stitching, and any mid-batch failure meant starting from scratch.   Problem  - Five platforms, five auth models, rate limits, and content schemas - a separate script each meant fiv ...

```
Q1: Can he pull data from TikTok and Instagram?
Q2: Has he built something that resumes after a crash?
Q3 (ID): Dia bisa mengambil data dari lima platform sosial?
```

## 15. Project: Google News Scraping + Gemini AI Aggregation

`google-news-scraping-gemini` | 1 doc, 3 chunk, 2572 chars

> Google News Scraping + Gemini AI Aggregation (2025) for Macro research desk. Always-on macro news scraping pipeline on Cloud Run combining Google CSE, APIFY, and Gemini parsing.  Categories: AI Engineer, Data Engineer. Stack: Gemini API, LLM Extraction, Prompt Engineering, Python, Cloud Run, Google CSE, APIFY, Web Scraping.  Context  A small research desk tracking monetary policy and FX needed continuous coverage of Bank Indonesia, inflation, and rupiah-related news across Indonesian and English sources. Manual scanning was the bottleneck - by the time an analyst had read enough to brief the team, intraday signals were already stale.   Problem  Three constraints shaped the build:  - Coverage ...

```
Q1: Any always on news collection system?
Q2: Has he used Google Custom Search?
Q3: Which client needed economic headlines watched continuously?
```

## 16. Project: Debt Collection Analytics & Agent Productivity

`debt-collection-analytics-ag` | 1 doc, 3 chunk, 2629 chars

> Debt Collection Analytics & Agent Productivity (2025) for Banking collections agency. Ten months of call center data turned into a dashboard + narrative report that reframed a data-quality issue as the real performance lever.  Categories: Data Analyst, Dashboard. Stack: Python, Pandas, Excel, Looker Studio.  Context  A banking collections operation: hundreds of agents, a large-scale delinquent portfolio. Monthly reports arrived as separate Excel files with no unified view across agents, regions, or time - so questions like "who are our consistent top performers" or "where is recovery coming from" meant rebuilding the analysis from scratch every month.   Problem  - Ten monthly Excel workbooks ...

```
Q1: Has he worked with call centre data?
Q2: Any project where the real problem turned out to be data quality?
Q3: Did he work for a bank collections team?
```

## 17. Project: Heavy Equipment Preventive Maintenance Dashboard

`heavy-equipment-preventive-m` | 1 doc, 3 chunk, 2437 chars

> Heavy Equipment Preventive Maintenance Dashboard (2025) for Mining contractor. Cylinder and hose lifecycle dashboard for a mining contractor's Komatsu excavator fleet.  Categories: Dashboard, Data Analyst. Stack: Looker Studio, Google Sheets, Excel, Python.  Context  A mining contractor running large Komatsu excavators (PC2000, PC1250, PC850 class) needed forward visibility into cylinder and hose rehousing demand. Lifetime data lived in the maintenance planner's workbooks, indexed by EGI (unit serial), section (boom, arm, bucket), and part number. Stock for overhaul (stock OVH) was tracked separately - the only way to confirm parts were ready when a cylinder hit its rehousing window was a ma ...

```
Q1: Any dashboards for mining?
Q2: Has he tracked equipment part lifecycles?
Q3 (ID): Dia pernah bikin dashboard untuk alat berat?
```

## 18. Project: Tableau for Data Analysis 101 Course

`tableau-for-data-analysis-10` | 1 doc, 3 chunk, 2432 chars

> Tableau for Data Analysis 101 Course (2025) for Indonesian EdTech platform. Instructor-built Tableau curriculum with a single retail case study (VoraStore) as the project arc.  Categories: Data Analyst, Dashboard. Stack: Tableau, PowerPoint, Curriculum Design.  Context  An EdTech platform for Indonesian working professionals needed a Tableau 101 course taking absolute beginners from install to a presentable dashboard. The market either over-indexed on tool clicks (no business framing) or jumped straight to advanced features. The brief: a course that respects a professional's time and lands them on something portfolio-worthy.   Problem  - Mixed backgrounds - some Excel-fluent, others from zer ...

```
Q1: Has he taught anything?
Q2: Has he put together teaching material for students?
Q3: Any Tableau training work?
```

## 19. Project: Government Statistics Data Platform Implementation

`government-statistics-data-p` | 1 doc, 3 chunk, 2430 chars

> Government Statistics Data Platform Implementation (2025) for Indonesian coordinating ministry. Reference catalog and ingestion layer mapping all 549 BPS office domains for a coordinating ministry.  Categories: Data Engineer. Stack: Python, BPS Web API, JavaScript, CSV, JSON.  Context  A coordinating ministry needed a clean, programmatically accessible catalog of statistical sources from Indonesia's central statistics agency (Badan Pusat Statistik / BPS), so analysts could pull regional and subject-specific data without manually navigating dozens of provincial and regency websites. Scoped as a 28-working-day implementation: deliver end-to-end, document, hand over.   Problem  - BPS publishes  ...

```
Q1: Has he worked with public sector data offices?
Q2: Did he map out where official figures live?
Q3: Did he deliver something for a ministry?
```

## 20. Project: Mobile App Analytics Data Pipeline

`mobile-app-analytics-data-pi` | 1 doc, 5 chunk, 4449 chars

> Mobile App Analytics Data Pipeline (2026) for Digital financial services app. Multi-source ingestion pipeline unifying AppsFlyer, MoEngage, Google Play Console, and App Store Connect into BigQuery for a Looker Studio monitoring dashboard.  Categories: Data Engineer, DevOps. Stack: Python, dbt, BigQuery, Cloud Run Jobs, Cloud Workflows, Cloud Scheduler, Cloud Build, GCP Secret Manager, Terraform, Looker Studio.  Context  A digital financial services app needed a single source of truth for their product analytics. Data was scattered across four platforms: AppsFlyer (attribution and events), MoEngage (push campaigns), Google Play Console (Android performance), and App Store Connect (iOS analyti ...

```
Q1: Has he brought several app marketing tools into one place?
Q2: Any Terraform in his projects?
Q3 (ID): Dia pernah menyatukan data dari beberapa sumber aplikasi mobile?
```

## 21. Project: Personal Portfolio Website with RAG Chat

`personal-portfolio-website-w` | 1 doc, 5 chunk, 4943 chars

> Personal Portfolio Website with RAG Chat (2026) for Self-initiated. A personal portfolio with an agentic RAG chat assistant powered by Gemini, combining semantic retrieval, tool calling, and guardrails to answer grounded questions about my projects, experience, and skills.  Categories: AI Engineer, Data Engineer. Stack: Gemini API, Agentic RAG, Vector Embeddings, Semantic Search, Tool Calling, Prompt Engineering, Guardrails, Python, Next.js, Vercel.  Context  I wanted a portfolio that could do more than just display static content. Most portfolios are one-way: the visitor has to read through everything and piece together the picture themselves. I wanted visitors to be able to ask questions a ...

```
Q1: Did he build his own website?
Q2: How does his chat assistant avoid being tricked?
Q3: Can his assistant actually do things, not only answer?
```

## 22. Project: Telco Customer Churn Prediction (Streamlit)

`telco-customer-churn-predict` | 1 doc, 5 chunk, 4437 chars

> Telco Customer Churn Prediction (Streamlit) (2022) for Self-initiated portfolio project. End-to-end ML app benchmarking Logistic Regression, Random Forest, and Gradient Boosting on telco subscriber data, deployed live on Streamlit.  Categories: Data Scientist, ML Engineer, Dashboard. Stack: Python, Streamlit, scikit-learn, Pandas, Logistic Regression, Random Forest, Gradient Boosting.  Context  A self-initiated project to practice the full ML lifecycle on a known business problem: predicting telecom subscriber churn. The dataset is DQLab's dummy data for a fictional "DQLab Telco" - customer state as of June 2020, ~6,800 rows, 21 attributes covering demographics, services, contract type, and  ...

```
Q1: Has he compared several classifiers on one dataset?
Q2: Any Streamlit app deployed publicly?
Q3 (ID): Dia pernah membuat model prediksi churn?
```
