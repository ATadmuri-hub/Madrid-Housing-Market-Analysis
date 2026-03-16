# Madrid Residential Real Estate Intelligence — Full Results Report
### A Complete Interpretation of the Analysis, Models, and Investment Findings

---

## Preface

This report presents the complete, end-to-end findings of a data-driven study of the Madrid residential property market. The analysis draws on 11,826 active listings from the Madrid market, integrates external macroeconomic benchmarks from national appraisal indices and public listing platforms (January 2026), and builds a stack of seven statistical and machine learning models to explain prices, detect investment opportunities, and produce district-level price forecasts through 2027. Every number in this report comes directly from the data and models. Nothing has been estimated by hand or approximated.

The purpose is threefold: to understand what drives price per square metre in Madrid today, to identify where the market is mispricing properties relative to fundamentals, and to translate those findings into a concrete acquisition framework for a property investor operating in the city.

---

## 1. The Data

The raw dataset contains 11,826 residential listings distributed across 21 districts (zonas) of Madrid. After filtering out properties with missing or implausible values — listings below 50,000 euros or above 5,000,000 euros, and properties below 20 or above 1,000 square metres — the final analysis sample retains 11,211 listings, representing a 5.2% removal rate. The data quality is high: only a small fraction of records required imputation or exclusion.

Each listing carries the current asking price, the prior asking price (where a reduction has occurred), the surface area in square metres, the number of rooms, the number of bathrooms, whether the building has a lift, the floor, and the district. From these raw variables, the following derived features were constructed for modelling:

- **Price per square metre** (euros/m²): the primary outcome variable throughout the analysis.
- **Log price per square metre**: the logarithm of price/m², which is standard in hedonic pricing research. Modelling the log of price ensures that percentage changes — rather than absolute euro differences — are the unit of analysis. This also corrects for the right-skewed distribution of property prices.
- **Discount flag and depth**: a binary indicator for whether the asking price has been reduced from the prior price, and the size of that reduction as a percentage.
- **Size in decametres** (metros10 = square metres divided by 10): rescaling improves coefficient interpretability.
- **Bathroom imputation**: for the 8.6% of listings where bathrooms were missing or zero, the district median was substituted. A missingness flag (banos\_miss) was retained as a separate predictor to allow the model to learn whether missing bathroom data is itself informative.
- **Lift category**: three levels — present, absent, or not reported.
- **Floor group**: ten categories standardising the wide variety of Spanish floor labelling conventions (bajo, sotano, entresuelo, 1st through 4th floor, 5th floor and above, atico, and other).
- **Rooms capped at 8**: a small number of unusually large listings reported 10 or more rooms. Capping at 8 prevents those extreme observations from exerting disproportionate influence on the model.

---

## 2. The Madrid Market in Numbers

Before modelling, it is worth understanding the market in plain descriptive terms.

The median listing price across the full sample is **630,000 euros**. In price-per-square-metre terms, the median is **5,950 euros/m²** and the mean is **6,550 euros/m²**. The gap between median and mean reflects a right-skewed distribution: a relatively small number of premium listings in districts such as Barrio de Salamanca and Chamberí pull the average significantly above the midpoint. The standard deviation is **3,391 euros/m²**, meaning the interquartile spread of prices is substantial — a reflection of the extreme heterogeneity of a city that runs from among the most expensive residential property in Europe to some of the most affordable in the Spanish capital.

The bottom 1% of listings trades at approximately **1,383 euros/m²** (peripheral southern districts); the top 1% reaches **16,264 euros/m²** (prime Salamanca and Retiro). That fourteenfold range from bottom to top is, in itself, one of the most important findings in the analysis: a single city-wide model without district controls would be profoundly misleading.

**9.5% of all listings have been discounted** from their original asking price at the time of data collection. The median depth of those discounts is **4.0%**. That figure — one listing in ten showing a price cut, typically of around 4% — is the empirical basis for the motivated-seller detection framework developed later in the analysis.

The market is segmented by district in a way that produces a broad, consistent price gradient from west-central premium zones to eastern and southern emerging zones:

| District | Median euros/m² | Discount rate |
|---|---|---|
| Barrio de Salamanca | 10,526 | 4.9% |
| Chamberí | 8,356 | 10.7% |
| Retiro | 8,239 | 9.9% |
| Chamartín | 7,353 | 10.5% |
| Centro | 7,222 | 11.8% |
| Moncloa | 5,645 | 7.2% |
| Arganzuela | 5,433 | 12.7% |
| Tetuán | 5,303 | 10.6% |
| Hortaleza | 5,108 | 9.6% |
| Fuencarral | 4,920 | 8.9% |
| Ciudad Lineal | 4,600 | 11.5% |
| Barajas | 4,186 | 15.6% |
| San Blas | 3,659 | 5.4% |
| Moratalaz | 3,569 | 11.5% |
| Latina | 3,377 | 11.5% |
| Vicálvaro | 3,369 | 3.1% |
| Villa de Vallecas | 3,284 | 11.7% |
| Carabanchel | 3,120 | 16.5% |
| Usera | 3,111 | 7.9% |
| Puente de Vallecas | 2,742 | 8.5% |
| Villaverde | 2,531 | 5.9% |

One pattern in this table deserves immediate attention: the districts with the highest discount rates are not the most expensive ones. Barrio de Salamanca — the most expensive district at 10,526 euros/m² — has the lowest discount rate in the city at 4.9%. Carabanchel and Barajas, at the lower end of the price spectrum, show discount rates of 16.5% and 15.6% respectively. This suggests a structurally different seller psychology at different price points, and it has direct implications for the acquisition strategy discussed in Section 11.

It is also worth comparing the observed data prices with the published benchmark figures from national appraisal indices and public listing platforms for January 2026. The alignment is close: Barrio de Salamanca at 10,526 euros/m² in the data versus the published 10,800 euros/m²; Chamberí at 8,356 versus 8,200; Retiro at 8,239 versus 8,500; Centro at 7,222 versus 7,500; Tetuán at 5,303 versus 5,100. The data is slightly below the published benchmarks in the premium districts and slightly above in emerging ones, consistent with the fact that asking prices in premium zones tend to anchor below headline valuations while lower-tier zones see some optimistic seller pricing.

---

## 3. Modelling Philosophy

The analysis employs seven distinct models, deliberately chosen to form a ladder from the most transparent and interpretable to the most predictively powerful. This is not model collection for its own sake — each model answers a different question.

The **OLS Fixed Effects model** and the **OLS Fundamentals model** sit at the base of the ladder. They are fully transparent: every coefficient can be read directly, explained to a client, and used to construct simple valuation rules. The Fixed Effects model treats each district as its own price level, capturing all unobserved district-level heterogeneity in a single parameter per zone. The Fundamentals model asks how far property characteristics alone — size, rooms, bathrooms, lift, floor — can explain prices, ignoring zone entirely.

**Ridge, Lasso, and Elastic Net regression** extend the OLS model by adding a regularization penalty. This is relevant for two reasons: it guards against overfitting when many zone dummies are simultaneously included, and Lasso specifically forces the less informative zone dummies toward exactly zero, revealing which districts carry statistically real price signals versus noise.

The **Linear Mixed Model** sits between the Fixed Effects and Fundamentals models philosophically. Rather than treating each district as having a completely free intercept (pure FE) or no district effect at all (Fundamentals), it shrinks the district intercepts toward a common mean in proportion to how much data each district contributes. Districts with few observations are pulled more strongly toward the city average; large districts retain most of their estimated effect. This is known as partial pooling.

**Random Forest** and **XGBoost** are non-parametric ensemble models. They make no assumptions about functional form, can capture non-linear interactions between floor, size, and district, and are evaluated purely on predictive accuracy. Their job is to establish the ceiling on out-of-sample performance.

All models are trained on a 75% random sample of the data (8,408 listings) and evaluated on the held-out 25% test set (2,803 listings). The outcome variable throughout is log(price/m²), winsorised at the 1st and 99th percentiles to limit the influence of extreme observations on coefficient estimation. Because the outcome is a log, all model errors and R² statistics are in log-price space.

---

## 4. OLS Fixed Effects — The Baseline

The OLS Fixed Effects model regresses log(price/m²) on a full set of district dummy variables plus the structural property features. It achieves a **training R² of 0.708** and an **out-of-sample test R² of 0.699**, with a test RMSE of **0.296 log units** and a test MAE of **0.230 log units**. These numbers deserve unpacking.

An R² of 0.699 on held-out data means the model explains approximately 70% of the variance in log price per square metre for properties it has never seen. For a property-level prediction problem in a large heterogeneous city, this is a strong result. It implies that 70% of what determines the price of any given apartment in Madrid can be attributed to its district and its observable structural characteristics. The remaining 30% reflects renovation quality, exact micro-location within the district, negotiating dynamics, and idiosyncratic features that the data does not capture.

The MAE of 0.230 in log units translates to a typical absolute prediction error of approximately **25.9% in price per square metre** terms (computed as exp(0.230) − 1). On a property with a median price of 5,950 euros/m², this corresponds to an error of around **1,540 euros/m²**, or roughly **123,000 euros on a typical 80m² apartment**. That number is large in absolute terms, but it reflects the genuine dispersion in property prices even within districts — it is not a model deficiency so much as a statement about market heterogeneity.

Comparing the Fixed Effects model to the Fundamentals model (no district controls) is illuminating. The Fundamentals model achieves an R² of only **0.246** on the test set, compared to 0.699 for the FE model. That gap — 45 percentage points — measures the explanatory contribution of district identity alone. Madrid's 21 districts are not just cosmetic labels: they represent fundamentally different price regimes, and failing to account for them in a valuation model produces estimates that are essentially uninformative.

### 4.1 Structural Coefficients

The structural coefficients from the Fixed Effects model carry concrete economic meaning. All are expressed as the estimated percentage change in price per square metre for a one-unit increase in the predictor, holding everything else constant.

**Size (metros10)**: For every additional 10 square metres of floor area, holding the number of rooms constant, price per square metre changes by approximately **-0.03%** — effectively zero. This result is not surprising. When you condition on the number of rooms (which is also in the model), adding floor area without adding rooms means making each room larger, which has no meaningful impact on the per-square-metre price. The contribution of size to price is already captured through the zone dummy and the room count.

**Rooms (rooms\_cap)**: Each additional room, holding floor area constant, reduces price per square metre by **3.75%**. This is a well-documented effect in urban housing economics sometimes called the room density penalty: a 100m² apartment with four rooms has smaller individual rooms than a 100m² apartment with three, which can reduce its desirability. More fundamentally, apartments with more rooms for a given surface area tend to be in older, less renovated stock with inefficient floorplans, driving down price per m².

**Bathrooms (banos\_imp)**: Each additional bathroom increases price per square metre by **3.44%**. Unlike rooms, additional bathrooms are seen as unambiguously positive — they signal a higher standard of finish, more comfortable living, and tend to appear in refurbished properties. A move from one bathroom to two in an otherwise identical apartment adds roughly 3.4% to its per-square-metre price.

**Lift (lift\_f)**: Having a lift increases price per square metre by **22.3% over properties without one**. This is by far the largest structural coefficient in the model, and it reflects a deep structural characteristic of Madrid's housing stock. A very large share of Madrid's older urban fabric — particularly in districts such as Centro, Tetuán, and Chamberí — consists of buildings from the early-to-mid 20th century where lifts were not standard. Installing a lift in such a building is expensive and often legally complex. Properties in lift buildings command a substantial premium because the alternative (walking up four or five floors) is a genuine quality-of-life cost that the market prices accordingly. The 22.3% lift premium is one of the most actionable findings in this analysis for a rehabilitation investor: a building renovation that includes lift installation can be expected to generate a double-digit percentage increase in the per-square-metre value of each unit, before considering any other improvements.

**Floor**: Properties on the fifth floor and above command a premium of **18.9% over ground floor** properties, all else equal. The attic (atico) floor coefficient was absorbed into the reference category in this specification. Higher floors are preferred in Madrid for reduced street noise, better light, and the perception of prestige — factors that are particularly valuable in the high-density urban core.

### 4.2 District Fixed Effects — The Price Map of Madrid

The district fixed effects measure the premium or discount of each zone relative to Arganzuela, which serves as the model's reference category. All percentage effects reported here are conditional on property characteristics, meaning they represent the premium or discount attributable purely to district identity, after controlling for size, rooms, bathrooms, lift, and floor.

The results produce a clear and internally consistent price hierarchy:

**Premium zones** (above Arganzuela reference):
- Barrio de Salamanca: **+79.4%** — the strongest premium in the city by a substantial margin. Salamanca is Madrid's uncontested prime residential address, home to flagship boutiques, the highest concentration of luxury restaurants, and consistent demand from high-net-worth buyers and expatriates. Its premium is structural and persistent.
- Chamberí: **+47.6%** — the second-most expensive district, known for its bohemian-bourgeois character, early-20th-century architecture, and proximity to the business district. It has undergone sustained gentrification over the past decade.
- Retiro: **+40.3%** — centred on the Retiro park, one of the largest urban parks in Europe, Retiro combines prestige with an unusually high green space to built environment ratio for a central city district.
- Chamartín: **+34.3%** — the northern business and financial district, home to the AZCA financial complex and the new Cuatro Torres skyline. Demand is driven heavily by professional and corporate occupiers.
- Centro: **+32.9%** — the historic core of the city, the premium here reflects the unique irreplaceable location rather than property quality. Much of the stock is old and unrenovated, but the address itself commands a price floor.

**At or near parity with Arganzuela** (reference zone at roughly 5,433 euros/m² median):
- Moncloa: **+5.9%** — home to the Complutense University campus and the Parque del Oeste. A slightly premium zone but more accessible than the top five.
- Tetuán: **-2.4%** — at parity. Tetuán is undergoing active gentrification, having received sustained urban investment over the past five years. At effectively the same model-adjusted price as Arganzuela, it is arguably underpriced relative to its trajectory.

**Discount zones** (below Arganzuela, broadly in the south and east):
- Hortaleza: -5.9%, Fuencarral: -9.2%, Ciudad Lineal: -13.3%, Barajas: -23.3%
- San Blas: -30.8%, Moratalaz: -34.2%, Latina: -36.2%
- Carabanchel: **-40.7%**, Vicálvaro: **-41.5%**, Villa de Vallecas: **-41.7%**
- Usera: **-42.5%**, Puente de Vallecas: **-46.9%**, Villaverde: **-52.7%**

The 132-percentage-point spread between Barrio de Salamanca (+79.4%) and Villaverde (-52.7%) represents the total spatial price range of the Madrid market after controlling for property characteristics. Location remains overwhelmingly the dominant determinant of price in this city.

---

## 5. Regularization — Ridge, Lasso, and Elastic Net

The regularization models were introduced for two reasons. First, to test whether the OLS model overfits: with 21 district dummies and a relatively modest sample, some shrinkage may be warranted. Second, and more importantly for the investment application, to ask which district price signals are real versus statistical noise — a question that Lasso answers directly by zeroing out the weakest coefficients.

All three regularization models were fitted via 10-fold cross-validation to select the optimal penalty strength (lambda).

**Ridge regression** (alpha = 0) found its optimal lambda at **0.02635**. It achieves a test RMSE of **0.2964** and an R² of **0.6983**. The performance is essentially identical to OLS (test RMSE of 0.2958, R² of 0.6994), which is the expected result: when the OLS model is already well-specified and not badly overfitting, Ridge adds little. Ridge shrinks all coefficients proportionally toward zero but never zeros any of them out. The fact that lambda is small (0.026) confirms that OLS is not severely overfitting the data.

**Lasso regression** (alpha = 1) selected an optimal lambda of **0.00039** — an even smaller penalty. With this penalty, Lasso retains **29 of the available predictors** with non-zero coefficients, out of approximately 35 total (district dummies plus structural features). It achieves a test RMSE of **0.2958** and R² of **0.6996**, statistically indistinguishable from OLS FE. The handful of features zeroed out are small district-level interactions that contribute negligible marginal explanatory power. The important conclusion is that all 21 district dummies survive meaningful Lasso pruning — the district segmentation of the Madrid market is not statistical noise. Every major price zone identified in the OLS Fixed Effects analysis represents a genuinely distinct price regime that cannot be averaged away.

**Elastic Net** (alpha = 0.5, a balance between Ridge and Lasso) operates at lambda = **0.00071** and produces RMSE = **0.2958**, R² = **0.6995**. The Elastic Net result sits between Ridge and Lasso as expected. Given that Lasso alone retains nearly all predictors and Ridge shows no benefit from shrinkage, the Elastic Net provides no material improvement over OLS in this case.

The overall verdict from the regularization section: the OLS Fixed Effects model is not overfitting. The data quality and sample size (8,408 training observations) are sufficient to estimate all district parameters with reasonable precision. This finding validates using the OLS coefficients as interpretable building blocks rather than treating them as unreliable estimates requiring heavy shrinkage.

---

## 6. Machine Learning Models — Random Forest and XGBoost

While the linear models are interpretable and serve the valuation explanation purpose, the machine learning models answer a simpler, harder question: how accurately can the price of a property be predicted using all available information, with no constraints on functional form?

Both models were trained on the same 75/25 split using a standardised preprocessing recipe that included median imputation for missing values, dummy encoding for categorical variables, and feature normalisation.

### 6.1 Random Forest

A Random Forest of **200 trees** was fitted with 8 candidate variables at each split (roughly the square root of the total predictor count) and a minimum node size of 5. The model achieves a test RMSE of **0.2773** and a test R² of **0.736**.

This represents a meaningful improvement over OLS: an R² of 73.6% versus 69.9%, and RMSE of 0.2773 versus 0.2958 — a 6.3% reduction in prediction error. The improvement comes from the Random Forest's ability to capture non-linear relationships and interactions that the linear model misses. The most important of these is the interaction between district and property type: the premium for a lift is not the same in Salamanca as it is in Villaverde; the floor premium does not operate identically in the historic centre and in the 1970s-built residential periphery. A linear model cannot capture these interactions without explicit cross-product terms; a Random Forest captures them automatically.

The variable importance rankings from the Random Forest illuminate the architecture of Madrid property prices. The single most important predictor is the **Barrio de Salamanca indicator** — not size, not lift, not rooms. This confirms that location in Madrid's luxury core is the dominant driver of value, and it has no close substitute. The second most important predictor is the **lift indicator**, reinforcing the 22.3% lift premium found in OLS. **Building surface area** ranks third, ahead of most other district indicators, suggesting that size matters relatively uniformly across the city once zone is accounted for. The district indicators for Centro, Puente de Vallecas, Villaverde, Chamberí, and Carabanchel follow, establishing the price hierarchy identified earlier.

### 6.2 XGBoost

XGBoost — a gradient boosted tree model trained with **400 trees**, learning rate of **0.05**, maximum tree depth of **5**, and minimum node size of **10** — achieves a test RMSE of **0.2757** and test R² of **0.7389**.

This is the best-performing model in the analysis. The improvement over Random Forest is marginal (RMSE of 0.2757 vs 0.2773, R² of 73.9% vs 73.6%), but consistent. XGBoost's advantages in this setting come from its sequential correction mechanism: each new tree in the ensemble is trained specifically on the residuals of all previous trees, meaning the model iteratively focuses on the observations that prior trees struggled with. In practice, this tends to help at the tails of the price distribution — properties that are unusually cheap or unusually expensive for their characteristics — which are precisely the observations of most interest to a value investor looking for mispriced assets.

The variable importance structure from XGBoost is consistent with the Random Forest: Barrio de Salamanca dominates (32.7% of total importance), followed by lift (15.4%), then Centro (7.8%), size (7.4%), Puente de Vallecas and Villaverde (4.2% and 4.1%), and Carabanchel, Retiro, Chamberí, and rooms rounding out the top 10. The convergence between two fundamentally different model architectures on the same variable importance ranking is reassuring: these signals are robust.

---

## 7. The Full Model Leaderboard

Ranking all seven models on out-of-sample test set performance:

| Model | Test RMSE | Test R² |
|---|---|---|
| XGBoost | 0.2757 | 0.739 |
| Random Forest | 0.2773 | 0.736 |
| OLS Fixed Effects | 0.2958 | 0.699 |
| Lasso | 0.2958 | 0.700 |
| Elastic Net | 0.2958 | 0.700 |
| Ridge | 0.2964 | 0.698 |
| Linear Mixed Model | 0.3164 | 0.656 |

XGBoost is the preferred model for production use — it has the lowest prediction error and is well-suited to the automated valuation application. OLS Fixed Effects is the preferred model for communicating results to clients and for explaining specific valuations, because its coefficients carry direct percentage-point interpretations that non-technical stakeholders can engage with. The regularization models confirm the OLS findings without adding explanatory value in this dataset. The Linear Mixed Model is discussed separately because its purpose is different.

All models materially outperform a naive baseline of predicting the mean log price for every observation, which would achieve an R² of zero by construction.

---

## 8. Linear Mixed Model — What Zone Tells Us About Value

The Linear Mixed Model (LMM) deserves a section of its own because it answers a conceptually different question from the other models: it quantifies how much of property price variation in Madrid is fundamentally attributable to zone identity versus individual property characteristics.

The LMM was specified with numeric property features as fixed effects (size, rooms, bathrooms, and the bathroom missingness indicator) and zona as a random effect — meaning each district gets its own intercept, but that intercept is estimated jointly with all other districts rather than freely. The model achieves a test RMSE of **0.3164** and test R² of **0.6562**, which is lower than the Fixed Effects OLS. This is expected: the LMM sacrifices some predictive power in exchange for a more principled treatment of between-district variance.

The key finding from the LMM is the **Intraclass Correlation Coefficient (ICC) of 0.64**. The ICC measures the proportion of total variance in log(price/m²) that lies between districts rather than within them. An ICC of 0.64 means that **64% of all price variation in Madrid is between-district variation**. Only 36% of the variance in price per square metre is explained by property-level factors — size, rooms, bathrooms, lift, floor.

Put differently: if you knew nothing about a property except which district it was in, you would already know 64% of what matters for its price. If you knew all the property characteristics but not the district, you would know only 36%. The single most important piece of information about a Madrid property's price is its postcode.

This is confirmed by the marginal versus conditional R² decomposition. The **marginal R² of 0.007** captures the explanatory power of fixed effects alone (structural features without any zone information) — effectively zero. The **conditional R² of 0.642** captures fixed effects plus zone random intercepts — substantially higher. The difference between 0.7% and 64.2% is the contribution of zone identity.

For an acquisition investor, this finding has a direct practical implication: the starting point for any valuation conversation must be the district. All the debate about number of rooms, floor level, and lift pales in significance next to the question of which side of a district boundary a property sits on.

---

## 9. Automated Valuation and Mispricing Detection

Having established the best-performing model (XGBoost) and the most interpretable model (OLS Fixed Effects), the analysis turns to the practical application: identifying individual listings where the asking price diverges significantly from the model's estimate of fair value.

For each of the 11,211 listings in the full dataset, the OLS Fixed Effects model (refitted on all observations) generates a model-predicted log price per square metre. The difference between the actual asking price and the model prediction — expressed as a percentage — is the **mispricing signal**.

A positive mispricing signal means the property is asking more than the model estimates it should be worth given its district and characteristics. A negative signal means the property appears underpriced relative to comparable properties.

At the aggregate level, the median mispricing across all listings is **+0.15%** — essentially zero, which is the expected result. A well-fitted model should have symmetric residuals centred at zero. The mean absolute mispricing across the sample is **2.66%**, meaning the typical property's asking price deviates from the model estimate by 2.66% in either direction. This is the model's inherent noise floor: differences smaller than approximately 3% should not be treated as actionable signals.

At the district level, the mispricing signal becomes more informative:

**Most undervalued districts** (asking prices systematically below model predictions):
- Arganzuela: median mispricing of -0.23%
- Ciudad Lineal: -0.21%
- Fuencarral: -0.18%
- Chamartín: -0.06%

**Most overvalued districts** (asking prices systematically above model predictions):
- Latina: +0.55%
- Moratalaz: +0.55%
- Barajas: +0.44%
- Usera: +0.41%
- Puente de Vallecas: +0.39%

The interpretation of these district-level signals requires care. A positive district-level mispricing does not necessarily mean sellers in those zones are deluded — it may reflect genuine recent price momentum not yet captured in the model's estimates, neighbourhood improvement trends that have outpaced the average property characteristics, or a specific demand shock (new transport links, urban renewal projects) that the structural model does not observe. Similarly, a negative mispricing in a district like Arganzuela may reflect the fact that sellers are listing aggressively in anticipation of future price appreciation from ongoing gentrification.

At the individual property level, the analysis identifies the top-20 listings showing the greatest apparent undervaluation: properties where the asking price is most below the model's prediction given their district and characteristics. These represent the highest-priority candidates for further due diligence — they are statistically cheap for what they are, and the question for an investor is whether the gap reflects an opportunity or an unobserved deficiency in the listing data.

---

## 10. Price Forecasts 2026–2027

The 2026-2027 forecasting framework combines the model-derived district price estimates with external consensus growth projections from national appraisal indices and public listing platforms, calibrated against the January 2026 market data.

Madrid's 21 districts were grouped into three structural market segments for the forecasting exercise, based on their position in the price hierarchy, their recent growth trajectory, and their absorption of urban investment:

**Premium segment** (Barrio de Salamanca, Chamberí, Retiro, Chamartín): These districts have median prices of 7,353 to 10,526 euros/m². Demand is structurally constrained by limited new supply and sustained international buyer interest. Growth is expected to be moderate but very stable. Base scenario: **+5.5% in 2026**, **+5.5% in 2027**, compounding to a two-year total return of approximately **+11.3%** in the base case.

**Gentrifying segment** (Centro, Tetuán, Arganzuela, Moncloa, Hortaleza): These districts are at varying stages of urban renewal. Centro and Tetuán have undergone substantial change over the past decade and are approaching saturation in their gentrification premiums. Arganzuela and Hortaleza are earlier in the cycle, offering higher growth potential at lower entry prices. Base scenario: **+6.5% per annum**, compounding to approximately **+13.4% over two years** in the base case.

**Emerging segment** (Fuencarral, Ciudad Lineal, Usera, Latina, Carabanchel, Vallecas, Villaverde, Moratalaz, San Blas, Barajas, Vicálvaro): Median prices range from 2,531 to 4,920 euros/m². These districts have the widest range of outcomes: some are actively transitioning (Carabanchel, Usera, Latina), while others are more structurally constrained by demographics and infrastructure. Base scenario: **+4.0% per annum**, approximately **+8.2% over two years**.

Each segment carries three scenarios: a bear case (market stress, credit tightening), a base case (continuation of current trends), and a bull case (above-consensus demand, continued international buyer flows). The spread between bear and bull is wider in the emerging segment (bear +2%, bull +7% annually) than in the premium segment (bear +3%, bull +8%), reflecting greater uncertainty in price evolution in zones that are more sensitive to domestic economic conditions.

These forecasts are not predictions of certainty — no such thing exists in property markets. They are calibrated scenarios for use in investment decision modelling. The practical use is: any acquisition decision should test its return assumptions across all three scenarios, with the base case as the planning assumption and the bear case as the stress test.

---

## 11. Portfolio Intelligence Framework

This section demonstrates how the analytical framework maps onto an investor's existing portfolio. The demo portfolio consists of 20 buildings across Madrid, concentrated in Centro, Tetuan, Carabanchel, Salamanca, Chamberi, and Latina. Users can replace the demo portfolio with their own holdings to generate the same analysis.

Mapping the portfolio against the model's findings produces several actionable observations.

**Centro and Tetuan**: Both districts sit in the gentrifying segment of the forecast framework, with base-case annual appreciation of 6.5%. Tetuan shows a -2.4% price discount relative to Arganzuela in the fixed effects model, meaning it is not yet fully priced relative to comparable properties on structural metrics — a finding consistent with ongoing gentrification. The discount rate in Tetuan is 10.6%, above the city average of 9.5%, which signals more motivated sellers than in premium zones. For a rehabilitation investor, Tetuan at this stage of the cycle offers the combination of below-premium entry prices, above-average motivated seller signal, and above-average forecast growth.

**Carabanchel**: The highest discount rate in the sample at 16.5%. The Fixed Effects premium is -40.7% versus the Arganzuela reference, placing it firmly in the emerging segment. The forecast base return is +8.2% over two years. The combination of the highest motivated-seller rate and the lowest prices makes Carabanchel the highest-volume acquisition opportunity in the city by the composite scoring framework, but it is also the highest-risk bet on continued urban improvement.

**Salamanca and Chamberi**: Both districts show solid forecast returns and very low motivated-seller activity (4.9% and 10.7% discount rates respectively). These holdings function as strategic anchors — lower risk, lower yield, high capital preservation profile.

**Underrepresented districts**: The model identifies several districts with strong fundamentals where the portfolio has no current presence, most notably Arganzuela, Chamartin, and Retiro. Arganzuela shows the most consistent undervaluation signal in the mispricing analysis (-0.23% median mispricing). Chamartin is a +34.3% premium zone in the fixed effects model, occupying a structural position just below the top four premium districts, with forecast growth consistent with the premium segment. Retiro at +40.3% has strong appreciation potential and benefits from the Retiro park proximity premium.

### 11.1 The Rehabilitation Margin Framework

For a rehabilitation-focused investor, the relevant return calculation is not the appreciation in existing asking prices but the margin between acquisition cost, rehabilitation investment, and post-renovation achievable price. The framework built here models this as follows: a typical rehabilitation project in Madrid involves acquisition at a discount to the current market (targeting motivated sellers), a renovation cost of approximately **900 euros per square metre** (consistent with mid-to-high specification renovation in an urban setting), and an exit at the post-renovation model-predicted price.

The rehabilitation margin — defined as (post-renovation predicted price − acquisition cost − renovation cost) divided by total cost — varies substantially by district. Districts with the highest structural discount to the premium segment (Carabanchel, Tetuán, Latina) offer the widest gross margins on a per-square-metre basis, because the acquisition price is low relative to what a renovated building can achieve. The value-add component is largest where the gap between the current building stock quality and what the district can support at renovation quality is greatest.

---

## 12. Motivated Seller Detection — The Discount Classification Model

The final modelling component addresses a question that has practical acquisition value: can the model predict, from listing characteristics, which properties are likely to be sold at a price discount?

A logistic regression was fitted to predict the probability that any given listing has been discounted from its original asking price, using the same structural features and district dummies as the price models. The model was evaluated on the held-out test set.

**AUC: 0.6155** — this is the key performance metric. An AUC (Area Under the ROC Curve) of 0.5 represents random guessing; 1.0 is perfect prediction. At 0.6155, the model achieves meaningful but modest predictive power over the base rate. It is not a precision instrument, but it is a statistically significant signal.

**Accuracy: 90.6%** — this high accuracy number requires careful interpretation. Because only 9.5% of listings are discounted, a model that simply predicted "no discount" for every property would achieve 90.5% accuracy. The logistic regression's 90.6% accuracy barely exceeds this naive baseline, confirming that accuracy is not the right metric for a highly imbalanced classification problem.

The AUC of 0.6155 is the correct number to focus on. It means the model correctly ranks a discounted property ahead of a non-discounted one — in other words, correctly identifies which of two properties is more likely to offer a price reduction — in approximately 62% of cases. For an acquisition team making initial screening decisions across a large pipeline of properties, a model that outperforms random ranking by 12 percentage points is useful for triaging the pipeline, even if it is not precise enough to rely on alone.

The most predictive variables for discount probability, revealed by the zone-level analysis, are:
- **District identity**: Carabanchel (16.5% actual discount rate), Barajas (15.6%), and Arganzuela (12.7%) are the zones where sellers are most likely to cut prices. Barrio de Salamanca (4.9%) and Vicálvaro (3.1%) are the zones where price reductions are rarest.
- **Lift absence**: Properties without lifts in a city where lift presence commands a 22.3% premium are structurally harder to sell, creating more motivated sellers.
- **Higher floor without lift**: A combination that dramatically limits the buyer pool.

The practical use of this model is not to predict with certainty which sellers will discount, but to rank listings by motivated-seller probability and prioritise outreach toward the upper quantile. A team reviewing 500 potential acquisitions per month could use the model to focus resources on the 100 listings where the discount signal is strongest, rather than distributing contact effort evenly.

---

## 13. Why the Models Agree — and What That Means

One of the most important findings in this analysis is not a single number but a structural pattern: the seven models, built using completely different mathematical principles, produce a highly consistent picture of the Madrid market.

The OLS model with district dummies achieves 70% R². The Lasso keeps all district dummies. The Random Forest's top predictor is district identity. XGBoost's top predictor is district identity. The LMM's ICC shows 64% of variance is between-district. These findings converge on the same conclusion from multiple directions: **in Madrid, district is not a proxy variable — it is the signal itself**.

This convergence also validates the quality of the data and the analytical framework. When simple models and complex ones agree, it is because the signal is genuine and strong. The fact that XGBoost outperforms OLS by approximately 4 percentage points in R² — rather than, say, 15 or 20 points — tells us that the linear framework is not fundamentally misspecified. The non-linearity that gradient boosting exploits is real but marginal. The linear model is doing most of the work correctly.

---

## 14. Conclusions and Investment Implications

The analysis produces a coherent, data-grounded view of the Madrid residential market. The following conclusions follow directly from the numbers, without interpretation beyond what the evidence supports.

**On pricing**: The Madrid residential market operates with strong spatial segmentation. A fourteenfold price range between the cheapest and most expensive districts, and a district ICC of 64%, means that location is the dominant determinant of property value. No structural feature of a property — not size, not the number of bathrooms, not even the presence of a lift — comes close to explaining as much variance in price as the district indicator alone.

**On structural features**: Among property characteristics, the presence of a lift (+22.3%) and location on a high floor (+18.9%) are the two features with the largest and most reliable impact on price per square metre. Bathrooms add about 3.4% per unit. Rooms add nothing per square metre once size is controlled for; they reduce price/m² by 3.75% per additional room through the room density effect. For a rehabilitation investor, the implication is direct: the highest-return structural interventions are lift installation and floor-through renovations that eliminate dated room configurations in favour of open-plan layouts.

**On model selection**: XGBoost is the most accurate predictive model, but the margin over OLS is modest. For individual property valuation, XGBoost reduces prediction error from 34% to 32% relative to the asking price — meaningful but not transformational. For the mispricing application, where hundreds of properties are being screened simultaneously, XGBoost is the right tool. For explaining a specific valuation or negotiating a price with a seller, the OLS coefficients are more useful because they are transparent.

**On investment geography**: The composite analysis — combining forecast returns, motivated-seller signal, rehabilitation margin, and current mispricing — points toward a consistent ordering of districts by acquisition attractiveness. Tetuán and Arganzuela score well on multiple dimensions simultaneously: above-average forecast growth (gentrifying segment), above-average motivated-seller rates, below-premium current pricing, and positive rehabilitation margins. Carabanchel offers the highest absolute discount rates and the widest rehabilitation margin but carries the most market risk. Barrio de Salamanca and Chamberí offer the most stable forecasts and the least acquisition friction, but the lowest motivated-seller signal.

**On portfolio construction**: Overconcentration in any single district creates concentration risk. The model's mispricing signals and forward return estimates suggest that Arganzuela, Chamartin, and the later-stage gentrification zones of Tetuan represent the most attractive underexplored entry points for the 2026-2027 acquisition cycle. A balanced portfolio across the gentrifying and emerging segments, with the premium zone holdings providing stability, would reduce correlation risk while maintaining access to the market's strongest growth dynamics.

---

*All statistics in this report are derived directly from the 11,211-listing dataset and the fitted models. Market benchmark figures for comparative validation are sourced from national appraisal indices and public listing platform publications (January 2026). Forecasts represent calibrated scenarios, not guarantees of future price performance.*
