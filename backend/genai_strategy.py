"""
PriceIQ — GenAI Strategy Engine
Generates dynamic pricing strategy narratives.
Option A: rule-based Python engine (works offline, no API key).
Option B: Google Gemini API (falls back to Option A on failure).
"""

import os
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

INDUSTRY_MARGIN_BENCHMARK = 0.25
ELASTIC_THRESHOLD = -1.0


# ── Option A — Rule-based engine ─────────────────────────────────────────────
def _rule_based_strategy(
    price: float,
    predicted_demand: float,
    elasticity: float,
    margin: float,
    item_type: str,
    outlet_type: str,
    optimal_price: float,
    competitor_avg_price: float,
) -> dict:
    gap = optimal_price - price
    gap_pct = (gap / price * 100) if price else 0
    direction = "below" if gap > 0 else "above"
    abs_gap_pct = abs(gap_pct)

    # Headline
    if abs_gap_pct < 3:
        headline = f"Pricing is near-optimal — within {abs_gap_pct:.0f}% of the sweet spot."
    elif gap > 0:
        headline = f"You are {abs_gap_pct:.0f}% below optimal — significant revenue left on the table."
    else:
        headline = f"You are {abs_gap_pct:.0f}% above optimal — demand erosion risk is high."

    # Insights
    insights = []

    # 1. Margin health
    if margin >= INDUSTRY_MARGIN_BENCHMARK:
        insights.append(
            f"Margin of {margin*100:.1f}% exceeds the industry benchmark of "
            f"{INDUSTRY_MARGIN_BENCHMARK*100:.0f}% — healthy profitability."
        )
    else:
        shortfall = (INDUSTRY_MARGIN_BENCHMARK - margin) * 100
        insights.append(
            f"Margin of {margin*100:.1f}% is {shortfall:.1f}pp below the "
            f"{INDUSTRY_MARGIN_BENCHMARK*100:.0f}% benchmark — consider cost optimization or repricing."
        )

    # 2. Elasticity
    if elasticity < ELASTIC_THRESHOLD:
        insights.append(
            f"{item_type} is highly elastic (ε = {elasticity:.2f}). "
            f"A 10% price hike could reduce demand by ~{abs(elasticity)*10:.1f}%. Price with caution."
        )
    elif elasticity < 0:
        insights.append(
            f"{item_type} shows moderate elasticity (ε = {elasticity:.2f}). "
            f"Some room to raise price, but monitor volume closely."
        )
    else:
        insights.append(
            f"{item_type} appears inelastic (ε = {elasticity:.2f}). "
            f"Consumers are less price-sensitive — pricing power is strong."
        )

    # 3. Gap to optimal
    if abs_gap_pct < 3:
        insights.append(
            f"Current price ₹{price:,.0f} is within ₹{abs(gap):,.0f} of the "
            f"model-optimal ₹{optimal_price:,.0f}. Fine-tune other levers like visibility & ratings."
        )
    else:
        revenue_at_current = price * predicted_demand
        projected_at_optimal = optimal_price * predicted_demand * (1 + gap_pct / 200)
        insights.append(
            f"Moving price from ₹{price:,.0f} to ₹{optimal_price:,.0f} ({direction} "
            f"by {abs_gap_pct:.0f}%) could shift projected demand-revenue from "
            f"₹{revenue_at_current:,.0f} toward ₹{projected_at_optimal:,.0f}."
        )

    # 4. Competitor context
    vs_comp = price - competitor_avg_price
    comp_dir = "above" if vs_comp > 0 else "below"
    insights.append(
        f"Your price sits ₹{abs(vs_comp):,.0f} {comp_dir} the category average "
        f"of ₹{competitor_avg_price:,.0f}. "
        + ("Consider maintaining premium positioning." if vs_comp > 0
           else "Competitive pricing may drive higher volume.")
    )

    # Tags
    tags = []
    if gap > 0:
        tags.append("#Underpriced")
    elif gap < -5:
        tags.append("#Overpriced")
    else:
        tags.append("#OptimalZone")

    tags.append("#ElasticCategory" if elasticity < ELASTIC_THRESHOLD else "#InelasticCategory")
    tags.append("#HealthyMargin" if margin >= INDUSTRY_MARGIN_BENCHMARK else "#MarginPressure")

    if competitor_avg_price > 0 and abs(vs_comp) / competitor_avg_price > 0.15:
        tags.append("#CompetitorGap")

    # Recommendation
    if abs_gap_pct < 3:
        recommendation = (
            f"Hold current pricing at ₹{price:,.0f} and focus on improving ratings "
            f"and product visibility to drive volume."
        )
    elif gap > 0:
        recommendation = (
            f"Increase price toward ₹{optimal_price:,.0f} in a phased manner "
            f"(+5% weekly) to capture ~{abs_gap_pct:.0f}% additional revenue."
        )
    else:
        recommendation = (
            f"Consider a tactical price reduction to ₹{optimal_price:,.0f} — "
            f"the volume uplift should more than compensate for the lower unit price."
        )

    return {
        "headline": headline,
        "insights": insights,
        "tags": tags,
        "recommendation": recommendation,
    }


# ── Option B — Gemini API ────────────────────────────────────────────────────
def _gemini_strategy(
    price: float,
    predicted_demand: float,
    elasticity: float,
    margin: float,
    item_type: str,
    outlet_type: str,
    optimal_price: float,
    competitor_avg_price: float,
) -> dict | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not GEMINI_AVAILABLE or not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""You are a retail/e-commerce pricing strategist AI. Given the following data, return a JSON object with exactly these keys:
- "headline": one bold strategic statement
- "insights": array of exactly 4 insight strings
- "tags": array of 3-4 hashtag strings
- "recommendation": one clear action sentence

DATA:
- Current Selling Price: ₹{price:.2f}
- Predicted Demand (rating count proxy): {predicted_demand:.0f}
- Price Elasticity: {elasticity:.2f}
- Profit Margin: {margin*100:.1f}%
- Industry Margin Benchmark: {INDUSTRY_MARGIN_BENCHMARK*100:.0f}%
- Category: {item_type}
- Channel: {outlet_type}
- Optimal Price (model-derived): ₹{optimal_price:.2f}
- Competitor Avg Price: ₹{competitor_avg_price:.2f}

RULES:
1. Every insight MUST reference at least one number from the data above.
2. Insight 1: margin health vs benchmark.
3. Insight 2: elasticity interpretation for {item_type}.
4. Insight 3: gap between current and optimal price.
5. Insight 4: competitor positioning.
Return ONLY valid JSON, no markdown."""

        response = model.generate_content(prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────
def generate_strategy(
    price: float,
    predicted_demand: float,
    elasticity: float,
    margin: float,
    item_type: str,
    outlet_type: str,
    optimal_price: float,
    competitor_avg_price: float,
) -> dict:
    """Return strategy dict. Uses Gemini if available, else rule-based."""
    result = _gemini_strategy(
        price, predicted_demand, elasticity, margin,
        item_type, outlet_type, optimal_price, competitor_avg_price,
    )
    if result:
        return result

    return _rule_based_strategy(
        price, predicted_demand, elasticity, margin,
        item_type, outlet_type, optimal_price, competitor_avg_price,
    )
