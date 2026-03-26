import os
import json
import time
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ---- Paths ----
SYNTHETIC_DIR = Path("data/synthetic")
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)

ARTICLES_DIR  = SYNTHETIC_DIR / "articles"
ARTICLES_DIR.mkdir(exist_ok=True)

PAIRS_PATH    = Path("data/training/pairs.jsonl")
DONE_PATH     = SYNTHETIC_DIR / "generated_topics.json"

# ---- Load progress ----
if DONE_PATH.exists():
    done_topics = set(json.loads(DONE_PATH.read_text()))
else:
    done_topics = set()

# ---- Article generator ----
def generate_article(topic):
    print(f"  Writing article: {topic}")
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Write a detailed 800-1200 word technical article about the following topic for Canadian commercial real estate lending professionals.

Topic: {topic}

Requirements:
- Use specific Canadian context (CMHC, OSFI, GoC spreads, provincial nuances)
- Include specific numbers, thresholds, and benchmarks where relevant
- Write as if for a senior analyst or underwriter
- Cover definitions, practical application, and key considerations
- Do not use headers or bullet points - write in flowing prose

Write the article now:"""
        }]
    )
    return response.content[0].text.strip()

# ---- Pair generator ----
def generate_pairs_from_article(topic, article):
    print(f"  Generating pairs...")
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": f"""Generate 15 high quality question/answer training pairs from this Canadian CRE article.

Topic: {topic}

Article:
{article}

Rules:
- Questions must be specific and technical
- Answers must be accurate, detailed, and use proper Canadian CRE terminology  
- Include specific numbers and thresholds from the article
- Vary question types: definitional, analytical, comparative, procedural
- Answers should be 3-6 sentences

Respond ONLY with a JSON array, no preamble, no markdown:
[{{"prompt": "...", "completion": "..."}}, ...]"""
        }]
    )
    
    raw = response.content[0].text.strip()
    
    # clean control characters
    cleaned = ""
    for char in raw:
        if char in ['\n', '\t']:
            cleaned += " "
        elif ord(char) >= 32:
            cleaned += char
    
    if "```" in cleaned:
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    
    return json.loads(cleaned)

# ---- Main runner ----
def run(topics):
    print(f"Total topics: {len(topics)}")
    print(f"Already done: {len(done_topics)}")
    print(f"Remaining: {len([t for t in topics if t not in done_topics])}")
    print()

    new_pairs  = 0
    newly_done = []

    for topic in topics:
        if topic in done_topics:
            print(f"Skipping (done): {topic}")
            continue

        print(f"\nTopic: {topic}")

        try:
            # generate article
            article = generate_article(topic)

            # save article
            safe_name = topic.lower().replace(" ", "_").replace("/", "_")[:60]
            article_path = ARTICLES_DIR / f"{safe_name}.txt"
            article_path.write_text(article, encoding="utf-8")

            # generate pairs
            pairs = generate_pairs_from_article(topic, article)

            # append to training pairs
            with open(PAIRS_PATH, "a") as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + "\n")

            new_pairs  += len(pairs)
            newly_done.append(topic)

            print(f"  ✓ {len(pairs)} pairs saved")

            # save progress after each topic
            all_done = list(done_topics) + newly_done
            DONE_PATH.write_text(json.dumps(all_done, indent=2))

            # small delay to avoid rate limits
            time.sleep(1)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue

    total_pairs = sum(1 for _ in open(PAIRS_PATH))
    print(f"\n✓ Generated {new_pairs} new pairs from {len(newly_done)} topics")
    print(f"✓ Total training pairs: {total_pairs:,}")

# ---- Topics - YOU FILL THIS IN ----
topics = [
    # ---- Underwriting Deep Dives ----
    "Breakeven occupancy analysis for Canadian multifamily properties",
    "Debt service coverage ratio sensitivity to interest rate changes in Canadian CRE",
    "Loan constant calculation and application in Canadian commercial mortgage underwriting",
    "Effective gross income multiplier as a valuation tool in Canadian CRE",
    "Gross rent multiplier analysis for Canadian multifamily properties",
    "Price per door analysis for Canadian apartment building acquisitions",
    "Replacement cost analysis in Canadian commercial real estate underwriting",
    "Insurable value vs market value in Canadian commercial real estate lending",
    "Going concern value vs liquidation value in Canadian CRE lending",
    "Stabilized net operating income projection methodology in Canadian CRE",
    "Above and below the line expense analysis in Canadian CRE underwriting",
    "Revenue gross up methodology in Canadian commercial real estate underwriting",
    "Straight line rent averaging in Canadian commercial lease underwriting",
    "Free rent and tenant improvement allowance treatment in Canadian CRE underwriting",
    "Percentage rent clauses and underwriting in Canadian retail properties",
    "Common area maintenance reconciliation in Canadian commercial real estate",
    "Triple net lease expense recovery analysis in Canadian commercial lending",
    "Modified gross lease underwriting in Canadian commercial real estate",
    "Absolute net lease underwriting and risk assessment in Canadian CRE",
    "Ground rent escalation and underwriting in Canadian commercial real estate",

    # ---- CMHC Deep Dives ----
    "CMHC Flex program requirements and application process in Canada",
    "CMHC Affordable Housing program criteria and underwriting",
    "CMHC Energy Efficiency program incentives and requirements",
    "CMHC Rental Construction Financing Initiative requirements",
    "CMHC National Housing Co-Investment Fund lending criteria",
    "CMHC Seed Funding program requirements and eligibility",
    "CMHC mortgage insurance premium calculation and impact on returns",
    "CMHC application process timeline and key milestones",
    "CMHC property inspection requirements and standards",
    "CMHC replacement reserve requirements and calculation",
    "CMHC debt service coverage requirements by property type",
    "CMHC maximum loan to value ratios by property type",
    "CMHC borrower eligibility requirements and assessment",
    "CMHC physical needs assessment requirements and standards",
    "CMHC market rent analysis requirements and methodology",
    "CMHC social housing underwriting criteria and requirements",
    "CMHC co-operative housing lending criteria and process",
    "CMHC indigenous housing lending programs and criteria",
    "CMHC Northern housing lending programs and considerations",
    "CMHUP program requirements and northern community lending",

    # ---- Provincial Nuances ----
    "Ontario Residential Tenancies Act impact on multifamily lending",
    "BC Residential Tenancy Act impact on multifamily underwriting",
    "Alberta Residential Tenancies Act and multifamily lending considerations",
    "Quebec Civil Code impact on commercial real estate lending",
    "Quebec TAL tribunal and rent dispute risk in multifamily lending",
    "Ontario land transfer tax and commercial real estate transactions",
    "BC property transfer tax and commercial real estate transactions",
    "Alberta land titles system and commercial mortgage registration",
    "Ontario personal property security act and commercial lending",
    "BC personal property security act and commercial lending",
    "Ontario Planning Act and development land lending",
    "BC Agricultural Land Reserve and development lending restrictions",
    "Alberta municipal development plans and commercial lending",
    "Quebec zoning regulations and commercial real estate lending",
    "Ontario rent control exemptions and new construction lending",
    "BC rent increase guidelines and multifamily underwriting",
    "Alberta rent regulation changes and multifamily lending impact",
    "Ontario vacant home tax impact on commercial real estate",
    "BC speculation and vacancy tax impact on commercial lending",
    "Toronto municipal land transfer tax and commercial transactions",

    # ---- Specific Deal Structures ----
    "Equity takeout refinancing in Canadian commercial real estate",
    "Cash out refinancing strategy and underwriting in Canadian CRE",
    "Portfolio cross collateralization strategy in Canadian commercial lending",
    "Blanket mortgage structure across multiple Canadian properties",
    "Partial release provisions in Canadian commercial mortgage structures",
    "Substitution of collateral in Canadian commercial mortgage lending",
    "Future advance mortgage provisions in Canadian commercial lending",
    "Construction to permanent mortgage conversion in Canadian lending",
    "Mini perm loan structure in Canadian commercial real estate",
    "Participating mortgage income sharing structures in Canadian CRE",
    "Convertible mortgage equity participation in Canadian CRE",
    "Mezzanine loan intercreditor agreement requirements in Canada",
    "Subordination non-disturbance and attornment agreements in Canada",
    "Recognition agreement requirements in Canadian commercial lending",
    "Estoppel certificate requirements in Canadian commercial lending",
    "Tenant direction to pay in Canadian commercial mortgage structures",
    "Cash management agreement structures in Canadian CRE lending",
    "Deposit account control agreement in Canadian commercial lending",
    "Lockbox structure in Canadian commercial real estate lending",
    "Cash sweep mechanism in Canadian commercial mortgage structures",

    # ---- Specific Asset Deep Dives ----
    "High rise apartment underwriting specific considerations in Canada",
    "Low rise apartment underwriting specific considerations in Canada",
    "Garden suite and laneway housing lending in Canadian cities",
    "Rooming house and shared accommodation lending in Canada",
    "Short term rental property underwriting and lending in Canada",
    "Furnished apartment underwriting and lending in Canada",
    "Affordable housing development underwriting in Canada",
    "Supportive housing lending criteria and underwriting in Canada",
    "Transitional housing underwriting and lending considerations",
    "Co-living development underwriting and lending in Canada",
    "Micro unit apartment underwriting and lending in Canada",
    "Seniors active lifestyle community underwriting in Canada",
    "Memory care facility underwriting and lending in Canada",
    "Retirement home underwriting and lending in Canada",
    "Long term care home lending criteria and underwriting in Canada",
    "Medical office building underwriting and lending in Canada",
    "Dental office building underwriting and lending in Canada",
    "Veterinary clinic real estate underwriting and lending",
    "Pharmacy real estate underwriting and lending in Canada",
    "Cannabis facility underwriting and lending in Canada",

    # ---- Industrial Deep Dives ----
    "Big box industrial underwriting and lending in Canada",
    "Multi tenant industrial underwriting and lending in Canada",
    "Cold storage industrial facility underwriting in Canada",
    "Food processing facility underwriting and lending in Canada",
    "Data centre underwriting and lending in Canada",
    "Flex industrial underwriting and lending in Canada",
    "Light industrial underwriting and lending in Canada",
    "Heavy industrial underwriting and lending in Canada",
    "Truck terminal underwriting and lending in Canada",
    "Distribution centre underwriting and lending in Canada",
    "Last mile logistics facility underwriting in Canada",
    "Industrial outdoor storage underwriting and lending in Canada",
    "Manufacturing facility underwriting and lending in Canada",
    "Research and development facility underwriting in Canada",
    "Airport industrial underwriting and lending in Canada",
    "Port adjacent industrial underwriting and lending in Canada",
    "Rail served industrial underwriting and lending in Canada",
    "Free trade zone industrial lending considerations in Canada",
    "Industrial strata underwriting and lending in Canada",
    "Industrial condo underwriting and lending in Canada",

    # ---- Retail Deep Dives ----
    "Power centre retail underwriting and lending in Canada",
    "Neighbourhood retail underwriting and lending in Canada",
    "Community retail centre underwriting and lending in Canada",
    "Regional mall underwriting and lending challenges in Canada",
    "Lifestyle centre retail underwriting and lending in Canada",
    "Urban street retail underwriting and lending in Canada",
    "Grocery anchored retail underwriting and lending in Canada",
    "Drug store anchored retail underwriting and lending in Canada",
    "Fast food and QSR property underwriting and lending in Canada",
    "Automotive dealership underwriting and lending in Canada",
    "Gas station and car wash underwriting and lending in Canada",
    "Car wash facility underwriting and lending in Canada",
    "Restaurant property underwriting and lending in Canada",
    "Fitness centre underwriting and lending in Canada",
    "Cinema and entertainment retail underwriting in Canada",
    "Big box anchor tenant analysis in Canadian retail lending",
    "Shadow anchor retail underwriting and lending in Canada",
    "Outparcel retail underwriting and lending in Canada",
    "Pad site retail underwriting and lending in Canada",
    "Dark anchor tenant risk in Canadian retail lending",

    # ---- Office Deep Dives ----
    "Downtown Class A office underwriting in Canadian cities",
    "Suburban office underwriting and lending in Canada",
    "Medical office building underwriting in Canadian markets",
    "Government leased office underwriting and lending in Canada",
    "Creative office space underwriting and lending in Canada",
    "Life sciences office underwriting and lending in Canada",
    "Tech office campus underwriting and lending in Canada",
    "Office conversion to residential underwriting in Canada",
    "Office to industrial conversion underwriting in Canada",
    "Single tenant office underwriting and lending in Canada",
    "Multi tenant office underwriting and lending in Canada",
    "Trophy office underwriting and lending in Canadian markets",
    "Class B office repositioning underwriting in Canada",
    "Class C office workout and lending considerations",
    "LEED certified office underwriting and green lending in Canada",
    "Net zero office underwriting and green financing in Canada",
    "Hybrid work impact on Canadian office underwriting",
    "Sublease space risk in Canadian office underwriting",
    "Tenant improvement allowance analysis in Canadian office lending",
    "Office lease expiry risk analysis in Canadian CRE lending",

    # ---- Construction Lending Deep Dives ----
    "Construction budget review and analysis in Canadian lending",
    "Construction contingency requirements in Canadian lending",
    "Hard cost vs soft cost analysis in Canadian construction lending",
    "Construction draw schedule review in Canadian lending",
    "Quantity surveyor role in Canadian construction lending",
    "Construction monitoring consultant role in Canadian lending",
    "Holdback release requirements in Canadian construction lending",
    "Construction completion guarantee in Canadian lending",
    "Cost overrun risk in Canadian construction lending",
    "Construction delay risk in Canadian commercial lending",
    "Pre-sale requirements for Canadian construction lending",
    "Pre-leasing requirements for Canadian commercial construction lending",
    "Construction loan interest reserve calculation in Canada",
    "Construction loan fee structure in Canadian lending",
    "Completion bond requirements in Canadian construction lending",
    "Performance bond requirements in Canadian construction lending",
    "Labour and material payment bond in Canadian construction lending",
    "Construction lien holdback requirements in Canadian provinces",
    "Substantial completion definition in Canadian construction lending",
    "Occupancy permit timing and construction loan repayment in Canada",

    # ---- Market Cycles and Economics ----
    "Canadian commercial real estate market cycle analysis and lending",
    "Cap rate expansion impact on Canadian CRE portfolio lending",
    "Rising interest rate impact on Canadian commercial real estate values",
    "Falling interest rate impact on Canadian commercial real estate lending",
    "Recession impact on Canadian commercial real estate lending",
    "Inflation impact on Canadian commercial real estate operating costs",
    "Supply chain disruption impact on Canadian construction lending",
    "Labour shortage impact on Canadian construction lending",
    "Material cost escalation risk in Canadian construction lending",
    "Population decline impact on secondary market CRE lending in Canada",
    "Remote work trend impact on Canadian commercial real estate lending",
    "E-commerce impact on Canadian retail and industrial real estate lending",
    "Climate change physical risk impact on Canadian CRE lending",
    "Energy transition impact on Canadian commercial real estate lending",
    "Demographic shift impact on Canadian commercial real estate lending",
    "Immigration wave impact on Canadian multifamily lending",
    "Baby boomer wealth transfer impact on Canadian CRE lending",
    "Millennial homeownership trends and Canadian multifamily lending",
    "Gen Z renter demand impact on Canadian multifamily lending",
    "Urbanization trends impact on Canadian commercial real estate lending",

    # ---- Advanced Financial Concepts ----
    "Waterfall distribution structures in Canadian CRE joint ventures",
    "Promoted interest structures in Canadian CRE partnerships",
    "Preferred return hurdle rates in Canadian CRE private equity",
    "Internal rate of return targets by asset class in Canadian CRE",
    "Equity multiple targets in Canadian commercial real estate investment",
    "Unlevered vs levered returns in Canadian CRE investment analysis",
    "Risk adjusted return analysis in Canadian commercial real estate",
    "Portfolio diversification strategy in Canadian commercial real estate",
    "Core vs core plus vs value add vs opportunistic in Canadian CRE",
    "Canadian REIT valuation methodology and NAV analysis",
    "Canadian REIT distribution sustainability analysis",
    "Canadian REIT debt metrics and lending considerations",
    "Private REIT structure and lending considerations in Canada",
    "Real estate limited partnership structure in Canadian CRE",
    "Co-investment structure in Canadian commercial real estate",
    "Fund of funds structure in Canadian commercial real estate",
    "Separate account structure in Canadian institutional CRE lending",
    "Open end fund vs closed end fund in Canadian CRE",
    "Redemption risk in Canadian open end real estate funds",
    "NAV discount and premium analysis for Canadian REITs",

    # ---- ESG and Sustainability ----
    "Green building certification impact on Canadian commercial lending",
    "LEED certification levels and Canadian CRE lending premiums",
    "BOMA BEST certification and Canadian commercial real estate lending",
    "Energy efficiency retrofit financing in Canadian commercial real estate",
    "Green mortgage programs for Canadian commercial real estate",
    "PACE financing for Canadian commercial real estate energy upgrades",
    "Carbon footprint reduction requirements in Canadian CRE lending",
    "Stranded asset risk from energy inefficiency in Canadian CRE",
    "ESG reporting requirements for Canadian institutional CRE lenders",
    "Climate risk disclosure in Canadian commercial real estate lending",
    "Flood risk assessment in Canadian commercial real estate lending",
    "Wildfire risk assessment in Canadian commercial real estate lending",
    "Earthquake risk assessment in Canadian commercial real estate lending",
    "Permafrost risk in Northern Canadian commercial real estate lending",
    "Transition risk from carbon pricing in Canadian CRE lending",
    "Social impact investing in Canadian commercial real estate",
    "Affordable housing mandate in Canadian institutional CRE lending",
    "Indigenous land rights and Canadian commercial real estate lending",
    "Community benefit agreements in Canadian commercial real estate",
    "Social procurement requirements in Canadian construction lending",

    # ---- Technology and Innovation ----
    "Artificial intelligence applications in Canadian commercial mortgage underwriting",
    "Machine learning for Canadian CRE risk assessment",
    "Automated valuation models in Canadian commercial real estate",
    "Blockchain applications in Canadian commercial real estate lending",
    "Digital mortgage platforms in Canadian commercial real estate",
    "Data analytics in Canadian commercial real estate portfolio management",
    "Proptech investment impact on Canadian commercial real estate",
    "Smart building technology impact on Canadian commercial lending",
    "Internet of things applications in Canadian commercial real estate",
    "Virtual reality in Canadian commercial real estate due diligence",
    "Drone technology in Canadian commercial real estate inspection",
    "Satellite data applications in Canadian commercial real estate analysis",
    "Big data analytics in Canadian commercial real estate market analysis",
    "Natural language processing in Canadian CRE document analysis",
    "Computer vision applications in Canadian commercial real estate",
    "Digital twin technology in Canadian commercial real estate",
    "Cybersecurity risk in Canadian commercial real estate lending",
    "Cloud computing adoption in Canadian commercial real estate operations",
    "API integration in Canadian commercial mortgage origination",
    "Fintech disruption in Canadian commercial real estate lending",

    # ---- Workout and Special Situations ----
    "Loan modification strategies for Canadian commercial mortgages",
    "Forbearance agreement structure in Canadian commercial lending",
    "Deed in lieu of foreclosure in Canadian commercial real estate",
    "Discounted payoff negotiation in Canadian commercial lending",
    "Receiver appointment process in Canadian commercial real estate",
    "CCAA protection and commercial real estate lending in Canada",
    "BIA bankruptcy and commercial real estate lending in Canada",
    "Sale of distressed commercial real estate assets in Canada",
    "Note sale and loan portfolio sale in Canadian commercial lending",
    "Real estate owned management in Canadian commercial lending",
    "Environmental liability in Canadian commercial mortgage workouts",
    "Guarantor enforcement in Canadian commercial mortgage defaults",
    "Cross default provisions in Canadian commercial lending",
    "Material adverse change clauses in Canadian commercial lending",
    "Cash trap provisions in Canadian commercial mortgage structures",
    "Cash management in distressed Canadian commercial properties",
    "Leasing strategy for vacant Canadian commercial properties",
    "Capital improvement strategy for distressed Canadian CRE assets",
    "Repositioning strategy for underperforming Canadian CRE assets",
    "Exit strategy options for distressed Canadian commercial lenders",

    # ---- International Comparisons ----
    "Canadian vs US commercial real estate lending standards comparison",
    "Canadian vs UK commercial real estate lending comparison",
    "Canadian vs Australian commercial real estate lending comparison",
    "Foreign lender entry into Canadian commercial real estate market",
    "US lender appetite for Canadian commercial real estate",
    "European institutional capital in Canadian commercial real estate",
    "Asian capital flows into Canadian commercial real estate",
    "Middle Eastern sovereign wealth fund investment in Canadian CRE",
    "Currency hedging costs for foreign investors in Canadian CRE",
    "Tax treaty implications for foreign investors in Canadian CRE",
    "FIRPTA equivalent rules for Canadian commercial real estate",
    "Foreign buyer restrictions in Canadian commercial real estate",
    "Repatriation of capital from Canadian commercial real estate investments",
    "Cross border financing structures for Canadian commercial real estate",
    "International accounting standards impact on Canadian CRE lending",
    "Basel III capital requirements impact on Canadian commercial lending",
    "IFRS 16 lease accounting impact on Canadian CRE tenant analysis",
    "Transfer pricing considerations in Canadian CRE corporate structures",
    "Thin capitalization rules and Canadian CRE corporate lending",
    "Withholding tax on Canadian commercial real estate income",
]

if __name__ == "__main__":
    if not topics:
        print("No topics defined - add topics to the topics list")
    else:
        run(topics)