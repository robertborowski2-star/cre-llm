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
    # ---- Underwriting Concepts ----
    "DSCR (Debt Service Coverage Ratio) in Canadian commercial real estate lending",
    "LTV (Loan to Value) ratio standards and thresholds in Canadian commercial lending",
    "Debt yield as an underwriting metric in Canadian CRE",
    "Cap rate fundamentals and application in Canadian commercial real estate",
    "NOI (Net Operating Income) calculation and underwriting adjustments in Canadian CRE",
    "Average market rents analysis for Canadian multifamily underwriting",
    "Sales per unit valuation methodology for Canadian multifamily properties",
    "Sales per square foot valuation methodology for Canadian commercial assets",
    "Vacancy and credit loss assumptions in Canadian CRE underwriting",
    "Operating expense ratios and benchmarks for Canadian commercial properties",
    "Effective gross income calculation in Canadian CRE underwriting",
    "Capital expenditure reserves and replacement costs in Canadian CRE",
    "Interest rate sensitivity analysis in Canadian commercial mortgage underwriting",
    "Stressed underwriting vs. in-place underwriting in Canadian CRE lending",
    "Amortization periods and their impact on Canadian commercial mortgage structuring",

    # ---- Canadian Lending Specific ----
    "CMHC MLI Select program requirements and application in Canadian multifamily lending",
    "CMHC standard insured mortgage program for Canadian apartment buildings",
    "OSFI B-20 guidelines and their impact on Canadian commercial real estate lending",
    "OSFI B-21 guidelines for residential mortgage insurance in Canada",
    "Government of Canada bond yield spreads and commercial mortgage pricing conventions",
    "RBC commercial real estate lending appetite and underwriting standards",
    "TD Bank commercial real estate lending criteria and risk appetite",
    "BMO commercial real estate lending programs and underwriting approach",
    "CIBC commercial real estate lending standards and target asset classes",
    "Scotiabank commercial real estate lending appetite and underwriting criteria",
    "National Bank commercial real estate lending focus and Quebec market expertise",
    "Equitable Bank commercial real estate lending programs and alternative lending appetite",
    "Laurentian Bank commercial real estate lending criteria and market focus",
    "Home Trust commercial real estate lending programs and underwriting standards",
    "Credit union commercial real estate lending in Canada - appetite and criteria",
    "Life company commercial real estate lending in Canada - lowest risk appetite",
    "Private debt funds in Canadian commercial real estate - highest risk appetite",
    "Bridge lenders in Canadian commercial real estate market",
    "Mortgage investment corporations (MICs) in Canadian real estate lending",
    "Canadian commercial mortgage broker role and lender relationships",
    "CMHC Canada Mortgage Fund (CMF) program structure and eligibility",
    "NHA Mortgage Backed Securities (MBS) in Canadian commercial real estate",
    "Syndicated mortgage lending in Canadian commercial real estate",

    # ---- Asset Classes ----
    "Multifamily apartment building underwriting in Canada",
    "Industrial real estate underwriting and lending in Canada",
    "Office building underwriting and lending in Canadian markets",
    "Retail real estate underwriting and lending in Canada",
    "Self storage facility underwriting and lending in Canada",
    "Senior residence and long term care facility lending in Canada",
    "Student residence underwriting and lending in Canadian university markets",
    "Hotel and hospitality real estate underwriting in Canada",
    "Development land underwriting and construction lending in Canada",
    "Mixed use commercial real estate underwriting in Canada",
    "Value-add multifamily underwriting and renovation lending in Canada",
    "New construction multifamily underwriting and CMHC construction financing",
    "Retail strip mall vs enclosed mall underwriting differences in Canada",
    "Net lease vs gross lease commercial real estate underwriting in Canada",
    "Single tenant vs multi tenant commercial property risk assessment in Canada",

    # ---- Markets ----
    "Toronto commercial real estate market fundamentals and lending considerations",
    "Vancouver commercial real estate market and lending landscape",
    "Calgary commercial real estate market conditions and lending appetite",
    "Montreal commercial real estate market and Quebec lending nuances",
    "Ottawa commercial real estate market fundamentals and lending",
    "Edmonton commercial real estate market and lending considerations",
    "Waterloo Region commercial real estate market and lending",
    "London Ontario commercial real estate market and lending",
    "Halifax commercial real estate market and Atlantic Canada lending",
    "Winnipeg commercial real estate market and lending considerations",
    "Saskatchewan commercial real estate markets - Regina and Saskatoon",
    "Canadian secondary markets commercial real estate lending considerations",
    "National Canadian commercial real estate market overview and trends 2024-2025",
    "Canadian multifamily rental market fundamentals and investment trends",
    "Canadian industrial real estate market fundamentals and cap rate trends",
    "Canadian office market challenges and lender risk appetite post-COVID",
    "Canadian retail real estate evolution and current lending standards",
    "Canadian commercial real estate cap rate trends by asset class 2020-2025",
    "Impact of Bank of Canada interest rate policy on Canadian commercial real estate",
    "Foreign investment in Canadian commercial real estate and lending implications",

    # ---- Deal Structure ----
    "First mortgage commercial lending structure in Canada",
    "Second mortgage and mezzanine financing in Canadian commercial real estate",
    "A-note and B-note commercial mortgage structure in Canada",
    "Construction loan structuring and holdback mechanisms in Canada",
    "Bridge loan structuring in Canadian commercial real estate",
    "Interest only periods in Canadian commercial mortgage structuring",
    "Loan syndication in Canadian commercial real estate",
    "Participating mortgage structures in Canadian commercial real estate",
    "Vendor take back mortgage structures in Canadian commercial real estate",
    "Preferred equity vs mezzanine debt in Canadian CRE capital stacks",
    "Recourse vs non-recourse commercial lending in Canada",
    "Open vs closed prepayment structures in Canadian commercial mortgages",
    "Yield maintenance prepayment penalties in Canadian commercial lending",
    "Canadian commercial mortgage term lengths and renewal risk",
    "Floating rate vs fixed rate commercial mortgage structuring in Canada",

    # ---- Risk Assessment ----
    "Credit analysis for commercial real estate borrowers in Canada",
    "Guarantor assessment and net worth analysis in Canadian CRE lending",
    "Market risk assessment in Canadian commercial real estate underwriting",
    "Property financial analysis and cash flow underwriting in Canadian CRE",
    "Appraisal value vs lender underwritten value in Canadian commercial lending",
    "Environmental risk assessment in Canadian commercial real estate lending",
    "Concentration risk in Canadian commercial real estate lending portfolios",
    "Sponsor track record assessment in Canadian commercial real estate lending",
    "Lease analysis and tenant credit assessment in Canadian commercial lending",
    "Rollover risk assessment in Canadian commercial real estate lending",
    "Construction risk assessment in Canadian commercial real estate lending",
    "Interest rate risk in Canadian commercial real estate lending",
    "Liquidity risk in Canadian commercial real estate lending",
    "Zoning and entitlement risk in Canadian commercial real estate lending",
    "Building condition assessment and deferred maintenance risk in Canadian CRE",
    "Rent roll analysis and in-place vs market rent assessment in Canadian CRE",
    "Stabilized vs unstabilized property underwriting in Canadian CRE lending",
    "Exit strategy analysis in Canadian commercial real estate lending",
    "Portfolio lending vs single asset lending risk in Canadian CRE",
    "Fraud risk mitigation in Canadian commercial real estate lending",

    # ---- Legal and Process ----
    "Commercial mortgage commitment letter structure and key terms in Canada",
    "Conditions precedent in Canadian commercial mortgage commitments",
    "Commercial mortgage closing process and documentation in Canada",
    "Discussion paper and preliminary credit assessment in Canadian CRE lending",
    "Term sheet structure and negotiation in Canadian commercial real estate",
    "Letter of intent in Canadian commercial real estate transactions",
    "Legal due diligence in Canadian commercial mortgage lending",
    "Title insurance in Canadian commercial real estate lending",
    "Commercial mortgage security and charge registration in Canada",
    "Assignment of rents and leases in Canadian commercial mortgage security",
    "Environmental indemnity in Canadian commercial real estate lending",
    "Personal guarantee structure and enforcement in Canadian CRE lending",
    "Corporate borrower structure and lending implications in Canada",
    "Canadian commercial mortgage default and enforcement process",
    "Power of sale vs foreclosure in Canadian commercial real estate",
    # ---- City x Asset Class Combinations ----
    "Toronto multifamily apartment cap rates and underwriting 2024-2025",
    "Toronto industrial real estate cap rates and lending standards 2024-2025",
    "Toronto office real estate lending and risk assessment 2024-2025",
    "Toronto retail real estate underwriting and lending 2024-2025",
    "Toronto mixed use development underwriting and lending",
    "Toronto construction lending and development financing 2024-2025",
    "Vancouver multifamily cap rates and underwriting 2024-2025",
    "Vancouver industrial real estate lending and underwriting 2024-2025",
    "Vancouver office real estate lending and risk assessment 2024-2025",
    "Vancouver retail real estate underwriting and lending 2024-2025",
    "Vancouver mixed use development underwriting and lending",
    "Vancouver construction lending and development financing 2024-2025",
    "Calgary multifamily cap rates and underwriting 2024-2025",
    "Calgary industrial real estate lending and underwriting 2024-2025",
    "Calgary office real estate lending and risk assessment 2024-2025",
    "Calgary retail real estate underwriting and lending 2024-2025",
    "Calgary mixed use development underwriting and lending",
    "Calgary construction lending and development financing 2024-2025",
    "Montreal multifamily cap rates and underwriting 2024-2025",
    "Montreal industrial real estate lending and underwriting 2024-2025",
    "Montreal office real estate lending and risk assessment 2024-2025",
    "Montreal retail real estate underwriting and lending 2024-2025",
    "Montreal mixed use development underwriting and lending",
    "Montreal construction lending and development financing 2024-2025",
    "Ottawa multifamily cap rates and underwriting 2024-2025",
    "Ottawa industrial real estate lending and underwriting 2024-2025",
    "Ottawa office real estate lending and risk assessment 2024-2025",
    "Ottawa retail real estate underwriting and lending 2024-2025",
    "Edmonton multifamily cap rates and underwriting 2024-2025",
    "Edmonton industrial real estate lending and underwriting 2024-2025",
    "Edmonton office real estate lending and risk assessment 2024-2025",
    "Edmonton retail real estate underwriting and lending 2024-2025",
    "Waterloo Region multifamily cap rates and underwriting 2024-2025",
    "Waterloo Region industrial real estate lending and underwriting 2024-2025",
    "London Ontario multifamily cap rates and underwriting 2024-2025",
    "London Ontario industrial real estate lending and underwriting 2024-2025",
    "Halifax multifamily cap rates and underwriting 2024-2025",
    "Halifax industrial real estate lending and underwriting 2024-2025",
    "Winnipeg multifamily cap rates and underwriting 2024-2025",
    "Winnipeg industrial real estate lending and underwriting 2024-2025",
    "Victoria BC commercial real estate lending and underwriting 2024-2025",
    "Kelowna BC commercial real estate lending and underwriting 2024-2025",
    "Hamilton Ontario multifamily lending and underwriting 2024-2025",
    "Mississauga commercial real estate lending and underwriting 2024-2025",
    "Brampton commercial real estate lending and underwriting 2024-2025",
    "Quebec City commercial real estate lending and underwriting 2024-2025",
    "Niagara Region commercial real estate lending and underwriting 2024-2025",
    "Kitchener Ontario commercial real estate lending and underwriting 2024-2025",
    "Moncton New Brunswick commercial real estate lending and underwriting",
    "Regina Saskatchewan commercial real estate lending and underwriting",

    # ---- Lender x Asset Class Combinations ----
    "CMHC insured multifamily lending underwriting criteria and process",
    "CMHC insured seniors housing lending criteria and underwriting",
    "CMHC insured student housing lending criteria and underwriting",
    "RBC multifamily lending appetite and underwriting standards",
    "RBC industrial real estate lending appetite and underwriting standards",
    "RBC office real estate lending appetite and underwriting standards",
    "RBC retail real estate lending appetite and underwriting standards",
    "RBC construction lending appetite and underwriting standards",
    "TD Bank multifamily lending appetite and underwriting standards",
    "TD Bank industrial real estate lending appetite and underwriting standards",
    "TD Bank office real estate lending appetite and underwriting standards",
    "TD Bank retail real estate lending appetite and underwriting standards",
    "TD Bank construction lending appetite and underwriting standards",
    "BMO multifamily lending appetite and underwriting standards",
    "BMO industrial real estate lending appetite and underwriting standards",
    "BMO office real estate lending appetite and underwriting standards",
    "BMO construction lending appetite and underwriting standards",
    "Scotiabank multifamily lending appetite and underwriting standards",
    "Scotiabank industrial real estate lending appetite and underwriting standards",
    "Scotiabank construction lending appetite and underwriting standards",
    "National Bank multifamily lending appetite and underwriting in Quebec",
    "National Bank industrial real estate lending appetite and underwriting",
    "Equitable Bank multifamily lending appetite and underwriting standards",
    "Equitable Bank commercial real estate bridge lending criteria",
    "Equitable Bank construction lending appetite and underwriting standards",
    "First National Financial multifamily lending criteria and underwriting",
    "First National Financial CMHC insured lending programs and process",
    "Romspen Mortgage Fund commercial real estate lending criteria",
    "KingSett Capital commercial real estate lending and investment criteria",
    "Trez Capital commercial real estate lending criteria and underwriting",
    "Peakhill Capital commercial real estate lending criteria and underwriting",
    "Atrium Mortgage Investment Corporation lending criteria and underwriting",
    "Firm Capital Mortgage Investment Corporation lending criteria",
    "Life company commercial real estate lending - Sun Life Financial criteria",
    "Life company commercial real estate lending - Manulife criteria",
    "Life company commercial real estate lending - Great West Life criteria",
    "Life company commercial real estate lending - Canada Life criteria",
    "Meridian Credit Union commercial real estate lending criteria Ontario",
    "Coast Capital commercial real estate lending criteria BC",
    "Libro Credit Union commercial real estate lending criteria Ontario",
    "Vancity commercial real estate lending criteria BC",
    "ATB Financial commercial real estate lending criteria Alberta",

    # ---- Underwriting Scenarios ----
    "Underwriting a value-add multifamily acquisition in Toronto with 20% vacancy",
    "Underwriting a stabilized industrial property in Calgary with single tenant",
    "Underwriting a retail strip mall in suburban Ontario with anchor tenant",
    "Underwriting a mixed use property in Vancouver with ground floor retail",
    "Underwriting a construction loan for a 100 unit multifamily in Ottawa",
    "Underwriting a bridge loan on an office building undergoing repositioning",
    "Underwriting a seniors housing facility in a secondary Canadian market",
    "Underwriting a student residence near a Canadian university",
    "Underwriting a hotel property in a Canadian resort market",
    "Underwriting a self storage facility in a growing Canadian suburb",
    "Underwriting a development land parcel in the Greater Toronto Area",
    "Underwriting a multifamily portfolio refinance across multiple Canadian cities",
    "Underwriting a commercial mortgage renewal with increased vacancy",
    "Underwriting a distressed commercial property acquisition in Canada",
    "Underwriting a net lease industrial property with investment grade tenant",
    "Underwriting a commercial property with environmental contamination history",
    "Underwriting a commercial mortgage with a highly leveraged sponsor",
    "Underwriting a commercial property with below market in-place rents",
    "Underwriting a new construction condo development in Vancouver",
    "Underwriting a purpose built rental apartment in Toronto",
    "Underwriting a commercial property purchase with vendor take back mortgage",
    "Underwriting a mezzanine loan on a Canadian commercial property",
    "Underwriting an A-note B-note structure on a large Canadian commercial deal",
    "Underwriting a syndicated commercial mortgage in Canada",
    "Underwriting a CMHC MLI Select application for a Toronto apartment building",

    # ---- Financial Analysis Deep Dives ----
    "Detailed NOI reconciliation and underwriting adjustments for Canadian multifamily",
    "Rent roll analysis techniques for Canadian apartment buildings",
    "Expense ratio benchmarking for Canadian office buildings",
    "Capital expenditure planning and reserve analysis for Canadian commercial properties",
    "Cash on cash return analysis for Canadian commercial real estate investments",
    "IRR and NPV analysis for Canadian commercial real estate investments",
    "Sensitivity analysis techniques for Canadian commercial mortgage underwriting",
    "Break even occupancy analysis for Canadian commercial properties",
    "Debt yield floor analysis in Canadian commercial lending",
    "Going in vs stabilized cap rate analysis in Canadian CRE underwriting",
    "Mark to market rent analysis for Canadian commercial properties",
    "Lease up projections and absorption assumptions in Canadian CRE lending",
    "Pro forma underwriting vs trailing 12 month analysis in Canadian CRE",
    "Operating expense escalation assumptions in Canadian CRE underwriting",
    "Management fee benchmarks and underwriting conventions in Canadian CRE",
    "Insurance cost benchmarks for Canadian commercial properties",
    "Property tax assessment and underwriting in Canadian provinces",
    "Utility cost benchmarking for Canadian commercial properties",
    "Maintenance and repair cost benchmarks for Canadian commercial properties",
    "Ground lease underwriting considerations in Canadian commercial real estate",

    # ---- Risk and Credit Deep Dives ----
    "Personal net worth statement analysis for Canadian CRE guarantors",
    "Corporate financial statement analysis for Canadian CRE borrowers",
    "Real estate portfolio analysis for Canadian CRE sponsors",
    "Liquidity analysis for Canadian commercial real estate guarantors",
    "Cash flow coverage analysis for Canadian CRE portfolio borrowers",
    "Cross collateralization structures in Canadian commercial lending",
    "Related party transaction risks in Canadian commercial real estate lending",
    "Fraud indicators and red flags in Canadian commercial mortgage applications",
    "Anti money laundering requirements in Canadian commercial mortgage lending",
    "KYC requirements for Canadian commercial real estate lenders",
    "Privacy and data protection in Canadian commercial mortgage lending",
    "Regulatory compliance requirements for Canadian commercial lenders",
    "Stress testing commercial mortgage portfolios in Canada",
    "Loan loss provisioning for Canadian commercial real estate portfolios",
    "Problem loan management in Canadian commercial real estate lending",
    "Loan modification and workout strategies in Canadian commercial lending",
    "Receiver and insolvency proceedings in Canadian commercial real estate",
    "IFRS accounting standards impact on Canadian commercial real estate lending",

    # ---- Market Analysis ----
    "Canadian commercial real estate investment volume trends 2020-2025",
    "Cross border capital flows into Canadian commercial real estate",
    "Institutional vs private investor activity in Canadian commercial real estate",
    "REIT vs private equity competition in Canadian commercial real estate",
    "Cap rate compression and expansion cycles in Canadian commercial real estate",
    "Canadian commercial real estate transaction volume by asset class 2024",
    "Impact of inflation on Canadian commercial real estate values and lending",
    "Canadian commercial real estate market liquidity and transaction volume",
    "Supply and demand dynamics in Canadian multifamily markets 2024-2025",
    "Supply and demand dynamics in Canadian industrial markets 2024-2025",
    "Canadian office market absorption and vacancy trends 2024-2025",
    "Canadian retail market evolution and omnichannel impact on lending",
    "Population growth impact on Canadian commercial real estate markets",
    "Immigration impact on Canadian multifamily rental demand and lending",
    "Interest rate cycle impact on Canadian commercial real estate valuations",
    "Canadian commercial real estate market outlook 2026",
    "ESG considerations in Canadian commercial real estate lending",
    "Climate risk assessment in Canadian commercial real estate lending",
    "Technology impact on Canadian commercial real estate markets",
    "Co-working and flexible office impact on Canadian office lending",

    # ---- Legal and Regulatory Deep Dives ----
    "Provincial differences in commercial mortgage enforcement across Canada",
    "Ontario commercial mortgage law and enforcement process",
    "BC commercial mortgage law and enforcement process",
    "Alberta commercial mortgage law and enforcement process",
    "Quebec commercial mortgage law and civil code considerations",
    "Standard charge terms in Canadian commercial mortgages",
    "Priority of charges and title searches in Canadian commercial lending",
    "Builders lien act implications for Canadian construction lending",
    "Occupancy permit requirements and construction lending in Canada",
    "Zoning bylaw compliance in Canadian commercial mortgage lending",
    "Heritage property considerations in Canadian commercial lending",
    "Strata and condominium lending considerations in Canadian commercial real estate",
    "Leasehold vs freehold commercial mortgage lending in Canada",
    "Ground lease financing structures in Canadian commercial real estate",
    "Air rights and density bonus considerations in Canadian commercial lending",
    "Easement and right of way considerations in Canadian commercial lending",
    "Survey requirements for Canadian commercial mortgage lending",
    "Phase 1 and Phase 2 environmental assessment requirements in Canada",
    "Record of site condition requirements in Ontario commercial lending",
    "Building inspection and condition assessment requirements in Canadian lending",

    # ---- Process and Operations ----
    "Commercial mortgage application and submission process in Canada",
    "Due diligence checklist for Canadian commercial mortgage underwriting",
    "Appraisal ordering and review process for Canadian commercial lenders",
    "Environmental report ordering and review in Canadian commercial lending",
    "Building condition assessment ordering and review in Canadian lending",
    "Legal instruction and title review process in Canadian commercial lending",
    "Commercial mortgage funding and drawdown process in Canada",
    "Post funding monitoring and reporting for Canadian commercial mortgages",
    "Annual review process for Canadian commercial mortgage portfolios",
    "Commercial mortgage renewal and refinancing process in Canada",
    "Broker submission requirements for Canadian commercial lenders",
    "Credit committee presentation best practices for Canadian CRE lending",
    "Investment committee memo writing for Canadian commercial real estate",
    "Deal screening and preliminary assessment in Canadian commercial lending",
    "Pipeline management for Canadian commercial mortgage lenders",
    "Loan administration and servicing for Canadian commercial mortgages",
    "Covenant compliance monitoring in Canadian commercial lending",
    "Insurance monitoring requirements for Canadian commercial mortgages",
    "Property tax monitoring requirements for Canadian commercial mortgages",
    "Construction draw process and monitoring in Canadian lending",

    # ---- Advanced Topics ----
    "Preferred equity structures in Canadian commercial real estate",
    "Joint venture structures in Canadian commercial real estate lending",
    "Convertible mortgage structures in Canadian commercial real estate",
    "Sale leaseback transactions and lending in Canadian commercial real estate",
    "Canadian commercial real estate securitization and CMBS",
    "Covered bond programs and commercial real estate in Canada",
    "Pension fund investment in Canadian commercial real estate",
    "Private REIT structures and lending in Canadian commercial real estate",
    "Limited partnership structures in Canadian commercial real estate lending",
    "Tax considerations in Canadian commercial real estate lending",
    "GST and HST implications in Canadian commercial real estate transactions",
    "Land transfer tax implications in Canadian commercial real estate",
    "Capital gains tax considerations in Canadian commercial real estate",
    "Corporate structure optimization for Canadian commercial real estate",
    "Estate planning considerations in Canadian commercial real estate lending",
    "Cross border lending considerations for US investors in Canadian CRE",
    "Currency hedging in Canadian commercial real estate lending",
    "Interest rate hedging strategies for Canadian commercial mortgages",
    "Portfolio optimization strategies for Canadian commercial real estate lenders",
    "Technology and proptech impact on Canadian commercial real estate lending",
]

if __name__ == "__main__":
    if not topics:
        print("No topics defined - add topics to the topics list")
    else:
        run(topics)