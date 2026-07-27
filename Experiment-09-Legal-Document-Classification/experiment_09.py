"""
Natural Language Processing Laboratory

Student Name    : Divya M
Register Number : 24AD0074
Experiment      : 09
Title           : Develop a Rule-Based Classifier to Categorize Legal Documents into
                  Different Types and Measure Its Accuracy Against a Maximum Entropy
                  Classifier
"""

import argparse
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
SPLIT_RATIO: float = 0.8
CATEGORY_ORDER: List[str] = ["contract", "judgment", "statute", "legal_notice"]

STUDENT_NAME: str = "Divya M"
REGISTER_NUMBER: str = "24AD0074"

EXPERIMENT_TITLE: str = (
    "Develop a Rule-Based Classifier to Categorize Legal Documents into\n"
    "Different Types and Measure Its Accuracy Against a Maximum Entropy\n"
    "Classifier"
)

AIM: str = (
    "To classify legal documents using a Rule-Based Classifier and compare\n"
    "its performance with a Maximum Entropy (MaxEnt) classifier."
)

DEFAULT_CUSTOM_TEXT: str = (
    "The parties agree that payment shall be completed within thirty days, "
    "and either party may terminate this agreement after a material breach."
)

# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------


def _ensure_nltk_resources() -> None:
    """Download only missing NLTK resources with readable error handling."""
    import nltk

    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except Exception as exc:
                print(f"Warning: could not download NLTK resource '{name}': {exc}")


# ---------------------------------------------------------------------------
# Embedded dataset  (80 fictional legal-document excerpts)
# ---------------------------------------------------------------------------


def build_dataset() -> List[Dict[str, str]]:
    """Return 80 short fictional legal-document excerpts (20 per category)."""

    data: List[Dict[str, str]] = [
        # ---- CONTRACT (20) ----
        {
            "id": "L01", "title": "Software License Agreement",
            "text": "This agreement is entered into between Apex Technologies and Brightway Inc. "
                    "The parties agree that Apex shall deliver the licensed software within fourteen days "
                    "of execution and Brightway shall remit payment of twelve thousand dollars upon delivery.",
            "category": "contract",
        },
        {
            "id": "L02", "title": "Service Level Agreement",
            "text": "GlobalServe Ltd hereby undertakes to provide managed hosting services to Orion Corp. "
                    "In consideration of monthly fees, GlobalServe guarantees 99.5 percent uptime and shall "
                    "indemnify Orion against losses arising from service interruptions.",
            "category": "contract",
        },
        {
            "id": "L03", "title": "Non-Disclosure Agreement",
            "text": "Both parties acknowledge that confidential information exchanged under this agreement "
                    "shall remain proprietary. Neither party may disclose trade secrets or business strategies "
                    "to any third party without prior written consent.",
            "category": "contract",
        },
        {
            "id": "L04", "title": "Employment Contract",
            "text": "Ms. Ananya Rao is hereby appointed as Senior Analyst at Vertex Analytics. "
                    "Her obligations include data reporting and client advisory. Compensation shall be "
                    "paid on the last working day of each month.",
            "category": "contract",
        },
        {
            "id": "L05", "title": "Supply Agreement",
            "text": "PrimeMaterials agrees to deliver five hundred units of raw silicone to NovaChem "
                    "by the fifteenth of each quarter. Failure to deliver shall constitute a material breach "
                    "and entitle NovaChem to terminate this agreement forthwith.",
            "category": "contract",
        },
        {
            "id": "L06", "title": "Joint Venture Agreement",
            "text": "StellarBuild and Pinnacle Realty enter this joint venture for the construction of "
                    "a commercial complex. Each party shall contribute equal capital and share profits "
                    "in proportion to their investment, subject to the terms herein.",
            "category": "contract",
        },
        {
            "id": "L07", "title": "Franchise Agreement",
            "text": "CafeBrew International grants Mr. Rohan Desai the right to operate a franchise "
                    "unit in Bengaluru. The franchisee shall adhere to brand standards and pay a royalty "
                    "of six percent of gross revenue monthly.",
            "category": "contract",
        },
        {
            "id": "L08", "title": "Lease Agreement",
            "text": "The lessor, Greenfield Estates, leases premises at Block 14 to TechNova for a period "
                    "of three years. The lessee shall pay rent of fifty thousand per month and maintain "
                    "the property in good condition throughout the term.",
            "category": "contract",
        },
        {
            "id": "L09", "title": "Consultancy Agreement",
            "text": "Meridian Consulting is retained by Aurum Financial to provide risk assessment services. "
                    "Payment of the consultancy fee shall be made within thirty days of invoice submission. "
                    "Either party may terminate upon sixty days written notice.",
            "category": "contract",
        },
        {
            "id": "L10", "title": "Distribution Agreement",
            "text": "WideReach Distributors is appointed as the exclusive distributor of Zenith Pharma products "
                    "in the southern region. The distributor shall maintain minimum stock levels and "
                    "indemnify Zenith against claims arising from improper storage.",
            "category": "contract",
        },
        {
            "id": "L11", "title": "Partnership Deed",
            "text": "Mr. Karthik and Ms. Priya agree to carry on business in partnership under the name "
                    "KP Traders. Profits and losses shall be shared equally. Neither partner shall incur "
                    "obligations exceeding two lakh rupees without the consent of the other.",
            "category": "contract",
        },
        {
            "id": "L12", "title": "Maintenance Contract",
            "text": "QuickFix Services agrees to perform quarterly maintenance of elevator systems at "
                    "Skyline Towers. The consideration for this contract is an annual fee payable in "
                    "advance. Breach of maintenance schedules entitles termination.",
            "category": "contract",
        },
        {
            "id": "L13", "title": "Construction Contract",
            "text": "IronClad Builders shall construct a warehouse for AgriStore within eighteen months. "
                    "Payment milestones are tied to completion stages. Liquidated damages of one percent "
                    "per week apply for delays attributable to the contractor.",
            "category": "contract",
        },
        {
            "id": "L14", "title": "Sponsorship Agreement",
            "text": "BrightStar Media sponsors the annual tech summit organised by InnoHub Foundation. "
                    "In consideration of brand visibility, BrightStar shall pay the sponsorship fee before "
                    "the event date. Both parties agree to the deliverables attached hereto.",
            "category": "contract",
        },
        {
            "id": "L15", "title": "Indemnity Agreement",
            "text": "The indemnitor, SafeHand Insurance, agrees to indemnify and hold harmless TransLink "
                    "Logistics from all claims, damages and liabilities arising from cargo transit. "
                    "This obligation survives the termination of the underlying transport agreement.",
            "category": "contract",
        },
        {
            "id": "L16", "title": "Confidentiality Agreement",
            "text": "All proprietary data shared between Quantum Labs and DataShield Inc during the "
                    "pilot programme shall be treated as confidential. Unauthorised disclosure constitutes "
                    "a breach entitling the aggrieved party to seek injunctive relief.",
            "category": "contract",
        },
        {
            "id": "L17", "title": "Outsourcing Agreement",
            "text": "CloudNine BPO shall handle customer support operations for SwiftTel Communications. "
                    "Service obligations include twenty-four-hour availability. Payment terms require "
                    "settlement within fifteen days of monthly invoice receipt.",
            "category": "contract",
        },
        {
            "id": "L18", "title": "Subscription Agreement",
            "text": "The subscriber, Mr. Arvind Shah, agrees to purchase two thousand equity shares of "
                    "GreenTech Ventures at a price of one hundred rupees per share. Payment shall be "
                    "remitted by bank transfer prior to allotment of shares.",
            "category": "contract",
        },
        {
            "id": "L19", "title": "Barter Agreement",
            "text": "TradeX Corp and MediaPulse agree to exchange advertising space for software licenses "
                    "of equivalent value. Each party warrants that it has the authority to enter into this "
                    "agreement and fulfill its respective obligations.",
            "category": "contract",
        },
        {
            "id": "L20", "title": "Termination Agreement",
            "text": "By mutual consent, NexGen Solutions and Mr. Vikram Iyer agree to terminate the "
                    "employment contract dated January 2024. All dues including severance and accrued "
                    "leave shall be settled within fourteen days of the effective date.",
            "category": "contract",
        },

        # ---- JUDGMENT (20) ----
        {
            "id": "L21", "title": "Civil Suit Judgment",
            "text": "The court finds that the respondent Mahesh Traders failed to deliver goods as per the "
                    "purchase order. After examining documentary evidence, the judge awards damages of "
                    "three lakh rupees to the petitioner Sunrise Exports.",
            "category": "judgment",
        },
        {
            "id": "L22", "title": "Criminal Appeal Order",
            "text": "The appeal filed by the accused Mr. Sunil Verma against conviction under Section 420 "
                    "is hereby dismissed. The High Court upholds the trial court's finding that the evidence "
                    "of forgery was conclusive and the sentence of two years stands.",
            "category": "judgment",
        },
        {
            "id": "L23", "title": "Writ Petition Ruling",
            "text": "The petitioner EcoGreen Foundation challenged the government notification permitting "
                    "mining in the Western Hills. The court held that the environmental clearance was "
                    "granted without proper assessment and quashed the notification.",
            "category": "judgment",
        },
        {
            "id": "L24", "title": "Family Court Decree",
            "text": "After hearing both parties, the family court grants a decree of divorce to Ms. Rekha "
                    "Nair. The respondent is ordered to pay monthly maintenance of fifteen thousand rupees "
                    "and the custody of the minor child is awarded to the petitioner.",
            "category": "judgment",
        },
        {
            "id": "L25", "title": "Consumer Dispute Judgment",
            "text": "The District Consumer Forum finds that AutoDrive Motors sold a defective vehicle to "
                    "the complainant. The forum ordered replacement of the vehicle and awarded compensation "
                    "of fifty thousand rupees for mental agony and harassment.",
            "category": "judgment",
        },
        {
            "id": "L26", "title": "Labour Tribunal Award",
            "text": "The tribunal held that the termination of Mr. Rajeev Kumar from Apex Steel was "
                    "wrongful and without due process. The respondent employer is ordered to reinstate "
                    "the petitioner with full back wages and continuity of service.",
            "category": "judgment",
        },
        {
            "id": "L27", "title": "Tax Appeal Decision",
            "text": "The Income Tax Appellate Tribunal reviewed the evidence submitted by the assessee "
                    "PrimeVentures LLP. The judge held that the disallowance of business expenditure "
                    "was unjustified and directed the respondent department to refund excess tax collected.",
            "category": "judgment",
        },
        {
            "id": "L28", "title": "Intellectual Property Ruling",
            "text": "The court examined the trademark infringement claim by BrightMark Corp against "
                    "CopyCraft Industries. Based on evidence of consumer confusion, the judge ordered "
                    "CopyCraft to cease using the contested logo and awarded damages.",
            "category": "judgment",
        },
        {
            "id": "L29", "title": "Motor Accident Claim",
            "text": "The Motor Accident Claims Tribunal finds the respondent driver negligent. "
                    "After considering medical evidence, the tribunal awarded compensation of five lakh "
                    "rupees to the petitioner for injuries sustained in the collision.",
            "category": "judgment",
        },
        {
            "id": "L30", "title": "Land Acquisition Judgment",
            "text": "The High Court held that the land acquisition by the State for a highway project "
                    "was valid but the compensation offered to the respondent landowners was inadequate. "
                    "The court ordered enhanced compensation based on prevailing market rates.",
            "category": "judgment",
        },
        {
            "id": "L31", "title": "Bail Order",
            "text": "After hearing submissions from both the prosecution and the accused, the court "
                    "grants bail to Mr. Faisal Ahmed on furnishing a surety bond. The judge observed "
                    "that the evidence at this stage does not necessitate continued detention.",
            "category": "judgment",
        },
        {
            "id": "L32", "title": "Arbitration Award",
            "text": "The arbitral tribunal examined the dispute between Delta Infra and Metro Rail Corp. "
                    "The tribunal held that the respondent breached the construction timeline and "
                    "ordered payment of damages amounting to twelve crore rupees to the petitioner.",
            "category": "judgment",
        },
        {
            "id": "L33", "title": "Acquittal Order",
            "text": "The sessions court acquits Mr. Deepak Joshi of all charges under Section 302. "
                    "The judge reasoned that the prosecution failed to establish guilt beyond reasonable "
                    "doubt. The evidence presented was circumstantial and unreliable.",
            "category": "judgment",
        },
        {
            "id": "L34", "title": "Contempt Proceedings",
            "text": "The court held the respondent Municipal Commissioner in contempt for failing to "
                    "comply with the earlier order directing removal of encroachments. A fine of one lakh "
                    "rupees is imposed and compliance is ordered within thirty days.",
            "category": "judgment",
        },
        {
            "id": "L35", "title": "Election Petition Verdict",
            "text": "The petitioner alleged irregularities in the ward election. After reviewing ballot "
                    "records and witness testimony, the court dismissed the petition holding that the "
                    "respondent's election was free and fair with no evidence of malpractice.",
            "category": "judgment",
        },
        {
            "id": "L36", "title": "Environmental Tribunal Order",
            "text": "The National Green Tribunal examined the pollution data and held that Chemix "
                    "Industries violated effluent discharge norms. The respondent is ordered to shut "
                    "operations until compliance is achieved, and damages are awarded to affected villagers.",
            "category": "judgment",
        },
        {
            "id": "L37", "title": "Cyber Crime Conviction",
            "text": "The court convicted the accused Mr. Nikhil Rana under the Information Technology Act "
                    "for unauthorized access to financial systems. The judge ordered imprisonment of "
                    "one year and a fine, noting the overwhelming digital evidence.",
            "category": "judgment",
        },
        {
            "id": "L38", "title": "Partition Suit Decree",
            "text": "The civil court decreed the partition of ancestral property among four siblings. "
                    "The judge examined revenue records and held that each petitioner is entitled to an "
                    "equal one-fourth share. The respondent's objections were overruled.",
            "category": "judgment",
        },
        {
            "id": "L39", "title": "Insurance Claim Judgment",
            "text": "The court held that StarLife Insurance wrongfully repudiated the health claim of "
                    "Mrs. Geeta Sharma. The evidence showed that the illness was not a pre-existing "
                    "condition. The respondent is ordered to settle the claim with interest.",
            "category": "judgment",
        },
        {
            "id": "L40", "title": "Appeal Dismissed",
            "text": "The appellate bench reviewed the trial record and found no merit in the appeal "
                    "filed by the respondent Zenith Constructions. The original judgment awarding "
                    "damages to the petitioner is upheld and costs are imposed on the appellant.",
            "category": "judgment",
        },

        # ---- STATUTE (20) ----
        {
            "id": "L41", "title": "Data Protection Act",
            "text": "Section 4 of the Data Protection Act 2024 defines personal data as any information "
                    "relating to an identifiable individual. Processing of such data without consent is "
                    "prohibited and shall attract a penalty not exceeding fifty lakh rupees.",
            "category": "statute",
        },
        {
            "id": "L42", "title": "Consumer Protection Regulation",
            "text": "Under Article 12 of the Consumer Protection Regulations, misleading advertisements "
                    "are prohibited. The regulatory authority shall have the power to impose fines and "
                    "direct withdrawal of offending material from all media.",
            "category": "statute",
        },
        {
            "id": "L43", "title": "Environmental Protection Act",
            "text": "Section 17 empowers the Central Pollution Control Authority to prescribe emission "
                    "standards for industrial units. Any person who contravenes these standards shall "
                    "be liable to imprisonment or a fine or both as defined in the Act.",
            "category": "statute",
        },
        {
            "id": "L44", "title": "Companies Amendment Act",
            "text": "Section 135 of the Companies Act requires every company with a net worth exceeding "
                    "five hundred crore rupees to constitute a Corporate Social Responsibility Committee. "
                    "The definition of eligible activities is provided in Schedule VII.",
            "category": "statute",
        },
        {
            "id": "L45", "title": "Right to Information Act",
            "text": "Section 6 of the Right to Information Act provides that any citizen may request "
                    "information from a public authority. The authority shall respond within thirty days. "
                    "Failure to comply attracts a penalty of two hundred fifty rupees per day.",
            "category": "statute",
        },
        {
            "id": "L46", "title": "Labour Code on Wages",
            "text": "Article 8 of the Code on Wages defines minimum wages for skilled and unskilled "
                    "workers. The appropriate government shall revise wage rates at intervals not "
                    "exceeding five years. Non-compliance is a punishable offence under this Act.",
            "category": "statute",
        },
        {
            "id": "L47", "title": "Cybersecurity Regulation",
            "text": "Section 22 of the Cybersecurity Act prohibits unauthorized access to protected "
                    "computer systems. The regulatory authority shall maintain a registry of critical "
                    "infrastructure. Penalties include imprisonment up to three years.",
            "category": "statute",
        },
        {
            "id": "L48", "title": "Anti-Corruption Act",
            "text": "Under Section 9, a public servant who accepts gratification other than legal "
                    "remuneration shall be guilty of criminal misconduct. The Act defines the "
                    "commencement date as the first day of the financial year following royal assent.",
            "category": "statute",
        },
        {
            "id": "L49", "title": "Goods and Services Tax Act",
            "text": "Section 16 of the GST Act provides the conditions for claiming input tax credit. "
                    "The registered person shall furnish returns as prescribed by regulation. "
                    "Contravention of this section attracts a penalty determined by the authority.",
            "category": "statute",
        },
        {
            "id": "L50", "title": "Motor Vehicles Amendment",
            "text": "Article 3 of the Motor Vehicles Amendment Act increases the penalty for drunk "
                    "driving to ten thousand rupees. The Act shall come into force on a date notified "
                    "by the Central Government. Definitions are provided in Section 2.",
            "category": "statute",
        },
        {
            "id": "L51", "title": "Food Safety Standards Act",
            "text": "Section 31 prohibits the manufacture and sale of adulterated food articles. "
                    "The Food Safety Authority shall license all food business operators. Violation "
                    "of prescribed standards attracts penalties ranging from one to five lakh rupees.",
            "category": "statute",
        },
        {
            "id": "L52", "title": "Insolvency and Bankruptcy Code",
            "text": "Under Section 7 of the Insolvency Code, a financial creditor may file an application "
                    "for initiating corporate insolvency resolution. The adjudicating authority shall "
                    "admit or reject the application within fourteen days of filing.",
            "category": "statute",
        },
        {
            "id": "L53", "title": "Telecommunications Regulation Act",
            "text": "Section 14 grants the Telecom Regulatory Authority power to regulate tariff "
                    "structures. Service providers shall comply with quality of service standards "
                    "as prescribed. The Act defines interconnection obligations in the schedule.",
            "category": "statute",
        },
        {
            "id": "L54", "title": "Real Estate Regulation Act",
            "text": "Article 5 of the Real Estate Act requires every promoter to register a project "
                    "before advertising or selling. The authority shall maintain a public database. "
                    "Penalties for non-registration include imprisonment and monetary fines.",
            "category": "statute",
        },
        {
            "id": "L55", "title": "Banking Regulation Amendment",
            "text": "Section 45 empowers the Reserve Bank to issue directions to banking companies "
                    "regarding lending practices. This section shall apply to cooperative banks from "
                    "the date of commencement of this amendment Act.",
            "category": "statute",
        },
        {
            "id": "L56", "title": "Arbitration and Conciliation Act",
            "text": "Under Section 11, the Supreme Court or High Court shall appoint an arbitrator "
                    "if parties fail to agree. The Act defines the applicability of international "
                    "commercial arbitration and prescribes limitation periods.",
            "category": "statute",
        },
        {
            "id": "L57", "title": "Competition Act",
            "text": "Section 3 of the Competition Act prohibits anti-competitive agreements including "
                    "price fixing and market allocation. The Competition Commission shall investigate "
                    "and penalise enterprises found in violation of this provision.",
            "category": "statute",
        },
        {
            "id": "L58", "title": "Intellectual Property Rights Act",
            "text": "Article 19 defines the duration of patent protection as twenty years from the "
                    "date of filing. The Controller of Patents shall grant or refuse applications "
                    "based on criteria of novelty and non-obviousness as per this Act.",
            "category": "statute",
        },
        {
            "id": "L59", "title": "Education Regulation Act",
            "text": "Section 8 of the Education Act mandates compulsory education for children between "
                    "six and fourteen years. The appropriate government shall establish schools within "
                    "a prescribed distance. Non-compliance by authorities attracts penalties.",
            "category": "statute",
        },
        {
            "id": "L60", "title": "Wildlife Protection Act",
            "text": "Section 51 prescribes penalties for hunting or trapping of animals listed in "
                    "Schedule I. The Chief Wildlife Warden shall have authority to grant permits in "
                    "exceptional circumstances. The Act commenced on the date of its publication.",
            "category": "statute",
        },

        # ---- LEGAL NOTICE (20) ----
        {
            "id": "L61", "title": "Payment Demand Notice",
            "text": "This notice is hereby served upon Mr. Ajay Mehra demanding payment of outstanding "
                    "dues amounting to two lakh rupees within fifteen days. Failure to comply shall "
                    "result in legal proceedings without further intimation.",
            "category": "legal_notice",
        },
        {
            "id": "L62", "title": "Cease and Desist Notice",
            "text": "You are hereby directed to immediately cease and desist from using the trademark "
                    "BrightSpark which is the registered property of LumiTech Corp. Failure to respond "
                    "within seven days shall compel us to initiate infringement proceedings.",
            "category": "legal_notice",
        },
        {
            "id": "L63", "title": "Eviction Notice",
            "text": "Notice is hereby given to the occupant of Flat 302 Riverside Apartments to vacate "
                    "the premises within thirty days. The outstanding rent of six months remains unpaid "
                    "despite repeated demands. Legal action shall follow non-compliance.",
            "category": "legal_notice",
        },
        {
            "id": "L64", "title": "Compliance Notice",
            "text": "The Municipal Health Department hereby notifies FreshBite Restaurant to comply "
                    "with sanitation standards within fourteen days. Inspection revealed violations "
                    "of food handling protocols. Non-compliance may result in licence cancellation.",
            "category": "legal_notice",
        },
        {
            "id": "L65", "title": "Hearing Notice",
            "text": "You are hereby summoned to appear before the District Rent Tribunal on the "
                    "twenty-fifth of August 2025 at ten in the morning. The hearing concerns the "
                    "eviction application filed by the landlord. Bring all relevant documents.",
            "category": "legal_notice",
        },
        {
            "id": "L66", "title": "Regulatory Warning Notice",
            "text": "The Securities Board hereby issues a warning to TradeFast Brokers for non-compliance "
                    "with margin reporting requirements. You must respond with corrective measures within "
                    "ten working days or face suspension of trading licence.",
            "category": "legal_notice",
        },
        {
            "id": "L67", "title": "Insurance Claim Notice",
            "text": "Notice is hereby given to SafeGuard Insurance Company demanding settlement of "
                    "the fire damage claim submitted on March 2025. The outstanding amount of eight "
                    "lakh rupees must be disbursed within twenty-one days of receipt of this notice.",
            "category": "legal_notice",
        },
        {
            "id": "L68", "title": "Defamation Notice",
            "text": "This legal notice is served upon Mr. Suresh Nair demanding immediate removal of "
                    "defamatory statements published on social media. You are hereby required to cease "
                    "all such activity and respond with an apology within seven days.",
            "category": "legal_notice",
        },
        {
            "id": "L69", "title": "Tenant Repair Notice",
            "text": "The landlord hereby notifies the tenant of Unit 5B to undertake repairs to the "
                    "damaged plumbing within ten days. The deadline for compliance is the fifteenth "
                    "of next month. Failure to comply will result in deduction from the security deposit.",
            "category": "legal_notice",
        },
        {
            "id": "L70", "title": "Debt Recovery Notice",
            "text": "Notice is hereby issued to Ms. Kavita Sinha to repay the outstanding loan balance "
                    "of three lakh fifty thousand rupees. You must respond within fifteen days failing "
                    "which recovery proceedings shall be initiated under applicable law.",
            "category": "legal_notice",
        },
        {
            "id": "L71", "title": "Noise Complaint Notice",
            "text": "The Residents Welfare Association hereby notifies the occupant of Villa 12 to cease "
                    "excessive noise after ten pm. Continued violations shall be reported to the local "
                    "authorities. You are required to comply immediately.",
            "category": "legal_notice",
        },
        {
            "id": "L72", "title": "Environmental Compliance Notice",
            "text": "The Pollution Control Board hereby demands that GreenChem Factory comply with "
                    "effluent treatment norms within thirty days. Failure to meet the deadline shall "
                    "result in closure orders. Respond with an action plan by the specified date.",
            "category": "legal_notice",
        },
        {
            "id": "L73", "title": "Trademark Infringement Notice",
            "text": "You are hereby put on notice that your use of the brand name QuickBite infringes "
                    "upon the registered trademark of QuickBite Foods Pvt Ltd. Cease all use immediately "
                    "and respond within fourteen days to avoid litigation.",
            "category": "legal_notice",
        },
        {
            "id": "L74", "title": "Tax Demand Notice",
            "text": "The Income Tax Department hereby demands payment of outstanding tax dues of "
                    "one lakh twenty thousand rupees for the assessment year 2023-24. You must comply "
                    "within thirty days. Failure to respond will attract interest and penalty.",
            "category": "legal_notice",
        },
        {
            "id": "L75", "title": "Construction Stop Notice",
            "text": "Notice is hereby served directing the owner of Plot 48 Sector 9 to cease all "
                    "construction activity immediately. The building plan has not received approval. "
                    "Respond with permit documents within seven days or face demolition proceedings.",
            "category": "legal_notice",
        },
        {
            "id": "L76", "title": "Arbitration Initiation Notice",
            "text": "Pursuant to clause 14 of the supply agreement, this notice hereby invokes "
                    "arbitration proceedings against EliteSteel Corp. The demand relates to outstanding "
                    "payment of four crore rupees. You must respond and nominate an arbitrator within fifteen days.",
            "category": "legal_notice",
        },
        {
            "id": "L77", "title": "Show Cause Notice",
            "text": "The regulatory authority hereby directs FastPharma Ltd to show cause within "
                    "twenty-one days why its manufacturing licence should not be suspended. "
                    "Inspection reports indicate non-compliance with drug safety standards.",
            "category": "legal_notice",
        },
        {
            "id": "L78", "title": "Boundary Dispute Notice",
            "text": "This notice is served upon Mr. Harish Gupta demanding removal of the encroachment "
                    "on the eastern boundary of Plot 22. You are hereby required to vacate the "
                    "encroached portion within fifteen days or face legal action.",
            "category": "legal_notice",
        },
        {
            "id": "L79", "title": "Warranty Claim Notice",
            "text": "Notice is hereby given to HomeAppliance Corp to honour the warranty on the "
                    "refrigerator model HX200 purchased in January 2025. The demand for free repair "
                    "or replacement must be fulfilled within ten days. Respond immediately.",
            "category": "legal_notice",
        },
        {
            "id": "L80", "title": "Salary Arrears Notice",
            "text": "This legal notice demands that QuickServe Hospitality release unpaid salary "
                    "arrears of ninety thousand rupees to Mr. Imran Ali within seven days. Failure "
                    "to comply will result in a complaint before the labour commissioner.",
            "category": "legal_notice",
        },
    ]
    return data


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess(text: str) -> Tuple[List[str], str]:
    """Shared preprocessing: lowercase, tokenize, filter, stopword removal, lemmatize.

    Returns (token_list, joined_string).
    """
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens, " ".join(tokens)


# ---------------------------------------------------------------------------
# Rule-Based Classifier
# ---------------------------------------------------------------------------

RULE_KEYWORDS: Dict[str, Dict[str, float]] = {
    "contract": {
        "agreement": 3.0, "party": 2.5, "consideration": 2.5, "payment": 2.0,
        "deliver": 2.0, "obligation": 3.0, "terminate": 2.5, "breach": 2.5,
        "indemnify": 3.0, "confidential": 2.5, "lessee": 2.0, "lessor": 2.0,
        "franchise": 2.0, "royalty": 2.0, "remit": 2.0,
    },
    "judgment": {
        "court": 2.5, "judge": 3.0, "petitioner": 3.0, "respondent": 2.5,
        "appeal": 2.5, "evidence": 2.0, "held": 2.5, "ordered": 2.5,
        "conviction": 3.0, "damages": 2.5, "acquit": 3.0, "decree": 2.5,
        "tribunal": 2.5, "dismissed": 2.0, "bail": 3.0,
    },
    "statute": {
        "section": 2.5, "article": 2.0, "act": 2.0, "regulation": 2.5,
        "shall": 1.5, "prohibited": 2.5, "authority": 2.0, "penalty": 2.5,
        "commencement": 3.0, "definition": 3.0, "prescribe": 2.5,
        "contravention": 3.0, "applicability": 3.0, "schedule": 2.0,
    },
    "legal_notice": {
        "notice": 3.0, "hereby": 2.5, "demand": 2.5, "comply": 2.5,
        "deadline": 3.0, "cease": 3.0, "respond": 2.0, "hearing": 2.0,
        "outstanding": 2.0, "vacate": 3.0, "served": 2.5, "desist": 3.0,
        "intimation": 2.5, "summoned": 3.0,
    },
}

RULE_PHRASES: Dict[str, Dict[str, float]] = {
    "contract": {
        "entered into": 4.0, "in consideration": 4.0, "material breach": 4.0,
        "written consent": 3.5, "mutual consent": 3.5,
    },
    "judgment": {
        "court finds": 4.0, "court held": 4.0, "is ordered": 4.0,
        "appeal dismissed": 4.5, "awarded damages": 4.5,
    },
    "statute": {
        "shall come into force": 5.0, "as defined in": 4.0,
        "prohibited conduct": 4.5, "regulatory authority": 4.0,
    },
    "legal_notice": {
        "hereby notifies": 4.5, "cease and desist": 5.0,
        "failure to comply": 4.0, "legal proceedings": 3.5,
        "hereby served": 4.0,
    },
}


def rule_based_predict(text: str) -> Tuple[str, Dict[str, float]]:
    """Classify a document using weighted keyword + phrase rules.

    Rule precedence:
      1. Multi-word phrase matches (higher weight) are scored first.
      2. Single keyword matches are added.
      3. The category with the highest total score wins.
      4. Tie-breaking: alphabetical category order (deterministic).
      5. Fallback: 'contract' when no keyword or phrase matches.

    Returns (predicted_category, score_dict).
    """
    lower_text = text.lower()
    scores: Dict[str, float] = {cat: 0.0 for cat in CATEGORY_ORDER}

    # Phrase scoring
    for cat, phrases in RULE_PHRASES.items():
        for phrase, weight in phrases.items():
            if phrase in lower_text:
                scores[cat] += weight

    # Keyword scoring (on lemmatized tokens)
    tokens, _ = preprocess(text)
    for cat, keywords in RULE_KEYWORDS.items():
        for kw, weight in keywords.items():
            for tok in tokens:
                if tok == kw or tok.startswith(kw):
                    scores[cat] += weight

    max_score = max(scores.values())
    if max_score == 0.0:
        return "contract", scores  # fallback

    # Deterministic tie-break: first in CATEGORY_ORDER (already alphabetical-like)
    best_cats = [c for c in CATEGORY_ORDER if scores[c] == max_score]
    return best_cats[0], scores


# ---------------------------------------------------------------------------
# ML helpers
# ---------------------------------------------------------------------------


def _build_tfidf():
    """Return a configured TfidfVectorizer."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_model(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Compute accuracy, macro/weighted precision/recall/F1."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0, labels=CATEGORY_ORDER),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0, labels=CATEGORY_ORDER),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=CATEGORY_ORDER),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0, labels=CATEGORY_ORDER),
    }


def per_category_report(y_true: List[str], y_pred: List[str]) -> str:
    """Return a per-category precision/recall/F1/support table."""
    from sklearn.metrics import classification_report
    return classification_report(y_true, y_pred, labels=CATEGORY_ORDER, zero_division=0)


def text_confusion_matrix(y_true: List[str], y_pred: List[str]) -> str:
    """Return a readable text confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=CATEGORY_ORDER)
    max_label_len = max(len(c) for c in CATEGORY_ORDER)
    header = " " * (max_label_len + 2) + "  ".join(f"{c:>{max_label_len}}" for c in CATEGORY_ORDER)
    lines = [f"{'Predicted ->':>{max_label_len + 2}}", header]
    for i, cat in enumerate(CATEGORY_ORDER):
        row_vals = "  ".join(f"{v:>{max_label_len}}" for v in cm[i])
        lines.append(f"{cat:>{max_label_len}}  {row_vals}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Experiment 09."""
    parser = argparse.ArgumentParser(description="Experiment 09 — Legal Document Classification")
    parser.add_argument("--text", type=str, default=DEFAULT_CUSTOM_TEXT,
                        help="Custom legal document text to classify")
    args = parser.parse_args()

    # NLTK resources
    _ensure_nltk_resources()

    # scikit-learn imports (after ensuring environment)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC

    # ---- Dataset ----
    dataset = build_dataset()
    texts = [d["text"] for d in dataset]
    labels = [d["category"] for d in dataset]
    ids = [d["id"] for d in dataset]
    titles = [d["title"] for d in dataset]

    total_count = len(dataset)
    cat_counts = Counter(labels)

    # ---- Split ----
    np.random.seed(RANDOM_SEED)
    indices = list(range(total_count))
    train_idx, test_idx = train_test_split(
        indices, test_size=1 - SPLIT_RATIO, random_state=RANDOM_SEED, stratify=labels
    )

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_ids = [ids[i] for i in test_idx]
    test_titles = [titles[i] for i in test_idx]

    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)

    # Preprocess for ML classifiers
    train_processed = [preprocess(t)[1] for t in train_texts]
    test_processed = [preprocess(t)[1] for t in test_texts]

    # ---- A. Rule-Based ----
    rb_train_time = 0.0
    t0 = time.time()
    rb_preds = [rule_based_predict(t)[0] for t in test_texts]
    rb_infer_time = time.time() - t0
    rb_metrics = evaluate_model(test_labels, rb_preds)
    rb_metrics["train_time"] = rb_train_time
    rb_metrics["infer_time"] = rb_infer_time

    # ---- B. Naive Bayes ----
    tfidf_nb = _build_tfidf()
    t0 = time.time()
    X_train_nb = tfidf_nb.fit_transform(train_processed)
    nb_model = MultinomialNB()
    nb_model.fit(X_train_nb, train_labels)
    nb_train_time = time.time() - t0
    t0 = time.time()
    X_test_nb = tfidf_nb.transform(test_processed)
    nb_preds = nb_model.predict(X_test_nb).tolist()
    nb_infer_time = time.time() - t0
    nb_metrics = evaluate_model(test_labels, nb_preds)
    nb_metrics["train_time"] = nb_train_time
    nb_metrics["infer_time"] = nb_infer_time

    # ---- C. Linear SVM ----
    tfidf_svm = _build_tfidf()
    t0 = time.time()
    X_train_svm = tfidf_svm.fit_transform(train_processed)
    svm_model = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED)
    svm_model.fit(X_train_svm, train_labels)
    svm_train_time = time.time() - t0
    t0 = time.time()
    X_test_svm = tfidf_svm.transform(test_processed)
    svm_preds = svm_model.predict(X_test_svm).tolist()
    svm_infer_time = time.time() - t0
    svm_metrics = evaluate_model(test_labels, svm_preds)
    svm_metrics["train_time"] = svm_train_time
    svm_metrics["infer_time"] = svm_infer_time

    # ---- D. Maximum Entropy (Logistic Regression) ----
    tfidf_me = _build_tfidf()
    t0 = time.time()
    X_train_me = tfidf_me.fit_transform(train_processed)
    me_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    me_model.fit(X_train_me, train_labels)
    me_train_time = time.time() - t0
    t0 = time.time()
    X_test_me = tfidf_me.transform(test_processed)
    me_preds = me_model.predict(X_test_me).tolist()
    me_infer_time = time.time() - t0
    me_metrics = evaluate_model(test_labels, me_preds)
    me_metrics["train_time"] = me_train_time
    me_metrics["infer_time"] = me_infer_time

    # ---- Select best model ----
    model_names = ["Rule-Based", "Naive Bayes", "Linear SVM", "MaxEnt (LogReg)"]
    all_metrics = [rb_metrics, nb_metrics, svm_metrics, me_metrics]
    all_preds = [rb_preds, nb_preds, svm_preds, me_preds]

    def _sort_key(idx: int) -> Tuple[float, float, float]:
        m = all_metrics[idx]
        return (m["macro_f1"], m["macro_recall"], m["accuracy"])

    best_idx = max(range(4), key=_sort_key)
    best_name = model_names[best_idx]
    best_preds = all_preds[best_idx]
    best_metrics = all_metrics[best_idx]

    # ---- Error analysis ----
    errors = []
    for i, (true, pred) in enumerate(zip(test_labels, best_preds)):
        if true != pred:
            errors.append({
                "id": test_ids[i],
                "title": test_titles[i],
                "excerpt": test_texts[i][:120] + ("..." if len(test_texts[i]) > 120 else ""),
                "true": true,
                "pred": pred,
            })
    errors = errors[:5]

    # ---- Custom document ----
    custom_text = args.text
    custom_tokens, custom_processed = preprocess(custom_text)
    custom_rb_pred, custom_rb_scores = rule_based_predict(custom_text)

    custom_nb_pred = nb_model.predict(tfidf_nb.transform([custom_processed]))[0]
    custom_svm_pred = svm_model.predict(tfidf_svm.transform([custom_processed]))[0]
    custom_me_pred = me_model.predict(tfidf_me.transform([custom_processed]))[0]

    nb_proba = nb_model.predict_proba(tfidf_nb.transform([custom_processed]))[0]
    nb_classes = nb_model.classes_.tolist()
    me_proba = me_model.predict_proba(tfidf_me.transform([custom_processed]))[0]
    me_classes = me_model.classes_.tolist()

    # Determine final prediction from best model
    best_custom_preds = [custom_rb_pred, custom_nb_pred, custom_svm_pred, custom_me_pred]
    final_custom_pred = best_custom_preds[best_idx]

    # ---- Rule-Based vs MaxEnt comparison ----
    rb_m = rb_metrics
    me_m = me_metrics
    acc_diff = abs(rb_m["accuracy"] - me_m["accuracy"])
    f1_diff = abs(rb_m["macro_f1"] - me_m["macro_f1"])
    rb_vs_me_better = "Rule-Based" if _sort_key(0) > _sort_key(3) else "MaxEnt (LogReg)"

    # ==================================================================
    # Generate output/output.txt
    # ==================================================================
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "output.txt"

    lines: List[str] = []

    def w(s: str = "") -> None:
        lines.append(s)

    w("=" * 72)
    w("EXPERIMENT 09 — LEGAL DOCUMENT CLASSIFICATION")
    w("=" * 72)
    w()
    w(f"Student Name    : {STUDENT_NAME}")
    w(f"Register Number : {REGISTER_NUMBER}")
    w()
    w("Title:")
    w(EXPERIMENT_TITLE)
    w()
    w("Aim:")
    w(AIM)
    w()
    w("-" * 72)
    w("NOTE ON ADDITIONAL ALGORITHMS")
    w("-" * 72)
    w("The original experiment requires a Rule-Based Classifier and a")
    w("Maximum Entropy (MaxEnt) Classifier. Two additional classifiers")
    w("— Multinomial Naive Bayes and Linear SVM — are included for a")
    w("broader comparative analysis as permitted by the staff.")
    w("Multinomial Logistic Regression is used as the Maximum Entropy")
    w("classifier because it is mathematically equivalent to MaxEnt.")
    w()
    w("-" * 72)
    w("DATASET SUMMARY")
    w("-" * 72)
    w(f"Total documents        : {total_count}")
    w(f"Training documents     : {len(train_idx)}")
    w(f"Testing documents      : {len(test_idx)}")
    w(f"Random seed            : {RANDOM_SEED}")
    w(f"Split ratio            : {SPLIT_RATIO:.0%} train / {1-SPLIT_RATIO:.0%} test (stratified)")
    w()
    w("Category counts:")
    w(f"  {'Category':<15} {'Total':>6} {'Train':>6} {'Test':>6}")
    for cat in CATEGORY_ORDER:
        w(f"  {cat:<15} {cat_counts[cat]:>6} {train_counts[cat]:>6} {test_counts[cat]:>6}")
    w()

    w("-" * 72)
    w("PREPROCESSING")
    w("-" * 72)
    w("Shared preprocessing applied to all ML classifiers:")
    w("  1. Convert text to lowercase")
    w("  2. Tokenize using nltk.word_tokenize")
    w("  3. Retain only alphabetic tokens")
    w("  4. Remove English stopwords (NLTK)")
    w("  5. Lemmatize using WordNet lemmatizer")
    w("  6. Remove tokens shorter than 2 characters")
    w()

    w("-" * 72)
    w("CLASSIFIER DESCRIPTIONS")
    w("-" * 72)
    w()
    w("A. Rule-Based Legal Classifier")
    w("   Deterministic weighted keyword and phrase matching.")
    w("   Multi-word phrases receive higher weights (4.0-5.0).")
    w("   Single keywords receive weights of 1.5-3.0.")
    w("   Tie-breaking: first category in predefined order.")
    w("   Fallback: 'contract' when no clue matches.")
    w("   Training time is 0 (predefined rules, no learning).")
    w()
    w("B. Multinomial Naive Bayes")
    w("   TF-IDF features with unigrams and bigrams, sublinear TF.")
    w("   MultinomialNB with default parameters.")
    w()
    w("C. Linear Support Vector Machine")
    w("   TF-IDF features with unigrams and bigrams, sublinear TF.")
    w("   LinearSVC with balanced class weights, random_state=42.")
    w()
    w("D. Maximum Entropy Classifier (Logistic Regression)")
    w("   TF-IDF features with unigrams and bigrams, sublinear TF.")
    w("   LogisticRegression (multinomial) with max_iter=2000,")
    w("   balanced class weights, random_state=42.")
    w("   Multinomial Logistic Regression is mathematically equivalent")
    w("   to the Maximum Entropy classifier.")
    w()

    w("-" * 72)
    w("EVALUATION TABLE")
    w("-" * 72)
    header = (f"  {'Model':<18} {'Acc':>6} {'M-Prec':>7} {'M-Rec':>6} "
              f"{'M-F1':>6} {'W-F1':>6} {'Train(s)':>9} {'Infer(s)':>9}")
    w(header)
    for name, m in zip(model_names, all_metrics):
        w(f"  {name:<18} {m['accuracy']:>6.4f} {m['macro_precision']:>7.4f} "
          f"{m['macro_recall']:>6.4f} {m['macro_f1']:>6.4f} {m['weighted_f1']:>6.4f} "
          f"{m['train_time']:>9.4f} {m['infer_time']:>9.4f}")
    w()

    w("-" * 72)
    w("RULE-BASED vs MAXIMUM ENTROPY COMPARISON")
    w("-" * 72)
    w(f"  Rule-Based accuracy       : {rb_m['accuracy']:.4f}")
    w(f"  MaxEnt accuracy           : {me_m['accuracy']:.4f}")
    w(f"  Rule-Based macro precision: {rb_m['macro_precision']:.4f}")
    w(f"  MaxEnt macro precision    : {me_m['macro_precision']:.4f}")
    w(f"  Rule-Based macro recall   : {rb_m['macro_recall']:.4f}")
    w(f"  MaxEnt macro recall       : {me_m['macro_recall']:.4f}")
    w(f"  Rule-Based macro F1       : {rb_m['macro_f1']:.4f}")
    w(f"  MaxEnt macro F1           : {me_m['macro_f1']:.4f}")
    w(f"  Accuracy difference       : {acc_diff:.4f}")
    w(f"  Macro F1 difference       : {f1_diff:.4f}")
    w(f"  Better model (RB vs ME)   : {rb_vs_me_better}")
    w()

    w("-" * 72)
    w(f"BEST OVERALL CLASSIFIER: {best_name}")
    w("-" * 72)
    w(f"  Macro F1  : {best_metrics['macro_f1']:.4f}")
    w(f"  Accuracy  : {best_metrics['accuracy']:.4f}")
    w()

    w("-" * 72)
    w(f"PER-CATEGORY METRICS ({best_name})")
    w("-" * 72)
    w(per_category_report(test_labels, best_preds))
    w()

    w("-" * 72)
    w(f"CONFUSION MATRIX ({best_name})")
    w("-" * 72)
    w(text_confusion_matrix(test_labels, best_preds))
    w()

    w("-" * 72)
    w("ERROR ANALYSIS (up to 5 misclassified test examples)")
    w("-" * 72)
    if errors:
        for e in errors:
            w(f"  ID: {e['id']}  Title: {e['title']}")
            w(f"    Excerpt : {e['excerpt']}")
            w(f"    True    : {e['true']}")
            w(f"    Predicted: {e['pred']}")
            w()
    else:
        w("  No misclassified test documents were found.")
    w()

    w("-" * 72)
    w("CUSTOM LEGAL DOCUMENT ANALYSIS")
    w("-" * 72)
    w(f"  Original text   : {custom_text}")
    w(f"  Processed tokens: {custom_tokens}")
    w()
    w(f"  Rule-Based prediction : {custom_rb_pred}")
    w(f"  Rule-Based scores     : {custom_rb_scores}")
    w(f"  Naive Bayes prediction: {custom_nb_pred}")
    w(f"  Linear SVM prediction : {custom_svm_pred}")
    w(f"  MaxEnt prediction     : {custom_me_pred}")
    w()
    w("  Naive Bayes class probabilities:")
    for cls in CATEGORY_ORDER:
        idx = nb_classes.index(cls)
        w(f"    {cls:<15}: {nb_proba[idx]:.4f}")
    w()
    w("  MaxEnt class probabilities:")
    for cls in CATEGORY_ORDER:
        idx = me_classes.index(cls)
        w(f"    {cls:<15}: {me_proba[idx]:.4f}")
    w()
    w(f"  Final category ({best_name}): {final_custom_pred}")
    w()

    w("-" * 72)
    w("RESULT")
    w("-" * 72)
    w(f"The best overall classifier is {best_name} with a macro F1-score")
    w(f"of {best_metrics['macro_f1']:.4f} and accuracy of {best_metrics['accuracy']:.4f}.")
    w(f"In the prescribed Rule-Based vs Maximum Entropy comparison,")
    w(f"{rb_vs_me_better} performed better.")
    w(f"These results are specific to the embedded 80-document dataset")
    w(f"and the stratified 80/20 split with random seed {RANDOM_SEED}.")
    w("=" * 72)

    report = "\n".join(lines)
    output_file.write_text(report, encoding="utf-8")

    # ==================================================================
    # Terminal summary
    # ==================================================================
    print("=" * 60)
    print("EXPERIMENT 09 — Legal Document Classification")
    print("=" * 60)
    print(f"Student : {STUDENT_NAME} ({REGISTER_NUMBER})")
    print(f"Dataset : {total_count} docs | Train: {len(train_idx)} | Test: {len(test_idx)}")
    print()
    print(f"{'Model':<18} {'Acc':>6} {'M-F1':>6} {'W-F1':>6} {'Train':>7} {'Infer':>7}")
    print("-" * 56)
    for name, m in zip(model_names, all_metrics):
        print(f"{name:<18} {m['accuracy']:>6.3f} {m['macro_f1']:>6.3f} "
              f"{m['weighted_f1']:>6.3f} {m['train_time']:>7.4f} {m['infer_time']:>7.4f}")
    print()
    print("Rule-Based vs MaxEnt:")
    print(f"  RB  acc={rb_m['accuracy']:.3f}  F1={rb_m['macro_f1']:.3f}")
    print(f"  ME  acc={me_m['accuracy']:.3f}  F1={me_m['macro_f1']:.3f}")
    print(f"  Better: {rb_vs_me_better}")
    print()
    print(f"Best overall: {best_name} (macro F1={best_metrics['macro_f1']:.4f})")
    print()
    print(f"Custom text: {custom_text[:80]}...")
    print(f"  Rule-Based : {custom_rb_pred}")
    print(f"  Naive Bayes: {custom_nb_pred}")
    print(f"  Linear SVM : {custom_svm_pred}")
    print(f"  MaxEnt     : {custom_me_pred}")
    print(f"  Final ({best_name}): {final_custom_pred}")
    print()
    print(f"Output: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
