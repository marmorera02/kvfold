"""E2.1 -- Synthetic supply-agreement corpus generator (spec E2.1).

Deterministic given --seed: boilerplate is string templates with slot
substitution only (no LLM), so ground-truth needle positions stay exact.
"""
import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

TOTAL_CLAUSES = 100
LIABILITY_POSITIONS = [8, 25, 50]
N1_POSITION = 33  # weird-but-benign
N2_POSITION = 87  # fluent-but-contradictory
N3_POSITION = 61  # genuinely novel
N4_POSITION = 45  # control (ordinary boilerplate, just tagged)

CATEGORIES = [
    "delivery", "warranty", "payment", "confidentiality",
    "force_majeure", "ip", "compliance", "termination",
]

JURISDICTIONS = [
    "the State of Delaware", "England and Wales", "the Province of Ontario",
    "Singapore", "the State of New York", "the Republic of Ireland",
]
DURATIONS = ["thirty (30) days", "forty-five (45) days", "sixty (60) days", "ninety (90) days"]
AMOUNTS = ["$5,000", "$10,000", "$25,000", "$50,000", "€8,000", "€20,000"]
PRODUCTS = [
    "precision-machined aluminum housings", "industrial control modules",
    "sealed bearing assemblies", "custom wiring harnesses",
    "injection-molded enclosures", "calibrated pressure sensors",
]
CURE_DAYS = [15, 30, 45, 60]
NOTICE_DAYS = [5, 10, 14, 21]
CURRENCIES = ["USD", "EUR", "GBP"]

TEMPLATES = {
    "delivery": [
        (
            "Supplier shall deliver the {product} to Buyer's designated receiving facility "
            "in accordance with the delivery schedule set out in Schedule 2, and time of "
            "delivery shall be of the essence. Should Supplier reasonably anticipate that a "
            "shipment will be delayed beyond the scheduled delivery date, Supplier shall "
            "notify Buyer in writing within {notice_days} days of becoming aware of the "
            "anticipated delay, specifying the revised delivery date and the cause of the "
            "delay. Partial shipments shall be permitted only with Buyer's prior written "
            "consent, and each partial shipment shall be separately invoiced. Risk of loss "
            "shall pass to Buyer upon confirmed receipt at the designated facility, and "
            "Supplier shall retain risk of loss for all goods in transit prior to that point. "
            "Title to the goods shall pass to Buyer upon full payment of the applicable "
            "invoice, notwithstanding the earlier passage of risk of loss, and Supplier shall "
            "retain a purchase money security interest in the goods until such payment is "
            "received in full. Supplier shall maintain adequate insurance covering loss or "
            "damage to goods in transit and shall provide evidence of such coverage to Buyer "
            "upon reasonable written request."
        ),
        (
            "All shipments of {product} under this Agreement shall be packaged in "
            "accordance with Supplier's standard export packaging specifications, or such "
            "other specifications as the parties may agree in writing, and shall be "
            "accompanied by a packing list identifying the purchase order number, quantity, "
            "and batch or lot identifier. Supplier shall provide Buyer with advance shipping "
            "notification not less than {notice_days} days prior to the scheduled ship date. "
            "Buyer shall inspect each shipment within {duration} of receipt and shall notify "
            "Supplier in writing of any discrepancy in quantity or apparent damage, failing "
            "which the shipment shall be deemed accepted as delivered. Supplier shall retain "
            "records of each shipment's packing list and carrier manifest for a period of not "
            "less than {duration}, and shall make such records available to Buyer upon "
            "reasonable written request for the purpose of resolving any delivery dispute. "
            "Title to rejected goods shall remain with Supplier pending disposition "
            "instructions from Supplier, which shall be provided within {notice_days} days of "
            "receiving notice of rejection."
        ),
    ],
    "warranty": [
        (
            "Supplier warrants that the {product} delivered under this Agreement shall, for "
            "a period of {duration} from the date of delivery, conform to the specifications "
            "set out in Schedule 1 and shall be free from defects in material and "
            "workmanship under normal use and service. This warranty does not extend to "
            "defects arising from misuse, unauthorized modification, or normal wear and "
            "tear. Buyer's sole and exclusive remedy for breach of this warranty shall be, "
            "at Supplier's option, repair, replacement, or refund of the purchase price for "
            "the nonconforming goods, provided that Buyer notifies Supplier in writing "
            "within {notice_days} days of discovering the defect. Supplier shall bear the "
            "cost of shipping any repaired or replacement goods to Buyer, and Buyer shall "
            "grant Supplier reasonable access to inspect the nonconforming goods prior to "
            "any repair, replacement, or refund. This warranty is in lieu of all other "
            "warranties, express or implied, including any implied warranty of "
            "merchantability or fitness for a particular purpose, except as expressly set "
            "out in this Clause."
        ),
        (
            "Supplier represents and warrants that all {product} supplied hereunder shall be "
            "new, of merchantable quality, and fit for the purpose for which such goods are "
            "customarily used, and shall comply in all material respects with the applicable "
            "specifications for a period of no less than {duration} following delivery. "
            "Supplier shall, at its own expense, correct any nonconformity identified during "
            "the warranty period within {cure_days} days of receiving written notice from "
            "Buyer describing the nonconformity in reasonable detail, provided that such "
            "correction is commercially and technically feasible. If Supplier is unable to "
            "correct the nonconformity within the {cure_days}-day period, Buyer may procure "
            "replacement goods from a third party and recover the reasonable additional cost "
            "of doing so from Supplier, provided Buyer has given Supplier a reasonable "
            "opportunity to cure. Nothing in this Clause shall limit any statutory warranty "
            "that cannot lawfully be excluded in {jurisdiction}."
        ),
    ],
    "payment": [
        (
            "Buyer shall pay all undisputed invoices within {duration} of the invoice date, "
            "by wire transfer to the bank account designated by Supplier in writing. Invoices "
            "not disputed in writing within {notice_days} days of receipt shall be deemed "
            "accepted. Amounts not paid when due shall accrue interest at the rate of one and "
            "one-half percent (1.5%) per month, or the maximum rate permitted by applicable "
            "law, whichever is lower. All amounts payable under this Agreement are exclusive "
            "of applicable value-added, sales, or similar taxes, which shall be borne by "
            "Buyer unless otherwise required by law. Buyer may withhold payment of any "
            "invoiced amount that is subject to a good-faith dispute, provided Buyer notifies "
            "Supplier in writing of the basis for the dispute within {notice_days} days of "
            "receipt of the invoice, and the parties shall use reasonable efforts to resolve "
            "the dispute within {duration}. Neither party shall be entitled to set off "
            "amounts owed under this Agreement against amounts owed under any other agreement "
            "between the parties absent written agreement to the contrary."
        ),
        (
            "Prices for the {product} are as set out in Schedule 1 and are denominated in "
            "{currency} unless otherwise agreed in writing. Supplier shall invoice Buyer upon "
            "shipment, and payment shall be due {duration} from the date of each invoice. "
            "Should Buyer fail to make payment when due, Supplier may, upon {notice_days} "
            "days' prior written notice, suspend further deliveries until all outstanding "
            "amounts, together with accrued interest, have been paid in full, without "
            "prejudice to any other rights or remedies available to Supplier. Any change in "
            "prices shall require not less than {notice_days} days' prior written notice to "
            "Buyer and shall apply only to purchase orders placed after the effective date of "
            "the change. All invoices shall be issued in {currency} and shall separately "
            "state the applicable purchase order number, quantity delivered, and unit price "
            "for each line item."
        ),
    ],
    "confidentiality": [
        (
            "Each party shall treat as confidential all technical, commercial, and financial "
            "information disclosed by the other party in connection with this Agreement, and "
            "shall not disclose such information to any third party without the prior written "
            "consent of the disclosing party, except to employees, contractors, and advisors "
            "with a need to know and who are bound by confidentiality obligations no less "
            "protective than those set out herein. This obligation shall survive termination "
            "of this Agreement for a period of {duration} and shall not apply to information "
            "that is or becomes publicly available through no fault of the receiving party. "
            "The receiving party may disclose Confidential Information to the extent required "
            "by law or a valid order of a court or governmental authority, provided that, to "
            "the extent legally permissible, it gives the disclosing party prompt written "
            "notice of such requirement so that the disclosing party may seek a protective "
            "order or other appropriate remedy at its own expense."
        ),
        (
            "Confidential Information disclosed under this Agreement shall be used by the "
            "receiving party solely for the purpose of performing its obligations hereunder, "
            "and shall be protected using at least the same degree of care that the receiving "
            "party uses to protect its own confidential information of similar importance, "
            "but in no event less than a reasonable degree of care. Upon termination of this "
            "Agreement, each party shall, within {duration} of the other party's written "
            "request, return or destroy all Confidential Information in its possession, "
            "except to the extent retention is required by applicable law or bona fide "
            "archival policy. Neither party shall use the other party's Confidential "
            "Information to develop products or services that compete with the disclosing "
            "party's business, and each party acknowledges that unauthorized disclosure of "
            "Confidential Information may cause irreparable harm for which monetary damages "
            "would be an inadequate remedy, entitling the disclosing party to seek injunctive "
            "relief in addition to any other available remedy."
        ),
    ],
    "force_majeure": [
        (
            "Neither party shall be liable for any failure or delay in the performance of "
            "its obligations under this Agreement, other than payment obligations, to the "
            "extent such failure or delay is caused by circumstances beyond its reasonable "
            "control, including acts of God, war, civil unrest, governmental action, "
            "pandemic, or failure of utilities or telecommunications infrastructure, provided "
            "that the affected party notifies the other party in writing within {notice_days} "
            "days of the onset of such circumstances and uses commercially reasonable efforts "
            "to mitigate the impact. If such circumstances continue for more than {duration}, "
            "either party may terminate this Agreement upon written notice. During the "
            "continuation of an event of force majeure, the unaffected party may, at its "
            "option and upon written notice, procure substitute goods or services from a "
            "third party to the extent reasonably necessary to mitigate the impact of the "
            "affected party's non-performance, without thereby waiving any right or remedy "
            "under this Agreement."
        ),
        (
            "A party affected by an event of force majeure shall promptly notify the other "
            "party of the nature and expected duration of the event, and shall be excused "
            "from performance, other than the obligation to make payments already due, for so "
            "long as the event of force majeure continues. Should an event of force majeure "
            "continue for a period exceeding {duration}, either party may terminate this "
            "Agreement without liability by giving not less than {notice_days} days' written "
            "notice to the other party, provided the event has not ceased prior to the "
            "expiry of such notice period. Neither party shall be entitled to terminate this "
            "Agreement for force majeure to the extent the affected party's own failure to "
            "maintain reasonable business continuity arrangements contributed materially to "
            "its inability to perform, and the parties shall cooperate in good faith to "
            "identify reasonable workaround arrangements before either party exercises any "
            "right of termination under this Clause."
        ),
    ],
    "ip": [
        (
            "All intellectual property rights in the {product}, including any designs, "
            "drawings, specifications, or know-how provided by Buyer to Supplier for the "
            "purpose of manufacture, shall remain the exclusive property of Buyer, and "
            "Supplier shall acquire no rights therein other than the limited right to use "
            "such materials solely for the performance of its obligations under this "
            "Agreement. Supplier shall not use, reproduce, or disclose such materials for any "
            "other purpose without Buyer's prior written consent, and shall return or destroy "
            "all such materials within {duration} of termination of this Agreement. Supplier "
            "shall ensure that any subcontractor engaged in the manufacture of the {product} "
            "is bound by written obligations of confidentiality and restricted use no less "
            "protective of Buyer's intellectual property than those set out in this Clause, "
            "and Supplier shall remain fully liable to Buyer for any breach of this Clause by "
            "such subcontractor."
        ),
        (
            "Any improvements, modifications, or derivative works created by Supplier in the "
            "course of manufacturing the {product} that relate specifically to Buyer's "
            "proprietary designs shall be owned by Buyer, and Supplier hereby assigns all "
            "right, title, and interest in such improvements to Buyer. Supplier retains "
            "ownership of any general manufacturing know-how, tooling designs, and process "
            "improvements that are not specific to Buyer's proprietary designs and that "
            "Supplier would have developed irrespective of this Agreement, within {duration} "
            "of first use. Where an improvement combines Buyer-specific design elements with "
            "Supplier's general manufacturing know-how in a manner that cannot reasonably be "
            "separated, the parties shall negotiate in good faith a reasonable allocation of "
            "rights, failing which Buyer shall be granted a perpetual, royalty-free license to "
            "use the improvement solely in connection with the {product}."
        ),
    ],
    "compliance": [
        (
            "Supplier shall comply with all applicable laws, regulations, and industry "
            "standards in the manufacture and supply of the {product}, including those "
            "relating to product safety, labor practices, and environmental protection in "
            "{jurisdiction}. Supplier shall, upon Buyer's reasonable written request, provide "
            "documentation evidencing such compliance, and shall permit Buyer or its "
            "designated auditor to conduct an on-site compliance audit upon not less than "
            "{notice_days} days' prior written notice, no more than once per {duration} "
            "period absent reasonable cause. Any audit conducted under this Clause shall be "
            "at Buyer's expense, shall be conducted during Supplier's normal business hours, "
            "and shall not unreasonably interfere with Supplier's manufacturing operations. "
            "Supplier shall address any material finding identified during an audit within "
            "{cure_days} days of receiving Buyer's written audit report."
        ),
        (
            "Supplier represents that it maintains a code of conduct consistent with "
            "generally accepted standards for responsible sourcing and shall not employ "
            "forced labor or child labor in the manufacture of the {product}. Supplier shall "
            "notify Buyer in writing within {notice_days} days of becoming aware of any "
            "material non-compliance with applicable law in {jurisdiction}, and shall "
            "implement a corrective action plan within {cure_days} days of identifying such "
            "non-compliance, subject to Buyer's reasonable approval. Buyer may, at its "
            "discretion and upon reasonable written notice, request evidence of Supplier's "
            "compliance with this Clause from any sub-tier supplier involved in the "
            "manufacture of the {product}, and Supplier shall use commercially reasonable "
            "efforts to obtain and provide such evidence within {notice_days} days."
        ),
    ],
    "termination": [
        (
            "Either party may terminate this Agreement for cause upon written notice if the "
            "other party commits a material breach of this Agreement and fails to cure such "
            "breach within {cure_days} days of receiving written notice describing the "
            "breach in reasonable detail. Either party may also terminate this Agreement for "
            "convenience upon not less than {duration} prior written notice to the other "
            "party. Termination of this Agreement shall not affect any rights or obligations "
            "accrued prior to the effective date of termination. Upon termination for "
            "convenience by Buyer, Buyer shall reimburse Supplier for reasonable costs "
            "incurred in reliance on outstanding purchase orders, including committed raw "
            "material costs, provided Supplier has used commercially reasonable efforts to "
            "mitigate such costs."
        ),
        (
            "This Agreement shall terminate automatically, without notice, in the event "
            "either party becomes insolvent, makes an assignment for the benefit of "
            "creditors, or becomes subject to proceedings under applicable insolvency law "
            "that are not dismissed within {cure_days} days. Upon termination for any reason, "
            "Buyer shall pay Supplier for all {product} delivered and accepted prior to the "
            "effective date of termination, and each party shall, within {duration}, return "
            "or destroy any Confidential Information of the other party then in its "
            "possession. The provisions of this Agreement relating to confidentiality, "
            "intellectual property, and limitation of liability shall survive termination or "
            "expiration of this Agreement for any reason, together with any other provision "
            "that by its nature is intended to survive."
        ),
    ],
}

LIABILITY_FRAME_TEXTS = [
    (
        "Supplier's aggregate liability under this Agreement shall be capped at 100% of the "
        "annual Contract Value; in no event shall Supplier be liable for consequential, "
        "incidental, or indirect damages of any kind arising out of or in connection with "
        "this Agreement, whether in contract, tort, or otherwise."
    ),
    (
        "Notwithstanding anything else in this Agreement, the total liability of Supplier "
        "arising out of or relating to this Agreement, whether in contract, tort, or "
        "otherwise, shall not exceed one hundred percent (100%) of the annual Contract "
        "Value, and Supplier shall have no liability whatsoever for any consequential, "
        "incidental, or indirect damages, however arising."
    ),
    (
        "For the avoidance of doubt, Supplier's cumulative liability to Buyer under this "
        "Agreement, however arising and regardless of the form of action, is limited to "
        "100% of the annual Contract Value, and excludes all consequential, indirect, "
        "special, or punitive damages of any kind."
    ),
]

N1_TEXT = (
    "All notices under this Agreement shall be sent by regustered mail to the address of "
    "the receiving party set out in Schedule 1, or by such other method as the parties may "
    "agree in writing from time to time, provided that notices sent by facsimile shall be "
    "deemed effective only upon confirmed transmission and a courtesy copy dispatched by "
    "ordinary post within two (2) Business Days thereof, notwithstanding the informal "
    "custom of certain regional offices to additionally telephone the recipient's front "
    "desk between the hours of nine and eleven in the forenoon, a practice that, while "
    "unusual, imposes no obligation on either party and shall not be construed to alter the "
    "notice periods specified elsewhere in this Agreement."
)

N2_TEXT = (
    "Notwithstanding any other provision of this Agreement, including without limitation "
    "the limitation of liability set forth in Clause 8, Supplier's liability for line-down "
    "damages arising from any failure to deliver Goods that causes a cessation of Buyer's "
    "production line at any Buyer production facility shall be unlimited in amount and "
    "shall not be subject to the limitations, caps, or exclusions set forth elsewhere in "
    "this Agreement, and Buyer shall be entitled to recover the full amount of any such "
    "line-down damages, including consequential and indirect losses, directly from Supplier "
    "without regard to the annual Contract Value."
)

N3_TEXT = (
    "The parties acknowledge that certain of the Goods may incorporate components subject "
    "to classification under export control regimes applicable to quantum-computing-"
    "enabling technologies, including cryogenic control electronics and superconducting "
    "qubit fabrication equipment, and Supplier shall, prior to shipment, obtain and retain "
    "all export classification determinations required by the export control authority of "
    "the country of origin, and shall notify Buyer in writing of any change in the Export "
    "Control Classification Number applicable to such components, it being understood that "
    "this Clause addresses a distinct regulatory subject matter unrelated to the delivery, "
    "warranty, or liability provisions set out elsewhere in this Agreement."
)

# Open-world control (E2 post-Phase-D directive, Task 2): a second,
# independent frame + contradiction pair, used only in seed 5's two-
# contradiction variant. Placed in sections with no other special content
# (section 2 and section 8) so it doesn't collide with N1/N3/N2a/N4/liability.
EXCLUSIVITY_POSITION = 12
N2B_POSITION = 78

EXCLUSIVITY_FRAME_TEXT = (
    "Buyer shall purchase the Goods exclusively from Supplier for the Term of "
    "this Agreement, and shall not engage any alternative source of supply "
    "for the Goods without Supplier's prior written consent."
)

N2B_TEXT = (
    "Notwithstanding the exclusivity undertaking set forth earlier in this "
    "Agreement, Buyer may procure the Goods from any third-party supplier at "
    "its sole discretion, without restriction and without incurring any "
    "liability to Supplier for doing so."
)


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        duration=rng.choice(DURATIONS),
        amount=rng.choice(AMOUNTS),
        currency=rng.choice(CURRENCIES),
        jurisdiction=rng.choice(JURISDICTIONS),
        product=rng.choice(PRODUCTS),
        cure_days=rng.choice(CURE_DAYS),
        notice_days=rng.choice(NOTICE_DAYS),
    )


def _base_special(seed: int) -> dict[int, tuple[str, str]]:
    """Seeds 0-2 (unchanged): standard one-contradiction config."""
    return {
        LIABILITY_POSITIONS[0]: ("liability_frame", LIABILITY_FRAME_TEXTS[0]),
        LIABILITY_POSITIONS[1]: ("liability_frame", LIABILITY_FRAME_TEXTS[1]),
        LIABILITY_POSITIONS[2]: ("liability_frame", LIABILITY_FRAME_TEXTS[2]),
        N1_POSITION: ("n1_weird_benign", N1_TEXT),
        N2_POSITION: ("n2_contradiction", N2_TEXT),
        N3_POSITION: ("n3_novel", N3_TEXT),
    }


def build_corpus(seed: int) -> tuple[list[dict], dict]:
    """Seeds 0-2: original one-contradiction-per-seed design (unchanged, byte
    for byte). Seeds 3-4 (open-world control, Task 2): zero contradictions --
    N2's slot reverts to ordinary boilerplate. Seed 5: two contradictions --
    the original liability one (N2a) plus an independent exclusivity one
    (N2b) in a different L2 section."""
    rng = random.Random(seed)

    special = _base_special(seed)
    needles = [
        {"id": N1_POSITION, "type": "n1_weird_benign"},
        {"id": N2_POSITION, "type": "n2_contradiction"},
        {"id": N3_POSITION, "type": "n3_novel"},
        {"id": N4_POSITION, "type": "n4_control"},
    ]
    liability_frame_ids = list(LIABILITY_POSITIONS)
    extra_frame_ids: list[int] = []

    if seed in (3, 4):
        del special[N2_POSITION]  # ordinary boilerplate takes the slot instead
        needles = [n for n in needles if n["type"] != "n2_contradiction"]
    elif seed == 5:
        special[EXCLUSIVITY_POSITION] = ("exclusivity_frame", EXCLUSIVITY_FRAME_TEXT)
        special[N2B_POSITION] = ("n2b_contradiction", N2B_TEXT)
        needles = [n if n["type"] != "n2_contradiction" else {"id": N2_POSITION, "type": "n2a_contradiction"}
                   for n in needles]
        needles.append({"id": N2B_POSITION, "type": "n2b_contradiction"})
        extra_frame_ids = [EXCLUSIVITY_POSITION]

    clauses = []
    cat_i = 0
    for clause_id in range(1, TOTAL_CLAUSES + 1):
        if clause_id in special:
            ctype, text = special[clause_id]
        else:
            category = CATEGORIES[cat_i % len(CATEGORIES)]
            cat_i += 1
            template = rng.choice(TEMPLATES[category])
            text = _fill(template, rng)
            ctype = "boilerplate"
        clauses.append({"id": clause_id, "type": ctype, "text": text})

    pos = 0
    for c in clauses:
        header = f"Clause {c['id']}. "
        full = header + c["text"]
        c["char_start"] = pos
        c["char_end"] = pos + len(full)
        c["full_text"] = full
        pos = c["char_end"] + 2  # "\n\n" separator

    ground_truth = {
        "seed": seed,
        "total_clauses": TOTAL_CLAUSES,
        "liability_frame_clause_ids": liability_frame_ids,
        "extra_frame_clause_ids": extra_frame_ids,
        "n_contradictions": sum(1 for n in needles if "contradiction" in n["type"]),
        "needles": needles,
    }
    return clauses, ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-corpus", type=Path, default=None)
    parser.add_argument("--out-truth", type=Path, default=None)
    args = parser.parse_args()

    clauses, ground_truth = build_corpus(args.seed)

    out_corpus = args.out_corpus or DATA_DIR / f"corpus_seed{args.seed}.jsonl"
    out_truth = args.out_truth or DATA_DIR / f"ground_truth_seed{args.seed}.json"
    out_corpus.parent.mkdir(parents=True, exist_ok=True)

    with open(out_corpus, "w", encoding="utf-8") as f:
        for c in clauses:
            f.write(json.dumps({
                "id": c["id"], "type": c["type"], "text": c["text"],
                "char_start": c["char_start"], "char_end": c["char_end"],
            }) + "\n")

    with open(out_truth, "w") as f:
        json.dump(ground_truth, f, indent=2)

    total_chars = sum(len(c["full_text"]) for c in clauses)
    print(f"wrote {len(clauses)} clauses, {total_chars} chars -> {out_corpus}")
    print(f"ground truth -> {out_truth}")


if __name__ == "__main__":
    main()
