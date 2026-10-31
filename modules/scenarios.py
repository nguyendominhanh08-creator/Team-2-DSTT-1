"""
Real-world application scenarios for two-player zero-sum matrix games.

Each scenario documents (as required by the assignment):
  * context and modelling assumptions,
  * interpretation of the matrix elements a_ij,
  * the meaning of the optimal strategies,
  * an economic / social analysis that goes beyond the computation.
Numbers shown in the static text refer to the PRESET payoff matrix; the app also shows
insights that are recomputed from whatever matrix is currently in the editor.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scenario:
    key: str
    title: str
    kind: str                       # which solution regime the preset illustrates
    row_player: str
    col_player: str
    row_names: list
    col_names: list
    A: np.ndarray
    unit: str
    context: str
    assumptions: list
    element_meaning: str
    strategy_meaning: str
    analysis: list
    limitations: str
    references: list = field(default_factory=list)


SCENARIOS = [
    # ------------------------------------------------------------------ 1
    Scenario(
        key="market",
        title="Scenario 1 - Market competition between two firms (3x3, pure saddle point)",
        kind="Pure strategies: the game has a saddle point",
        row_player="Firm A",
        col_player="Firm B",
        row_names=["X1: Aggressive price promotion", "X2: Digital / influencer campaign", "X3: Premium product launch"],
        col_names=["Y1: Match A's prices", "Y2: Loyalty programme", "Y3: Keep current strategy"],
        A=np.array([[3.0, 2.0, 4.0],
                    [1.0, 0.0, 5.0],
                    [2.0, -1.0, 3.0]]),
        unit="percentage points of market share gained by Firm A per quarter",
        context=(
            "Two firms dominate a mature consumer-goods market (for example, packaged coffee or "
            "mobile data plans). Customers rarely leave the market, so the quarterly decision of each "
            "firm is essentially a fight for existing customers. At the start of each quarter both firms "
            "commit simultaneously to one marketing strategy, without seeing the other's choice."
        ),
        assumptions=[
            "The market size is fixed over the quarter and the two firms share all of it, so Firm A's "
            "gain in market share (in percentage points) is exactly Firm B's loss: the game is constant-sum, "
            "which is strategically equivalent to zero-sum.",
            "Each firm chooses exactly one of three marketing strategies per quarter, simultaneously and independently.",
            "Both firms are rational payoff maximisers (A maximises its share gain, B minimises it) and know the payoff matrix, "
            "for example from past quarters or market research.",
            "The payoffs are expected changes in share, already averaged over random demand shocks.",
        ],
        element_meaning=(
            "a_ij is the expected change in Firm A's market share, in percentage points, when A plays row i and B plays column j. "
            "Positive entries are gains for A; the negative entry (-1) in row X3 / column Y2 means that a premium product launch "
            "against a loyalty programme costs A one point of share. Row X1 is the most robust for A: it never earns less than +2 pp."
        ),
        strategy_meaning=(
            "A pure optimal strategy means that each firm should commit to one marketing instrument and there is no gain from "
            "randomising or from guessing the opponent's move. Here (X1, Y2) is a saddle point: A's price promotion guarantees "
            "at least +2 pp whatever B does, and B's loyalty programme holds A to at most +2 pp whatever A does. "
            "Neither firm can profit from a unilateral deviation, so the saddle point is also a pure-strategy Nash equilibrium. "
            "The game value v = 2 is the share A can count on in equilibrium."
        ),
        analysis=[
            "Dominance explains the result. X3 is strictly worse than X1 in every column, so a rational firm never launches the "
            "premium product in this game; once X3 is gone, Y1 and Y3 are strictly worse for B than Y2 (they give A more), and "
            "finally X2 falls to X1. The competitive structure therefore collapses by iterated dominance to a single outcome. "
            "In market terms: when rivals are close substitutes, price-based promotion is hard to resist, and the only defence "
            "that holds the loss to a minimum is to raise switching costs, which is exactly what a loyalty programme does.",
            "Consumer welfare. In equilibrium, customers enjoy promotions from A and rewards from B. Price promotions and loyalty "
            "schemes transfer surplus to consumers in the short run, which regulators generally welcome. The long-run effect is "
            "ambiguous: loyalty schemes lock customers in and can reduce future price sensitivity, softening competition after the "
            "initial fight (the standard switching-cost argument in industrial organisation).",
            "Strategic lesson. The saddle point shows that security levels coincide: A's maximin guarantee equals B's minimax "
            "ceiling, so no information about the opponent is worth anything. This is the 'stable' regime in which competition is "
            "predictable and market shares are determined by structure, not by surprise.",
        ],
        limitations=(
            "Market share is zero-sum, profit is not. Both firms may end up with lower margins when both spend on promotions, a "
            "prisoner's-dilemma effect a zero-sum model cannot capture. The model also ignores entry by new competitors, "
            "dynamic (multi-quarter) effects and market expansion."
        ),
    ),
    # ------------------------------------------------------------------ 2
    Scenario(
        key="patrol",
        title="Scenario 2 - Randomised police patrol against opportunistic theft (2x2, mixed strategies)",
        kind="Mixed strategies: 2x2 closed-form solution",
        row_player="Police (defender)",
        col_player="Thief (attacker)",
        row_names=["X1: Patrol Downtown market", "X2: Patrol Industrial park"],
        col_names=["Y1: Target Downtown market", "Y2: Target Industrial park"],
        A=np.array([[0.0, -5.0],
                    [-8.0, 0.0]]),
        unit="million VND of goods stolen (negative = loss for the police / community)",
        context=(
            "A district police unit has one patrol car per night and two areas to protect: a Downtown retail market "
            "(goods worth about 8 million VND are exposed per night) and an Industrial park "
            "(about 5 million VND exposed). One patrol car cannot be in both places. An experienced opportunistic thief "
            "chooses one of the two areas each night and knows the value at stake in both."
        ),
        assumptions=[
            "The car can patrol exactly one area per night, and the thief attacks exactly one area; both decide without seeing the other's choice.",
            "If the thief attacks the patrolled area he is deterred or caught and nothing is stolen; "
            "if he attacks the unpatrolled area he steals the full exposed value of that area.",
            "Losses to the community equal the thief's gain (a transfer of goods), so the game is zero-sum in monetary terms.",
            "The thief learns the police's long-run patrol frequencies by observation, but not tonight's actual move; "
            "therefore only the probabilities matter.",
        ],
        element_meaning=(
            "a_ij is the police's payoff = minus the value stolen. a_11 = a_22 = 0: the thief meets the patrol and steals nothing. "
            "a_12 = -5: the police are Downtown, the thief hits the Industrial park and takes 5 million VND. "
            "a_21 = -8: the police are in the Industrial park and the thief hits Downtown and takes 8 million VND. "
            "There is no saddle point: the maximin is -5 (always patrol Downtown) but the minimax is 0, so each side wants to outguess the other."
        ),
        strategy_meaning=(
            "With D = (0 + 0) - (-5 - 8) = 13, the optimal mixed strategy of the police is x* = (8/13, 5/13): patrol Downtown on "
            "about 61.5% of nights and the Industrial park on 38.5%. The thief's optimal strategy is y* = (5/13, 8/13): attack Downtown "
            "on 38.5% of nights and the Industrial park on 61.5%. The value is v = -40/13 = -3.08 million VND per night: even with optimal "
            "randomisation the community cannot prevent all losses, but it limits the expected loss to about 3.08 million VND."
        ),
        analysis=[
            "Predictability is expensive. If the police always protect the high-value market (the maximin pure strategy), a "
            "rational thief simply attacks the Industrial park every night and the loss is 5 million VND. Randomising according to "
            "x* cuts the expected loss to 3.08 million VND, a reduction of about 38%. The gain is produced purely by unpredictability, "
            "not by extra resources. This is why modern security operations (airport and port security, ticket inspections, "
            "customs and tax audits) use randomised schedules: a published fixed route is a free instruction manual for the offender.",
            "Counter-intuitive equilibrium. The police should patrol the more valuable area more often (61.5% vs 38.5%), but the thief "
            "attacks it less often (38.5%). The reason is that the thief's mix is chosen to make the police indifferent between "
            "their two patrols, so the high-stakes area is attacked less frequently precisely because it is protected more. "
            "Any policy evaluation that only counts where crimes happen would therefore mistake deterrence for safety.",
            "Social and equity dimension. The equilibrium allocates scarce public resources by monetary value at risk. "
            "This is efficient for property loss but not necessarily fair: an area with lower asset value but vulnerable residents "
            "would receive fewer patrols if value alone is used. In practice the payoff matrix should be reweighted to include harm "
            "to people, not just goods. Displacement is also a social effect: protecting one area may shift crime to another, which "
            "this game captures through the thief's choice. The value of the game also gives a defensible number for budget "
            "decisions: adding a second car would reduce the expected loss to zero in this model (both areas could be covered), "
            "which can be compared with the car's running cost.",
        ],
        limitations=(
            "The model uses a single night, one patrol, one thief and perfect knowledge of the values at stake. Real offenders "
            "have different valuations, deterrence also has psychological effects, and the thief usually observes the defender "
            "before acting (a Stackelberg security game) instead of choosing simultaneously."
        ),
        references=["Tambe (2011)"],
    ),
    # ------------------------------------------------------------------ 3
    Scenario(
        key="cyber",
        title="Scenario 3 - Cybersecurity investment: company vs. attacker (3x4, general LP)",
        kind="Mixed strategies: general m x n game solved by linear programming",
        row_player="Company (defender)",
        col_player="Attacker",
        row_names=["X1: Staff training + email filtering", "X2: Backups + endpoint detection (EDR)", "X3: DDoS protection + zero-trust access"],
        col_names=["Y1: Phishing", "Y2: Ransomware", "Y3: DDoS", "Y4: Insider misuse"],
        A=np.array([[-1.0, -6.0, -5.0, -4.0],
                    [-4.0, -1.0, -6.0, -3.0],
                    [-5.0, -4.0, -1.0, -6.0]]),
        unit="million USD of expected loss (negative) for the company in one planning period",
        context=(
            "A mid-size online company has a fixed security budget that lets it emphasise one of three defensive investments "
            "this year. A professional attacker, who can probe the company's technology and typical practices, "
            "selects the attack vector he expects to be the most damaging: phishing, ransomware, a denial-of-service (DDoS) "
            "attack or abuse of an insider's access. The company cannot defend everything equally with a finite budget."
        ),
        assumptions=[
            "The budget allows one dominant defensive focus; the company may randomise its focus (or split budget in proportion) across the three options.",
            "The attacker picks one vector and the loss to the company is the attacker's gain (a transfer such as ransom or fraud). "
            "This zero-sum treatment ignores deadweight losses and is therefore a simplification.",
            "Both sides know the loss table, for instance from industry incident reports and penetration-test results.",
            "Each a_ij is an expected annual loss in million USD, already including the probability that the attack succeeds.",
        ],
        element_meaning=(
            "a_ij = - expected loss when the company focuses on defence i and the attacker uses vector j. The diagonal-like "
            "entries (-1) mark the 'matched' pairings (training vs phishing, backups/EDR vs ransomware, DDoS protection vs DDoS) "
            "where the defence is most effective. Off-diagonal entries reach -6, the cost of facing an attack that the chosen "
            "investment barely covers. The matrix has 3 rows and 4 columns, so it is not square: the number of defences and of attack vectors need not match."
        ),
        strategy_meaning=(
            "No pure strategy is safe: any single focus can be exploited, so the optimal strategy is a portfolio. Solving the LP gives "
            "x* = (1/14, 8/14, 5/14), i.e. 7.1% / 57.1% / 35.7% of the effort on X1 / X2 / X3, and the attacker's optimal mix "
            "y* = (1/14, 0, 5/14, 8/14). The value of the game is v = -29/7, about -4.14 million USD of expected loss, "
            "compared with -6 million USD if the company chose any single defence and the attacker best-responded."
        ),
        analysis=[
            "Deterrence, not observation. Ransomware gets probability 0 in the attacker's equilibrium mix, but this does not mean "
            "that the backup / EDR investment is wasteful: X2 receives the largest share (57%) exactly because the attacker "
            "is deterred from ransomware by it. If the company stopped investing in X2 the ransomware column would immediately "
            "become one of the attacker's best responses. Security spending is therefore justified by the attacks that do NOT happen. "
            "Security managers who cut a control because 'no ransomware attacks were seen' commit exactly this logical error.",
            "Diversification and the economics of security. The equilibrium resembles a portfolio: the company spreads effort "
            "so that every attack vector in the attacker's support yields the same expected loss (-4.14), and spends nothing "
            "unnecessarily on vectors that are already covered. The classical Gordon-Loeb model also argues that it is not optimal "
            "to defend assets perfectly; the optimal security spending is at most about 37% of the expected loss from a breach. "
            "The matrix-game result agrees with this spirit: the equilibrium accepts a residual expected loss of about 4.14 million USD "
            "instead of trying to remove all risk.",
            "Social perspective. Breaches harm customers and third parties (identity theft, service outages), "
            "costs that the firm does not fully internalise. Since the firm optimises its own loss table, its equilibrium level of protection is "
            "typically lower than the socially optimal one, which is the economic rationale for breach-notification laws, "
            "minimum security standards and cyber-insurance requirements. The asymmetry of the game is also visible: "
            "the attacker needs to find one weakness, while the company must spread resources over all of them.",
        ],
        limitations=(
            "Real attackers do not have exactly opposite incentives (the attacker's gain from ransom differs from the victim's total loss), "
            "attack success probabilities change with technology, and defence is sequential and adaptive. The numbers are illustrative and "
            "should be estimated from incident data before being used for real budget decisions."
        ),
        references=["Gordon and Loeb (2002)"],
    ),
]


def get_scenario(key):
    return next(s for s in SCENARIOS if s.key == key)


def scenario_insights(row_names, col_names, report, unit=""):
    """Plain-language insights recomputed from the CURRENT matrix and its solution (markdown lines)."""
    A = report.A
    sp = report.saddle
    v = report.value
    lines = []

    lines.append(
        f"**Security levels.** Player 1 can guarantee at least **{sp['maximin']:.4f}** with a pure strategy (Maximin); "
        f"Player 2 can hold Player 1 to at most **{sp['minimax']:.4f}** with a pure strategy (Minimax)."
    )
    if sp["has_saddle_point"]:
        lines.append("Maximin = Minimax: a pure saddle point exists, so randomising brings no benefit to either player.")
    else:
        g1 = v - sp["maximin"]
        g2 = sp["minimax"] - v
        lines.append(
            f"**Value of randomisation.** Mixing raises Player 1's guaranteed payoff from {sp['maximin']:.4f} to "
            f"{v:.4f} (gain {g1:.4f}) and lowers Player 2's ceiling from {sp['minimax']:.4f} to {v:.4f} (gain {g2:.4f})."
        )

    def describe(names, p, who):
        used = [f"{nm} ({pr * 100:.1f}%)" for nm, pr in zip(names, p) if pr > 1e-9]
        unused = [nm for nm, pr in zip(names, p) if pr <= 1e-9]
        txt = f"**{who} plays** " + ", ".join(used) + "."
        if unused:
            txt += f" Never used in equilibrium: {', '.join(unused)}."
        return txt

    lines.append(describe(row_names, report.x, "Player 1 (rows)"))
    lines.append(describe(col_names, report.y, "Player 2 (columns)"))
    lines.append(f"**Expected payoff** x^T A y = {report.verification['xAy']:.4f}{(' ' + unit) if unit else ''}.")
    return lines
