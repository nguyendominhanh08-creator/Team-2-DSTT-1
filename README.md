# GAME-THEORY-SOLVER.TEAM2 — Applications of Matrices in Game Theory

Two-player **zero-sum matrix games**: payoff matrix `A = (a_ij)`, pure and mixed strategies, saddle points,
the Minimax Theorem, the equivalent linear-programming model, solution algorithms and real-world scenarios.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py        # web interface (any m x n matrix: input + computation only)
python main.py              # terminal demo of the three scenarios
python tests/test_solver.py # automated checks (or: python -m pytest tests -q)
```

## Structure

| File | Content |
|---|---|
| `modules/saddle_point.py` | Maximin / Minimax, all pure saddle points |
| `modules/dominance.py` | Iterated elimination of strictly dominated strategies |
| `modules/solve_2x2.py` | 2x2 closed form (indifference principle) + graphical envelope method |
| `modules/lp_solver.py` | LP model of a matrix game (HiGHS), LaTeX of the LP models, optimality check |
| `modules/simplex.py` | Own tableau-simplex implementation (Bland's rule), returns every tableau |
| `modules/solver.py` | Full algorithm: saddle test -> 2x2 / dominance + LP -> verification -> cross-check |
| `modules/scenarios.py` | 3 real-world scenario matrices (presets). The written analysis is kept here for the report, not shown in the app |
| `app.py` | Streamlit UI: payoff matrix input and computation of the solution |

## Scenarios

1. Market competition, 3x3 — pure saddle point (value 2 pp).
2. Randomised police patrol vs. theft, 2x2 — mixed strategies, x* = (8/13, 5/13), v = -40/13.
3. Cybersecurity investment, 3x4 — general LP, x* = (1, 8, 5)/14, y* = (1, 0, 5, 8)/14, v = -29/7.

## References

Used for the report (Osborne 2004; Taha 2017; von Neumann 1928; Dantzig 1951; Ferguson 2014;
Owen 2013; Luenberger & Ye 2016; Gordon & Loeb 2002; Tambe 2011).
