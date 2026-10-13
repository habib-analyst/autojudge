# autojudge — STATUS

**Started:** 2026-10-13 ~00:40 PKT (subagent build)
**Invention:** significance-tested improvement audit — an LLM-judge comparison
harness that refuses to declare a win until the improvement is statistically
real (paired permutation test + bootstrap CI + practical effect-size gate).

- [x] venv + deps (numpy, pytest)
- [ ] package (stats, evaluate, diagnostics, report)
- [ ] tests
- [ ] kill-gate experiments (3 scenarios + type-I calibration, >=200 null sims)
- [ ] paper
- [ ] README (brand kit)
- [ ] LinkedIn draft
- [x] push payload
- [x] DONE

## DONE 2026-10-13 ~01:05 PKT
- Kill-gate PASS: 3/3 scenario verdicts correct (WIN / NO_SIGNIFICANT_DIFFERENCE / NOT_PRACTICALLY_SIGNIFICANT)
- Scenario (c): p=0.0022 significant but d=0.146 < 0.2 -> correctly refused; naive mean-comparison would ship it
- Type-I calibration: 0.045 over 200 null sims (bar [0.02, 0.08]) — calibrated
- 13/13 unit tests green; full suite ~1s CPU
- REPO NEEDED: autojudge
