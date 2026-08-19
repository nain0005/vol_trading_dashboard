"""Correctness tests for risk_tool.pricing.

These check the pricing engine against independently-known facts, not
against itself: a published textbook example, an algebraic identity
(put-call parity) that must hold exactly regardless of inputs, and a
known qualitative fact about American vs. European exercise.
"""
import math

import pytest

from risk_tool.pricing import (
    binomial_crr_price,
    black_scholes_price,
    compute_d1_d2,
    implied_volatility,
    itm_probability,
)


# Hull, "Options, Futures, and Other Derivatives" — the standard textbook
# example: S=42, K=40, r=10%, sigma=20%, T=0.5y, no dividend.
# Published answers: call ~= 4.76, put ~= 0.81.
HULL_PARAMS = dict(S=42.0, K=40.0, T=0.5, r=0.10, q=0.0, sigma=0.20)


def test_black_scholes_matches_known_textbook_value():
    call = black_scholes_price(**HULL_PARAMS, option_type="call")
    put = black_scholes_price(**HULL_PARAMS, option_type="put")
    assert call == pytest.approx(4.76, abs=0.02)
    assert put == pytest.approx(0.81, abs=0.02)


@pytest.mark.parametrize(
    "S,K,T,r,q,sigma",
    [
        (100, 100, 1.0, 0.05, 0.0, 0.2),
        (55, 60, 0.25, 0.03, 0.02, 0.35),
        (200, 150, 2.0, 0.01, 0.0, 0.5),
        (42, 40, 0.5, 0.10, 0.0, 0.2),
    ],
)
def test_put_call_parity_holds_exactly(S, K, T, r, q, sigma):
    """C - P = S*e^-qT - K*e^-rT must hold algebraically for any inputs —
    this isn't an approximation, so the tolerance is tight."""
    call = black_scholes_price(S, K, T, r, q, sigma, "call")
    put = black_scholes_price(S, K, T, r, q, sigma, "put")
    lhs = call - put
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_itm_probability_is_bounded_and_complementary():
    """N(d2) for a call and N(-d2) for a put must sum to 1 — they're
    complementary events under the same risk-neutral measure."""
    params = dict(S=100, K=105, T=0.5, r=0.04, q=0.0, sigma=0.25)
    p_call_itm = itm_probability(**params, option_type="call")
    p_put_itm = itm_probability(**params, option_type="put")
    assert 0.0 <= p_call_itm <= 1.0
    assert 0.0 <= p_put_itm <= 1.0
    assert p_call_itm + p_put_itm == pytest.approx(1.0, abs=1e-12)


def test_compute_d1_d2_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        compute_d1_d2(S=100, K=100, T=0, r=0.05, q=0, sigma=0.2)  # T=0
    with pytest.raises(ValueError):
        compute_d1_d2(S=100, K=100, T=1, r=0.05, q=0, sigma=0)  # sigma=0


class TestBinomialTree:
    params = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25)

    def test_european_binomial_converges_to_black_scholes_no_dividend(self):
        for option_type in ("call", "put"):
            bs = black_scholes_price(**self.params, q=0.0, option_type=option_type)
            crr = binomial_crr_price(**self.params, q=0.0, option_type=option_type, american=False, n_steps=500)
            assert crr == pytest.approx(bs, abs=0.05)

    def test_european_binomial_converges_to_black_scholes_with_dividend(self):
        for option_type in ("call", "put"):
            bs = black_scholes_price(**self.params, q=0.03, option_type=option_type)
            crr = binomial_crr_price(**self.params, q=0.03, option_type=option_type, american=False, n_steps=500)
            assert crr == pytest.approx(bs, abs=0.05)

    def test_american_call_equals_european_when_no_dividend(self):
        """It is never optimal to early-exercise a call on a non-dividend
        stock (you'd throw away remaining time value for nothing) — so the
        American and European CRR prices must match exactly here. If they
        don't, the early-exercise check in binomial_crr_price is broken."""
        european = binomial_crr_price(**self.params, q=0.0, option_type="call", american=False, n_steps=300)
        american = binomial_crr_price(**self.params, q=0.0, option_type="call", american=True, n_steps=300)
        assert american == pytest.approx(european, abs=1e-9)

    def test_american_put_has_positive_early_exercise_premium(self):
        """Puts (unlike calls) can have real early-exercise value — a deep
        ITM put on a non-dividend stock is the classic case, since you'd
        rather collect the strike now and earn interest on it. American
        should therefore be strictly worth more than European here."""
        deep_itm_params = dict(S=50.0, K=100.0, T=1.0, r=0.08, sigma=0.20)
        european = binomial_crr_price(**deep_itm_params, q=0.0, option_type="put", american=False, n_steps=300)
        american = binomial_crr_price(**deep_itm_params, q=0.0, option_type="put", american=True, n_steps=300)
        assert american > european

    def test_rejects_non_positive_steps(self):
        with pytest.raises(ValueError):
            binomial_crr_price(**self.params, q=0.0, option_type="call", n_steps=0)


class TestImpliedVolatility:
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("true_sigma", [0.15, 0.30, 0.60, 1.10])
    def test_round_trip_recovers_known_sigma(self, option_type, true_sigma):
        """Price at a known sigma, then invert — should recover it, whether
        Newton-Raphson converges cleanly or falls back to Brent's method."""
        params = dict(S=100.0, K=100.0, T=0.5, r=0.03, q=0.0)
        price = black_scholes_price(**params, sigma=true_sigma, option_type=option_type)
        recovered = implied_volatility(price, **params, option_type=option_type)
        assert recovered == pytest.approx(true_sigma, abs=1e-4)

    def test_low_vega_regime_still_recovers_correctly(self):
        """Short-dated, meaningfully OTM options sit in the low-vega regime
        where Newton-Raphson is unreliable and the Brent's-method fallback
        matters — but the price here is still large enough to be a real,
        quotable price (not underflowed numerical noise), so recovery must
        still be accurate. (Going even further OTM makes the true price
        underflow toward machine noise, at which point IV is genuinely
        unrecoverable from the price alone — a real market maker couldn't
        back it out either, so that's not a case worth asserting on.)"""
        params = dict(S=100.0, K=115.0, T=0.05, r=0.03, q=0.0)
        true_sigma = 0.15
        price = black_scholes_price(**params, sigma=true_sigma, option_type="call")
        recovered = implied_volatility(price, **params, option_type="call")
        assert recovered == pytest.approx(true_sigma, abs=1e-3)
