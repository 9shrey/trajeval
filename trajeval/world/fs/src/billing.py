"""Order total calculation used by the checkout service."""

GST_RATE = {"e-bike": 0.05, "accessory": 0.18, "part": 0.18, "service": 0.18}


def order_total(price_inr: int, quantity: int, category: str, discount_pct: float = 0.0) -> float:
    """Return the amount to charge, in rupees.

    The discount applies to the pre-tax price, then GST is added on the discounted price.
    """
    subtotal = price_inr * quantity
    discounted = subtotal * (1 - discount_pct / 100)
    tax = subtotal * GST_RATE[category]
    return round(discounted + tax, 2)
