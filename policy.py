"""Refund policy, enforced in code. The model never decides whether a refund is allowed."""

from datetime import date

import store
from schemas import Order, RefundDecision

POLICY_SOURCE_ID = "POLICY-REFUND"
POLICY_TEXT = (
    f"Refunds are allowed only when the order status is 'delivered', it was delivered no more than "
    f"{store.REFUND_WINDOW_DAYS} days ago, and the order total is ${store.MAX_AUTO_REFUND:.0f} or less. "
    f"Refunds over ${store.MAX_AUTO_REFUND:.0f} need a human agent, so a support ticket is opened."
)


def check_refund_eligibility(order: Order, today: date = store.TODAY) -> RefundDecision:
    """Checks run in order: status, then the 30-day window, then the amount.

    The amount check comes last on purpose: a human can approve a large refund,
    but not one the basic policy already rules out, so no ticket is opened for those.
    """
    oid = order.order_id
    if order.status == "refunded":
        return RefundDecision(allowed=False, code="already_refunded", reason=f"Order {oid} has already been refunded.")
    if order.status != "delivered" or order.delivered_on is None:
        return RefundDecision(
            allowed=False,
            code="not_delivered",
            reason=f"Order {oid} has status '{order.status}'. Refunds are only possible after delivery.",
        )

    days = (today - order.delivered_on).days
    if days > store.REFUND_WINDOW_DAYS:
        return RefundDecision(
            allowed=False,
            code="outside_window",
            reason=(
                f"Order {oid} was delivered on {order.delivered_on.isoformat()} ({days} days ago), "
                f"outside the {store.REFUND_WINDOW_DAYS}-day refund window."
            ),
            days_since_delivery=days,
        )
    if order.total > store.MAX_AUTO_REFUND:
        return RefundDecision(
            allowed=False,
            code="over_limit",
            reason=(
                f"Order {oid} totals ${order.total:.2f}, over the ${store.MAX_AUTO_REFUND:.0f} limit for "
                f"automatic refunds, so a human agent must approve it."
            ),
            days_since_delivery=days,
        )
    return RefundDecision(
        allowed=True, code="eligible", reason=f"Order {oid} meets the refund policy.", days_since_delivery=days
    )
