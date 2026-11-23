"""The task suite itself is tested: every gold trajectory replays cleanly against a fresh
world and its gold answer passes the checker, and key expected values are re-derived
independently from the fixtures (so a typo in a YAML literal cannot hide)."""

import json

import pytest

from trajeval.tasks import check_task, extract_numbers, load_suite
from trajeval.tools.base import OK, TOOL_ERROR
from trajeval.tools.world import _fs_fixture, make_env

TASKS = load_suite()


def test_suite_shape():
    cats = {}
    for t in TASKS:
        cats[t.category] = cats.get(t.category, 0) + 1
        if t.is_judged:
            assert t.rubric and t.reference, t.id
        else:
            assert t.gold_answer is not None, t.id
    assert cats["open_ended"] >= 20 and len(TASKS) >= 60, cats


@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_gold_trajectory_replays_and_passes(task):
    env = make_env()
    for i, c in enumerate(task.gold_calls):
        rec = env.call(c["tool"], c["args"], step=i)
        allowed = (OK, TOOL_ERROR) if task.category == "trap" else (OK,)
        assert rec.status in allowed, (task.id, c, rec.result)
    if task.is_judged:
        return
    correct, details = check_task(task, task.gold_answer, env.world)
    assert correct, (task.id, details)


@pytest.mark.parametrize("task", [t for t in TASKS if not t.is_judged and t.category != "action"],
                         ids=lambda t: t.id)
def test_empty_answer_fails(task):
    """No checker may pass an empty answer on an untouched world (guards against vacuous checks)."""
    env = make_env()
    correct, _ = check_task(task, "", env.world)
    assert correct is False


def test_expected_values_independently():
    fs = _fs_fixture()
    log = fs["/logs/app.log"].splitlines()
    errors = [ln for ln in log if " ERROR " in ln]
    assert len(errors) == 9
    assert sum(" payment " in ln for ln in errors) == 4
    inv = [json.loads(fs[p]) for p in fs if p.startswith("/invoices/")]
    rate = {"INR": 1, "USD": 84.0, "EUR": 91.5}
    assert sum(i["amount"] * rate[i["currency"]] for i in inv if i["status"] == "unpaid") == 1048825
    stock = [ln.split(",") for ln in fs["/data/inventory.csv"].splitlines()[1:]]
    assert sum(int(u) for p, _, u in stock if p == "Kestrel Helmet") == 225
    beng = {p: int(u) for p, w, u in stock if w == "Bengaluru"}
    assert min(beng, key=beng.get) == "Trailblazer X1 Pro" and beng["Trailblazer X1 Pro"] == 6

    db = make_env().world.db
    q = lambda s: db.execute(s).fetchall()  # noqa: E731
    j = "FROM orders o JOIN customers c ON c.id=o.customer_id JOIN products p ON p.id=o.product_id WHERE o.status!='cancelled'"
    assert q(f"SELECT SUM(o.quantity*p.price_inr) {j} AND c.city='Bengaluru'")[0][0] == 984480
    top = q(f"SELECT c.name, SUM(o.quantity*p.price_inr) s {j} GROUP BY c.id ORDER BY s DESC LIMIT 2")
    assert top[0] == ("Lakshmi Krishnan", 239997) and top[1][1] == 239995  # a 2-rupee margin, on purpose
    assert q("SELECT COUNT(*) FROM tickets WHERE status IN ('open','in_progress')")[0][0] == 9


@pytest.mark.parametrize("text,nums", [
    ("Rs 2,39,997 in total", [239997]),
    ("about ₹1,048,825.", [1048825]),
    ("on 2025-05-19 it was 5 orders, 3 pending", [5, 3]),
    ("at 19:40 IST, 8999.85 rupees", [8999.85]),
    ("-3 and 6.5 kg", [-3, 6.5]),
])
def test_extract_numbers(text, nums):
    assert extract_numbers(text) == nums
