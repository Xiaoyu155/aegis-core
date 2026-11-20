"""Domain invariant tripwires plus baseline-differential L3 integration."""
import subprocess

import pytest

from aegis_core.invariant_guard import inspect_source, scan_changed_files


def function(body, name="run", args=""):
    normalized = [line[4:] if line.startswith("    ") else line for line in body.splitlines()]
    return f"def {name}({args}):\n" + "\n".join("    " + line for line in normalized) + "\n"


VIOLATIONS = [
    (function("client.submit_order(quantity=quantity)", "submit_order", "quantity"), "FINANCE_UNGUARDED_SIZE"),
    (function("exchange.create_order(amount=amount)", "place_order", "amount"), "FINANCE_UNGUARDED_SIZE"),
    (function("broker.place_order(size=size)", "create_order", "size"), "FINANCE_UNGUARDED_SIZE"),
    (function("return capital / leverage", "calculate_leverage", "capital, leverage"), "FINANCE_UNGUARDED_LEVERAGE"),
    (function("return float(balance)", "balance_value", "balance"), "FINANCE_BINARY_FLOAT_MONEY"),
    (function("return float(price)", "price_value", "price"), "FINANCE_BINARY_FLOAT_MONEY"),
    (function("return float(amount)", "amount_value", "amount"), "FINANCE_BINARY_FLOAT_MONEY"),
    (function("return float(quantity)", "quantity_value", "quantity"), "FINANCE_BINARY_FLOAT_MONEY"),
    (function("return float(equity)", "equity_value", "equity"), "FINANCE_BINARY_FLOAT_MONEY"),
    (function("return float(notional)", "notional_value", "notional"), "FINANCE_BINARY_FLOAT_MONEY"),
    (function("return float(pnl)", "pnl_value", "pnl"), "FINANCE_BINARY_FLOAT_MONEY"),
    (function("return float(fee)", "fee_value", "fee"), "FINANCE_BINARY_FLOAT_MONEY"),
    (function("db.query('INSERT INTO trades(id) VALUES (1)')\n"
              "db.query('UPDATE trades SET status = \\'open\\' WHERE id=1')", args="db"),
     "DATABASE_MULTI_WRITE_WITHOUT_TRANSACTION"),
    (function("db.execute('DELETE FROM records WHERE id=1')\n"
              "db.execute('INSERT INTO records(id) VALUES (1)')", args="db"),
     "DATABASE_MULTI_WRITE_WITHOUT_TRANSACTION"),
    (function("db.run('UPDATE jobs SET status=\\'done\\' WHERE id=1')\n"
              "db.run('INSERT INTO audit_log(id) VALUES (1)')", args="db"),
     "DATABASE_MULTI_WRITE_WITHOUT_TRANSACTION"),
    (function("db.query('SELECT * FROM users')", args="db"), "DATABASE_UNBOUNDED_SCAN"),
    (function("db.execute('SELECT * FROM orders')", args="db"), "DATABASE_UNBOUNDED_SCAN"),
    (function("db.query('SELECT * FROM tasks')", args="db"), "DATABASE_UNBOUNDED_SCAN"),
    (function("db.query('SELECT * FROM sessions')", args="db"), "DATABASE_UNBOUNDED_SCAN"),
    (function("db.query('SELECT * FROM records')", args="db"), "DATABASE_UNBOUNDED_SCAN"),
    (function("db.query('UPDATE orders SET status=\\'done\\' WHERE id=order_id')", args="db, order_id"),
     "STATE_TRANSITION_NOT_COMPARE_AND_SET"),
    (function("db.query('UPDATE tasks SET status=\\'done\\' WHERE task_id=task_id')", args="db, task_id"),
     "STATE_TRANSITION_NOT_COMPARE_AND_SET"),
    (function("db.query('UPDATE jobs SET status=\\'closed\\' WHERE id=job_id')", args="db, job_id"),
     "STATE_TRANSITION_NOT_COMPARE_AND_SET"),
    (function("db.query('UPDATE trades SET status=\\'closed\\' WHERE id=trade_id')", args="db, trade_id"),
     "STATE_TRANSITION_NOT_COMPARE_AND_SET"),
    (function("requests.get(url)", args="url"), "NETWORK_REQUEST_NO_TIMEOUT"),
    (function("requests.post(url, json=payload)", args="url, payload"), "NETWORK_REQUEST_NO_TIMEOUT"),
    (function("httpx.get(url)", args="url"), "NETWORK_REQUEST_NO_TIMEOUT"),
    (function("httpx.request('GET', url)", args="url"), "NETWORK_REQUEST_NO_TIMEOUT"),
    (function("urllib3.request('GET', url)", args="url"), "NETWORK_REQUEST_NO_TIMEOUT"),
    (function("fetch(url)", args="url"), "NETWORK_FETCH_NO_ABORT_SIGNAL"),
    (function("for attempt in range(4):\n        requests.post(url, json=payload)", args="url, payload"),
     "NETWORK_REQUEST_NO_TIMEOUT"),
    (function("for attempt in range(4):\n        requests.get(url, timeout=1)", args="url"),
     "NETWORK_RETRY_WITHOUT_BACKOFF"),
    ("API_KEY = 'live-secret-value'\n", "HARDCODED_CREDENTIAL"),
    ("password = 'literal-password'\n", "HARDCODED_CREDENTIAL"),
    ("settings = {'private_key': 'private-value'}\n", "HARDCODED_CREDENTIAL"),
    (function("handle = open(path)", args="path"), "RESOURCE_OPEN_NOT_CLOSED"),
    (function("lock.acquire()", args="lock"), "RESOURCE_LOCK_NOT_RELEASED"),
    (function("connection = pool.connect()", args="pool"), "RESOURCE_CONNECTION_NOT_RELEASED"),
    (function("socket = client.create_connection(address)", args="client, address"),
     "RESOURCE_CONNECTION_NOT_RELEASED"),
]


@pytest.mark.parametrize("source,expected_code", VIOLATIONS)
def test_high_risk_domain_pattern_has_location_and_severity(source, expected_code):
    findings = inspect_source(source, "finance_or_service.py")
    if expected_code is None:
        assert any(item["code"] == "FINANCE_UNGUARDED_LEVERAGE" for item in findings)
        return
    finding = next(item for item in findings if item["code"] == expected_code)
    assert finding["path"] == "finance_or_service.py"
    assert finding["line"] > 0
    assert finding["severity"] in {"high", "medium"}
    assert finding["domain"] in {"finance", "database", "external_api", "resource"}


BENIGN = [
    function("if quantity <= 0:\n        raise ValueError('quantity')\n    return client.submit_order(quantity=quantity)",
             "submit_order", "quantity, client"),
    function("if amount <= 0:\n        raise ValueError('amount')\n    return exchange.create_order(amount=amount)",
             "place_order", "amount, exchange"),
    function("if leverage <= 0:\n        raise ValueError('leverage')\n    return capital / leverage",
             "calculate_leverage", "capital, leverage"),
    function("return Decimal(balance)", "balance_value", "balance"),
    function("return amount * Decimal('1.0001')", "amount_value", "amount"),
    function("with transaction(db):\n        db.query('INSERT INTO trades(id) VALUES (1)')\n"
             "        db.query('UPDATE trades SET status=\\'open\\' WHERE id=1 AND status=\\'new\\'')", args="db"),
    function("db.query('SELECT * FROM orders WHERE user_id=? LIMIT 25')", args="db"),
    function("db.query('SELECT * FROM users WHERE id=?')", args="db"),
    function("db.query('UPDATE jobs SET status=\\'done\\' WHERE id=? AND status=\\'running\\'')", args="db"),
    function("requests.get(url, timeout=2)", args="url"),
    function("httpx.post(url, timeout=2)", args="url"),
    function("fetch(url, signal=controller.signal)", args="url, controller"),
    function("for attempt in range(4):\n        requests.get(url, timeout=1)\n        time.sleep(0.1)", args="url"),
    function("with open(path, 'rb') as handle:\n        return handle.read(10)", args="path"),
    function("lock.acquire()\n    try:\n        return work()\n    finally:\n        lock.release()", args="lock"),
    function("connection = pool.connect()\n    try:\n        return connection.execute('SELECT 1')\n    finally:\n        connection.close()", args="pool"),
    "API_KEY = os.environ['API_KEY']\n",
    "secret_env = 'REQUIRED_TOKEN'\n",
    "api_key_name = 'API_KEY'\n",
    "password_path = '/run/secrets/database'\n",
    "settings = {'password': os.environ.get('DB_PASSWORD')}\n",
    function("return value / count if count > 0 else 0", "average", "value, count"),
    function("return fetch(url, signal=abort_controller.signal)", args="url, abort_controller"),
    function("db.query('DELETE FROM records WHERE id=?')", args="db"),
    function("lock.release()", args="lock"),
    function("return int(balance)", "balance_units", "balance"),
    "def example():\n    return {'access_token': token}\n",
    function("if size < 0:\n        raise ValueError()\n    return broker.place_order(size=size)",
             "create_order", "size, broker"),
    function("if margin == 0:\n        raise ValueError()\n    return capital / margin", "calc", "capital, margin"),
    function("if leverage == 0:\n        raise ValueError()\n    return capital / leverage",
             "calculate_leverage", "capital, leverage"),
    function("if not math.isfinite(leverage) or leverage <= 0:\n        raise ValueError()\n    return capital / leverage",
             "calculate_leverage", "capital, leverage"),
    function("requests.get(url, timeout=3, params={'page': page})", args="url, page"),
    function("fetch(url, signal=signal)", args="url, signal"),
    function("with open(path) as stream:\n        return stream.readline()", args="path"),
    function("connection = pool.connect()\n    connection.close()", args="pool"),
    function("db.query('SELECT * FROM tasks WHERE done=false LIMIT 10')", args="db"),
    function("db.query('UPDATE tasks SET status=\\'done\\' WHERE id=1 AND status=\\'running\\'')", args="db"),
]


@pytest.mark.parametrize("source", BENIGN)
def test_guard_does_not_report_visibly_guarded_or_bounded_operations(source):
    assert inspect_source(source, "safe.py") == []


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _init_repo(root, source):
    root.mkdir()
    (root / "service.py").write_text(source, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Guard Test")
    _git(root, "config", "user.email", "guard@example.invalid")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "baseline")
    return _git(root, "rev-parse", "HEAD")


def test_l3_audit_blocks_new_network_invariant_at_physical_file_line(tmp_path):
    from aegis_core.reviewer import audit_candidate

    root = tmp_path / "repo"
    base = _init_repo(root, "def call_api(url):\n    return requests.get(url, timeout=2)\n")
    (root / "service.py").write_text("def call_api(url):\n    return requests.get(url)\n", encoding="utf-8")
    report = audit_candidate(root, base, ["service.py"])
    assert report["passed"] is False
    finding = next(item for item in report["findings"] if item["code"] == "NETWORK_REQUEST_NO_TIMEOUT")
    assert finding["path"] == "service.py" and finding["line"] == 2
    assert report["invariant_guard"]["passed"] is False


def test_baseline_invariant_findings_are_not_reintroduced_as_candidate_regressions(tmp_path):
    root = tmp_path / "repo"
    base = _init_repo(root, "def call_api(url):\n    return requests.get(url)\n")
    report = scan_changed_files(root, ["service.py"], baseline=base)
    assert report["passed"] is True
    assert report["scanned_files"] == ["service.py"]


def test_l3_accepts_multiline_finite_positive_leverage_guard(tmp_path):
    from aegis_core.reviewer import audit_candidate

    root = tmp_path / "repo"
    base = _init_repo(root, "def max_stake(notional, leverage):\n    return notional / leverage\n")
    (root / "service.py").write_text(
        "import math\n"
        "def max_stake(notional, leverage):\n"
        "    if isinstance(leverage, bool) or not isinstance(leverage, (int, float)) or not math.isfinite(leverage):\n"
        "        raise ValueError('invalid leverage')\n"
        "    if leverage <= 0:\n"
        "        raise ValueError('invalid leverage')\n"
        "    return notional / leverage\n",
        encoding="utf-8",
    )

    report = audit_candidate(root, base, ["service.py"])

    assert report["passed"] is True
    assert report["findings"] == []


@pytest.mark.parametrize("dangerous", VIOLATIONS[:20])
def test_differential_guard_compares_only_new_findings(tmp_path, dangerous):
    source, _ = dangerous
    root = tmp_path / "repo"
    base = _init_repo(root, source)
    report = scan_changed_files(root, ["service.py"], baseline=base)
    assert report["passed"] is True


@pytest.mark.parametrize("source", ["def broken(:\n", "async def broken(:\n"])
def test_invalid_python_is_reported_at_the_parser_location(source):
    findings = inspect_source(source, "broken.py")
    assert findings[0]["code"] == "SOURCE_SYNTAX_ERROR"
    assert findings[0]["line"] == 1


def test_guard_skips_non_python_and_symlinked_candidate_files(tmp_path):
    root = tmp_path / "repo"
    base = _init_repo(root, "value = 1\n")
    (root / "safe.js").write_text("const value = 1;\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("password = 'ignored'\n", encoding="utf-8")
    (root / "linked.py").symlink_to(outside)
    report = scan_changed_files(root, ["safe.js", "linked.py"], baseline=base)
    assert report["passed"] is True
    assert report["scanned_files"] == ["safe.js"]

