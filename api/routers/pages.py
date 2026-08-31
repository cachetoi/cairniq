import importlib.util
import logging
import os
import re
from datetime import datetime
import json
from pathlib import Path
from datetime import date

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from agent.logger import log_to_component
from agent.version import __version__
from tools.user_profile import get_active_profile, is_demo_mode, list_available_profiles

router = APIRouter()
templates = Jinja2Templates(directory="templates")
# Exposed as a Jinja global, not a per-page context key: the version label lives
# in base.html, which every page extends, so a route that builds its own context
# (login, the graph view) would otherwise render a blank version.
templates.env.globals["app_version"] = __version__
_TRUTHY = {"1", "true", "yes", "y", "on"}
_SUPPORTED_CURRENCIES = {"USD", "CAD", "EUR", "GBP", "AUD", "JPY"}
_CURRENCY_SYMBOLS = {
    "USD": "$",
    "CAD": "$",
    "EUR": "â‚¬",
    "GBP": "Â£",
    "AUD": "$",
    "JPY": "Â¥",
}
_LOCALE_DEFAULT_CURRENCY = {
    "English (Canada)": "CAD",
    "French (Canada)": "CAD",
    "English (United States)": "USD",
    "English (United Kingdom)": "GBP",
    "German (Frankfurt)": "EUR",
    "Japanese (Tokyo)": "JPY",
}


def _env_truthy(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in _TRUTHY


def _normalize_currency(value: str | None, default: str | None = None) -> str | None:
    code = str(value or "").strip().upper()
    if code in _SUPPORTED_CURRENCIES:
        return code
    return default


def _persisted_config_value(key: str) -> str | None:
    """Read a non-secret config value from the persisted .env FILE first.

    os.environ is per-process: a Settings save mutates only the handling worker's
    env, so other workers keep a stale snapshot. Reading the shared .env file makes
    locale/currency render the authoritative saved value regardless of which worker
    serves the request â€” fixing the intermittent "currency reset to USD". Falls back
    to os.environ when the file or key is absent.
    """
    try:
        from dotenv import dotenv_values
        val = dotenv_values(os.path.join(os.getcwd(), "user_data", ".env")).get(key)
        if val is not None and str(val).strip() != "":
            return val
    except Exception:
        pass
    return os.environ.get(key)


def _configured_locale() -> str:
    return _persisted_config_value("REGIONAL_LOCALE") or "English (Canada)"


def _configured_base_currency() -> str:
    """Delegates to tools.memory so the page and the memory store cannot diverge.

    They did diverge: this resolved an unset currency through the persisted .env
    and a locale default (CAD here), while `get_profile_base_currency` hardcoded
    USD â€” so a profile that never stated one read CAD on screen and was stamped
    USD in storage. Harmless until 4.5's wealth goal shipped, at which point a
    target typed as CAD would be scored as USD. One resolver, one answer.
    """
    from tools.memory import configured_base_currency
    return configured_base_currency()


def _guru_picks_enabled() -> bool:
    if not _env_truthy("CAIRNIQ_ENABLE_GURU_PICKS"):
        return False

    module_name = os.environ.get("CAIRNIQ_GURU_PICKS_TOOL", "").strip()
    if not module_name:
        return True
    return importlib.util.find_spec(module_name) is not None


def _format_annual_income(value: object, currency: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "Not Set"

    number_text = re.sub(r"[^0-9.]", "", text)
    if number_text:
        try:
            amount = float(number_text)
            symbol = _CURRENCY_SYMBOLS.get(currency, "$")
            return f"{symbol}{amount:,.0f} {currency}"
        except ValueError:
            pass
    return f"{text} {currency}"

def _format_sync_time(value: object) -> str:
    """Render a portfolio snapshot's as-of time for display.

    Prices shown in the UI come from a cached summary that can fall back to a
    Last-Known-Good snapshot of any age, so the page must state when the data
    was actually captured rather than implying it is live.
    """
    text = str(value or "").strip()
    if not text:
        return "Unknown"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return "Unknown"

    now = datetime.now(stamp.tzinfo) if stamp.tzinfo else datetime.now()
    if stamp.date() == now.date():
        return stamp.strftime("Today %H:%M")
    return stamp.strftime("%b %d, %H:%M")


def _mask_key(key: str) -> str:
    if not key or len(key) < 8: return key
    return key[:4] + "*" * (len(key) - 8) + key[-4:]

def _mask_secret(name: str) -> str:
    """Masked display for a secret, read keychain-aware.

    Secrets live in the OS keychain; the .env secret lines are typically empty
    after migration, so reading os.environ alone shows a misleading blank even
    when the key is set. Fall back to the keychain so a stored secret shows as
    set (masked) instead of empty.
    """
    val = os.environ.get(name, "")
    if not val:
        try:
            from tools.secrets_store import get_secret
            val = get_secret(name) or ""
        except Exception:
            val = ""
    return _mask_key(val)

def _env_settings() -> dict:
    # Non-secret config vars (model IDs, endpoints, provider) are read from the
    # .env file directly so Settings always shows the current saved state, not the
    # stale os.environ snapshot from server startup.  Secret vars (API keys) are
    # read keychain-aware via _mask_secret (os.environ, then the OS keychain) so a
    # stored secret shows as set even when its .env line is empty after migration.
    try:
        from dotenv import dotenv_values
        _file: dict = dotenv_values(os.path.join(os.getcwd(), "user_data", ".env"))
    except Exception:
        _file = {}

    def _cfg(key: str, default: str = "") -> str:
        """Read a non-secret config var: .env file first, os.environ fallback."""
        v = _file.get(key)
        return v if v is not None else os.environ.get(key, default)

    # Broker credentials are per-profile (the default profile keeps the legacy
    # global values), so resolve them through the broker accessor rather than
    # the global .env / os.environ so Settings shows the active profile's state.
    from tools.broker_credentials import get_broker_secret, get_broker_setting

    def _mask_broker_secret(name: str) -> str:
        return _mask_key(get_broker_secret(name) or "")

    base_currency = _configured_base_currency()
    settings = {
        "LLM_PROVIDER": _cfg("LLM_PROVIDER", "bedrock"),
        "AIDLC_MODEL_ID": _cfg("AIDLC_MODEL_ID", ""),
        "AIDLC_SONNET_MODEL_ID": _cfg("AIDLC_SONNET_MODEL_ID", ""),
        "AIDLC_EMBED_MODEL_ID": _cfg("AIDLC_EMBED_MODEL_ID", ""),
        "REGIONAL_LOCALE": _configured_locale(),
        "BASE_CURRENCY": base_currency,
        "CAIRNIQ_ENABLE_GURU_PICKS": "true" if _guru_picks_enabled() else "false",
        "OPENAI_API_KEY": _mask_secret("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": _mask_secret("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": _mask_secret("GOOGLE_API_KEY"),
        "GOOGLE_SERVICE_ACCOUNT_KEY": _mask_secret("GOOGLE_SERVICE_ACCOUNT_KEY"),
        "GOOGLE_CLOUD_PROJECT": _cfg("GOOGLE_CLOUD_PROJECT", ""),
        "GOOGLE_CLOUD_LOCATION": _cfg("GOOGLE_CLOUD_LOCATION", ""),
        "AZURE_OPENAI_API_KEY": _mask_secret("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_ENDPOINT": _cfg("AZURE_OPENAI_ENDPOINT", ""),
        "AZURE_OPENAI_API_VERSION": _cfg("AZURE_OPENAI_API_VERSION", ""),
        "AZURE_OPENAI_API_KEY_FAST": _mask_secret("AZURE_OPENAI_API_KEY_FAST"),
        "AZURE_OPENAI_ENDPOINT_FAST": _cfg("AZURE_OPENAI_ENDPOINT_FAST", ""),
        "ALPHA_VANTAGE_API_KEY": _mask_secret("ALPHA_VANTAGE_API_KEY"),
        "FMP_API_KEY": _mask_secret("FMP_API_KEY"),
        "FRED_API_KEY": _mask_secret("FRED_API_KEY"),
        "AWS_ACCESS_KEY_ID": _mask_secret("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": _mask_secret("AWS_SECRET_ACCESS_KEY"),
        "AWS_REGION": _cfg("AWS_REGION", "us-east-1"),
        "QUESTRADE_REFRESH_TOKEN": _mask_broker_secret("QUESTRADE_REFRESH_TOKEN"),
        "QUESTRADE_ENABLED": get_broker_setting("QUESTRADE_ENABLED", "false"),
        "QUESTRADE_ACCOUNT_OWNER": get_broker_setting("QUESTRADE_ACCOUNT_OWNER", ""),
        "TAVILY_API_KEY": _mask_secret("TAVILY_API_KEY"),
        "FINNHUB_API_KEY": _mask_secret("FINNHUB_API_KEY"),
        "POLYGON_API_KEY": _mask_secret("POLYGON_API_KEY"),
        "ALPACA_API_KEY": _mask_broker_secret("ALPACA_API_KEY"),
        "ALPACA_SECRET_KEY": _mask_broker_secret("ALPACA_SECRET_KEY"),
        "ALPACA_PAPER_MODE": get_broker_setting("ALPACA_PAPER_MODE", "true"),
        "SCHEDULER_ENABLED": get_broker_setting("SCHEDULER_ENABLED", "false"),
    }

    from tools.scheduler import get_scheduler_cooldowns, get_scheduler_settings
    scheduler_jobs = get_scheduler_settings()
    scheduler_cooldowns = get_scheduler_cooldowns()
    for job_name, enabled in scheduler_jobs.items():
        settings[f"SCHEDULER_JOB_{job_name}"] = "true" if enabled else "false"
        if job_name in scheduler_cooldowns:
            settings[f"SCHEDULER_COOLDOWN_{job_name}"] = str(int(scheduler_cooldowns[job_name]))

    # Per-provider remembered model ids, so switching LLM_PROVIDER in Settings never
    # loses the models configured for another provider. Any scoped var that is unset
    # falls back to the generic AIDLC_MODEL_ID (migration: existing users only had
    # the generic var before per-provider scoped vars were introduced, typically
    # containing a Bedrock model id). After one save all scoped vars are written.
    for prov in ("bedrock", "openai", "anthropic", "azure", "google", "vertexai"):
        suffix = prov.upper()
        primary = _cfg(f"AIDLC_MODEL_ID_{suffix}", "")
        fast = _cfg(f"AIDLC_SONNET_MODEL_ID_{suffix}", "")
        embed = _cfg(f"AIDLC_EMBED_MODEL_ID_{suffix}", "")
        # Apply generic fallback to every provider whose scoped var is empty.
        # This restores model ids for providers that were configured before
        # per-provider scoped vars existed (e.g. Bedrock when switching to Azure).
        primary = primary or settings["AIDLC_MODEL_ID"]
        fast = fast or settings["AIDLC_SONNET_MODEL_ID"]
        embed = embed or settings["AIDLC_EMBED_MODEL_ID"]
        settings[f"AIDLC_MODEL_ID_{suffix}"] = primary
        settings[f"AIDLC_SONNET_MODEL_ID_{suffix}"] = fast
        settings[f"AIDLC_EMBED_MODEL_ID_{suffix}"] = embed

    if is_demo_mode():
        for key in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "GOOGLE_SERVICE_ACCOUNT_KEY",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY_FAST",
            "ALPHA_VANTAGE_API_KEY",
            "FMP_API_KEY",
            "FRED_API_KEY",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "QUESTRADE_REFRESH_TOKEN",
            "TAVILY_API_KEY",
            "FINNHUB_API_KEY",
            "POLYGON_API_KEY",
            "ALPACA_API_KEY",
            "ALPACA_SECRET_KEY",
        ):
            settings[key] = ""
        settings["QUESTRADE_ENABLED"] = "false"
        settings["QUESTRADE_ACCOUNT_OWNER"] = "Demo User"
        settings["ALPACA_PAPER_MODE"] = "true"
        settings["CAIRNIQ_ENABLE_GURU_PICKS"] = "false"

    return settings

def get_dashboard_context(request: Request):
    from tools.memory import get_active_theses, load_memory
    from tools.portfolio_csv import get_portfolio_summary

    try:
        # Optimization: On initial page load (GET /), we don't block for the portfolio summary
        # unless it's already in the cache. This prevents the "stuck" home page.
        is_api_call = request.url.path.startswith("/api/")
        summary = get_portfolio_summary(force=is_api_call)
    except Exception as e:
        log_to_component("server", "Dashboard", f"Dashboard Context Error: {e}", level=logging.ERROR)
        summary = {"holdings": [], "total_value_cad": 0, "percent_return": 0, "sync_errors": [str(e)]}

    total_val = summary.get("total_value_cad", 0)
    pct_return = summary.get("percent_return", 0)
    theses = get_active_theses()
    memory = load_memory()

    # --- DYNAMIC PROFILE ENRICHMENT ---
    # Only re-extract from facts if the field is truly empty (None or empty string)
    # This ensures manual context updates take absolute precedence.
    profile = memory.get("user_profile", {}).copy()
    facts = memory.get("key_facts", [])

    # Age Re-extraction (Fallback only)
    # NOTE: `re` is imported at module scope on purpose. A local `import re`
    # inside any branch here makes `re` function-local for the WHOLE function,
    # so a profile that skips this branch (age set) but enters a later one
    # (income unset) hits an UnboundLocalError and 500s the dashboard.
    if not profile.get("age"):
        for fact in facts:
            match = re.search(r"(?i)(?:I\s+am|age\s+is|age)\s+(\d+)", fact)
            if not match:
                match = re.search(r"(?i)(\d+)\s+years\s+old", fact)
            if match:
                profile["age"] = match.group(1)
                break

    # Income Re-extraction (Fallback only)
    if not profile.get("annual_income"):
        for fact in facts:
            match = re.search(r"(?i)(?:income|make)\s+(?:of|close\s+to|around)?\s*[\$]?\s*([\d,]+)", fact)
            if match:
                profile["annual_income"] = f"${match.group(1)}"
                break

    if not profile.get("retirement_age"):
        for fact in facts:
            match = re.search(r"(?i)(?:retire\s+at)\s+(\d+)", fact)
            if not match:
                match = re.search(r"(?i)(?:retire\s+in|retirement\s+goal)\s+(\d+)\s+years", fact)
            if not match:
                match = re.search(r"(?i)(?:retire\s+in)\s+(\d{4})", fact)

            if match:
                profile["retirement_age"] = match.group(1)
                break

    base_currency = _normalize_currency(profile.get("base_currency"), _configured_base_currency())
    profile["base_currency"] = base_currency
    profile["annual_income_currency"] = base_currency
    profile["annual_income_display"] = _format_annual_income(profile.get("annual_income"), base_currency)

    # --- GRAPH DATA INJECTION ---
    # Force reload from disk to pick up any changes without server restart
    import networkx as nx

    from tools.graph_memory import graph_memory
    graph_memory.load()
    graph_data = nx.node_link_data(graph_memory.graph)

    # Filter graph data for cleaner D3 visualization â€” remove Unknown-typed nodes
    all_nodes = graph_data.get("nodes", [])
    clean_nodes = [n for n in all_nodes if n.get("type") != "Unknown"]
    clean_node_ids = {n["id"] for n in clean_nodes}
    all_links = graph_data.get("links") or graph_data.get("edges") or []
    clean_links = [l for l in all_links
                   if l.get("source") in clean_node_ids and l.get("target") in clean_node_ids]

    # Build a symbol â†’ current_price map for thesis upside calculations.
    # current_price is a formatted string ("$150.00"); current_price_raw is its
    # numeric twin. Fall back to parsing the string so Last-Known-Good snapshots
    # written before current_price_raw existed still populate the map.
    holdings_price_map = {}
    for h in summary.get("holdings", []):
        sym = (h.get("symbol") or "").upper()
        if not sym:
            continue
        price = h.get("current_price_raw")
        if not isinstance(price, (int, float)):
            try:
                price = float(str(h.get("current_price") or "").replace("$", "").replace(",", ""))
            except ValueError:
                continue
        if price > 0:
            holdings_price_map[sym] = price

    return {
        "request": request,
        "total_value_cad": f"${total_val:,.0f}" if total_val else "$0",
        "percent_return": f"{pct_return:+.1f}%" if pct_return else "+0.0%",
        "sync_errors": summary.get("sync_errors", []),
        "last_sync_display": _format_sync_time(summary.get("last_sync_time")),
        "is_stale": bool(summary.get("is_stale")),
        "theses": theses,
        "holdings": summary.get("holdings", []),
        "holdings_price_map": holdings_price_map,
        # Positions excluded from total_value_cad above because nothing could value
        # them. Surfaced so the editor can say which rows the headline figure is
        # missing â€” a total quietly short by a pension is the failure this prevents.
        "unvalued_holdings": summary.get("unvalued_holdings", []),
        "unvalued_notice": summary.get("unvalued_notice", ""),
        "profile": profile,
        # Roadmap 3.7. Read straight off memory rather than through get_playbook()
        # so a template default can never diverge from what the sentinel reads
        # back during a drawdown â€” {} renders every field blank, which is the
        # honest state when nothing has been agreed.
        "playbook": memory.get("drawdown_playbook") or {},
        # Roadmap 2.2/4.4's caps, read raw for the same reason as the playbook:
        # {} renders every field blank, which is the honest state, and a template
        # default here would be the one thing this store forbids.
        "risk_constraints": memory.get("risk_constraints") or {},
        # Roadmap 4.4's drift target, read raw for the same reason as the two
        # above: {} renders blank, and blank is the honest state for a store
        # nothing may default.
        "target_allocation": memory.get("target_allocation") or {},
        "lessons": memory.get("lessons_learned", []),
        "facts": facts,
        "graph_nodes": clean_nodes,
        "graph_links": clean_links,
        # Expose active profile to all templates
        "active_profile": get_active_profile(),
        "available_profiles": list_available_profiles(),
        "logged_in_user": getattr(request.state, "user", None),
        "demo_mode": is_demo_mode(),
        "env_settings": _env_settings(),
        "enable_guru_picks": _guru_picks_enabled(),
    }


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    """Standalone login page â€” no sidebar, no dashboard context."""
    return templates.TemplateResponse(request=request, name="login.html", context={"request": request})


@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context=get_dashboard_context(request))

@router.get("/journal", response_class=HTMLResponse)
def journal(request: Request):
    from tools.trade_journal import get_trade_history
    context = get_dashboard_context(request)
    context["journal_history"] = get_trade_history()
    return templates.TemplateResponse(request=request, name="trade_journal.html", context=context)
BRIDGE_SETTINGS_PATH = r"C:\autotrader\bridge_settings.json"
BRIDGE_LEDGER_PATH = r"C:\autotrader\trade_ledger.json"


def _bridge_defaults():
    return {
        "environment": "PAPER",
        "execution_mode": "MANUAL",
        "order_size": 100.0,
        "allowed_action": "BUY",
        "minimum_confidence": "HIGH",
        "recommendation_age": "TODAY",
        "max_trades_per_day": 3,
        "max_dollars_per_day": 300.0,
        "duplicate_protection": True,
        "kill_switch": False,
    }


def _load_bridge_settings():
    import json as _json
    from pathlib import Path as _Path

    settings = _bridge_defaults()
    path = _Path(BRIDGE_SETTINGS_PATH)

    if path.exists():
        try:
            saved = _json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(saved, dict):
                settings.update(saved)
        except (_json.JSONDecodeError, OSError):
            pass

    # Hard safety invariants: UI cannot unlock these.
    settings["environment"] = "PAPER"
    settings["duplicate_protection"] = True
    settings["allowed_action"] = "BUY"
    settings["recommendation_age"] = "TODAY"

    return settings


def _save_bridge_settings(settings):
    import json as _json
    from pathlib import Path as _Path

    path = _Path(BRIDGE_SETTINGS_PATH)
    path.write_text(
        _json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )


def _bridge_context(request, saved=False, error=None):
    import json as _json
    from pathlib import Path as _Path
    from datetime import date as _date

    context = get_dashboard_context(request)
    settings = _load_bridge_settings()

    trades = []
    ledger_path = _Path(BRIDGE_LEDGER_PATH)

    if ledger_path.exists():
        try:
            loaded = _json.loads(ledger_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, list):
                trades = loaded
        except (_json.JSONDecodeError, OSError):
            trades = []

    today = _date.today().isoformat()
    todays_trades = [
        trade for trade in trades
        if str(trade.get("processed_at", "")).startswith(today)
    ]

    deployed_today = sum(
        float(trade.get("paper_amount", 0) or 0)
        for trade in todays_trades
    )

    recommendation_checks = []
    recommendation_error = None

    try:
        import requests as _requests

        response = _requests.get(
            "http://localhost:8000/api/export/advisor-scorecard?format=json",
            timeout=4,
        )
        response.raise_for_status()
        recommendation_data = response.json()
        recommendations = recommendation_data.get("recommendations", [])

        confidence_rank = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
        }

        required_confidence = str(
            settings.get("minimum_confidence", "HIGH")
        ).upper()

        required_rank = confidence_rank.get(
            required_confidence,
            3,
        )

        processed_ids = {
            str(entry.get("recommendation_id", ""))
            for entry in trades
        }

        # Show newest recommendations first.
        indexed_recommendations = list(enumerate(recommendations))
        indexed_recommendations.sort(
            key=lambda item: (
                str(item[1].get("date", "")),
                item[0],
            ),
            reverse=True,
        )

        for _, rec in indexed_recommendations[:10]:
            rec_date = str(rec.get("date", "")).strip()
            ticker = str(rec.get("ticker", "")).strip().upper()
            action = str(rec.get("action", "")).strip().upper()
            confidence = str(
                rec.get("confidence_grade", "")
            ).strip().upper()
            executed = rec.get("executed")

            rec_id = "|".join([
                rec_date,
                ticker,
                action,
                str(rec.get("price_at_advice", "")),
                confidence,
                str(rec.get("horizon", "")),
            ])

            blocked = []

            if settings.get("kill_switch"):
                blocked.append("Kill switch is ON")

            if rec_date != today:
                blocked.append("Recommendation is not from today")

            if action != "BUY":
                blocked.append("Action must be BUY")

            if confidence_rank.get(confidence, 0) < required_rank:
                blocked.append(
                    f"Confidence is below {required_confidence}"
                )

            if executed not in (None, False, "", 0):
                blocked.append("Recommendation is already marked executed")

            if rec_id in processed_ids:
                blocked.append("Already processed by the bridge")

            if len(todays_trades) >= int(
                settings.get("max_trades_per_day", 3)
            ):
                blocked.append("Daily trade limit reached")

            if (
                deployed_today
                + float(settings.get("order_size", 100) or 0)
                > float(settings.get("max_dollars_per_day", 300) or 0)
            ):
                blocked.append("Daily dollar limit would be exceeded")

            recommendation_checks.append({
                "ticker": ticker or "—",
                "action": action or "—",
                "confidence": confidence or "—",
                "date": rec_date or "—",
                "reason": rec.get("reason") or "",
                "price": rec.get("price_at_advice"),
                "eligible": not blocked,
                "blocked_reasons": blocked,
            })

    except Exception as exc:
        recommendation_error = str(exc)

    context.update({
        "bridge_settings": settings,
        "bridge_trades_today": len(todays_trades),
        "bridge_deployed_today": deployed_today,
        "bridge_last_trade": trades[-1] if trades else None,
        "bridge_settings_saved": saved,
        "bridge_settings_error": error,
        "bridge_recommendation_checks": recommendation_checks,
        "bridge_recommendation_error": recommendation_error,
    })

    return context


@router.get("/trading-bridge", response_class=HTMLResponse)
def trading_bridge(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="trading_bridge.html",
        context=_bridge_context(request),
    )


@router.post("/trading-bridge", response_class=HTMLResponse)
async def save_trading_bridge_settings(request: Request):
    from urllib.parse import parse_qs

    try:
        raw_body = (await request.body()).decode("utf-8")
        values = {
            key: items[-1]
            for key, items in parse_qs(raw_body).items()
            if items
        }

        execution_mode = str(values.get("execution_mode", "MANUAL")).upper()
        if execution_mode not in {"MANUAL", "AUTO"}:
            raise ValueError("Invalid trading mode.")

        minimum_confidence = str(
            values.get("minimum_confidence", "HIGH")
        ).upper()
        if minimum_confidence not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("Invalid confidence setting.")

        order_size = float(values.get("order_size", 100))
        max_trades = int(values.get("max_trades_per_day", 3))
        max_dollars = float(values.get("max_dollars_per_day", 300))

        if not 1 <= order_size <= 10000:
            raise ValueError("Order size must be between $1 and $10,000.")

        if not 1 <= max_trades <= 100:
            raise ValueError("Max trades per day must be between 1 and 100.")

        if not 1 <= max_dollars <= 100000:
            raise ValueError(
                "Max daily deployment must be between $1 and $100,000."
            )

        if order_size > max_dollars:
            raise ValueError(
                "Order size cannot exceed the max daily deployment."
            )

        settings = {
            "environment": "PAPER",
            "execution_mode": execution_mode,
            "order_size": round(order_size, 2),
            "allowed_action": "BUY",
            "minimum_confidence": minimum_confidence,
            "recommendation_age": "TODAY",
            "max_trades_per_day": max_trades,
            "max_dollars_per_day": round(max_dollars, 2),
            "duplicate_protection": True,
            "kill_switch": values.get("kill_switch") == "on",
        }

        _save_bridge_settings(settings)

        return templates.TemplateResponse(
            request=request,
            name="trading_bridge.html",
            context=_bridge_context(request, saved=True),
        )

    except (ValueError, TypeError) as exc:
        return templates.TemplateResponse(
            request=request,
            name="trading_bridge.html",
            context=_bridge_context(request, error=str(exc)),
            status_code=400,
        )

@router.get("/recommendations", response_class=HTMLResponse)
def recommendations(request: Request):
    return templates.TemplateResponse(request=request, name="recommendations.html", context=get_dashboard_context(request))

@router.get("/alerts", response_class=HTMLResponse)
def alerts(request: Request):
    return templates.TemplateResponse(request=request, name="alerts.html", context=get_dashboard_context(request))

@router.get("/review", response_class=HTMLResponse)
def weekly_review(request: Request):
    """The weekly one-page review.

    Rendered server-side rather than fetched: this page is meant to be read once
    a week and printed or kept, so it must not depend on client JS having run.
    """
    from tools.weekly_review import build_weekly_review

    ctx = get_dashboard_context(request)
    ctx["review"] = build_weekly_review()
    return templates.TemplateResponse(request=request, name="weekly_review.html", context=ctx)

@router.get("/context", response_class=HTMLResponse)
def context(request: Request):
    from tools.memory import LESSON_CAP
    from tools.pending_lessons import list_pending_lessons

    ctx = get_dashboard_context(request)
    # Lessons drafted from low-rated turns and from the 1.7 consolidation pass.
    # They are NOT in effect â€” this page is the human-confirmation gate roadmap
    # 1.4 requires before anything auto-drafted can reach the prompt.
    ctx["pending_lessons"] = list_pending_lessons()
    # 1.7: the store no longer evicts to make room, so how full it is has to be
    # visible BEFORE a confirmation is refused.
    ctx["lesson_cap"] = LESSON_CAP
    return templates.TemplateResponse(request=request, name="context_and_graph.html", context=ctx)

@router.get("/portfolio", response_class=HTMLResponse)
def portfolio(request: Request):
    return templates.TemplateResponse(request=request, name="portfolio_editor.html", context=get_dashboard_context(request))

@router.get("/monitor", response_class=HTMLResponse)
def monitor(request: Request):
    """Read-only surfaces over the holdings â€” fund flows (5.5) and event radar (3.5b).

    Split off /portfolio, which is an editor: it has a Save button, a CSV
    upload and a reconciliation panel whose cause cell POSTs. Nothing here
    writes. The two are visited on completely different rhythms â€” the grid
    monthly, this page daily â€” and while they shared a template every visit to
    correct a share count paid for two engine fetches it did not ask for.
    """
    return templates.TemplateResponse(request=request, name="monitor.html", context=get_dashboard_context(request))

@router.get("/settings", response_class=HTMLResponse)
def settings(request: Request):
    return templates.TemplateResponse(request=request, name="terminal_settings.html", context=get_dashboard_context(request))

@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html", context=get_dashboard_context(request))


