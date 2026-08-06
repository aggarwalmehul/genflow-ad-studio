"""Token usage logging to BigQuery for Genflow."""
import datetime, logging, contextvars
log = logging.getLogger("token_logger")
current_user_email = contextvars.ContextVar("current_user_email", default="unknown")
try:
    from google.cloud import bigquery
    _bq = bigquery.Client(project="ltm-craftstudio-poc")
except Exception as e:
    _bq = None
    log.warning("TOKENLOG: bq init failed: %s", e)

_TABLE = "ltm-craftstudio-poc.token_usage.usage"

def log_tokens(platform, model, resp, user_email="unknown"):
    if _bq is None: return
    u = getattr(resp, "usage_metadata", None)
    if u is None: return
    row = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform, "user_email": user_email, "model": str(model),
        "tokens_in": int(getattr(u,"prompt_token_count",0) or 0),
        "tokens_out": int(getattr(u,"candidates_token_count",0) or 0),
        "total": int(getattr(u,"total_token_count",0) or 0),
    }
    try:
        errs = _bq.insert_rows_json(_TABLE, [row])
        if errs: log.warning("TOKENLOG: insert errors: %s", errs)
        else: log.info("TOKENLOG: inserted %s tokens for %s", row["total"], model)
    except Exception as e:
        log.warning("TOKENLOG: insert exception: %s", e)
