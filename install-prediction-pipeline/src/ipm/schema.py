"""Canonical description of the raw dataset schema.

Centralising the schema means the loader, the featurizer and the serving layer
all agree on column names and types. If adjoe adds/renames a column we change it
in exactly one place.
"""
from __future__ import annotations

# --- Raw column names, exactly as they appear in dataset.parquet -------------
USER_ID = "user_id"
COUNTRY = "country"
DEVICE_OS = "device_os"
COUNT_IMPRESSIONS_7 = "count_user_impressions_7"
APPID = "appid"
SDKAPPID = "sdkappid"
MEMORY_TOTAL = "memory_total"
COUNT_CLICKS_7 = "count_user_clicks_7"
SESSION_COUNT_7D = "session_count_7d"
USER_INSTALL_PROFILE = "user_install_profile"
POSTBACK = "postback"          # target
TIMESTAMP = "timestamp"        # utc epoch milliseconds

TARGET = POSTBACK

# Columns we expect to read from disk. We list them explicitly so a column
# projection can be pushed down to the parquet reader (load only what we use).
RAW_STRING_COLUMNS = [USER_ID, COUNTRY, DEVICE_OS, APPID, SDKAPPID, USER_INSTALL_PROFILE]
RAW_NUMERIC_COLUMNS = [
    COUNT_IMPRESSIONS_7,
    MEMORY_TOTAL,
    COUNT_CLICKS_7,
    SESSION_COUNT_7D,
    TIMESTAMP,
]
ALL_COLUMNS = RAW_STRING_COLUMNS + RAW_NUMERIC_COLUMNS + [TARGET]

# Sentinel used to replace missing categorical values *before* one-hot encoding.
# Keeping it explicit (rather than relying on the encoder) guarantees the exact
# same value is produced at training and at serving time.
CATEGORICAL_MISSING = "__MISSING__"
