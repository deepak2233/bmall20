"""Smoke-test the running REST service (extra credit demo).

Start the server first:
    make serve            # or: uvicorn ipm.serve:app --port 8080
Then:
    python scripts/example_request.py
"""
import json
import sys
import urllib.request

PAYLOAD = {
    "instances": [
        {
            "user_id": "u-123",
            "country": "US",
            "device_os": "Android",
            "count_user_impressions_7": 8,
            "appid": "jade.uniform.spark",
            "sdkappid": "haven.silk.fern",
            "memory_total": 4_000_000_000,
            "count_user_clicks_7": 2,
            "session_count_7d": 5,
            "user_install_profile": "a.b c.d e.f g.h",
            "timestamp": 1_774_872_991_349,
        },
        # A minimal request with unknown app + missing fields — must still score.
        {"country": "ZZ", "device_os": "iOS", "appid": "brand.new.app"},
    ]
}


def main(url: str = "http://localhost:8080/predict") -> None:
    req = urllib.request.Request(
        url, data=json.dumps(PAYLOAD).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        print(json.dumps(json.loads(resp.read()), indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/predict")
