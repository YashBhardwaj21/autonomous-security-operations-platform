#!/usr/bin/env python3
"""ISOLATED demo seed — NON-PRODUCTION, outside the app/ML path.

Creates a couple of demo users (bcrypt-hashed) and a tiny real-shaped asset topology
so you can exercise the API locally. This is NOT application data and is never
imported by src/. It only wires an InMemoryUserProvider and a twin at runtime.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.api.app import set_user_provider
from src.api.auth import InMemoryUserProvider

def demo_users():
    return InMemoryUserProvider({
        "analyst": {"username": "analyst",
                    "password_hash": InMemoryUserProvider.hash_password(os.environ.get("DEMO_PW", "changeme")),
                    "roles": ["soc_analyst"]},
    })

def main():
    if not os.environ.get("JWT_SECRET"):
        print("Set JWT_SECRET before running the API (no baked default).")
    set_user_provider(demo_users())
    print("demo user provider set (analyst / $DEMO_PW). This is non-production.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
