"""
Unit tests for Router component.
"""

import pytest
from agentkit.components.router import Router
from agentkit.types import RouteConfig
from agentkit.engines.rule_engine import RuleEngine, KeywordRule
from agentkit.exceptions import ConfigError


def test_router_normal_routing():
    engine = RuleEngine(rules=[
        KeywordRule(name="billing_rule", keywords=["payment", "refund", "invoice"], verdict="billing"),
        KeywordRule(name="tech_rule", keywords=["bug", "crash", "error"], verdict="technical"),
    ])
    routes = {
        "billing": RouteConfig(name="billing", description="Billing and payment inquiries"),
        "technical": RouteConfig(name="technical", description="Technical issues and bugs"),
        "general": RouteConfig(name="general", description="General questions"),
    }
    router = Router(engine=engine, routes=routes, fallback="general")

    res_billing = router.route("I need a refund for my invoice")
    assert res_billing.verdict == "billing"

    res_tech = router.route("The mobile app is crashing on startup")
    assert res_tech.verdict == "technical"


def test_router_requires_min_two_routes():
    engine = RuleEngine()
    routes = {"billing": RouteConfig(name="billing", description="Billing")}
    with pytest.raises(ConfigError):
        Router(engine=engine, routes=routes)


def test_router_invalid_fallback():
    engine = RuleEngine()
    routes = {
        "billing": RouteConfig(name="billing", description="Billing"),
        "tech": RouteConfig(name="tech", description="Tech"),
    }
    with pytest.raises(ConfigError):
        Router(engine=engine, routes=routes, fallback="unknown_route")


def test_router_empty_query_raises_value_error():
    engine = RuleEngine()
    routes = {
        "a": RouteConfig(name="a", description="Route A"),
        "b": RouteConfig(name="b", description="Route B"),
    }
    router = Router(engine=engine, routes=routes)
    with pytest.raises(ValueError):
        router.route("")


def test_router_route_and_execute():
    executed = []

    def handle_billing(q):
        executed.append(q)
        return "billing_handled"

    routes = {
        "billing": RouteConfig(name="billing", description="payment", handler=handle_billing),
        "general": RouteConfig(name="general", description="everything else"),
    }
    engine = RuleEngine(rules=[KeywordRule(name="pay", keywords=["payment"], verdict="billing")])
    router = Router(engine=engine, routes=routes)

    res, output = router.route_and_execute("My payment failed")
    assert res.verdict == "billing"
    assert output == "billing_handled"
    assert len(executed) == 1
