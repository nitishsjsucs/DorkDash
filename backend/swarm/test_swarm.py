"""Quick integration test for the Dasher Swarm module."""
import asyncio
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.swarm.dasher_agent import DasherAgent, DasherPersona
from backend.swarm.marketplace_env import MarketplaceEnvironment
from backend.swarm.swarm_runner import DasherSwarmRunner


async def test_single_agent():
    print("=== Test 1: Single DasherAgent (rule-based) ===")
    persona = DasherPersona(
        agent_id=0, name="TestDasher", home_zone=2,
        tenure_days=100, base_elasticity=0.6,
        schedule_preference="flexible", income_target_daily=80.0,
        risk_tolerance=0.5, fatigue_rate=0.2,
    )
    agent = DasherAgent(persona=persona)  # no API key => rule-based
    offer = {
        "zone_id": 2,
        "incentive_amount": 3.0,
        "time_of_day": "afternoon",
        "zone_demand": 0.7,
        "neighbour_incentives": {1: 2.0, 3: 4.0},
    }
    decision = await agent.decide(offer)
    print(f"  Accepted: {decision['accepted']}")
    print(f"  Reason: {decision['reason']}")
    print(f"  Would switch: {decision['would_switch_zone']}")
    print(f"  Agent state: {agent.get_state()}")
    assert "accepted" in decision
    assert "reason" in decision
    print("  PASSED\n")


async def test_marketplace_env():
    print("=== Test 2: MarketplaceEnvironment (3 steps) ===")
    env = MarketplaceEnvironment(n_zones=5, n_dashers=10, seed=42)
    env.reset()
    assert len(env.zones) == 5
    assert len(env.agents) == 10
    print(f"  Zones: {len(env.zones)}, Agents: {len(env.agents)}")

    env.set_incentives({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: 5.0})
    for i in range(3):
        result = await env.step()
        print(f"  Step {result['step']}: accept={result['n_accepted']}/{result['total_agents']}, "
              f"rate={result['acceptance_rate']:.2f}, switches={result['agent_switches']}")
    state = env.get_state()
    assert state["step_count"] == 3
    print(f"  Avg earnings: ${state['agent_summary']['avg_earnings']:.2f}")
    print("  PASSED\n")


async def test_swarm_runner():
    print("=== Test 3: DasherSwarmRunner (2-day sim) ===")
    runner = DasherSwarmRunner(n_dashers=20, n_zones=10, seed=42)
    result = await runner.run(n_days=2)
    agg = result["aggregate"]
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Dashers: {result['n_dashers']}, Zones: {result['n_zones']}")
    print(f"  Mean accept rate: {agg['mean_acceptance_rate']:.3f} +/- {agg['std_acceptance_rate']:.3f}")
    print(f"  Zone switches: {agg['total_zone_switches']}")
    print(f"  Daily summaries: {len(result['daily_summaries'])}")
    print(f"  Agent states: {len(result['agent_states'])}")
    assert result["total_steps"] == 12  # 2 days * 6 steps
    assert len(result["daily_summaries"]) == 2
    assert len(result["agent_states"]) == 20
    assert 0 <= agg["mean_acceptance_rate"] <= 1

    # Check runner state
    state = runner.get_state()
    assert state["is_complete"] is True
    print("  PASSED\n")


async def test_oasis_bridge():
    print("=== Test 4: OASISBridge (availability check) ===")
    from backend.swarm.swarm_runner import OASISBridge
    bridge = OASISBridge()
    available = await bridge.is_available()
    print(f"  OASIS sidecar available: {available}")
    print("  PASSED (bridge instantiates; sidecar connection is optional)\n")


async def main():
    print("\n" + "=" * 60)
    print("  DASHER SWARM MODULE — INTEGRATION TESTS")
    print("=" * 60 + "\n")

    await test_single_agent()
    await test_marketplace_env()
    await test_swarm_runner()
    await test_oasis_bridge()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
