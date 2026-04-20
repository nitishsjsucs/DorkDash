"""
LLM Meta-Agent for experiment management.
Uses OpenAI function calling to interact with the bandit/BO/causal systems.

Three capabilities:
1. Arm Management — propose new arms, retire underperformers
2. Experiment Summarization — produce PM-readable summaries after each batch
3. Non-stationarity Detection — flag regime changes, suggest prior resets

References:
  - Bouneffouf & Feraud, "Bandits, LLMs, and Agentic AI" (AAAI 2026)
  - "Agent A/B" (arXiv 2504.09723) — LLM agents for experimentation
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


SYSTEM_PROMPT = """You are an AI experiment management agent for DoorDash's Supply Optimization team.
You analyze multi-armed bandit experiments and Bayesian optimization runs for Dasher incentive optimization.

Your responsibilities:
1. **Arm Management**: Analyze posterior distributions and recommend adding/retiring arms.
   - Retire arms that have been consistently underperforming for multiple days
   - Propose new arm values based on GP posterior (unexplored promising regions)

2. **Experiment Summarization**: After each batch cycle, produce clear, actionable summaries.
   - Report which arms are winning, what the optimal incentive appears to be
   - Highlight interesting patterns (zone-specific effects, time-of-day effects)
   - Quantify business impact in dollars (savings per activation × scale)

3. **Non-stationarity Detection**: Monitor for regime changes.
   - Flag when treatment effect posteriors are drifting significantly
   - Detect sudden shifts (competitor entry, holiday, weather events)
   - Recommend prior resets or increased exploration when detected

4. **Network Analysis**: Assess spillover effects between zones.
   - Identify zones where incentives create negative externalities on neighbours
   - Recommend coordinated incentive strategies across zone clusters

5. **Dose-Response Analysis**: Interpret continuous treatment effect curves.
   - Identify diminishing-returns thresholds (concavity breakpoints)
   - Recommend per-zone optimal incentive levels from the dose-response curve

6. **Constraint Monitoring**: Track budget utilisation and allocation fairness.
   - Flag when daily/weekly budgets are nearly exhausted
   - Ensure exploration and control floors are maintained

Always ground your recommendations in the data. Cite specific numbers.
Format responses for a PM or EM audience — clear, concise, actionable.
When computing business impact, assume DoorDash processes ~10M Dasher activations per month.

**IMPORTANT**: Any write actions (arm changes, prior resets) are proposals that require human approval.
Never state you have *made* a change — always say you are *proposing* it."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_bandit_state",
            "description": "Get current state of the Thompson Sampling bandit including arm posteriors, pull counts, and cumulative regret",
            "parameters": {
                "type": "object",
                "properties": {
                    "algorithm": {
                        "type": "string",
                        "enum": ["vanilla_ts", "contextual_ts"],
                        "description": "Which bandit algorithm to query"
                    }
                },
                "required": ["algorithm"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bo_state",
            "description": "Get Bayesian Optimization state including GP posterior, optimal incentive estimate, and observation history",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_id": {
                        "type": "integer",
                        "description": "Zone ID to query (0-9)"
                    }
                },
                "required": ["zone_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_causal_insights",
            "description": "Get heterogeneous treatment effect estimates by Dasher segment (tenure, zone deficit, peak/off-peak)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_simulation_snapshot",
            "description": "Get current marketplace state including zone supply/demand, deficits, and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propose_arm_change",
            "description": "Propose adding or retiring an incentive arm",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "retire"],
                        "description": "Whether to add a new arm or retire an existing one"
                    },
                    "arm_id": {
                        "type": "string",
                        "description": "Arm identifier (e.g., '$3.50')"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Justification for the change"
                    }
                },
                "required": ["action", "arm_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flag_nonstationarity",
            "description": "Flag a detected non-stationarity event in a zone",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_id": {
                        "type": "integer",
                        "description": "Affected zone"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "How severe the detected shift is"
                    },
                    "description": {
                        "type": "string",
                        "description": "Description of what changed"
                    },
                    "recommendation": {
                        "type": "string",
                        "description": "Suggested action (e.g., reset priors, increase exploration)"
                    }
                },
                "required": ["zone_id", "severity", "description", "recommendation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_networked_bandit_state",
            "description": "Get Networked LinTS state including per-zone regret, global/local decomposition, and sharing strength",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_id": {
                        "type": "integer",
                        "description": "Optional zone ID for zone-specific stats. Omit for global view."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_change_detection_state",
            "description": "Get non-stationarity detection state: CUSUM, sliding window, and posterior drift events across all zones",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_reward_worker_state",
            "description": "Get batch reward worker state: treatment effects relative to control, batch history, doubly-robust estimates",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_dose_response",
            "description": "Get dose-response curve: expected response probability as a function of incentive amount, with confidence intervals",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_interference_effects",
            "description": "Get interference-aware estimates: direct effect (own-zone) vs indirect spillover effect, per zone",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_id": {
                        "type": "integer",
                        "description": "Optional zone ID. Omit for global estimate."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_allocator_state",
            "description": "Get constraint-aware allocator state: budget utilisation, per-zone spend, constraint configuration",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]


class ExperimentAgent:
    """LLM-powered meta-agent for adaptive experiment management."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.has_openai = HAS_OPENAI and self.api_key and self.api_key != "sk-your-key-here"
        if self.has_openai:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None

        self.conversation_history: List[Dict] = []
        self.proposed_changes: List[Dict] = []
        self.nonstationarity_flags: List[Dict] = []

        # Data sources — set externally by the API layer
        self._data_sources: Dict[str, Any] = {}

    def set_data_sources(self, sources: Dict[str, Any]):
        """Inject references to bandit, BO, causal, and simulator objects."""
        self._data_sources = sources

    def _execute_tool_call(self, function_name: str, arguments: Dict) -> str:
        """Execute a tool call and return the result as a JSON string."""
        sources = self._data_sources

        if function_name == "get_bandit_state":
            algo = arguments.get("algorithm", "vanilla_ts")
            if algo == "vanilla_ts" and "vanilla_ts" in sources:
                return json.dumps(sources["vanilla_ts"].get_state())
            elif algo == "contextual_ts" and "contextual_ts" in sources:
                return json.dumps(sources["contextual_ts"].get_state())
            return json.dumps({"error": f"Algorithm {algo} not found"})

        elif function_name == "get_bo_state":
            if "bo" in sources:
                zone_id = arguments.get("zone_id", 0)
                state = sources["bo"].get_state()
                posterior = sources["bo"].get_posterior(zone_id)
                optimal = sources["bo"].get_optimal_incentive(zone_id)
                return json.dumps({**state, "posterior_sample": {
                    "zone_id": zone_id,
                    "optimal": optimal,
                    "n_posterior_points": len(posterior.get("mean", [])),
                }})
            return json.dumps({"error": "BO not initialized"})

        elif function_name == "get_causal_insights":
            if "causal_segments" in sources:
                return json.dumps(sources["causal_segments"])
            return json.dumps({"error": "Causal model not fitted"})

        elif function_name == "get_simulation_snapshot":
            if "simulator" in sources:
                return json.dumps(sources["simulator"].get_snapshot())
            return json.dumps({"error": "Simulator not initialized"})

        elif function_name == "propose_arm_change":
            change = {
                "timestamp": datetime.now().isoformat(),
                **arguments,
            }
            self.proposed_changes.append(change)
            return json.dumps({"status": "proposed", "change": change})

        elif function_name == "flag_nonstationarity":
            flag = {
                "timestamp": datetime.now().isoformat(),
                **arguments,
            }
            self.nonstationarity_flags.append(flag)
            return json.dumps({"status": "flagged", "event": flag})

        elif function_name == "get_networked_bandit_state":
            if "net_ts" in sources:
                state = sources["net_ts"].get_state()
                zone_id = arguments.get("zone_id")
                if zone_id is not None:
                    state["focused_zone"] = {
                        "zone_id": zone_id,
                        "zone_regret": state.get("zone_regret", {}).get(str(zone_id), 0),
                        "zone_arms": state.get("zone_stats", {}).get(str(zone_id), []),
                    }
                return json.dumps(state)
            return json.dumps({"error": "Networked bandit not initialized"})

        elif function_name == "get_change_detection_state":
            if "change_detector" in sources:
                return json.dumps(sources["change_detector"].get_state())
            return json.dumps({"error": "Change detector not initialized"})

        elif function_name == "get_reward_worker_state":
            if "reward_worker" in sources:
                return json.dumps(sources["reward_worker"].get_state())
            return json.dumps({"error": "Reward worker not initialized"})

        elif function_name == "get_dose_response":
            if "dose_response" in sources:
                return json.dumps(sources["dose_response"].get_curve())
            return json.dumps({"error": "Dose-response not fitted"})

        elif function_name == "get_interference_effects":
            if "interference_est" in sources:
                state = sources["interference_est"].get_state()
                zone_id = arguments.get("zone_id")
                if zone_id is not None and "interference_by_zone" in sources:
                    state["zone_detail"] = sources["interference_by_zone"].get(int(zone_id), {})
                return json.dumps(state)
            return json.dumps({"error": "Interference estimator not fitted"})

        elif function_name == "get_allocator_state":
            if "allocator" in sources:
                return json.dumps(sources["allocator"].get_state())
            return json.dumps({"error": "Allocator not initialized"})

        return json.dumps({"error": f"Unknown function: {function_name}"})

    async def chat(self, user_message: str) -> Dict:
        """
        Process a user message through the LLM agent.
        Returns the agent's response and any tool calls made.
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        if not self.has_openai:
            return self._mock_response(user_message)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.conversation_history[-10:],  # last 10 messages for context
        ]

        tool_calls_made = []

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.3,
            )

            message = response.choices[0].message

            # Handle tool calls
            while message.tool_calls:
                messages.append(message)

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)
                    result = self._execute_tool_call(fn_name, fn_args)

                    tool_calls_made.append({
                        "function": fn_name,
                        "arguments": fn_args,
                        "result": json.loads(result),
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                # Get follow-up response
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                )
                message = response.choices[0].message

            assistant_message = message.content or ""
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message,
            })

            return {
                "response": assistant_message,
                "tool_calls": tool_calls_made,
                "proposed_changes": self.proposed_changes[-5:],
                "nonstationarity_flags": self.nonstationarity_flags[-5:],
            }

        except Exception as e:
            return {
                "response": f"Error communicating with LLM: {str(e)}",
                "tool_calls": [],
                "error": str(e),
            }

    def _mock_response(self, user_message: str) -> Dict:
        """Generate a mock response when OpenAI is not available."""
        sources = self._data_sources

        # Build a data-driven mock response
        parts = ["**Experiment Summary** (Mock Agent — set OPENAI_API_KEY for full capability)\n"]

        if "vanilla_ts" in sources:
            state = sources["vanilla_ts"].get_state()
            parts.append(f"- **Vanilla TS**: {state['total_pulls']} total pulls, "
                        f"cumulative regret: {state['cumulative_regret']:.2f}")
            for arm in state["arms"][:3]:
                parts.append(f"  - Arm {arm['arm_id']}: mean={arm['mean']:.3f}, pulls={arm['n_pulls']}")

        if "contextual_ts" in sources:
            state = sources["contextual_ts"].get_state()
            parts.append(f"- **Contextual LinTS**: {state['total_pulls']} total pulls, "
                        f"cumulative regret: {state['cumulative_regret']:.2f}")

        if "net_ts" in sources:
            state = sources["net_ts"].get_state()
            parts.append(f"- **Networked LinTS**: {state['total_pulls']} total pulls, "
                        f"cumulative regret: {state['cumulative_regret']:.2f}, "
                        f"{state['n_zones']} zones, sharing={state['sharing_strength']:.2f}")

        if "bo" in sources:
            state = sources["bo"].get_state()
            parts.append(f"- **Bayesian Optimization**: {state['total_observations']} observations")
            for zid, zstate in state.get("zones", {}).items():
                opt = zstate.get("optimal", {})
                if opt.get("optimal_incentive") is not None:
                    parts.append(f"  - Zone {zid}: optimal incentive ≈ ${opt['optimal_incentive']:.2f}")

        if "causal_segments" in sources:
            segs = sources["causal_segments"]
            overall = segs.get("overall", {})
            if overall:
                parts.append(f"- **Causal ML**: ATE = {overall.get('mean_cate', 0):.4f}")

        if "change_detector" in sources:
            cd = sources["change_detector"].get_state()
            parts.append(f"- **Change Detection**: {cd['total_events']} events detected")

        if "reward_worker" in sources:
            rw = sources["reward_worker"].get_state()
            parts.append(f"- **Reward Worker**: {rw['batch_count']} batches, "
                        f"{rw['total_outcomes_processed']} outcomes processed")

        if "interference_est" in sources:
            ie = sources["interference_est"].get_state()
            if ie.get("is_fitted"):
                parts.append(f"- **Interference**: direct={ie['direct_effect']:.4f}, "
                            f"indirect(spillover)={ie['indirect_effect']:.4f}")

        response = "\n".join(parts)
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
        })

        return {
            "response": response,
            "tool_calls": [],
            "proposed_changes": [],
            "nonstationarity_flags": [],
            "mock": True,
        }

    def get_state(self) -> Dict:
        """Return agent state."""
        return {
            "has_openai": self.has_openai,
            "conversation_length": len(self.conversation_history),
            "proposed_changes": self.proposed_changes,
            "nonstationarity_flags": self.nonstationarity_flags,
        }
