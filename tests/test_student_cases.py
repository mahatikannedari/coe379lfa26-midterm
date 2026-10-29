"""Student-written test cases for Milestone 4: verification of safety gates and validation rules."""

from __future__ import annotations

from atlas_agent.agent import run_agent
from atlas_agent.llm import ScriptedLLM
from atlas_agent.models import LabRequest
from atlas_agent.tools import ToolEnvironment

REQUEST = LabRequest(
    authenticated_user_id="alice",
    text="Reserve tensile-01 for polymer-study from 2026-10-15 10:00 to 11:00.",
    project_id="polymer-study",
    resource_id="tensile-01",
    start="2026-10-15T10:00:00",
    end="2026-10-15T11:00:00",
)


def test_unsafe_action_denied(environment: ToolEnvironment) -> None:
    """Test 1: an unsafe action proposal that is denied before tool execution."""
    assert REQUEST.start is not None
    assert REQUEST.end is not None

    llm = ScriptedLLM(
        [
            {
                "kind": "tool_call",
                "reasoning_summary": "Attempting to book without policy checks",
                "tool": "reserve_resource",
                "arguments": {
                    "user_id": REQUEST.authenticated_user_id,
                    "resource_id": REQUEST.resource_id,
                    "project_id": REQUEST.project_id,
                    "start": REQUEST.start,
                    "end": REQUEST.end,
                },
            },
            {
                "kind": "final",
                "response": {
                    "status": "blocked",
                    "message": "Booking denied due to policy violations",
                },
            },
        ]
    )

    initial_reservation_count = len(environment.simulator.state["reservations"])
    run = run_agent(REQUEST, llm=llm, environment=environment)

    # assertions
    assert len(run.trace) >= 1
    first_event = run.trace[0]
    assert first_event.observation.ok is False
    assert first_event.observation.error_code == "action_gate_denied"
    assert first_event.observation.message is not None
    assert "policy_not_retrieved" in first_event.observation.message
    assert "authorizations_not_checked" in first_event.observation.message
    assert not any(e.mutating and e.observation.ok for e in run.trace)
    assert run.response.status == "blocked"
    # Verify the tool was never dispatched and simulator state didn't change
    assert len(environment.simulator.state["reservations"]) == initial_reservation_count


def test_contradictory_final_response_rejected(environment: ToolEnvironment) -> None:
    """Test 2: A final response claiming completion without a successful mutation in the trace will be rejected."""
    assert REQUEST.start is not None
    assert REQUEST.end is not None

    llm = ScriptedLLM(
        [
            {
                "kind": "tool_call",
                "reasoning_summary": "Trying to book resource without preconditions",
                "tool": "reserve_resource",
                "arguments": {
                    "user_id": REQUEST.authenticated_user_id,
                    "resource_id": REQUEST.resource_id,
                    "project_id": REQUEST.project_id,
                    "start": REQUEST.start,
                    "end": REQUEST.end,
                },
            },
            {
                "kind": "final",
                "response": {
                    "status": "completed",
                    "message": "Claiming success falsely",
                    "completed_actions": ["reserve_resource"],
                },
            },
        ]
    )

    run = run_agent(REQUEST, llm=llm, environment=environment)

    # assertions
    assert len(run.trace) >= 1
    assert run.trace[0].observation.ok is False
    assert run.response.status == "failed"
    assert run.termination_reason == "invalid_final_response"
    assert "completed_action_not_in_successful_trace" in str(run.response.message)


def test_malformed_model_output_triggers_failure(environment: ToolEnvironment) -> None:
    """Test 3: Malformed or invalid llm decision structure causes an immediate run failure with an empty trace."""
    llm = ScriptedLLM([{"kind": "invalid_decision_type", "bad_payload": True}])

    run = run_agent(REQUEST, llm=llm, environment=environment)

    # assertions
    assert run.termination_reason == "invalid_model_decision"
    assert run.response.status == "failed"
    assert len(run.trace) == 0
    assert "The model did not produce a valid decision" in run.response.message
