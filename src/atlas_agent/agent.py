"""Milestone 2: complete the bounded decision–action–observation loop."""

# Ruff suppressions cover names intentionally reserved for the TODOs. Remove
# this line after completing Milestone 2.


from __future__ import annotations

import json

from pydantic import ValidationError

from .agent_models import (
    ClarificationDecision,
    DecisionAdapter,
    FinalDecision,
    ToolCallDecision,
)
from .llm import StructuredLLM
from .models import AgentResponse, AgentRun, LabRequest, ToolObservation, TraceEvent
from .prompts import initialize_messages
from .tools import MUTATING_TOOLS, ToolEnvironment
from .validation import validate_final_response, validate_proposed_tool_call


def _failed(message: str, trace: list[TraceEvent], reason: str) -> AgentRun:
    return AgentRun(
        response=AgentResponse(status="failed", message=message),
        trace=trace,
        termination_reason=reason,
    )


def run_agent(
    request: LabRequest,
    *,
    llm: StructuredLLM,
    environment: ToolEnvironment,
    max_steps: int = 10,
) -> AgentRun:
    messages = initialize_messages(request)
    trace: list[TraceEvent] = []

    for step in range(1, max_steps + 1):
        try:
            # TODO 2.1: Ask llm for one decision validated by DecisionAdapter.
            decision = llm.generate(messages=messages, response_model=DecisionAdapter)

        except (ValidationError, ValueError, TypeError, RuntimeError) as exc:
            return _failed(
                f"The model did not produce a valid decision: {exc}",
                trace,
                "invalid_model_decision",
            )

        if isinstance(decision, ClarificationDecision):
            # TODO 2.2: Return clarification_required without calling a tool.
            return AgentRun(
                response=AgentResponse(
                    status="clarification_required",
                    message=decision.question,
                ),
                trace=trace,
                termination_reason="clarification_required",
            )

        if isinstance(decision, FinalDecision):
            # TODO 2.3: Validate the final response. Return a failed run if it is
            # inconsistent; otherwise return the model's final response.
            # raise NotImplementedError
            # start with returning an error
            # validation_error = validate_final_response(decision.response, trace)
            # if validation_error:
            #     return _failed(
            #         f"The model's final response is inconsistent: {validation_error}",
            #         trace,
            #         "invalid_final_response",
            #     )
            # return AgentRun(
            #     response=decision.response,
            #     trace=trace,
            #     termination_reason=decision.response.status,
            # )

            errors = validate_final_response(decision.response, trace)
            if errors:
                return _failed(
                    f"The model's final response is inconsistent: {errors}",
                    trace,
                    "invalid_final_response",
                )
            return AgentRun(
                response=decision.response,
                trace=trace,
                termination_reason="model_finished",
            )

        if not isinstance(decision, ToolCallDecision):
            return _failed(
                "Unsupported decision type.", trace, "invalid_model_decision"
            )

        # TODO 2.4: Call validate_proposed_tool_call. If it returns errors,
        # create an action_gate_denied observation. Otherwise dispatch the tool.
        # intialixe the gate error we will receive
        errors = validate_proposed_tool_call(
            decision, request=request, trace=trace
        )  # needed to change this to match the validation.py file
        if errors:
            observation = ToolObservation(
                ok=False,
                tool=decision.tool,
                error_code="action_gate_denied",
                message="; ".join(errors),
            )
        else:
            observation = environment.dispatch(decision.tool, decision.arguments)

        # TODO 2.5: Create and append a TraceEvent. Set mutating by consulting
        # MUTATING_TOOLS rather than hard-coding one tool name.
        # creating the event
        event = TraceEvent(
            step=step,
            decision=decision.model_dump(),
            observation=observation,
            mutating=decision.tool in MUTATING_TOOLS,
        )
        trace.append(event)

        # TODO 2.6: Append the serialized decision as an assistant message and
        # the serialized observation as a tool message.
        # seralized decision -> assitant message
        messages.append({"role": "assistant", "content": decision.model_dump_json()})
        # serialized output -> tool message
        messages.append({"role": "tool", "content": observation.model_dump_json()})

    # TODO 2.7: Return an explicit maximum_steps_exceeded failure.
    # need to write messages to output the max steps exceeded
    return _failed(
        "The agent did not reach a final decision within the maximum number of steps.",
        trace,
        "maximum_steps_exceeded",
    )
