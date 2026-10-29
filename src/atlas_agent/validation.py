"""Milestone 3: implement three selected validation responsibilities."""

from __future__ import annotations

# from xml.parsers.expat import errors
from datetime import datetime

from .agent_models import ToolCallDecision
from .models import AgentResponse, LabRequest, TraceEvent
from .policy_context import ReservationContext, build_reservation_context


def validate_request_binding(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
) -> list[str]:
    # """TODO 3.1: Prevent identity and resolved-field substitution."""
    # output the errors received
    errors: list[str] = []
    args = decision.arguments

    # need to map the request fields to expected tool to validate a request
    bindings = [
        ("authenticated_user_id", "user_id", request.authenticated_user_id),
        ("resource_id", "resource_id", request.resource_id),
        ("project_id", "project_id", request.project_id),
        ("start", "start", request.start),
        ("end", "end", request.end),
    ]

    is_reserve = decision.tool == "reserve_resource"

    # check each binding to see if the decision arguments match the request fields
    for req_field, arg_key, req_val in bindings:
        # if req_val is not None and arg_key in args:
        arg_val = args.get(arg_key)

        # if the request value is None, we skip the check for that field
        if req_val is None:
            if is_reserve and arg_val is not None:
                errors.append(f"{arg_key}_unresolved_in_request")
            continue

        # next case, if request has a value, but model output is none
        if arg_val is None:
            continue

        # check if the request value and argument value match, if not, add an error
        if req_field in ("start", "end"):
            try:
                req_match = datetime.fromisoformat(
                    str(req_val)
                ) == datetime.fromisoformat(str(arg_val))
            except (ValueError, TypeError):
                req_match = str(req_val) == str(arg_val)
            # convert to datetime for comparison
        else:
            req_match = str(req_val) == str(arg_val)

        if not req_match:
            if arg_key == "user_id":
                errors.append("user_id_must_match_authenticated_user")
            else:
                errors.append(f"{arg_key}_does_not_match_request")

    return errors


def validate_reservation_preconditions(context: ReservationContext) -> list[str]:
    """TODO 3.2: Return an error for every required context fact that is false."""
    errors: list[str] = []

    # all possible errors received
    if not context.arguments_complete:
        errors.append("arguments_incomplete")
    if not context.policy_retrieved:
        errors.append("policy_not_retrieved")
    if not context.authorizations_checked:
        errors.append("authorizations_not_checked")
    if not context.resource_status_checked:
        errors.append("resource_status_not_checked")
    if not context.project_authorized:
        errors.append("project_not_authorized")
    if not context.resource_available:
        errors.append("resource_not_available")
    if not context.qualification_current:
        errors.append("required_qualification_missing")
    if not context.calibration_current:
        errors.append("calibration_not_current")
    if not context.maintenance_clear:
        errors.append("resource_under_maintenance")
    if not context.after_hours_approval_satisfied:
        errors.append("after_hours_approval_not_satisfied")

    return errors


def validate_proposed_tool_call(
    decision: ToolCallDecision,
    *,
    request: LabRequest,
    trace: list[TraceEvent],
) -> list[str]:
    """Supplied wrapper. Do not modify this function."""
    errors = validate_request_binding(decision, request=request)
    if decision.tool == "reserve_resource":
        context = build_reservation_context(decision, trace)
        errors.extend(validate_reservation_preconditions(context))
    return errors


def retrieved_passage_ids(trace: list[TraceEvent]) -> set[str]:
    """Supplied helper. Do not modify this function."""
    identifiers: set[str] = set()
    for event in trace:
        if event.observation.tool != "search_lab_documents" or not event.observation.ok:
            continue
        identifiers.update(
            passage["passage_id"]
            for passage in event.observation.data.get("passages", [])
        )
    return identifiers


def validate_final_response(
    response: AgentResponse, trace: list[TraceEvent]
) -> list[str]:
    """TODO 3.3: Validate citations, claimed actions, and final status."""
    # del response, trace
    # return []
    # finalized arguments
    errors: list[str] = []
    retrieved_ids = retrieved_passage_ids(trace)

    # check the successful muutations
    successful_mutations = [
        event.observation.tool
        for event in trace
        if event.mutating and event.observation.ok  # main logic
    ]

    # following repo guidelines
    # step 3a, check indv citations
    invalid_citations = set(response.citations) - retrieved_ids
    if invalid_citations:
        errors.append(f"citations_not_retrieved: {sorted(invalid_citations)}")

    # step 3b, check claimed actions completed_actions -> successful_mutation
    invalid_actions = set(response.completed_actions) - set(successful_mutations)
    if invalid_actions:
        errors.append("completed_action_not_in_successful_trace")

    # step 3c, check final status one status is completed. should have claimed action or successful excecution
    if response.status == "completed":
        if not successful_mutations:
            errors.append("completed_without_successful_mutation")

        if not response.completed_actions:
            errors.append("status_completed_requires_completed_action")

    # step 3d, check final status one status is failed. should have no successful excecution or claimed action
    if response.status != "completed":
        if response.completed_actions:
            errors.append("non_completed_status_cannot_have_completed_actions")

        if successful_mutations:
            errors.append("unreported_successful_mutation")

    return errors
