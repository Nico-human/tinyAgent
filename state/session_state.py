from .custom_state import TodoState, UsageTrackState, ContextCompactState


# -- type hint for IDE --
class SessionState(TodoState, UsageTrackState, ContextCompactState):
    pass