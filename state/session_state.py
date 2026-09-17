from .custom_state import ContextCompactState, TodoState, UsageTrackState


# -- type hint for IDE --
class SessionState(TodoState, UsageTrackState, ContextCompactState):
    pass
