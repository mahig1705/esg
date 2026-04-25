from enum import Enum

class AgentStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    NULL_RESULT = "NULL_RESULT"
    FAILED = "FAILED"
