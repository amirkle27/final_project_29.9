"""
Error types, error responses, and helpers for turning application errors
into FastAPI HTTP responses with a consistent JSON body.

- Define domain (business) exceptions with HTTP status + error code.
- Provide a unified `ErrorResponse` schema for clients.
- Offer helpers: `http_error`, `http_error_code`, and `log_and_raise`.

Notes:
- Back-compat aliases are provided for older misspelled names:
  PolinomialMaxMinError               → PolynomialMaxMinError
  PolinomialNotDFError                → PolynomialNotDFError
  PolinomialForClassificationError    → PolynomialForClassificationError
"""

from __future__ import annotations
from http import HTTPStatus
from typing import Optional
import logging

from fastapi import HTTPException
from pydantic import BaseModel

logger = logging.getLogger("ml_server.errors")


# --------- Low-level (technical) exceptions used by training/prediction code ---------

class TargetColumnNotFoundError(Exception):
    """Raised when the required target/label column is missing from a DataFrame."""
    def __init__(self, column_name: str):
        super().__init__(f"Target column '{column_name}' not found in dataframe.")


class InvalidPenaltySolverCombination(Exception):
    """Raised when an invalid (penalty, solver) pairing is used for LogisticRegression."""
    def __init__(self, penalty: str, solver: str):
        super().__init__(f"Invalid combination of penalty: '{penalty}' and solver: {solver}.")


class NotFittedError(Exception):
    """Raised when `predict` is called on a model that has not been fitted yet."""
    def __init__(self):
        super().__init__("Model must be fit before predicting.")


class MultipleFeaturesPolyError(Exception):
    """Raised when polynomial regression receives multiple features instead of a single feature."""
    def __init__(self):
        super().__init__("Polynomial regression expects a single feature column.")


class NumericTargetRequiredError(Exception):
    """Raised when a classification-like target is used with linear regression."""
    def __init__(self):
        super().__init__("Linear Regression cannot be used for classification. Target must be numeric.")


# Polynomial-related exceptions (canonical names)
class PolynomialMaxMinError(Exception):
    """Raised when polynomial feature range (min/max) configuration is invalid."""
    pass


class PolynomialNotDFError(Exception):
    """Raised when a non-DataFrame object is provided where a DataFrame is required."""
    pass


class PolynomialForClassificationError(Exception):
    """Raised when polynomial regression flow is (incorrectly) used for classification targets."""
    pass


class LinearForClassificationError(Exception):
    """Raised when linear-regression flow is used for a classification target."""
    pass


# Back-compat aliases for earlier misspellings (OLD → NEW)
PolinomialMaxMinError = PolynomialMaxMinError
PolinomialNotDFError = PolynomialNotDFError
PolinomialForClassificationError = PolynomialForClassificationError


# --------- Wire-level unified error payload ---------

class ErrorResponse(BaseModel):
    """
    Standard error response payload returned to clients.

    Attributes:
        error_code: Stable machine-readable code (e.g., "MODEL_NOT_FOUND").
        message:    Human-readable message explaining the error.
        details:    Optional structured context (dict) for debugging or UI.
    """
    error_code: str
    message: str
    details: Optional[dict] = None


MESSAGES = {
    # Authorization
    "AUTH_BAD_CREDENTIALS": "Username or Password incorrect.",
    "AUTH_TOKEN_MISSING": "No token provided.",
    "AUTH_TOKEN_EXPIRED": "Token expired.",
    "AUTH_TOKEN_INVALID": "Token invalid.",
    "AUTH_USER_NOT_FOUND": "User not found.",

    # Tokens/Cost
    "TOKENS_NOT_ENOUGH": "Tokens insufficient for the desired action.",
    "TOKENS_COST_INVALID": "The 'cost' parameter must be a positive integer value.",

    # Input/Parameters/Files
    "INPUT_INVALID_JSON": "JSON Field invalid.",
    "INPUT_MISSING_COLUMNS": "Missing columns in DB.",
    "INPUT_FILE_UNSUPPORTED": "Unsupported file format.",
    "INPUT_MODEL_PARAMS_INVALID": "model_params must be a JSON object (dict).",

    # Models
    "MODEL_NOT_FOUND": "Model not found.",
    "MODEL_KIND_MISMATCH": "Model kind not suitable for this action.",
    "MODEL_TRAIN_FAILED": "Model train failed.",
    "MODEL_PREDICT_FAILED": "Model predict failed.",
    "MODEL_BUNDLE_CORRUPT": "Model file corrupt/missing data.",

    # General Error
    "INTERNAL_ERROR": "Internal error. Please try later.",
}


# --------- Application-level (business) exceptions with HTTP mapping ---------

class AppError(Exception):
    """
    Base class for application errors that map directly to HTTP responses.

    Subclasses should set:
        http_status (HTTPStatus): The HTTP status to return.
        error_code (str):         A key into MESSAGES (or a custom stable code).
    """
    http_status: HTTPStatus = HTTPStatus.BAD_REQUEST
    error_code: str = "INTERNAL_ERROR"
    details: Optional[dict]

    def __init__(self, details: Optional[dict] = None, message: Optional[str] = None):
        self.details = details
        self.message = message or MESSAGES.get(self.error_code, "Error")
        super().__init__(self.message)


# Auth
class AuthBadCredentials(AppError):
    """401: Username or password incorrect."""
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_BAD_CREDENTIALS"


class AuthTokenMissing(AppError):
    """401: No bearer token provided."""
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_TOKEN_MISSING"


class AuthTokenExpired(AppError):
    """401: Bearer token has expired."""
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_TOKEN_EXPIRED"


class AuthTokenInvalid(AppError):
    """401: Bearer token invalid or malformed."""
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_TOKEN_INVALID"


class AuthUserNotFound(AppError):
    """401: User referenced by token or request not found."""
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_USER_NOT_FOUND"


# Tokens
class TokensNotEnough(AppError):
    """402: Action requires more tokens than the user currently has."""
    http_status = HTTPStatus.PAYMENT_REQUIRED
    error_code = "TOKENS_NOT_ENOUGH"


class TokensCostInvalid(AppError):
    """400: Token cost must be a positive integer."""
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "TOKENS_COST_INVALID"


# Inputs
class InputInvalidJSON(AppError):
    """400: Provided JSON field is invalid or malformed."""
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "INPUT_INVALID_JSON"


class InputMissingColumns(AppError):
    """400: Expected columns are missing in the uploaded data."""
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "INPUT_MISSING_COLUMNS"


class InputFileUnsupported(AppError):
    """415: Unsupported file/media type for upload."""
    http_status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    error_code = "INPUT_FILE_UNSUPPORTED"


class InputModelParamsInvalid(AppError):
    """400: model_params must be a JSON object (dict)."""
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "INPUT_MODEL_PARAMS_INVALID"


# Models
class ModelNotFound(AppError):
    """404: Requested model does not exist (or not owned by user)."""
    http_status = HTTPStatus.NOT_FOUND
    error_code = "MODEL_NOT_FOUND"


class ModelKindMismatch(AppError):
    """400: The model's kind is not compatible with the requested action."""
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "MODEL_KIND_MISMATCH"


class ModelTrainFailed(AppError):
    """400: Training failed due to invalid data, parameters, or internal error."""
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "MODEL_TRAIN_FAILED"


class ModelPredictFailed(AppError):
    """400: Prediction failed due to invalid inputs or model state."""
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "MODEL_PREDICT_FAILED"


class ModelBundleCorrupt(AppError):
    """500: Model bundle on disk is missing/corrupted or unreadable."""
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "MODEL_BUNDLE_CORRUPT"


class InternalError(AppError):
    """500: Unexpected server error."""
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "INTERNAL_ERROR"


# --------- Helpers (Pydantic v1/v2 JSON) ---------

def _to_json(model: BaseModel) -> str:
    """Return JSON for a Pydantic model in both v1 and v2 environments."""
    # v2
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json()
    # v1
    if hasattr(model, "json"):
        return model.json()
    # Fallback (shouldn't happen)
    return str(model.dict() if hasattr(model, "dict") else {})


# --------- HTTPException adapters ---------

def http_error(exc: AppError) -> HTTPException:
    """
    Build a FastAPI HTTPException from an AppError, with a unified JSON body.
    """
    body = _to_json(ErrorResponse(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details
    ))
    return HTTPException(status_code=int(exc.http_status), detail=body)


def http_error_code(
    code: str,
    status: HTTPStatus,
    details: Optional[dict] = None,
    message: Optional[str] = None
) -> HTTPException:
    """
    Create an HTTPException directly from code/status without defining a subclass.
    """
    msg = message or MESSAGES.get(code, code)
    body = _to_json(ErrorResponse(error_code=code, message=msg, details=details))
    return HTTPException(status_code=int(status), detail=body)


# --------- Logging helper ---------

def log_and_raise(exc: AppError) -> None:
    """
    Log an AppError and raise the corresponding HTTPException.
    """
    logger.warning(f"{exc.error_code}: {exc.message}; details={exc.details}")
    raise http_error(exc)
