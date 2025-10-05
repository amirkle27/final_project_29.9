from __future__ import annotations
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
from http import HTTPStatus
import logging

logger = logging.getLogger("ml_server.errors")

class TargetColumnNotFoundError(Exception):
    def __init__(self, column_name: str):
        super().__init__(f"Target column '{column_name}' not found in dataframe.")

class InvalidPenaltySolverCombination(Exception):
    def __init__(self, penalty: str, solver: str):
        super().__init__(f"Invalid combination of penalty: '{penalty}' and solver: {solver}.")

class NotFittedError(Exception):
    def __init__(self):
        super().__init__("Model must be fit before predicting.")

class MultipleFeaturesPolyError(Exception):
    def __init__(self):
        super().__init__("Polynomial regression expects a single feature column.")

class NumericTargetRequiredError(Exception):
    def __init__(self):
        super().__init__("Linear Regression cannot be used for classification. Target must be numeric.")

class TargetColumnNotFoundError(Exception):
    """Raised when the target/label column is missing from DataFrame."""

class InvalidPenaltySolverCombination(Exception):
    """Raised when LogisticRegression penalty/solver combo is invalid."""

class NotFittedError(Exception):
    """Raised when predict is called before fitting a model."""

class MultipleFeaturesPolyError(Exception):
    """Raised when polynomial transform gets invalid feature spec."""

class PolinomialMaxMinError(Exception):
    pass
PolynomialMaxMinError = PolinomialMaxMinError 

class PolinomialNotDFError(Exception):
    pass
PolynomialNotDFError = PolinomialNotDFError  

class PolinomialForClassificationError(Exception):
    pass
PolynomialForClassificationError = PolinomialForClassificationError  

class LinearForClassificationError(Exception):
    """Raised if linear-regression flow is used for classification target."""
    pass

class ErrorResponse(BaseModel):
    error_code: str         
    message: str            
    details: Optional[dict] = None  

MESSAGES = {
    #  Authorization
    "AUTH_BAD_CREDENTIALS": "Username or Password incorrect.",
    "AUTH_TOKEN_MISSING": "No token provided.",
    "AUTH_TOKEN_EXPIRED": "Token expired.",
    "AUTH_TOKEN_INVALID": "Token invalid.",
    "AUTH_USER_NOT_FOUND": "User not found.",

    # Tokens/Cost
    "TOKENS_NOT_ENOUGH": "Tokens insufficient for the desired action.",
    "TOKENS_COST_INVALID": "העלות (cost) חייבת להיות מספר שלם חיובי.",

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

# --- מחלקות חריגים אפליקטיביות (Business) ---

class AppError(Exception):
    """בסיס לכל חריג אפליקטיבי עם קוד ומידע משלים."""
    http_status: HTTPStatus = HTTPStatus.BAD_REQUEST
    error_code: str = "INTERNAL_ERROR"
    details: Optional[dict]

    def __init__(self, details: Optional[dict] = None, message: Optional[str] = None):
        self.details = details
        self.message = message or MESSAGES.get(self.error_code, "Error")
        super().__init__(self.message)

class AuthBadCredentials(AppError):
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_BAD_CREDENTIALS"

class AuthTokenMissing(AppError):
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_TOKEN_MISSING"

class AuthTokenExpired(AppError):
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_TOKEN_EXPIRED"

class AuthTokenInvalid(AppError):
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_TOKEN_INVALID"

class AuthUserNotFound(AppError):
    http_status = HTTPStatus.UNAUTHORIZED
    error_code = "AUTH_USER_NOT_FOUND"

class TokensNotEnough(AppError):
    http_status = HTTPStatus.PAYMENT_REQUIRED
    error_code = "TOKENS_NOT_ENOUGH"

class TokensCostInvalid(AppError):
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "TOKENS_COST_INVALID"

class InputInvalidJSON(AppError):
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "INPUT_INVALID_JSON"

class InputMissingColumns(AppError):
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "INPUT_MISSING_COLUMNS"

class InputFileUnsupported(AppError):
    http_status = HTTPStatus.UNSUPPORTED_MEDIA_TYPE
    error_code = "INPUT_FILE_UNSUPPORTED"

class InputModelParamsInvalid(AppError):
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "INPUT_MODEL_PARAMS_INVALID"

class ModelNotFound(AppError):
    http_status = HTTPStatus.NOT_FOUND
    error_code = "MODEL_NOT_FOUND"

class ModelKindMismatch(AppError):
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "MODEL_KIND_MISMATCH"

class ModelTrainFailed(AppError):
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "MODEL_TRAIN_FAILED"

class ModelPredictFailed(AppError):
    http_status = HTTPStatus.BAD_REQUEST
    error_code = "MODEL_PREDICT_FAILED"

class ModelBundleCorrupt(AppError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "MODEL_BUNDLE_CORRUPT"

class InternalError(AppError):
    http_status = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code = "INTERNAL_ERROR"

# --- מתאמי HTTPException קצרים (למקומות שעדיין רוצים raise HTTPException) ---

def http_error(exc: AppError) -> HTTPException:
    """בונה HTTPException עם גוף אחיד לפי ErrorResponse."""
    body = ErrorResponse(error_code=exc.error_code, message=exc.message, details=exc.details).model_json()
    return HTTPException(status_code=int(exc.http_status), detail=body)

def http_error_code(code: str, status: HTTPStatus, details: Optional[dict] = None, message: Optional[str] = None) -> HTTPException:
    msg = message or MESSAGES.get(code, code)
    body = ErrorResponse(error_code=code, message=msg, details=details).model_json()
    return HTTPException(status_code=int(status), detail=body)

# --- עזר ללוגים ---
def log_and_raise(exc: AppError) -> None:
    logger.warning(f"{exc.error_code}: {exc.message}; details={exc.details}")
    raise http_error(exc)

