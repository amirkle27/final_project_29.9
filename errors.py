# errors.py
from __future__ import annotations
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional
from http import HTTPStatus
import logging

logger = logging.getLogger("ml_server.errors")

# errors.py — הוסף/י את זה (למעלה או למטה, לא משנה):
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

# --- Compat for older/newer modules expecting these names ---

class TargetColumnNotFoundError(Exception):
    """Raised when the target/label column is missing from DataFrame."""


class InvalidPenaltySolverCombination(Exception):
    """Raised when LogisticRegression penalty/solver combo is invalid."""


class NotFittedError(Exception):
    """Raised when predict is called before fitting a model."""


class MultipleFeaturesPolyError(Exception):
    """Raised when polynomial transform gets invalid feature spec."""


# שימור כתיב שגוי 'Polinomial*' לצורך תאימות, + אליאס ל-Polynomial*
class PolinomialMaxMinError(Exception):
    pass
PolynomialMaxMinError = PolinomialMaxMinError  # alias


class PolinomialNotDFError(Exception):
    pass
PolynomialNotDFError = PolinomialNotDFError  # alias


class PolinomialForClassificationError(Exception):
    pass
PolynomialForClassificationError = PolinomialForClassificationError  # alias


class LinearForClassificationError(Exception):
    """Raised if linear-regression flow is used for classification target."""
    pass

# --- מודל תגובת שגיאה אחיד ל-API ---
class ErrorResponse(BaseModel):
    error_code: str          # קוד פנימי קצר ויציב, לדוג': AUTH_BAD_CREDENTIALS
    message: str             # הודעה ידידותית למשתמש/מורה
    details: Optional[dict] = None  # מידע משלים (לא חובה), לדוג': {"missing": ["age","salary"]}

# --- מאגר הודעות שגיאה (אפשר לתרגם/לשנות במקום אחד) ---
MESSAGES = {
    # אימות/הרשאות
    "AUTH_BAD_CREDENTIALS": "שם משתמש או סיסמה שגויים.",
    "AUTH_TOKEN_MISSING": "לא סופק טוקן.",
    "AUTH_TOKEN_EXPIRED": "הטוקן פג תוקף.",
    "AUTH_TOKEN_INVALID": "טוקן לא תקין.",
    "AUTH_USER_NOT_FOUND": "משתמש לא נמצא.",

    # טוקנים/תמחור
    "TOKENS_NOT_ENOUGH": "אין מספיק טוקנים לביצוע הפעולה.",
    "TOKENS_COST_INVALID": "העלות (cost) חייבת להיות מספר שלם חיובי.",

    # קלט/פרמטרים/קבצים
    "INPUT_INVALID_JSON": "שדה JSON לא תקין.",
    "INPUT_MISSING_COLUMNS": "חסרות עמודות בקובץ הנתונים.",
    "INPUT_FILE_UNSUPPORTED": "פורמט קובץ לא נתמך.",
    "INPUT_MODEL_PARAMS_INVALID": "model_params חייב להיות אובייקט JSON (dict).",

    # מודלים
    "MODEL_NOT_FOUND": "מודל לא נמצא.",
    "MODEL_KIND_MISMATCH": "סוג המודל לא תואם לפעולה.",
    "MODEL_TRAIN_FAILED": "אימון המודל נכשל.",
    "MODEL_PREDICT_FAILED": "חיזוי נכשל.",
    "MODEL_BUNDLE_CORRUPT": "קובץ המודל השמור פגום/חסר רכיב.",

    # כללי
    "INTERNAL_ERROR": "שגיאה פנימית. נסה שוב מאוחר יותר.",
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
