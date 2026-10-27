class AppError(Exception):
    """Bazowy wyjątek aplikacji."""


class NoUserDataError(AppError):
    """Brak wpisów dla użytkownika."""


class InsufficientHistoryError(AppError):
    """Za mało danych historycznych do wykonania predykcji."""


class PredictionError(AppError):
    """Błąd podczas wykonywania predykcji modelu."""


class EmailAlreadyExistsError(AppError):
    """Podany adres e-mail jest już zajęty."""


class InvalidCredentialsError(AppError):
    """Niepoprawne dane logowania."""