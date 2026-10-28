export type User = {
  id: string
  email: string
  created_at: string
}

type TokenResponse = {
  access_token: string
  token_type: 'bearer'
}

async function request<T>(
  path: string,
  options: RequestInit,
): Promise<T> {
  let response: Response

  try {
    response = await fetch(`/api${path}`, options)
  } catch {
    throw new Error(
      'Nie udało się połączyć z serwerem. Spróbuj ponownie.',
    )
  }

  const body = await response.json().catch(() => null)

  if (!response.ok) {
    if (response.status >= 500) {
      throw new Error(
        'Serwer jest chwilowo niedostępny. Spróbuj ponownie.',
      )
    }

    if (response.status === 422) {
      throw new Error('Sprawdź poprawność wprowadzonych danych.')
    }

    const message =
      typeof body?.detail === 'string'
        ? body.detail
        : 'Nie udało się wykonać żądania.'

    throw new Error(message)
  }

  if (body === null) {
    throw new Error('Serwer zwrócił nieprawidłową odpowiedź.')
  }

  return body as T
}

export function login(
  email: string,
  password: string,
): Promise<TokenResponse> {
  return request<TokenResponse>('/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email: email.trim(),
      password,
    }),
  })
}

export function getCurrentUser(token: string): Promise<User> {
  return request<User>('/me', {
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })
}

export function register(
  email: string,
  password: string,
): Promise<User> {
  return request<User>('/auth/register', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email: email.trim(),
      password,
    }),
  })
}