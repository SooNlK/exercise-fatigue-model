export type User = {
  id: string
  email: string
  created_at: string
}

export class UnauthorizedError extends Error {}

export type DailyLogInput = {
  date: string
  fatigue: number
  soreness: number | null
  readiness: number | null
  sleep_duration: number | null
  sleep_quality: number | null
  mood: number | null
  stress: number | null
  daily_srpe: number | null
}

export type DailyLog = DailyLogInput & {
  id: number
  user_id: string
}

export type Prediction = {
  user_id: string
  prediction_date: string
  target_date: string
  current_fatigue: number
  deterioration_probability: number
  model_id: string
  method: string
  probability_source: string
  training_rows_for_person: number
  known_person_with_20_training_pairs: boolean
  intense_day: boolean
  intensity_status: string
  target_definition: string
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers)
  const method = (options.method ?? 'GET').toUpperCase()

  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    headers.set('X-CSRF-Protection', '1')
  }

  let response: Response

  try {
    response = await fetch(`/api${path}`, {
      ...options,
      headers,
      credentials: 'same-origin',
      cache: 'no-store',
    })
  } catch {
    throw new Error(
      'Nie udało się połączyć z serwerem. Spróbuj ponownie.',
    )
  }

  const body = await response.json().catch(() => null)

  if (!response.ok) {
    if (response.status === 401) {
      throw new UnauthorizedError(
        typeof body?.detail === 'string'
          ? body.detail
          : 'Zaloguj się ponownie.',
      )
    }

    if (response.status >= 500) {
      throw new Error(
        'Serwer jest chwilowo niedostępny. Spróbuj ponownie.',
      )
    }

    if (response.status === 422) {
      throw new Error('Sprawdź poprawność wprowadzonych danych.')
    }

    throw new Error(
      typeof body?.detail === 'string'
        ? body.detail
        : 'Nie udało się wykonać żądania.',
    )
  }

  if (body === null) {
    throw new Error('Serwer zwrócił nieprawidłową odpowiedź.')
  }

  return body as T
}

export function login(
  email: string,
  password: string,
): Promise<{ message: string }> {
  return request('/auth/login', {
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

export function logout(): Promise<{ message: string }> {
  return request('/auth/logout', {
    method: 'POST',
  })
}

export function getCurrentUser(): Promise<User> {
  return request<User>('/me')
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

export function getDailyLogs(
  filters: {
    dateFrom?: string
    dateTo?: string
  } = {},
): Promise<DailyLog[]> {
  const params = new URLSearchParams()

  if (filters.dateFrom) {
    params.set('date_from', filters.dateFrom)
  }

  if (filters.dateTo) {
    params.set('date_to', filters.dateTo)
  }

  const query = params.toString()

  return request<DailyLog[]>(
    `/me/daily-logs${query ? `?${query}` : ''}`,
  )
}

export async function getDailyLog(
  date: string,
): Promise<DailyLog | null> {
  const logs = await getDailyLogs({
    dateFrom: date,
    dateTo: date,
  })

  return logs[0] ?? null
}

export function saveDailyLog(
  entry: DailyLogInput,
): Promise<DailyLog> {
  return request<DailyLog>('/me/daily-log', {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(entry),
  })
}

export function getPrediction(
  date: string,
): Promise<Prediction> {
  const params = new URLSearchParams({
    prediction_date: date,
  })

  return request<Prediction>(
    `/me/predict?${params.toString()}`,
  )
}
