import { useState } from 'react'
import type { FormEvent } from 'react'
import { getCurrentUser, login } from './api'
import type { User } from './api'

type Session = {
  token: string
  user: User
}

function App() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [session, setSession] = useState<Session | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (isLoading) return

    setError('')
    setIsLoading(true)

    try {
      // Najpierw pobieramy token.
      const tokenResponse = await login(email, password)

      // Następnie sprawdzamy token i pobieramy użytkownika.
      const user = await getCurrentUser(tokenResponse.access_token)

      setSession({
        token: tokenResponse.access_token,
        user,
      })

      setPassword('')
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : 'Wystąpił nieoczekiwany błąd.',
      )
    } finally {
      setIsLoading(false)
    }
  }

  function handleLogout() {
    setSession(null)
    setEmail('')
    setPassword('')
    setError('')
  }

  if (session) {
    return (
      <main className="page">
        <section className="card" aria-labelledby="account-title">
          <h1 id="account-title">Twoje konto</h1>

          <p>
            Zalogowano jako <strong>{session.user.email}</strong>
          </p>

          <button type="button" onClick={handleLogout}>
            Wyloguj
          </button>
        </section>
      </main>
    )
  }

  return (
    <main className="page">
      <section className="card" aria-labelledby="login-title">
        <h1 id="login-title">Zaloguj się</h1>
        <p>Uzyskaj dostęp do swoich raportów i prognoz.</p>

        <form onSubmit={handleLogin} aria-busy={isLoading}>
          <fieldset disabled={isLoading}>
            <label htmlFor="email">Adres e-mail</label>
            <input
              id="email"
              name="email"
              type="email"
              autoComplete="username"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              maxLength={254}
              required
            />

            <label htmlFor="password">Hasło</label>
            <input
              id="password"
              name="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              maxLength={128}
              required
            />

            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}

            <button type="submit">
              {isLoading ? 'Logowanie…' : 'Zaloguj'}
            </button>
          </fieldset>
        </form>
      </section>
    </main>
  )
}

export default App