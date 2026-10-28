import { useState } from 'react'
import type { FormEvent } from 'react'
import { getCurrentUser, login, register } from './api'
import type { User } from './api'

type Session = {
  token: string
  user: User
}

type Mode = 'login' | 'register'

function App() {
  const [mode, setMode] = useState<Mode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [session, setSession] = useState<Session | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')

  const isRegister = mode === 'register'

  function changeMode() {
    setMode(isRegister ? 'login' : 'register')
    setPassword('')
    setConfirmPassword('')
    setError('')
    setMessage('')
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (isLoading) return

    setError('')
    setMessage('')

    // Powtórzenie hasła sprawdzamy tylko podczas rejestracji.
    if (isRegister) {
      if (password.length < 8) {
        setError('Hasło musi mieć co najmniej 8 znaków.')
        return
      }

      if (password !== confirmPassword) {
        setError('Podane hasła muszą być takie same.')
        return
      }
    }

    setIsLoading(true)

    try {
      if (isRegister) {
        const user = await register(email, password)

        // Po utworzeniu konta wracamy do logowania.
        setMode('login')
        setEmail(user.email)
        setPassword('')
        setConfirmPassword('')
        setMessage('Konto zostało utworzone. Możesz się zalogować.')
      } else {
        const tokenResponse = await login(email, password)
        const user = await getCurrentUser(tokenResponse.access_token)

        setSession({
          token: tokenResponse.access_token,
          user,
        })

        setPassword('')
        setConfirmPassword('')
      }
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
    setMode('login')
    setEmail('')
    setPassword('')
    setConfirmPassword('')
    setError('')
    setMessage('')
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
      <section className="card" aria-labelledby="auth-title">
        <h1 id="auth-title">
          {isRegister ? 'Utwórz konto' : 'Zaloguj się'}
        </h1>

        <p>
          {isRegister
            ? 'Załóż konto, aby zapisywać raporty i sprawdzać prognozy.'
            : 'Uzyskaj dostęp do swoich raportów i prognoz.'}
        </p>

        {message && (
          <p className="success" role="status">
            {message}
          </p>
        )}

        <form onSubmit={handleSubmit} aria-busy={isLoading}>
          <fieldset disabled={isLoading}>
            <label htmlFor="email">Adres e-mail</label>
            <input
              id="email"
              name="email"
              type="email"
              autoComplete={isRegister ? 'email' : 'username'}
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
              autoComplete={
                isRegister ? 'new-password' : 'current-password'
              }
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              minLength={isRegister ? 8 : undefined}
              maxLength={128}
              aria-describedby={isRegister ? 'password-hint' : undefined}
              required
            />

            {isRegister && (
              <>
                <p id="password-hint" className="hint">
                  Hasło musi mieć od 8 do 128 znaków.
                </p>

                <label htmlFor="confirm-password">
                  Powtórz hasło
                </label>
                <input
                  id="confirm-password"
                  name="confirm-password"
                  type="password"
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(event) =>
                    setConfirmPassword(event.target.value)
                  }
                  minLength={8}
                  maxLength={128}
                  required
                />
              </>
            )}

            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}

            <button type="submit">
              {isLoading
                ? isRegister
                  ? 'Tworzenie konta…'
                  : 'Logowanie…'
                : isRegister
                  ? 'Utwórz konto'
                  : 'Zaloguj'}
            </button>
          </fieldset>
        </form>

        <button
          className="secondary"
          type="button"
          onClick={changeMode}
          disabled={isLoading}
        >
          {isRegister
            ? 'Masz już konto? Zaloguj się'
            : 'Nie masz konta? Zarejestruj się'}
        </button>
      </section>
    </main>
  )
}

export default App