import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { Link, useLocation, useNavigate } from 'react-router'
import { getCurrentUser, login, register } from './api'
import type { User } from './api'

type Props = {
  mode: 'login' | 'register'
  onLogin: (user: User) => void
}

type NavigationState = {
  email?: string
  message?: string
}

export default function AuthPage({ mode, onLogin }: Props) {
  const navigate = useNavigate()
  const location = useLocation()
  const state = location.state as NavigationState | null

  const [email, setEmail] = useState(state?.email ?? '')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [message, setMessage] = useState(state?.message ?? '')
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const active = useRef(true)
  const isRegister = mode === 'register'

  useEffect(() => {
    active.current = true

    return () => {
      active.current = false
    }
  }, [])

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (isLoading) return

    setError('')
    setMessage('')

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

        if (!active.current) return

        navigate('/login', {
          replace: true,
          state: {
            email: user.email,
            message: 'Konto zostało utworzone. Możesz się zalogować.',
          },
        })
      } else {
        await login(email, password)

        if (!active.current) return

        const user = await getCurrentUser()

        if (!active.current) return

        onLogin(user)
        navigate('/reports', { replace: true })
      }
    } catch (error) {
      if (!active.current) return

      setError(
        error instanceof Error
          ? error.message
          : 'Wystąpił nieoczekiwany błąd.',
      )
    } finally {
      if (active.current) {
        setIsLoading(false)
      }
    }
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

        <Link
          className="auth-switch"
          to={isRegister ? '/login' : '/register'}
          aria-disabled={isLoading}
          onClick={(event) => {
            if (isLoading) event.preventDefault()
          }}
        >
          {isRegister
            ? 'Masz już konto? Zaloguj się'
            : 'Nie masz konta? Zarejestruj się'}
        </Link>
      </section>
    </main>
  )
}
