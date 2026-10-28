import { useCallback, useEffect, useState } from 'react'
import {
  Navigate,
  NavLink,
  Outlet,
  Route,
  Routes,
} from 'react-router'
import type { User } from './api'
import { getCurrentUser, logout, UnauthorizedError } from './api'
import AuthPage from './AuthPage'
import ReportsPanel from './ReportsPanel'
import PredictionPanel from './PredictionPanel'

type Session = {
  user: User
}

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [dataVersion, setDataVersion] = useState(0)
  const [isSaving, setIsSaving] = useState(false)
  const [isRestoring, setIsRestoring] = useState(true)
  const [restoreError, setRestoreError] = useState('')
  const [restoreAttempt, setRestoreAttempt] = useState(0)
  const [isLoggingOut, setIsLoggingOut] = useState(false)
  const [logoutError, setLogoutError] = useState('')

  const handleLogin = useCallback((user: User) => {
    setSession({ user })
    setDataVersion(0)
    setIsSaving(false)
    setLogoutError('')
  }, [])

  const clearSession = useCallback(() => {
    setSession(null)
    setDataVersion(0)
    setIsSaving(false)
    setLogoutError('')
  }, [])

  async function handleLogout() {
    if (isLoggingOut || isSaving) return

    setIsLoggingOut(true)
    setLogoutError('')

    try {
      await logout()
      clearSession()
    } catch (error) {
      setLogoutError(
        error instanceof Error
          ? error.message
          : 'Nie udało się wylogować.',
      )
    } finally {
      setIsLoggingOut(false)
    }
  }

  const handleSaved = useCallback(() => {
    setDataVersion((value) => value + 1)
  }, [])

  // Nie przekierowujemy na logowanie, zanim API nie sprawdzi ciasteczka.
  useEffect(() => {
    let ignore = false

    async function restoreSession() {
      setIsRestoring(true)
      setRestoreError('')

      try {
        const user = await getCurrentUser()

        if (!ignore) setSession({ user })
      } catch (error) {
        if (ignore) return

        if (error instanceof UnauthorizedError) {
          setSession(null)
        } else {
          setRestoreError(
            error instanceof Error
              ? error.message
              : 'Nie udało się sprawdzić sesji.',
          )
        }
      } finally {
        if (!ignore) setIsRestoring(false)
      }
    }

    void restoreSession()

    return () => {
      ignore = true
    }
  }, [restoreAttempt])

  if (isRestoring) {
    return (
      <main className="page">
        <p role="status">Sprawdzanie sesji…</p>
      </main>
    )
  }

  if (restoreError) {
    return (
      <main className="page">
        <section className="card">
          <h1>Nie można sprawdzić sesji</h1>
          <p className="error" role="alert">{restoreError}</p>
          <button
            type="button"
            onClick={() => setRestoreAttempt((value) => value + 1)}
          >
            Spróbuj ponownie
          </button>
        </section>
      </main>
    )
  }

  const home = session ? '/reports' : '/login'

  return (
    <Routes>
      <Route
        path="/"
        element={<Navigate to={home} replace />}
      />

      <Route
        path="/login"
        element={
          session ? (
            <Navigate to="/reports" replace />
          ) : (
            <AuthPage
              key="login"
              mode="login"
              onLogin={handleLogin}
            />
          )
        }
      />

      <Route
        path="/register"
        element={
          session ? (
            <Navigate to="/reports" replace />
          ) : (
            <AuthPage
              key="register"
              mode="register"
              onLogin={handleLogin}
            />
          )
        }
      />

      <Route
        element={
          session ? (
            <main className="page">
              <section
                className="card dashboard-card"
                aria-labelledby="account-title"
              >
                <h1 id="account-title">Twoje konto</h1>

                <p>
                  Zalogowano jako{' '}
                  <strong>{session.user.email}</strong>
                </p>

                <nav className="main-nav" aria-label="Menu główne">
                  <NavLink
                    to="/reports"
                    aria-disabled={isSaving || isLoggingOut}
                    onClick={(event) => {
                      if (isSaving || isLoggingOut) event.preventDefault()
                    }}
                  >
                    Raporty
                  </NavLink>

                  <NavLink
                    to="/prediction"
                    aria-disabled={isSaving || isLoggingOut}
                    onClick={(event) => {
                      if (isSaving || isLoggingOut) event.preventDefault()
                    }}
                  >
                    Prognoza
                  </NavLink>

                  <button
                    type="button"
                    onClick={() => void handleLogout()}
                    disabled={isSaving || isLoggingOut}
                  >
                    {isLoggingOut ? 'Wylogowywanie…' : 'Wyloguj'}
                  </button>
                </nav>

                {logoutError && (
                  <p className="error" role="alert">{logoutError}</p>
                )}

                <Outlet />
              </section>
            </main>
          ) : (
            <Navigate to="/login" replace />
          )
        }
      >
        <Route
          path="/reports"
          element={
            session && (
              <ReportsPanel
                refreshVersion={dataVersion}
                isSaving={isSaving}
                onSaved={handleSaved}
                onSavingChange={setIsSaving}
                onUnauthorized={clearSession}
              />
            )
          }
        />

        <Route
          path="/prediction"
          element={
            session && (
              <PredictionPanel
                dataVersion={dataVersion}
                disabled={isSaving}
                onUnauthorized={clearSession}
              />
            )
          }
        />
      </Route>

      <Route
        path="*"
        element={<Navigate to={home} replace />}
      />
    </Routes>
  )
}
