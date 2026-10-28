import { useCallback, useState } from 'react'
import {
  Navigate,
  NavLink,
  Outlet,
  Route,
  Routes,
} from 'react-router'
import type { User } from './api'
import AuthPage from './AuthPage'
import ReportsPanel from './ReportsPanel'
import PredictionPanel from './PredictionPanel'

type Session = {
  token: string
  user: User
}

export default function App() {
  const [session, setSession] = useState<Session | null>(null)
  const [dataVersion, setDataVersion] = useState(0)
  const [isSaving, setIsSaving] = useState(false)

  const handleLogin = useCallback((token: string, user: User) => {
    setSession({ token, user })
    setDataVersion(0)
    setIsSaving(false)
  }, [])

  const handleLogout = useCallback(() => {
    setSession(null)
    setDataVersion(0)
    setIsSaving(false)
  }, [])

  const handleSaved = useCallback(() => {
    setDataVersion((value) => value + 1)
  }, [])

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
                    aria-disabled={isSaving}
                    onClick={(event) => {
                      if (isSaving) event.preventDefault()
                    }}
                  >
                    Raporty
                  </NavLink>

                  <NavLink
                    to="/prediction"
                    aria-disabled={isSaving}
                    onClick={(event) => {
                      if (isSaving) event.preventDefault()
                    }}
                  >
                    Prognoza
                  </NavLink>

                  <button
                    type="button"
                    onClick={handleLogout}
                    disabled={isSaving}
                  >
                    Wyloguj
                  </button>
                </nav>

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
                token={session.token}
                refreshVersion={dataVersion}
                isSaving={isSaving}
                onSaved={handleSaved}
                onSavingChange={setIsSaving}
                onUnauthorized={handleLogout}
              />
            )
          }
        />

        <Route
          path="/prediction"
          element={
            session && (
              <PredictionPanel
                token={session.token}
                dataVersion={dataVersion}
                disabled={isSaving}
                onUnauthorized={handleLogout}
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