import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { getDailyLogs, UnauthorizedError } from './api'
import type { DailyLog } from './api'

type Props = {
  refreshVersion: number
  editingDisabled: boolean
  onEdit: (date: string) => void
  onUnauthorized: () => void
}

function displayNumber(value: number | null): string {
  return value === null ? '—' : value.toLocaleString('pl-PL')
}

function displayDate(date: string): string {
  return date.split('-').reverse().join('.')
}

export default function DailyLogHistory({
  refreshVersion,
  editingDisabled,
  onEdit,
  onUnauthorized,
}: Props) {
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')

  // Pola formularza stosujemy dopiero po kliknięciu „Pokaż”.
  const [filters, setFilters] = useState({
    dateFrom: '',
    dateTo: '',
  })

  const [logs, setLogs] = useState<DailyLog[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState('')
  const [filterError, setFilterError] = useState('')
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    let ignore = false

    async function load() {
      setIsLoading(true)
      setError('')

      try {
        const entries = await getDailyLogs(filters)

        if (ignore) return

        // Najnowszy raport wyświetlamy jako pierwszy.
        setLogs(
          [...entries].sort((a, b) => b.date.localeCompare(a.date)),
        )
      } catch (error) {
        if (ignore) return

        if (error instanceof UnauthorizedError) {
          onUnauthorized()
          return
        }

        setError(
          error instanceof Error
            ? error.message
            : 'Nie udało się pobrać historii.',
        )
      } finally {
        if (!ignore) {
          setIsLoading(false)
        }
      }
    }

    void load()

    return () => {
      ignore = true
    }
  }, [filters, refreshVersion, retry, onUnauthorized])

  function handleFilter(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (dateFrom && dateTo && dateFrom > dateTo) {
      setFilterError(
        'Data początkowa nie może być późniejsza od końcowej.',
      )
      return
    }

    setFilterError('')
    setFilters({ dateFrom, dateTo })
  }

  return (
    <section className="history" aria-labelledby="history-title">
      <h2 id="history-title">Historia raportów</h2>

      <form onSubmit={handleFilter}>
        <div className="date-range">
          <div className="form-field">
            <label htmlFor="history-from">Od dnia</label>
            <input
              id="history-from"
              type="date"
              value={dateFrom}
              onChange={(event) => setDateFrom(event.target.value)}
            />
          </div>

          <div className="form-field">
            <label htmlFor="history-to">Do dnia</label>
            <input
              id="history-to"
              type="date"
              value={dateTo}
              onChange={(event) => setDateTo(event.target.value)}
            />
          </div>
        </div>

        <button type="submit">Pokaż</button>

        {filterError && (
          <p className="error" role="alert">
            {filterError}
          </p>
        )}
      </form>

      {isLoading && <p role="status">Wczytywanie historii…</p>}

      {!isLoading && error && (
        <>
          <p className="error" role="alert">
            {error}
          </p>
          <button
            type="button"
            onClick={() => setRetry((value) => value + 1)}
          >
            Spróbuj ponownie
          </button>
        </>
      )}

      {!isLoading && !error && logs.length === 0 && (
        <p>Brak raportów w wybranym zakresie dat.</p>
      )}

      {!isLoading && !error && logs.length > 0 && (
        <div className="table-scroll">
          <table>
            <caption>
              Zapisane raporty. Zmęczenie: 1 = duże zmęczenie,
              5 = pełna świeżość. „—” oznacza brak danych.
            </caption>

            <thead>
              <tr>
                <th scope="col">Data</th>
                <th scope="col">Zmęczenie</th>
                <th scope="col">Sen [h]</th>
                <th scope="col">Gotowość</th>
                <th scope="col">sRPE</th>
                <th scope="col">Akcja</th>
              </tr>
            </thead>

            <tbody>
              {logs.map((log) => (
                <tr key={log.id}>
                  <th scope="row">{displayDate(log.date)}</th>
                  <td>{displayNumber(log.fatigue)}</td>
                  <td>{displayNumber(log.sleep_duration)}</td>
                  <td>{displayNumber(log.readiness)}</td>
                  <td>{displayNumber(log.daily_srpe)}</td>
                  <td>
                    <button
                      type="button"
                      disabled={editingDisabled}
                      onClick={() => onEdit(log.date)}
                      aria-label={
                        `Edytuj raport z ${displayDate(log.date)}`
                      }
                    >
                      Edytuj
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
