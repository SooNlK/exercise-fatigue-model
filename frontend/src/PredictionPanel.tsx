import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { getPrediction, UnauthorizedError } from './api'
import type { Prediction } from './api'

type Props = {
  dataVersion: number
  disabled: boolean
  onUnauthorized: () => void
}

type Result = {
  prediction: Prediction
  dataVersion: number
}

function localToday(): string {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')

  return `${year}-${month}-${day}`
}

function displayDate(date: string): string {
  return date.split('-').reverse().join('.')
}

export default function PredictionPanel({
  dataVersion,
  disabled,
  onUnauthorized,
}: Props) {
  const [date, setDate] = useState(localToday)
  const [result, setResult] = useState<Result | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const requestId = useRef(0)

  // Po zmianie danych lub rozpoczęciu zapisu ignorujemy
  // odpowiedź z wcześniejszego żądania.
  useEffect(() => {
    requestId.current += 1
    setIsLoading(false)
    setError('')

    return () => {
      requestId.current += 1
    }
  }, [dataVersion, disabled])

  function changeDate(value: string) {
    requestId.current += 1
    setDate(value)
    setResult(null)
    setError('')
    setIsLoading(false)
  }

  async function handlePredict(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault()

    if (!date || disabled || isLoading) return

    const currentRequest = ++requestId.current

    setIsLoading(true)
    setError('')
    setResult(null)

    try {
      const prediction = await getPrediction(date)

      if (currentRequest !== requestId.current) return

      setResult({
        prediction,
        dataVersion,
      })
    } catch (error) {
      if (currentRequest !== requestId.current) return

      if (error instanceof UnauthorizedError) {
        onUnauthorized()
        return
      }

      setError(
        error instanceof Error
          ? error.message
          : 'Nie udało się obliczyć prognozy.',
      )
    } finally {
      if (currentRequest === requestId.current) {
        setIsLoading(false)
      }
    }
  }

  const isOutdated =
    result !== null && result.dataVersion !== dataVersion

  const prediction =
    result && !isOutdated && !disabled
      ? result.prediction
      : null

  const percentage = prediction
    ? new Intl.NumberFormat('pl-PL', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }).format(prediction.deterioration_probability)
    : ''

  return (
    <section
      className="prediction-panel"
      aria-labelledby="prediction-title"
    >
      <h2 id="prediction-title">Prognoza na następny dzień</h2>

      <p>
        Wybierz dzień, dla którego masz zapisany raport.
        Prognoza będzie dotyczyć dnia następnego.
      </p>

      <form onSubmit={handlePredict} aria-busy={isLoading}>
        <fieldset disabled={disabled}>
          <label htmlFor="prediction-date">
            Dzień, na podstawie którego obliczamy prognozę
          </label>

          <input
            id="prediction-date"
            type="date"
            value={date}
            onChange={(event) => changeDate(event.target.value)}
            required
          />

          <button
            type="submit"
            disabled={isLoading || !date}
          >
            {isLoading ? 'Obliczanie…' : 'Oblicz prognozę'}
          </button>
        </fieldset>
      </form>

      <p className="hint">
        Historia musi obejmować wybrany dzień i co najmniej
        28 wcześniejszych dni. Potrzebna jest ocena zmęczenia
        z wybranego dnia oraz co najmniej 7 takich ocen
        z poprzednich 21 dni. Nie musisz mieć kompletnych
        raportów ze wszystkich 29 dni.
      </p>

      {disabled && (
        <p role="status">
          Trwa zapis raportu. Zaczekaj na jego zakończenie.
        </p>
      )}

      {!disabled && isLoading && (
        <p role="status">Trwa obliczanie prognozy…</p>
      )}

      {!disabled && isOutdated && (
        <p className="notice" role="status">
          Dane zostały zmienione. Oblicz prognozę ponownie.
        </p>
      )}

      {!disabled && error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {prediction && (
        <div className="prediction-result" role="status">
          <h3>
            Prognoza na {displayDate(prediction.target_date)}
          </h3>

          <p>
            Prawdopodobieństwo pogorszenia oceny zmęczenia:
          </p>

          <p className="prediction-value">{percentage}</p>

          <p>
            Pogorszenie oznacza spadek oceny o co najmniej
            1 punkt względem dnia{' '}
            {displayDate(prediction.prediction_date)}.
            Niższa ocena oznacza większe zmęczenie.
          </p>

          <dl>
            <dt>Ocena w dniu prognozy</dt>
            <dd>{prediction.current_fatigue} / 5</dd>

            <dt>Intensywność dnia</dt>
            <dd>
              {prediction.intensity_status !== 'available'
                ? 'Brak wystarczających danych do oceny.'
                : prediction.intense_day
                  ? 'Dzień intensywny względem wcześniejszych obciążeń.'
                  : 'Dzień nie przekroczył progu intensywności.'}
            </dd>
          </dl>

          {prediction.current_fatigue === 1 && (
            <p className="hint">
              Przy ocenie 1 dalszy spadek w tej skali jest
              niemożliwy, dlatego prawdopodobieństwo tego
              zdarzenia wynosi 0%.
            </p>
          )}
        </div>
      )}
    </section>
  )
}
