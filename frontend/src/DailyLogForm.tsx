import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import {
  getDailyLog,
  saveDailyLog,
  UnauthorizedError,
} from './api'
import type { DailyLogInput } from './api'

type Props = {
  initialDate?: string
  onSaved: () => void
  onSavingChange: (saving: boolean) => void
  onUnauthorized: () => void
}

type Values = Record<
  Exclude<keyof DailyLogInput, 'date'>,
  string
>

type Field = {
  name: keyof Values
  label: string
  min: number
  max?: number
  step: number | 'any'
  required?: boolean
  hint?: string
}

const fields: Field[] = [
  {
    name: 'fatigue',
    label: 'Jak oceniasz swoje zmęczenie? (1–5)',
    min: 1,
    max: 5,
    step: 1,
    required: true,
    hint:
      '1 = bardzo duże zmęczenie, ' +
      '3 = normalne samopoczucie, ' +
      '5 = pełna świeżość.',
  },
  {
    name: 'readiness',
    label: 'Gotowość do treningu (1–10)',
    min: 1,
    max: 10,
    step: 1,
    hint:
      '1 = brak gotowości do treningu, ' +
      '10 = bardzo duża gotowość.',
  },
  {
    name: 'soreness',
    label: 'Samopoczucie mięśni (1–5)',
    min: 1,
    max: 5,
    step: 1,
    hint:
      '1 = bardzo duża bolesność, ' +
      '3 = normalne samopoczucie mięśni, ' +
      '5 = bardzo dobre samopoczucie mięśni.',
  },
  {
    name: 'sleep_duration',
    label: 'Długość snu poprzedniej nocy [h]',
    min: 0,
    max: 24,
    step: 'any',
    hint:
      'Podaj liczbę godzin. ' +
      'Przykład: 7,5 godziny oznacza 7 godzin i 30 minut.',
  },
  {
    name: 'sleep_quality',
    label: 'Jakość snu poprzedniej nocy (1–5)',
    min: 1,
    max: 5,
    step: 1,
    hint:
      '1 = bardzo zły sen lub brak snu, ' +
      '3 = normalny sen, ' +
      '5 = bardzo dobry, spokojny sen.',
  },
  {
    name: 'mood',
    label: 'Nastrój (1–5)',
    min: 1,
    max: 5,
    step: 1,
    hint:
      '1 = bardzo zły nastrój, ' +
      '3 = normalny nastrój, ' +
      '5 = bardzo dobry nastrój.',
  },
  {
    name: 'stress',
    label: 'Samopoczucie związane ze stresem (1–5)',
    min: 1,
    max: 5,
    step: 1,
    hint:
      '1 = bardzo duży stres, ' +
      '3 = normalne samopoczucie, ' +
      '5 = duże odprężenie.',
  },
  {
    name: 'daily_srpe',
    label: 'Łączne obciążenie treningowe dnia — sRPE',
    min: 0,
    step: 'any',
    hint:
      'Dla każdej sesji pomnóż czas w minutach przez RPE (1–10), ' +
      'a następnie zsumuj wyniki. ' +
      'Przykład: 60 minut × RPE 6 = 360. ' +
      'Jeżeli nie znasz obciążenia, pozostaw pole puste.',
  },
]

function emptyValues(): Values {
  return {
    fatigue: '',
    soreness: '',
    readiness: '',
    sleep_duration: '',
    sleep_quality: '',
    mood: '',
    stress: '',
    daily_srpe: '',
  }
}

function localToday(): string {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')

  return `${year}-${month}-${day}`
}

function optionalNumber(value: string): number | null {
  return value.trim() === '' ? null : Number(value)
}

export default function DailyLogForm({
  initialDate,
  onSaved,
  onSavingChange,
  onUnauthorized,
}: Props) {
  const [date, setDate] = useState(() => initialDate || localToday())
  const [values, setValues] = useState<Values>(emptyValues)
  const [loaded, setLoaded] = useState(false)
  const [existing, setExisting] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [reload, setReload] = useState(0)

  useEffect(() => {
    let ignore = false

    async function load() {
      setLoaded(false)
      setError('')
      setMessage('')

      try {
        const entry = await getDailyLog(date)

        // Nie stosujemy odpowiedzi dotyczącej poprzedniej daty.
        if (ignore) return

        const nextValues = emptyValues()

        if (entry) {
          for (const field of fields) {
            const value = entry[field.name]
            nextValues[field.name] =
              value === null ? '' : String(value)
          }
        }

        setValues(nextValues)
        setExisting(entry !== null)
        setLoaded(true)
      } catch (error) {
        if (ignore) return

        if (error instanceof UnauthorizedError) {
          onUnauthorized()
          return
        }

        setError(
          error instanceof Error
            ? error.message
            : 'Nie udało się wczytać raportu.',
        )
      }
    }

    if (date) {
      void load()
    }

    return () => {
      ignore = true
    }
  }, [date, reload, onUnauthorized])

  function changeDate(value: string) {
    setDate(value)
    setLoaded(false)
    setValues(emptyValues())
    setError('')
    setMessage('')
  }

  async function handleSave(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!loaded || isSaving) return

    setError('')
    setMessage('')
    setIsSaving(true)
    onSavingChange(true)

    const entry: DailyLogInput = {
      date,
      fatigue: Number(values.fatigue),
      soreness: optionalNumber(values.soreness),
      readiness: optionalNumber(values.readiness),
      sleep_duration: optionalNumber(values.sleep_duration),
      sleep_quality: optionalNumber(values.sleep_quality),
      mood: optionalNumber(values.mood),
      stress: optionalNumber(values.stress),
      daily_srpe: optionalNumber(values.daily_srpe),
    }

    try {
      await saveDailyLog(entry)

      setExisting(true)
      setMessage(`Zapisano raport za dzień ${date}.`)
      onSaved()
    } catch (error) {
      if (error instanceof UnauthorizedError) {
        onUnauthorized()
        return
      }

      setError(
        error instanceof Error
          ? error.message
          : 'Nie udało się zapisać raportu.',
      )
    } finally {
      setIsSaving(false)
      onSavingChange(false)
    }
  }

  return (
      <section className="daily-log" aria-labelledby="daily-log-title">
        <h2 id="daily-log-title">Raport dzienny</h2>

        <p className="hint">
          Wypełnij raport dla wybranego dnia.
          Data i ocena zmęczenia są wymagane.
          Wyższe oceny samopoczucia oznaczają korzystniejszy stan.
          Sen dotyczy poprzedniej nocy, a obciążenie treningowe
          obejmuje wszystkie sesje wybranego dnia.
        </p>

        <p className="hint">
          Nieznane wartości pozostaw puste.
          Zero oznacza znaną wartość równą zero, a nie brak informacji.
          Prognozę wykonuj po uzupełnieniu obciążenia całego dnia,
          jeśli je znasz.
        </p>

        <label htmlFor="report-date">Data raportu</label>
        <input
            id="report-date"
            type="date"
            value={date}
            onChange={(event) => changeDate(event.target.value)}
            disabled={isSaving}
            required
        />

        {!date && <p>Wybierz datę raportu.</p>}

        {date && !loaded && !error && (
            <p role="status">Wczytywanie raportu…</p>
        )}

        {error && (
            <p className="error" role="alert">
              {error}
            </p>
        )}

        {date && !loaded && error && (
            <button
                type="button"
                onClick={() => setReload((value) => value + 1)}
            >
              Spróbuj wczytać ponownie
            </button>
        )}

        {loaded && (
            <form onSubmit={handleSave} aria-busy={isSaving}>
              <p className="hint">
                {existing
                    ? 'Edytujesz zapisany raport. Zapis zastąpi jego dane.'
                    : 'Dla tej daty nie ma jeszcze raportu.'}
              </p>

              <fieldset disabled={isSaving}>
                {fields.map((field) => (
                    <div className="form-field" key={field.name}>
                      <label htmlFor={`report-${field.name}`}>
                        {field.label}
                        {field.required ? ' *' : ''}
                      </label>

                      <input
                          id={`report-${field.name}`}
                          name={field.name}
                          type="number"
                          min={field.min}
                          max={field.max}
                          step={field.step}
                          required={field.required}
                          value={values[field.name]}
                          aria-describedby={
                            field.hint ? `hint-${field.name}` : undefined
                          }
                          onChange={(event) => {
                            setValues((current) => ({
                              ...current,
                              [field.name]: event.target.value,
                            }))
                            setMessage('')
                          }}
                      />

                      {field.hint && (
                          <p id={`hint-${field.name}`} className="hint">
                            {field.hint}
                          </p>
                      )}
                    </div>
                ))}

                <button type="submit">
                  {isSaving ? 'Zapisywanie…' : 'Zapisz raport'}
                </button>
              </fieldset>
            </form>
        )}

        {message && (
            <p className="success" role="status">
              {message}
            </p>
        )}
      </section>
  )
}
