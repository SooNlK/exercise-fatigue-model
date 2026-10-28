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
    label: 'Ocena zmęczenia (1–5)',
    min: 1,
    max: 5,
    step: 1,
    required: true,
    hint: '1 = duże zmęczenie, 5 = pełna świeżość.',
  },
  {
    name: 'readiness',
    label: 'Gotowość do treningu (1–10)',
    min: 1,
    max: 10,
    step: 1,
  },
  {
    name: 'soreness',
    label: 'Ocena bolesności mięśni (1–5)',
    min: 1,
    max: 5,
    step: 1,
  },
  {
    name: 'sleep_duration',
    label: 'Długość snu w godzinach',
    min: 0,
    max: 24,
    step: 'any',
  },
  {
    name: 'sleep_quality',
    label: 'Jakość snu (1–5)',
    min: 1,
    max: 5,
    step: 1,
  },
  {
    name: 'mood',
    label: 'Nastrój (1–5)',
    min: 1,
    max: 5,
    step: 1,
  },
  {
    name: 'stress',
    label: 'Ocena stresu (1–5)',
    min: 1,
    max: 5,
    step: 1,
  },
  {
    name: 'daily_srpe',
    label: 'Dzienne obciążenie sRPE',
    min: 0,
    step: 'any',
    hint: 'Podaj obciążenie sRPE, a nie samą ocenę RPE.',
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
        Data i ocena zmęczenia są wymagane.
        Pozostałe pola możesz pozostawić puste.
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
