import { useCallback, useState } from 'react'
import DailyLogForm from './DailyLogForm'
import DailyLogHistory from './DailyLogHistory'
import PredictionPanel from './PredictionPanel'

type Props = {
  token: string
  onUnauthorized: () => void
}

export default function ReportsPanel({
  token,
  onUnauthorized,
}: Props) {
  const [refreshVersion, setRefreshVersion] = useState(0)
  const [isSaving, setIsSaving] = useState(false)

  const [selection, setSelection] = useState({
    date: '',
    version: 0,
  })

  const handleSaved = useCallback(() => {
    // Odświeżamy historię i unieważniamy poprzednią prognozę.
    setRefreshVersion((value) => value + 1)
  }, [])

  function handleEdit(date: string) {
    if (isSaving) return

    setSelection((current) => ({
      date,
      version: current.version + 1,
    }))

    document
      .getElementById('daily-log-title')
      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  return (
    <>
      <div className="reports-layout">
        <DailyLogForm
          key={selection.version}
          token={token}
          initialDate={selection.date}
          onSaved={handleSaved}
          onSavingChange={setIsSaving}
          onUnauthorized={onUnauthorized}
        />

        <DailyLogHistory
          token={token}
          refreshVersion={refreshVersion}
          editingDisabled={isSaving}
          onEdit={handleEdit}
          onUnauthorized={onUnauthorized}
        />
      </div>

      <PredictionPanel
        token={token}
        dataVersion={refreshVersion}
        disabled={isSaving}
        onUnauthorized={onUnauthorized}
      />
    </>
  )
}