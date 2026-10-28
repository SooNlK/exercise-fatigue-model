import { useState } from 'react'
import DailyLogForm from './DailyLogForm'
import DailyLogHistory from './DailyLogHistory'

type Props = {
  refreshVersion: number
  isSaving: boolean
  onSaved: () => void
  onSavingChange: (saving: boolean) => void
  onUnauthorized: () => void
}

export default function ReportsPanel({
  refreshVersion,
  isSaving,
  onSaved,
  onSavingChange,
  onUnauthorized,
}: Props) {
  const [selection, setSelection] = useState({
    date: '',
    version: 0,
  })

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
    <div className="reports-layout">
      <DailyLogForm
        key={selection.version}
        initialDate={selection.date}
        onSaved={onSaved}
        onSavingChange={onSavingChange}
        onUnauthorized={onUnauthorized}
      />

      <DailyLogHistory
        refreshVersion={refreshVersion}
        editingDisabled={isSaving}
        onEdit={handleEdit}
        onUnauthorized={onUnauthorized}
      />
    </div>
  )
}
