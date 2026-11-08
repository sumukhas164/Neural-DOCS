import React, { useCallback, useEffect, useState } from 'react'

function csrf() {
  return document.cookie.split('; ').find(part => part.startsWith('csrftoken='))?.split('=')[1] || ''
}

async function api(path, options = {}) {
  let response
  try {
    response = await fetch(path, {
      credentials: 'same-origin',
      ...options,
      headers: { 'X-CSRFToken': csrf(), ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }), ...options.headers },
    })
  } catch {
    throw new Error('Cannot reach the app server. Check that Django is running on 127.0.0.1:8000, then refresh.')
  }
  if (!response.ok) {
    let body = {}
    try { body = await response.json() } catch { /* non-JSON error */ }
    const authRequired = response.status === 401 || (response.status === 403 &&
      typeof body.detail === 'string' && /authentication credentials were not provided|not authenticated/i.test(body.detail))
    const error = new Error(authRequired
      ? 'Sign in with a regular account to view and process your invoices. No superuser is needed.'
      : body.detail || `Request failed (${response.status}). Check the Django terminal for details.`)
    error.authRequired = authRequired
    throw error
  }
  return response.json()
}

function Status({ value }) {
  return <span className={`status ${value || 'pending'}`}>{(value || 'pending').replaceAll('_', ' ')}</span>
}

function downloadStructuredItems(run) {
  const file = new Blob([JSON.stringify(run.structured_items || [], null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(file)
  const link = document.createElement('a')
  link.href = url
  link.download = `invoice-${run.id}-items.json`
  document.body.appendChild(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

function PageImage({ runId, page, variant }) {
  const [url, setUrl] = useState('')
  useEffect(() => {
    let active = true
    fetch(`/api/invoices/runs/${runId}/pages/${page}/image/${variant === 'original' ? '?variant=original' : ''}`, { credentials: 'same-origin' })
      .then(response => { if (!response.ok) throw new Error('Image unavailable'); return response.blob() })
      .then(blob => { if (active) setUrl(URL.createObjectURL(blob)) })
      .catch(() => setUrl(''))
    return () => { active = false }
  }, [runId, page, variant])
  useEffect(() => () => { if (url) URL.revokeObjectURL(url) }, [url])
  return url ? <img className="source-image" src={url} alt={`${variant === 'original' ? 'Original' : 'Processed'} invoice page ${page}`} /> : <p className="muted">Page preview unavailable.</p>
}

function ItemEditor({ item, groups, save, cancel }) {
  const [form, setForm] = useState({ name: item.name, group_id: item.group_id || '', quantity: item.quantity || '', unit_rate: item.unit_rate || '', gross_amount: item.gross_amount || '', discount: item.discount || '', net_amount: item.net_amount || '', note: '' })
  const set = (key, value) => setForm(current => ({ ...current, [key]: value }))
  return <form className="edit-form" onSubmit={event => { event.preventDefault(); save({ ...form, group_id: form.group_id || null, quantity: form.quantity || null, unit_rate: form.unit_rate || null, gross_amount: form.gross_amount || null, discount: form.discount || null, net_amount: form.net_amount || null }) }}>
    <label>Description<input value={form.name} onChange={event => set('name', event.target.value)} /></label>
    <label>Billing group<select value={form.group_id} onChange={event => set('group_id', event.target.value)}><option value="">Unassigned</option>{groups.map(group => <option key={group.stable_id} value={group.stable_id}>{group.name}</option>)}</select></label>
    <div className="edit-grid">{['quantity', 'unit_rate', 'gross_amount', 'discount', 'net_amount'].map(key => <label key={key}>{key.replaceAll('_', ' ')}<input type="number" step="0.01" value={form[key]} onChange={event => set(key, event.target.value)} /></label>)}</div>
    <label>Reason for correction<input required value={form.note} onChange={event => set('note', event.target.value)} placeholder="What did you verify on the invoice?" /></label>
    <div className="actions"><button type="submit">Save correction</button><button type="button" className="secondary" onClick={cancel}>Cancel</button></div>
  </form>
}

function NoteAction({ label, placeholder, onSubmit, children }) {
  const [open, setOpen] = useState(false)
  const [note, setNote] = useState('')
  if (!open) return <button className="text-button" onClick={() => setOpen(true)}>{label}</button>
  return <form className="note-action" onSubmit={async event => { event.preventDefault(); if (await onSubmit(note)) { setOpen(false); setNote('') } }}>
    {children}
    <input required value={note} onChange={event => setNote(event.target.value)} placeholder={placeholder} />
    <div className="actions"><button type="submit">{label}</button><button type="button" className="secondary" onClick={() => setOpen(false)}>Cancel</button></div>
  </form>
}

const REVIEW_STATUS = { auto_extracted: 'auto extracted', needs_review: 'needs review', human_corrected: 'human corrected', human_added: 'added by reviewer' }

function pageOf(target, rows) {
  const match = /^page-(\d+)$/.exec(target || '')
  if (match) return Number(match[1])
  return rows.find(row => row.stable_id === target)?.page_number || null
}

function RowRef({ item }) {
  return <small className="row-ref">p{item.page_number}{item.table_number ? ` · table ${item.table_number}` : ''} · row {item.row_order}{item.serial ? ` · S.No ${item.serial}` : ''} · {item.extraction_method}</small>
}

function Review({ run, refresh }) {
  const [page, setPage] = useState(1)
  const [variant, setVariant] = useState('processed')
  const [focus, setFocus] = useState(null)
  const [editing, setEditing] = useState('')
  const [candidateGroup, setCandidateGroup] = useState({})
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const call = async (path, method, body) => {
    try {
      await api(path, { method, body: JSON.stringify(body) })
      setError(''); refresh(); return true
    } catch (err) { setError(err.message); return false }
  }
  const save = async (id, body) => { if (await call(`/api/invoices/runs/${run.id}/items/${id}/`, 'PATCH', body)) setEditing('') }
  const decide = async decision => { if (await call(`/api/invoices/runs/${run.id}/decision/`, 'POST', { decision, note })) setNote('') }
  const resolve = (finding, text) => call(`/api/invoices/runs/${run.id}/findings/${finding.id}/resolve/`, 'POST', { note: text })
  const candidate = (row, action, text) => call(`/api/invoices/runs/${run.id}/candidates/${row.id}/`, 'POST', { action, note: text, group_id: candidateGroup[row.id] || null })
  const show = (pageNumber, item) => { if (pageNumber) setPage(pageNumber); setFocus(item || null) }
  const awaitingDecision = run.status === 'needs_review' || run.status === 'auto_approved'
  const groups = run.groups || []
  const items = run.items || []
  const findings = run.findings || []
  const lineItems = items.filter(item => item.kind === 'item')
  const sectionRows = items.filter(item => item.kind === 'item' || item.kind === 'subtotal')
  const otherRows = items.filter(item => item.kind !== 'item' && item.kind !== 'subtotal')
  const groupNames = Object.fromEntries(groups.map(group => [group.stable_id, group.name]))
  const openFindings = findings.filter(finding => !finding.resolved && finding.severity !== 'info')
  const completeness = openFindings.filter(finding => finding.category === 'completeness')
  const rowWarnings = Object.groupBy ? Object.groupBy(openFindings, finding => finding.target_id) : openFindings.reduce((map, finding) => ({ ...map, [finding.target_id]: [...(map[finding.target_id] || []), finding] }), {})
  const candidates = (run.extraction?.candidate_rows || []).filter(row => row.status === 'pending')
  const summary = run.validation_summary || {}
  const pageCount = run.ocr_job_pages || Math.max(1, ...items.map(item => item.page_number))
  const findingRow = finding => <div key={finding.id} className={`finding ${finding.resolved ? 'resolved' : ''}`}>
    <Status value={finding.resolved ? 'resolved' : finding.severity} /><strong>{finding.code.replaceAll('_', ' ')}</strong><small className="category">{finding.category}</small>
    <p>{finding.message}</p>{finding.evidence && <pre className="evidence">{finding.evidence}</pre>}
    <small>{finding.target_id}{finding.resolved && ` · resolved: ${finding.resolution_note}`}</small>
    <div className="actions">{pageOf(finding.target_id, items) && <button className="text-button" onClick={() => show(pageOf(finding.target_id, items), items.find(item => item.stable_id === finding.target_id))}>Show on page {pageOf(finding.target_id, items)}</button>}
      {!finding.resolved && <NoteAction label="Mark checked" placeholder="What did you compare against the invoice?" onSubmit={text => resolve(finding, text)} />}</div>
  </div>
  const itemRow = item => <div key={item.stable_id} className={`item ${focus?.stable_id === item.stable_id ? 'focused' : ''}`}>
    <div className="item-title"><strong>{item.name || <em>no description</em>}</strong><span>{item.net_amount || item.gross_amount || '—'}</span></div>
    <small>{item.kind} · {item.relationship} · <span className={`review-status ${item.review_status}`}>{REVIEW_STATUS[item.review_status] || item.review_status}</span></small><RowRef item={item} />
    {(rowWarnings[item.stable_id] || []).map(finding => <p className="row-warning" key={finding.id}>⚠ {finding.message}</p>)}
    <div className="actions"><button className="text-button" onClick={() => show(item.page_number, item)}>Compare with page</button>
      {editing === item.stable_id ? null : <button className="text-button" onClick={() => { show(item.page_number, item); setEditing(item.stable_id) }}>Correct row or group</button>}</div>
    {editing === item.stable_id && <ItemEditor item={item} groups={groups} save={body => save(item.stable_id, body)} cancel={() => setEditing('')} />}
  </div>
  return <section className="review">
    <div className="review-head"><div><p className="eyebrow">Extraction review</p><h2>Invoice run #{run.id}</h2><p className="muted">Schema {run.schema_version} · {run.stage}</p></div><Status value={run.status} /></div>
    {error && <p className="alert" role="alert">{error}</p>}
    {run.error_message && <p className="alert">{run.error_message}</p>}
    <div className="summary">{[['Provider', run.extraction?.provider_name], ['Invoice', run.extraction?.invoice_number], ['Date', run.extraction?.invoice_date], ['Patient', run.extraction?.patient_name], ['Grand total', run.extraction?.grand_total], ['Confidence', run.confidence_score == null ? null : `${run.confidence_score} / 100 (threshold ${run.confidence?.threshold ?? '—'})`]].map(([label, value]) => <div key={label}><small>{label}</small><strong>{value || 'Not found'}</strong></div>)}</div>
    {run.confidence_score != null && (run.confidence?.reasons?.length > 0 || run.confidence?.blockers?.length > 0) && <ul className="confidence-reasons">{[...(run.confidence.blockers || []), ...(run.confidence.reasons || [])].map(reason => <li key={reason}>{reason}</li>)}</ul>}
    {completeness.length > 0 && <section className="completeness" aria-label="Completeness warnings"><h3>Possible missing or incomplete rows ({completeness.length})</h3><p className="muted">Each warning must be checked against the source page. The invoice will not be approved automatically while any remain.</p>{completeness.map(findingRow)}</section>}
    {candidates.length > 0 && <section className="candidates" aria-label="Candidate rows"><h3>Rows found only by targeted table OCR ({candidates.length})</h3><p className="muted">These were read from a full-resolution crop but not from the page OCR. Accept a row only after finding it on the source page.</p>
      {candidates.map(row => <div className="item candidate" key={row.id}><div className="item-title"><strong>{row.name || <em>no description</em>}</strong><span>{row.net_amount || row.gross_amount || '—'}</span></div><small>page {row.page_number} · qty {row.quantity ?? '—'} · rate {row.unit_rate ?? '—'} · {row.extraction_method}</small><pre className="evidence">{row.source_text}</pre>
        <div className="actions"><button className="text-button" onClick={() => show(row.page_number)}>Show page {row.page_number}</button>
          <NoteAction label="Accept as item" placeholder="Where is this row on the invoice?" onSubmit={text => candidate(row, 'accept', text)}><select aria-label="Billing group" value={candidateGroup[row.id] || ''} onChange={event => setCandidateGroup(current => ({ ...current, [row.id]: event.target.value }))}><option value="">Unassigned</option>{groups.map(group => <option key={group.stable_id} value={group.stable_id}>{group.name}</option>)}</select></NoteAction>
          <NoteAction label="Dismiss" placeholder="Why is this not a separate charge?" onSubmit={text => candidate(row, 'dismiss', text)} /></div></div>)}
    </section>}
    <section className="structured-output" aria-label="Structured invoice items">
      <div className="structured-head"><div><h3>Structured invoice items</h3><p className="muted">Amount uses the net amount when present; otherwise it uses the gross amount. Select a row to compare it with the page.</p></div><button className="secondary" disabled={!lineItems.length} onClick={() => downloadStructuredItems(run)}>Download JSON</button></div>
      {lineItems.length ? <div className="table-scroll"><table className="items-table"><thead><tr><th>Source</th><th>Item name</th><th>Billing group</th><th>Qty</th><th>Rate</th><th>Amount</th><th>Status</th></tr></thead><tbody>{lineItems.map(item => <tr key={item.stable_id} className={rowWarnings[item.stable_id] ? 'warn' : ''} onClick={() => show(item.page_number, item)}><td>p{item.page_number}·r{item.row_order}</td><td>{item.name || '—'}{rowWarnings[item.stable_id] && <span className="warn-mark" title={rowWarnings[item.stable_id].map(f => f.message).join('\n')}> ⚠</span>}</td><td>{groupNames[item.group_id] || 'Unassigned'}</td><td>{item.quantity ?? '—'}</td><td>{item.unit_rate ?? '—'}</td><td>{item.net_amount ?? item.gross_amount ?? '—'}</td><td><span className={`review-status ${item.review_status}`}>{REVIEW_STATUS[item.review_status] || item.review_status}</span></td></tr>)}</tbody></table></div> : <p className="empty">No item rows extracted yet. Check the OCR result and source page.</p>}
      {Object.keys(summary).length > 0 && <p className="check-summary">{Object.entries(summary).map(([check, counts]) => <span key={check}>{check.replaceAll('_', ' ')}: {counts.passed} passed, {counts.failed} failed, {counts.unavailable} unavailable</span>)}</p>}
    </section>
    <div className="review-columns"><div>
      <h3>Source rows and corrections</h3>
      {groups.filter(group => sectionRows.some(item => item.group_id === group.stable_id)).map(group => <div className="group" key={group.stable_id}><h4>{group.name} <small>page {group.page_number}</small></h4>{sectionRows.filter(item => item.group_id === group.stable_id).map(itemRow)}</div>)}
      {sectionRows.filter(item => !item.group_id).length > 0 && <div className="group"><h4>Unassigned</h4>{sectionRows.filter(item => !item.group_id).map(itemRow)}</div>}
      {otherRows.length > 0 && <details className="group other-rows"><summary>Totals, taxes, payments and summary lines ({otherRows.length}) — not counted as items</summary>{otherRows.map(itemRow)}</details>}
      {!items.length && <p className="empty">No item rows found. Review the OCR page before making a decision.</p>}
      <h3>Validation findings</h3>{findings.filter(finding => finding.category !== 'completeness').length ? findings.filter(finding => finding.category !== 'completeness').map(findingRow) : <p className="muted">No deterministic discrepancies found.</p>}
      {findings.some(finding => finding.category === 'completeness' && (finding.resolved || finding.severity === 'info')) && <details><summary>Resolved and informational completeness notes</summary>{findings.filter(finding => finding.category === 'completeness' && (finding.resolved || finding.severity === 'info')).map(findingRow)}</details>}
      <h3>Judge findings</h3>{run.judge_findings?.length ? run.judge_findings.map((finding, index) => <div className="finding" key={index}><Status value={finding.verdict} /><p>{finding.reason}</p><small>{finding.target_id}</small></div>) : <p className="muted">No judge verdicts recorded.</p>}
      <h3>Targeted OCR retries</h3>{run.retry_attempts?.length ? run.retry_attempts.map((attempt, index) => <p className="audit" key={index}>Page {attempt.page_number} · {attempt.coverage} crop [{attempt.region.join(', ')}] · <Status value={attempt.status} /> · {attempt.reasons.join(', ')}{attempt.error && <small>{attempt.error}</small>}</p>) : <p className="muted">No pages needed a targeted re-read.</p>}
      <h3>Reviewer decision</h3><textarea value={note} onChange={event => setNote(event.target.value)} placeholder="Record the evidence for your decision" /><div className="actions"><button disabled={!note || !awaitingDecision} onClick={() => decide('approve')}>Approve</button><button className="secondary" disabled={!note || !awaitingDecision} onClick={() => decide('reject')}>Keep for review</button></div>
      <h3>Audit history</h3>{run.review_events?.length ? run.review_events.map((event, index) => <p className="audit" key={index}>{event.actor} · {event.action} · {event.note} <small>{new Date(event.created_at).toLocaleString()}</small></p>) : <p className="muted">No reviewer changes yet.</p>}
    </div><aside className="source"><div className="source-head"><h3>Source page</h3><select aria-label="Source page" value={page} onChange={event => { setPage(Number(event.target.value)); setFocus(null) }}>{Array.from({ length: pageCount }, (_, index) => <option key={index + 1} value={index + 1}>Page {index + 1}</option>)}</select></div>
      <div className="variant-toggle" role="group" aria-label="Image version">{['processed', 'original'].map(name => <button key={name} className={variant === name ? '' : 'secondary'} onClick={() => setVariant(name)}>{name}</button>)}</div>
      {focus && focus.page_number === page && <div className="focus-row"><small>Compare this extracted row with the page</small><strong>{focus.name}</strong><RowRef item={focus} /><pre className="evidence">{focus.source_text}</pre></div>}
      <PageImage runId={run.id} page={page} variant={variant} /><p className="muted">Page references come from OCR. Region boxes are unavailable; use the row text above to locate it.</p></aside></div>
  </section>
}

function latestFor(rows, matches) {
  return rows.filter(matches).reduce((latest, row) => !latest || row.id > latest.id ? row : latest, null)
}

function DocumentCard({ doc, preprocess, ocr, runs, busy, start, openReview }) {
  const prep = latestFor(preprocess, job => job.document === doc.id)
  const ocrJob = latestFor(ocr, job => job.preprocessing_job === prep?.id)
  const run = latestFor(runs, item => item.ocr_job === ocrJob?.id)
  const canStartOcr = prep?.status === 'completed' && (!ocrJob || ocrJob.status === 'failed')
  const canExtract = ocrJob?.status === 'completed' && (!run || run.status === 'failed')
  return <article className="doc">
    <div className="doc-top"><div><strong>{doc.original_filename}</strong><small>{doc.page_count} page{doc.page_count === 1 ? '' : 's'} · #{doc.id}</small></div><Status value={run?.status || ocrJob?.status || prep?.status} /></div>
    <div className="steps"><span>1 Preprocess <Status value={prep?.status} /></span><span>2 Nanonets OCR <Status value={ocrJob?.status} /></span><span>3 Review <Status value={run?.status} /></span></div>
    {ocrJob?.status === 'running' && <p className="muted">OCR page {ocrJob.pages_done} of {prep.pages_total} completed. Large invoices can take a while on CPU.</p>}
    {prep?.status === 'failed' && <p className="alert">{prep.error_message}</p>}
    {ocrJob?.status === 'failed' && <p className="alert">{ocrJob.error_message}</p>}
    {run?.status === 'failed' && <p className="alert">{run.error_message}</p>}
    <div className="actions">
      {canStartOcr && <button disabled={busy} onClick={() => start(`/api/preprocessing/jobs/${prep.id}/ocr/`)}>{ocrJob ? 'Retry OCR' : 'Start OCR'}</button>}
      {canExtract && <button disabled={busy} onClick={() => start(`/api/ocr/jobs/${ocrJob.id}/extract/`)}>{run ? 'Retry extraction' : 'Extract invoice'}</button>}
      {run && run.status !== 'failed' && <button onClick={() => openReview(run)}>Open review</button>}
    </div>
  </article>
}

export default function App() {
  const [documents, setDocuments] = useState([])
  const [preprocess, setPreprocess] = useState([])
  const [ocr, setOcr] = useState([])
  const [runs, setRuns] = useState([])
  const [selected, setSelected] = useState(null)
  const [error, setError] = useState('')
  const [runtime, setRuntime] = useState(null)
  const [authRequired, setAuthRequired] = useState(false)
  const [busy, setBusy] = useState(false)
  useEffect(() => { api('/api/runtime/').then(setRuntime).catch(() => {}) }, [])
  const load = useCallback(async () => {
    if (authRequired) return
    try {
      const docs = await api('/api/documents/')
      const [prep, ocrJobs, invoiceRuns] = await Promise.all([
        api('/api/preprocessing/jobs/'), api('/api/ocr/jobs/'), api('/api/invoices/runs/'),
      ])
      setDocuments(docs.results || docs); setPreprocess(prep.results || prep); setOcr(ocrJobs.results || ocrJobs); setRuns(invoiceRuns.results || invoiceRuns)
      setSelected(current => current ? (invoiceRuns.results || invoiceRuns).find(run => run.id === current.id) || current : null)
      setError('')
    } catch (err) { setError(err.message); if (err.authRequired) setAuthRequired(true) }
  }, [authRequired])
  useEffect(() => {
    if (authRequired) return
    load()
    const timer = setInterval(load, 12000)
    return () => clearInterval(timer)
  }, [load, authRequired])
  const upload = async event => {
    event.preventDefault(); const file = event.currentTarget.elements.file.files[0]; if (!file) return
    const form = new FormData(); form.append('file', file)
    setBusy(true)
    try { await api('/api/documents/', { method: 'POST', body: form }); await load() } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  const start = async path => {
    setBusy(true)
    try { await api(path, { method: 'POST', body: '{}' }); await load() } catch (err) { setError(err.message) } finally { setBusy(false) }
  }
  return <div className="shell"><header><div className="brand"><span className="mark">N</span><div><strong>Neural DOCS</strong><small>Medical invoice workspace</small></div></div><nav>{runtime?.local_demo_mode ? <span className="mode-badge">Local demo</span> : <a href="/api-auth/login/?next=/">Sign in</a>}<button className="secondary" onClick={() => authRequired ? setAuthRequired(false) : load()}>Refresh</button></nav></header>
    <main><div className="intro"><div><p className="eyebrow">Invoice operations</p><h1>Review every charge with its source.</h1><p>Upload a medical invoice, run Nanonets OCR2, and resolve findings before accepting the result.</p></div><form onSubmit={upload} className="upload"><label htmlFor="file">New medical invoice PDF</label><input id="file" name="file" type="file" accept="application/pdf" required /><button disabled={busy || authRequired}>Upload invoice</button></form></div>
      {error && <p className="alert" role="alert">{error} {authRequired && !runtime?.local_demo_mode && <a href="/api-auth/login/?next=/">Sign in</a>}</p>}
      <div className="layout"><section className="documents"><div className="section-title"><h2>Documents</h2><span>{documents.length}</span></div>{documents.length ? documents.map(doc => <DocumentCard key={doc.id} doc={doc} preprocess={preprocess} ocr={ocr} runs={runs} busy={busy} start={start} openReview={setSelected} />) : <p className="empty">No invoices yet. Upload a PDF to begin.</p>}</section><section className="detail">{selected ? <Review run={selected} refresh={load} /> : <div className="welcome"><div className="welcome-icon">⌁</div><h2>Review workspace</h2><p>Choose an extracted invoice to inspect its billing groups, source pages, checks, and reviewer history.</p></div>}</section></div>
    </main><footer>Neural DOCS · Medical invoices only · Human approval required</footer></div>
}
