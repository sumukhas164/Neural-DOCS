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

function PageImage({ runId, page }) {
  const [url, setUrl] = useState('')
  useEffect(() => {
    let active = true
    fetch(`/api/invoices/runs/${runId}/pages/${page}/image/`, { credentials: 'same-origin' })
      .then(response => { if (!response.ok) throw new Error('Image unavailable'); return response.blob() })
      .then(blob => { if (active) setUrl(URL.createObjectURL(blob)) })
      .catch(() => setUrl(''))
    return () => { active = false }
  }, [runId, page])
  useEffect(() => () => { if (url) URL.revokeObjectURL(url) }, [url])
  return url ? <img className="source-image" src={url} alt={`Processed invoice page ${page}`} /> : <p className="muted">Page preview unavailable.</p>
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

function Review({ run, refresh }) {
  const [page, setPage] = useState(1)
  const [editing, setEditing] = useState('')
  const [note, setNote] = useState('')
  const [error, setError] = useState('')
  const save = async (id, body) => {
    try {
      await api(`/api/invoices/runs/${run.id}/items/${id}/`, { method: 'PATCH', body: JSON.stringify(body) })
      setEditing(''); setError(''); refresh()
    } catch (err) { setError(err.message) }
  }
  const decide = async decision => {
    try {
      await api(`/api/invoices/runs/${run.id}/decision/`, { method: 'POST', body: JSON.stringify({ decision, note }) })
      setNote(''); setError(''); refresh()
    } catch (err) { setError(err.message) }
  }
  const groups = run.groups || []
  const items = run.items || []
  const lineItems = items.filter(item => item.kind === 'item')
  const groupNames = Object.fromEntries(groups.map(group => [group.stable_id, group.name]))
  return <section className="review">
    <div className="review-head"><div><p className="eyebrow">Extraction review</p><h2>Invoice run #{run.id}</h2><p className="muted">Schema {run.schema_version} · {run.stage}</p></div><Status value={run.status} /></div>
    {error && <p className="alert" role="alert">{error}</p>}
    {run.error_message && <p className="alert">{run.error_message}</p>}
    <div className="summary">{[['Provider', run.extraction?.provider_name], ['Invoice', run.extraction?.invoice_number], ['Date', run.extraction?.invoice_date], ['Patient', run.extraction?.patient_name], ['Grand total', run.extraction?.grand_total]].map(([label, value]) => <div key={label}><small>{label}</small><strong>{value || 'Not found'}</strong></div>)}</div>
    <section className="structured-output" aria-label="Structured invoice items">
      <div className="structured-head"><div><h3>Structured invoice items</h3><p className="muted">Amount uses the net amount when present; otherwise it uses the gross amount.</p></div><button className="secondary" disabled={!lineItems.length} onClick={() => downloadStructuredItems(run)}>Download JSON</button></div>
      {lineItems.length ? <div className="table-scroll"><table className="items-table"><thead><tr><th>Item name</th><th>Billing group</th><th>Qty</th><th>Rate</th><th>Amount</th></tr></thead><tbody>{lineItems.map(item => <tr key={item.stable_id}><td>{item.name}</td><td>{groupNames[item.group_id] || 'Unassigned'}</td><td>{item.quantity ?? '—'}</td><td>{item.unit_rate ?? '—'}</td><td>{item.net_amount ?? item.gross_amount ?? '—'}</td></tr>)}</tbody></table></div> : <p className="empty">No item rows extracted yet. Check the OCR result and source page.</p>}
    </section>
    <div className="review-columns"><div>
      <h3>Source rows and corrections</h3>
      {groups.map(group => <div className="group" key={group.stable_id}><h4>{group.name} <small>page {group.page_number}</small></h4>{items.filter(item => item.group_id === group.stable_id).map(item => <div className="item" key={item.stable_id}><div className="item-title"><strong>{item.name}</strong><span>{item.net_amount || item.gross_amount || '—'}</span></div><small>{item.kind} · page {item.page_number} · {item.relationship}</small>{editing === item.stable_id ? <ItemEditor item={item} groups={groups} save={body => save(item.stable_id, body)} cancel={() => setEditing('')} /> : <button className="text-button" onClick={() => { setPage(item.page_number); setEditing(item.stable_id) }}>Correct row or group</button>}</div>)}</div>)}
      {items.filter(item => !item.group_id).length > 0 && <div className="group"><h4>Unassigned and totals</h4>{items.filter(item => !item.group_id).map(item => <div className="item" key={item.stable_id}><div className="item-title"><strong>{item.name}</strong><span>{item.net_amount || item.gross_amount || '—'}</span></div><small>{item.kind} · page {item.page_number}</small>{editing === item.stable_id ? <ItemEditor item={item} groups={groups} save={body => save(item.stable_id, body)} cancel={() => setEditing('')} /> : <button className="text-button" onClick={() => { setPage(item.page_number); setEditing(item.stable_id) }}>Correct row or group</button>}</div>)}</div>}
      {!items.length && <p className="empty">No item rows found. Review the OCR page before making a decision.</p>}
      <h3>Validation findings</h3>{run.findings?.length ? run.findings.map((finding, index) => <div className="finding" key={index}><Status value={finding.severity} /><strong>{finding.code.replaceAll('_', ' ')}</strong><p>{finding.message}</p><small>{finding.target_id}</small></div>) : <p className="muted">No deterministic discrepancies found.</p>}
      <h3>Judge findings</h3>{run.judge_findings?.length ? run.judge_findings.map((finding, index) => <div className="finding" key={index}><Status value={finding.verdict} /><p>{finding.reason}</p><small>{finding.target_id}</small></div>) : <p className="muted">No judge verdicts recorded.</p>}
      <h3>Reviewer decision</h3><textarea value={note} onChange={event => setNote(event.target.value)} placeholder="Record the evidence for your decision" /><div className="actions"><button disabled={!note || run.status !== 'needs_review'} onClick={() => decide('approve')}>Approve</button><button className="secondary" disabled={!note || run.status !== 'needs_review'} onClick={() => decide('reject')}>Keep for review</button></div>
      <h3>Audit history</h3>{run.review_events?.length ? run.review_events.map((event, index) => <p className="audit" key={index}>{event.actor} · {event.action} · {event.note} <small>{new Date(event.created_at).toLocaleString()}</small></p>) : <p className="muted">No reviewer changes yet.</p>}
    </div><aside className="source"><div className="source-head"><h3>Source page</h3><select aria-label="Source page" value={page} onChange={event => setPage(Number(event.target.value))}>{Array.from({ length: run.ocr_job_pages || Math.max(1, ...items.map(item => item.page_number)) }, (_, index) => <option key={index + 1} value={index + 1}>Page {index + 1}</option>)}</select></div><PageImage runId={run.id} page={page} /><p className="muted">Page references come from OCR. Region boxes are unavailable.</p></aside></div>
  </section>
}

export default function App() {
  const [documents, setDocuments] = useState([])
  const [preprocess, setPreprocess] = useState([])
  const [ocr, setOcr] = useState([])
  const [runs, setRuns] = useState([])
  const [selected, setSelected] = useState(null)
  const [error, setError] = useState('')
  const [authRequired, setAuthRequired] = useState(false)
  const [busy, setBusy] = useState(false)
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
  return <div className="shell"><header><div className="brand"><span className="mark">N</span><div><strong>Neural DOCS</strong><small>Medical invoice workspace</small></div></div><nav><a href="/api-auth/login/?next=/">Sign in</a><button className="secondary" onClick={() => authRequired ? setAuthRequired(false) : load()}>Refresh</button></nav></header>
    <main><div className="intro"><div><p className="eyebrow">Invoice operations</p><h1>Review every charge with its source.</h1><p>Upload a medical invoice, run Nanonets OCR2, and resolve findings before accepting the result.</p></div><form onSubmit={upload} className="upload"><label htmlFor="file">New medical invoice PDF</label><input id="file" name="file" type="file" accept="application/pdf" required /><button disabled={busy || authRequired}>Upload invoice</button></form></div>
      {error && <p className="alert" role="alert">{error} {authRequired && <a href="/api-auth/login/?next=/">Sign in</a>}</p>}
      <div className="layout"><section className="documents"><div className="section-title"><h2>Documents</h2><span>{documents.length}</span></div>{documents.length ? documents.map(doc => { const prep = preprocess.find(job => job.document === doc.id); const ocrJob = ocr.find(job => job.preprocessing_job === prep?.id); const run = runs.find(item => item.ocr_job === ocrJob?.id); return <article className="doc" key={doc.id}><div className="doc-top"><div><strong>{doc.original_filename}</strong><small>{doc.page_count} page{doc.page_count === 1 ? '' : 's'} · #{doc.id}</small></div><Status value={run?.status || ocrJob?.status || prep?.status} /></div><div className="steps"><span>1 Preprocess <Status value={prep?.status} /></span><span>2 Nanonets OCR <Status value={ocrJob?.status} /></span><span>3 Review <Status value={run?.status} /></span></div><div className="actions">{prep?.status === 'completed' && !ocrJob && <button disabled={busy} onClick={() => start(`/api/preprocessing/jobs/${prep.id}/ocr/`)}>Start OCR</button>}{ocrJob?.status === 'completed' && !run && <button disabled={busy} onClick={() => start(`/api/ocr/jobs/${ocrJob.id}/extract/`)}>Extract invoice</button>}{run && <button onClick={() => setSelected(run)}>Open review</button>}</div></article> }) : <p className="empty">No invoices yet. Upload a PDF to begin.</p>}</section><section className="detail">{selected ? <Review run={selected} refresh={load} /> : <div className="welcome"><div className="welcome-icon">⌁</div><h2>Review workspace</h2><p>Choose an extracted invoice to inspect its billing groups, source pages, checks, and reviewer history.</p></div>}</section></div>
    </main><footer>Neural DOCS · Medical invoices only · Human approval required</footer></div>
}
