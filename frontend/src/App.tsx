import { useEffect, useState, type FormEvent } from 'react'
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleHelp,
  House,
  LoaderCircle,
  Moon,
  Play,
  Sun,
} from 'lucide-react'
import { LossChart } from './components/LossChart'
import { getOptions, predictPrice, trainModel } from './lib-api'
import type {
  DatasetOptions,
  LossName,
  OptimizerName,
  PredictionResult,
  PropertyFeatures,
  TrainingResult,
} from './types/api'

const emptyProperty: PropertyFeatures = {
  Rooms: 3,
  Distance: 8,
  Bathroom: 2,
  Car: 1,
  Type: 'h',
  Region: '',
}

const currency = new Intl.NumberFormat('en-AU', {
  style: 'currency',
  currency: 'AUD',
  maximumFractionDigits: 0,
})

function App() {
  const [options, setOptions] = useState<DatasetOptions | null>(null)
  const [property, setProperty] = useState(emptyProperty)
  const [loss, setLoss] = useState<LossName>('mse')
  const [method, setMethod] = useState<OptimizerName>('batch')
  const [learningRate, setLearningRate] = useState('0.03')
  const [epochs, setEpochs] = useState('300')
  const [training, setTraining] = useState<TrainingResult | null>(null)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [busy, setBusy] = useState<'train' | 'predict' | null>(null)
  const [error, setError] = useState('')
  const [dark, setDark] = useState(false)
  const [loadingOptions, setLoadingOptions] = useState(true)

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
  }, [dark])

  useEffect(() => {
    getOptions()
      .then((data) => {
        setOptions(data)
        setProperty((current) => ({
          ...current,
          Region: data.regions.includes(current.Region) ? current.Region : data.regions[0] ?? '',
          Type: data.property_types.some((type) => type.value === current.Type)
            ? current.Type
            : data.property_types[0]?.value ?? '',
        }))
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoadingOptions(false))
  }, [])

  async function handleTrain(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy('train')
    setError('')
    setPrediction(null)
    try {
      const result = await trainModel({ loss, method, learning_rate: Number(learningRate), epochs: Number(epochs) })
      setTraining(result)
      setOptions((current) => current ? { ...current, model_ready: true } : current)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Training failed. Please try again.')
    } finally {
      setBusy(null)
    }
  }

  async function handlePredict(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy('predict')
    setError('')
    try {
      setPrediction(await predictPrice(property))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Prediction failed. Please try again.')
    } finally {
      setBusy(null)
    }
  }

  const modelReady = options?.model_ready ?? false
  const lossChange = training && training.first_epoch_loss !== 0
    ? ((training.first_epoch_loss - training.final_loss) / training.first_epoch_loss) * 100
    : 0

  return (
    <div className="app-frame">
      <header className="topbar flex items-center justify-between">
        <a className="brand flex items-center" href="#top" aria-label="Melbourne Model Studio home">
          <span className="brand-symbol"><House size={17} strokeWidth={2} aria-hidden="true" /></span>
          <span className="brand-name">melbourne<span> / </span>model studio</span>
        </a>
        <nav className="top-navigation" aria-label="Main navigation">
          <a className="nav-link active" href="#workspace">Workspace</a>
          <a className="nav-link" href="#learning-curve">Training</a>
        </nav>
        <div className="topbar-actions flex items-center">
          <div className={`api-status ${error && !options ? 'offline' : 'online'}`}>
            <span className="status-dot" />
            <span>{loadingOptions ? 'Connecting' : error && !options ? 'API offline' : 'Model API'}</span>
          </div>
          <button className="theme-toggle" type="button" onClick={() => setDark((value) => !value)} aria-label={`Switch to ${dark ? 'light' : 'dark'} theme`}>
            {dark ? <Sun size={17} /> : <Moon size={17} />}
          </button>
        </div>
      </header>

      <main id="top" className="page-content">
        <section className="page-intro">
          <div className="intro-copy">
            <p className="location-line"><span className="location-dot" /> Melbourne, Australia <span className="location-separator">/</span> Housing model</p>
            <h1>Explore what shapes<br className="desktop-break" /> a home's estimated value.</h1>
            <p className="intro-description">Train a linear regression model from scratch, follow its learning curve, then estimate a property using the features it learned.</p>
          </div>
          <div className="dataset-note" aria-label="Dataset details">
            <span className="dataset-label">Training dataset</span>
            <strong>{options ? `${options.rows.toLocaleString('en-AU')} homes` : 'Melbourne housing'}</strong>
            <span className="dataset-source">Cleaned property records</span>
          </div>
        </section>

        {error && (
          <div className="error-banner" role="alert">
            <CircleHelp size={18} aria-hidden="true" />
            <span>{error}</span>
            <button type="button" onClick={() => setError('')} aria-label="Dismiss error">×</button>
          </div>
        )}

        <section id="workspace" className="workspace-grid grid" aria-label="Model workspace">
          <form className="surface-panel property-panel" onSubmit={handlePredict}>
            <div className="panel-heading">
              <div>
                <div className="section-heading-row"><House size={19} strokeWidth={1.8} aria-hidden="true" /><h2>Property details</h2></div>
                <p className="section-copy">Describe a home to estimate its market value.</p>
              </div>
              <span className="step-label">01 <span>/</span> Input</span>
            </div>

            <div className="field-grid grid grid-cols-2">
              <label className="field">
                <span>Rooms <span className="field-unit">count</span></span>
                <input type="number" min="1" max="30" step="1" required value={property.Rooms} onChange={(event) => setProperty({ ...property, Rooms: Number(event.target.value) })} />
              </label>
              <label className="field">
                <span>Bathrooms <span className="field-unit">count</span></span>
                <input type="number" min="0" max="30" step="0.5" required value={property.Bathroom} onChange={(event) => setProperty({ ...property, Bathroom: Number(event.target.value) })} />
              </label>
              <label className="field">
                <span>Distance to CBD <span className="field-unit">km</span></span>
                <input type="number" min="0" max="500" step="0.1" required value={property.Distance} onChange={(event) => setProperty({ ...property, Distance: Number(event.target.value) })} />
              </label>
              <label className="field">
                <span>Car spaces <span className="field-unit">count</span></span>
                <input type="number" min="0" max="30" step="1" required value={property.Car} onChange={(event) => setProperty({ ...property, Car: Number(event.target.value) })} />
              </label>
              <label className="field">
                <span>Property type</span>
                <span className="select-wrap">
                  <select required value={property.Type} onChange={(event) => setProperty({ ...property, Type: event.target.value })} disabled={!options}>
                    {options?.property_types.map((type) => <option key={type.value} value={type.value}>{type.label}</option>)}
                  </select>
                  <ChevronDown size={16} aria-hidden="true" />
                </span>
              </label>
              <label className="field">
                <span>Melbourne region</span>
                <span className="select-wrap">
                  <select required value={property.Region} onChange={(event) => setProperty({ ...property, Region: event.target.value })} disabled={!options}>
                    {options?.regions.map((region) => <option key={region} value={region}>{region}</option>)}
                  </select>
                  <ChevronDown size={16} aria-hidden="true" />
                </span>
              </label>
            </div>

            <div className="property-footer">
              <span className="form-note"><Activity size={15} aria-hidden="true" /> Uses the model's learned feature weights</span>
              <button className="button-primary" type="submit" disabled={!modelReady || busy !== null || !options}>
                {busy === 'predict' ? <LoaderCircle className="spin" size={17} /> : <span>Estimate price <ArrowUpRight size={16} /></span>}
                {busy === 'predict' && 'Estimating…'}
              </button>
            </div>

            {!modelReady && <p className="inline-hint">Train the model first to enable an estimate.</p>}

            {prediction && (
              <div className="prediction-result" aria-live="polite">
                <div>
                  <span className="result-label">Estimated property value</span>
                  <strong>{currency.format(prediction.price)}</strong>
                  <span className="result-context">Based on your trained {prediction.loss.toUpperCase()} · {prediction.method === 'batch' ? 'Batch GD' : 'SGD'} model</span>
                </div>
                <span className="result-check"><Check size={19} aria-hidden="true" /></span>
              </div>
            )}
          </form>

          <form className="training-panel" onSubmit={handleTrain}>
            <div className="training-heading">
              <div className="section-heading-row"><Activity size={19} strokeWidth={1.8} aria-hidden="true" /><h2>Train the model</h2></div>
              <p>Choose how the model measures error and updates its weights.</p>
            </div>

            <label className="training-field">
              <span>Loss function</span>
              <span className="select-wrap dark-select">
                <select value={loss} onChange={(event) => setLoss(event.target.value as LossName)}>
                  <option value="mse">Mean squared error</option>
                  <option value="mae">Mean absolute error</option>
                  <option value="huber">Huber loss</option>
                </select>
                <ChevronDown size={16} aria-hidden="true" />
              </span>
            </label>
            <div className="training-field">
              <span className="control-label">Optimizer</span>
              <div className="segmented-control" role="group" aria-label="Gradient descent method">
                <button type="button" className={method === 'batch' ? 'selected' : ''} onClick={() => { setMethod('batch'); if (learningRate === '0.001') setLearningRate('0.03') }}>Batch GD</button>
                <button type="button" className={method === 'sgd' ? 'selected' : ''} onClick={() => { setMethod('sgd'); if (learningRate === '0.03') setLearningRate('0.001') }}>SGD</button>
              </div>
            </div>
            <div className="training-row">
              <label className="training-field">
                <span>Learning rate</span>
                <input type="number" min="0.00001" max="1" step="any" required value={learningRate} onChange={(event) => setLearningRate(event.target.value)} />
              </label>
              <label className="training-field">
                <span>Epochs</span>
                <input type="number" min="1" max="2000" step="1" required value={epochs} onChange={(event) => setEpochs(event.target.value)} />
              </label>
            </div>
            <button className="button-train" type="submit" disabled={busy !== null || !options}>
              {busy === 'train' ? <><LoaderCircle className="spin" size={17} /> Training on {options?.rows.toLocaleString('en-AU')} homes…</> : <><Play size={16} fill="currentColor" /> Train model</>}
            </button>
            <p className="training-footnote"><span className="status-dot" /> SGD updates once per home and can take longer than Batch GD.</p>
          </form>
        </section>

        <section id="learning-curve" className="learning-section">
          <div className="learning-section-header">
            <div>
              <p className="location-line">Training progress</p>
              <h2>Watch the model learn.</h2>
            </div>
            {training && (
              <div className="run-summary" aria-live="polite">
                <div><span>Epoch 1 loss</span><strong>{training.first_epoch_loss.toFixed(4)}</strong></div>
                {lossChange > 0 ? <ArrowDownRight size={17} aria-hidden="true" /> : lossChange < 0 ? <ArrowUpRight size={17} aria-hidden="true" /> : <span aria-hidden="true">→</span>}
                <div><span>Final loss</span><strong>{training.final_loss.toFixed(4)}</strong></div>
                <div className="loss-improvement"><span>{lossChange > 0 ? 'Lower since epoch 1' : lossChange < 0 ? 'Higher since epoch 1' : 'No change since epoch 1'}</span><strong>{Math.abs(lossChange).toFixed(1)}%</strong></div>
              </div>
            )}
          </div>
          <LossChart history={training?.history ?? []} label={training?.loss.toUpperCase() ?? 'Training loss'} />
          <div className="transparency-note">
            <span className="note-mark"><Activity size={16} aria-hidden="true" /></span>
            <p><strong>Reading this curve</strong> — It shows the selected loss on training data after each epoch. A falling curve means the model is fitting those examples better; it does not measure accuracy on unseen homes.</p>
          </div>
        </section>

        <footer className="page-footer">
          <span>Melbourne Housing Model Studio</span>
          <span>Linear Regression · NumPy · {options?.rows.toLocaleString('en-AU') ?? '—'} training records</span>
        </footer>
      </main>
    </div>
  )
}

export default App
