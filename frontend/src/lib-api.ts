import type {
  DatasetOptions,
  PredictionResult,
  PropertyFeatures,
  TrainingRequest,
  TrainingResult,
} from './types/api'

const API_URL = (import.meta.env.VITE_API_URL ?? 'http://localhost:8000').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: { 'Content-Type': 'application/json', ...init?.headers },
    })
  } catch {
    throw new Error('Could not reach the model API. Start the FastAPI server and try again.')
  }

  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(typeof body.detail === 'string' ? body.detail : 'The request could not be completed.')
  }
  return body as T
}

export const getOptions = () => request<DatasetOptions>('/api/options')
export const trainModel = (payload: TrainingRequest) =>
  request<TrainingResult>('/api/train', { method: 'POST', body: JSON.stringify(payload) })
export const predictPrice = (payload: PropertyFeatures) =>
  request<PredictionResult>('/api/predict', { method: 'POST', body: JSON.stringify(payload) })
