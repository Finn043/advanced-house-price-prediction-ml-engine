export type LossName = 'mse' | 'mae' | 'huber'
export type OptimizerName = 'batch' | 'sgd'

export interface PropertyTypeOption {
  value: string
  label: string
}

export interface DatasetOptions {
  regions: string[]
  property_types: PropertyTypeOption[]
  rows: number
  model_ready: boolean
  numeric_features: string[]
}

export interface TrainingRequest {
  loss: LossName
  method: OptimizerName
  learning_rate: number
  epochs: number
}

export interface TrainingResult {
  loss: LossName
  method: OptimizerName
  epochs: number
  rows_trained: number
  first_epoch_loss: number
  final_loss: number
  history: number[]
}

export interface PropertyFeatures {
  Rooms: number
  Distance: number
  Bathroom: number
  Car: number
  Type: string
  Region: string
}

export interface PredictionResult {
  price: number
  currency: 'AUD'
  loss: LossName
  method: OptimizerName
}
