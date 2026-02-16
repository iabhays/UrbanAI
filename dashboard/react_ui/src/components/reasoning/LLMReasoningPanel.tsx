import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Brain, 
  MessageSquare, 
  Target, 
  AlertTriangle, 
  CheckCircle,
  Clock,
  Eye,
  TrendingUp,
  FileText,
  Zap
} from 'lucide-react'
import { clsx } from '../../utils/clsx'

interface ReasoningStep {
  id: string
  step: number
  title: string
  description: string
  evidence: string[]
  confidence: number
  reasoningType: 'detection' | 'analysis' | 'inference' | 'prediction'
  timestamp: Date
}

interface DetectedEvent {
  id: string
  timestamp: Date
  eventType: 'suspicious_behavior' | 'threat_detection' | 'anomaly' | 'emergency'
  location: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  confidence: number
}

interface ModelInterpretation {
  id: string
  modelName: string
  version: string
  inputFeatures: Array<{ name: string; value: number; importance: number }>
  outputProbabilities: Array<{ class: string; probability: number }>
  attentionWeights: Array<{ token: string; weight: number }>
  shapValues: Array<{ feature: string; contribution: number }>
}

interface RiskAssessment {
  id: string
  overallRisk: number
  riskFactors: Array<{
    factor: string
    weight: number
    score: number
    description: string
  }>
  temporalTrend: 'increasing' | 'decreasing' | 'stable'
  spatialSpread: number
  mitigationOptions: Array<{
    option: string
    effectiveness: number
    cost: number
    feasibility: number
  }>
}

interface RecommendedResponse {
  id: string
  action: string
  priority: number
  estimatedEffectiveness: number
  resourceRequirements: Array<{ resource: string; quantity: number }>
  executionTime: number
  sideEffects: string[]
  alternatives: Array<{
    action: string
    effectiveness: number
    tradeoffs: string[]
  }>
}

interface LLMReasoningPanelProps {
  className?: string
  eventId?: string
}

const LLMReasoningPanel: React.FC<LLMReasoningPanelProps> = ({ 
  className,
  eventId 
}) => {
  const [selectedEvent, setSelectedEvent] = useState<DetectedEvent | null>(null)
  const [reasoningSteps, setReasoningSteps] = useState<ReasoningStep[]>([])
  const [modelInterpretation, setModelInterpretation] = useState<ModelInterpretation | null>(null)
  const [riskAssessment, setRiskAssessment] = useState<RiskAssessment | null>(null)
  const [recommendedResponse, setRecommendedResponse] = useState<RecommendedResponse | null>(null)
  const [expandedStep, setExpandedStep] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'reasoning' | 'interpretation' | 'risk' | 'response'>('reasoning')

  useEffect(() => {
    // Generate mock detected events
    const mockEvent: DetectedEvent = {
      id: 'event-001',
      timestamp: new Date(),
      eventType: 'suspicious_behavior',
      location: 'Zone 4 - Commercial District',
      severity: 'high',
      description: 'Multiple individuals exhibiting coordinated suspicious movement patterns near high-value infrastructure',
      confidence: 87
    }

    setSelectedEvent(mockEvent)

    // Generate mock reasoning steps
    const mockReasoningSteps: ReasoningStep[] = [
      {
        id: 'step-1',
        step: 1,
        title: 'Object Detection',
        description: 'YOLOv8-L model detected 4 human figures in restricted area with 94.2% confidence',
        evidence: [
          'Bounding boxes: [(245, 120, 380, 450), (410, 115, 545, 445), (580, 118, 715, 448), (750, 122, 885, 452)]',
          'Class probabilities: person: 94.2%, 93.8%, 94.5%, 93.9%',
          'Temporal consistency: 12 consecutive frames'
        ],
        confidence: 94.2,
        reasoningType: 'detection',
        timestamp: new Date(Date.now() - 5000)
      },
      {
        id: 'step-2',
        step: 2,
        title: 'Behavioral Analysis',
        description: 'DeepSORT tracking identified coordinated movement patterns with abnormal trajectory clustering',
        evidence: [
          'Track IDs: TRK-001, TRK-002, TRK-003, TRK-004',
          'Velocity correlation: 0.87 (threshold: 0.6)',
          'Movement pattern: convergent trajectory toward secure facility',
          'Behavior classification: coordinated_suspicious_movement (89.7% confidence)'
        ],
        confidence: 89.7,
        reasoningType: 'analysis',
        timestamp: new Date(Date.now() - 4000)
      },
      {
        id: 'step-3',
        step: 3,
        title: 'Contextual Inference',
        description: 'Geospatial analysis indicates proximity to critical infrastructure with historical incident correlation',
        evidence: [
          'Location: 50m from Data Center (critical asset)',
          'Time: 02:30 AM (low activity period)',
          'Historical incidents: 3 similar patterns in past 6 months',
          'Environmental factors: low visibility, minimal civilian presence'
        ],
        confidence: 82.3,
        reasoningType: 'inference',
        timestamp: new Date(Date.now() - 3000)
      },
      {
        id: 'step-4',
        step: 4,
        title: 'Threat Prediction',
        description: 'LSTM temporal model predicts 78% probability of escalation within 15 minutes',
        evidence: [
          'Temporal features: movement acceleration, formation changes',
          'Pattern matching: 92% similarity to historical escalation events',
          'Risk factors: proximity, timing, coordination level',
          'Predicted outcome: unauthorized access attempt'
        ],
        confidence: 78.0,
        reasoningType: 'prediction',
        timestamp: new Date(Date.now() - 2000)
      }
    ]

    setReasoningSteps(mockReasoningSteps)

    // Generate mock model interpretation
    const mockInterpretation: ModelInterpretation = {
      id: 'interp-001',
      modelName: 'YOLOv8-L',
      version: 'v8.0.0',
      inputFeatures: [
        { name: 'pixel_intensity', value: 0.73, importance: 0.15 },
        { name: 'edge_density', value: 0.68, importance: 0.22 },
        { name: 'texture_variance', value: 0.54, importance: 0.18 },
        { name: 'motion_vector', value: 0.89, importance: 0.31 },
        { name: 'depth_estimate', value: 0.41, importance: 0.14 }
      ],
      outputProbabilities: [
        { class: 'person', probability: 0.942 },
        { class: 'vehicle', probability: 0.038 },
        { class: 'animal', probability: 0.012 },
        { class: 'object', probability: 0.008 }
      ],
      attentionWeights: [
        { token: 'suspicious', weight: 0.23 },
        { token: 'movement', weight: 0.19 },
        { token: 'coordinated', weight: 0.17 },
        { token: 'restricted', weight: 0.15 },
        { token: 'area', weight: 0.12 },
        { token: 'multiple', weight: 0.08 },
        { token: 'individuals', weight: 0.06 }
      ],
      shapValues: [
        { feature: 'coordination_score', contribution: 0.34 },
        { feature: 'proximity_critical', contribution: 0.28 },
        { feature: 'time_anomaly', contribution: 0.21 },
        { feature: 'velocity_variance', contribution: 0.17 }
      ]
    }

    setModelInterpretation(mockInterpretation)

    // Generate mock risk assessment
    const mockRiskAssessment: RiskAssessment = {
      id: 'risk-001',
      overallRisk: 0.78,
      riskFactors: [
        { factor: 'Proximity to Critical Infrastructure', weight: 0.35, score: 0.92, description: 'Within 100m of data center facility' },
        { factor: 'Temporal Anomaly', weight: 0.25, score: 0.85, description: 'Activity during low-traffic hours' },
        { factor: 'Behavioral Coordination', weight: 0.30, score: 0.87, description: 'Highly synchronized movement patterns' },
        { factor: 'Historical Precedent', weight: 0.10, score: 0.73, description: 'Similar patterns in incident history' }
      ],
      temporalTrend: 'increasing',
      spatialSpread: 0.45,
      mitigationOptions: [
        { option: 'Immediate Patrol Deployment', effectiveness: 0.89, cost: 0.65, feasibility: 0.92 },
        { option: 'Drone Surveillance', effectiveness: 0.76, cost: 0.35, feasibility: 0.88 },
        { option: 'Access Point Lockdown', effectiveness: 0.94, cost: 0.45, feasibility: 0.78 }
      ]
    }

    setRiskAssessment(mockRiskAssessment)

    // Generate mock recommended response
    const mockResponse: RecommendedResponse = {
      id: 'response-001',
      action: 'Deploy Mobile Patrol Unit with Drone Support',
      priority: 1,
      estimatedEffectiveness: 0.87,
      resourceRequirements: [
        { resource: 'Patrol Unit', quantity: 1 },
        { resource: 'Officers', quantity: 2 },
        { resource: 'Surveillance Drone', quantity: 1 },
        { resource: 'Radio Channel', quantity: 1 }
      ],
      executionTime: 8,
      sideEffects: ['Potential escalation', 'Civilian disruption', 'Resource allocation'],
      alternatives: [
        {
          action: 'Drone-Only Surveillance',
          effectiveness: 0.72,
          tradeoffs: ['Lower deterrence', 'Limited intervention capability', 'Reduced risk to personnel']
        },
        {
          action: 'Remote Monitoring',
          effectiveness: 0.45,
          tradeoffs: ['Minimal deterrence', 'Delayed response', 'Resource efficient']
        }
      ]
    }

    setRecommendedResponse(mockResponse)
  }, [])

  const getReasoningTypeColor = (type: string) => {
    switch (type) {
      case 'detection': return 'text-blue-400 border-blue-500/30'
      case 'analysis': return 'text-yellow-400 border-yellow-500/30'
      case 'inference': return 'text-purple-400 border-purple-500/30'
      case 'prediction': return 'text-red-400 border-red-500/30'
      default: return 'text-gray-400 border-gray-500/30'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-400'
    if (confidence >= 75) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div className={clsx('h-full flex flex-col bg-gray-900/50 rounded-lg border border-cyan-500/20', className)}>
      {/* Header */}
      <div className="bg-gray-800/90 backdrop-blur-sm border-b border-cyan-500/20 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Brain className="text-cyan-400 w-5 h-5" />
            <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">
              LLM Reasoning & Explainability
            </h3>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-green-400 text-xs font-mono">ACTIVE</span>
            </div>
          </div>
          
          {selectedEvent && (
            <div className="flex items-center space-x-2 text-xs">
              <span className="text-gray-400">Event:</span>
              <span className="text-cyan-400 font-mono">{selectedEvent.id}</span>
              <span className={clsx(
                'px-2 py-1 rounded font-mono uppercase',
                selectedEvent.severity === 'critical' ? 'text-red-400 bg-red-900/20' :
                selectedEvent.severity === 'high' ? 'text-orange-400 bg-orange-900/20' :
                selectedEvent.severity === 'medium' ? 'text-yellow-400 bg-yellow-900/20' :
                'text-blue-400 bg-blue-900/20'
              )}>
                {selectedEvent.severity}
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Event Summary */}
      {selectedEvent && (
        <div className="bg-gray-800/50 border-b border-cyan-500/20 p-3">
          <div className="flex items-start justify-between">
            <div>
              <h4 className="text-gray-200 font-mono text-sm mb-1">{selectedEvent.description}</h4>
              <div className="flex items-center space-x-4 text-xs">
                <span className="text-gray-400">{selectedEvent.location}</span>
                <span className="text-gray-400">{selectedEvent.timestamp.toLocaleTimeString()}</span>
                <span className={clsx('font-mono', getConfidenceColor(selectedEvent.confidence))}>
                  {selectedEvent.confidence}% confidence
                </span>
              </div>
            </div>
            <div className="flex items-center space-x-1">
              <Target className="text-cyan-400 w-4 h-4" />
              <span className="text-cyan-400 text-xs font-mono uppercase">{selectedEvent.eventType.replace('_', ' ')}</span>
            </div>
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="flex border-b border-cyan-500/20">
        {[
          { id: 'reasoning', label: 'Reasoning', icon: MessageSquare },
          { id: 'interpretation', label: 'Model', icon: Eye },
          { id: 'risk', label: 'Risk', icon: AlertTriangle },
          { id: 'response', label: 'Response', icon: CheckCircle }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={clsx(
              'flex-1 flex items-center justify-center space-x-2 py-3 text-xs font-mono uppercase tracking-wider transition-all',
              activeTab === tab.id
                ? 'text-cyan-400 border-b-2 border-cyan-400 bg-cyan-400/10'
                : 'text-gray-500 hover:text-gray-300'
            )}
          >
            <tab.icon className="w-4 h-4" />
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {activeTab === 'reasoning' && (
            <motion.div
              key="reasoning"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="h-full overflow-y-auto p-4"
            >
              <div className="space-y-3">
                {reasoningSteps.map((step, index) => (
                  <motion.div
                    key={step.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className={clsx(
                      'border rounded-lg p-3 cursor-pointer transition-all',
                      expandedStep === step.id 
                        ? 'border-cyan-400 bg-cyan-400/10' 
                        : 'border-gray-700 hover:border-gray-600'
                    )}
                    onClick={() => setExpandedStep(expandedStep === step.id ? null : step.id)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <div className="flex items-center justify-center w-6 h-6 rounded-full bg-cyan-400/20 text-cyan-400 text-xs font-mono">
                          {step.step}
                        </div>
                        <div>
                          <h5 className="text-gray-200 font-mono text-sm">{step.title}</h5>
                          <p className="text-gray-400 text-xs mt-1">{step.description}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <span className={clsx(
                          'text-xs px-2 py-1 rounded border font-mono uppercase',
                          getReasoningTypeColor(step.reasoningType)
                        )}>
                          {step.reasoningType}
                        </span>
                        <span className={clsx('text-xs font-mono', getConfidenceColor(step.confidence))}>
                          {step.confidence.toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <AnimatePresence>
                      {expandedStep === step.id && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="mt-3 pt-3 border-t border-gray-700"
                        >
                          <div className="space-y-2">
                            <div className="flex items-center space-x-2">
                              <FileText className="text-cyan-400 w-3 h-3" />
                              <span className="text-cyan-400 text-xs font-mono uppercase">Evidence</span>
                            </div>
                            <ul className="space-y-1">
                              {step.evidence.map((evidence, evidenceIndex) => (
                                <li key={evidenceIndex} className="text-gray-300 text-xs flex items-start space-x-2">
                                  <div className="w-1 h-1 bg-cyan-400 rounded-full mt-1 flex-shrink-0" />
                                  <span>{evidence}</span>
                                </li>
                              ))}
                            </ul>
                            <div className="flex items-center space-x-2 text-xs text-gray-500">
                              <Clock className="w-3 h-3" />
                              <span>{step.timestamp.toLocaleTimeString()}</span>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {activeTab === 'interpretation' && modelInterpretation && (
            <motion.div
              key="interpretation"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="h-full overflow-y-auto p-4"
            >
              <div className="space-y-4">
                {/* Model Info */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-2">Model Information</h5>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-gray-400">Model:</span>
                      <span className="ml-2 text-gray-300">{modelInterpretation.modelName}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Version:</span>
                      <span className="ml-2 text-gray-300">{modelInterpretation.version}</span>
                    </div>
                  </div>
                </div>

                {/* Input Features */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Input Features</h5>
                  <div className="space-y-2">
                    {modelInterpretation.inputFeatures.map((feature, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        <span className="text-gray-400 text-xs w-24">{feature.name}</span>
                        <div className="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                          <motion.div
                            className="h-full bg-cyan-400"
                            initial={{ width: 0 }}
                            animate={{ width: `${feature.value * 100}%` }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                          />
                        </div>
                        <span className="text-gray-300 text-xs w-12 text-right">{(feature.value * 100).toFixed(0)}%</span>
                        <span className="text-cyan-400 text-xs w-8 text-right">{(feature.importance * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Output Probabilities */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Output Probabilities</h5>
                  <div className="space-y-2">
                    {modelInterpretation.outputProbabilities.map((output, index) => (
                      <div key={index} className="flex items-center space-x-3">
                        <span className="text-gray-400 text-xs w-16">{output.class}</span>
                        <div className="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                          <motion.div
                            className={clsx(
                              'h-full',
                              output.probability > 0.8 ? 'bg-green-400' :
                              output.probability > 0.5 ? 'bg-yellow-400' :
                              'bg-red-400'
                            )}
                            initial={{ width: 0 }}
                            animate={{ width: `${output.probability * 100}%` }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                          />
                        </div>
                        <span className={clsx(
                          'text-xs w-12 text-right font-mono',
                          output.probability > 0.8 ? 'text-green-400' :
                          output.probability > 0.5 ? 'text-yellow-400' :
                          'text-red-400'
                        )}>
                          {(output.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Attention Weights */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Attention Weights</h5>
                  <div className="flex flex-wrap gap-2">
                    {modelInterpretation.attentionWeights.map((weight, index) => (
                      <div
                        key={index}
                        className={clsx(
                          'px-2 py-1 rounded text-xs font-mono',
                          weight.weight > 0.2 ? 'bg-cyan-400/20 text-cyan-300 border border-cyan-400/30' :
                          'bg-gray-700 text-gray-400 border border-gray-600'
                        )}
                      >
                        {weight.token} ({(weight.weight * 100).toFixed(0)}%)
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'risk' && riskAssessment && (
            <motion.div
              key="risk"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="h-full overflow-y-auto p-4"
            >
              <div className="space-y-4">
                {/* Overall Risk */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Overall Risk Assessment</h5>
                  <div className="flex items-center space-x-4">
                    <div className="relative w-20 h-20">
                      <svg className="transform -rotate-90 w-20 h-20">
                        <circle
                          cx="40"
                          cy="40"
                          r="36"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="none"
                          className="text-gray-700"
                        />
                        <motion.circle
                          cx="40"
                          cy="40"
                          r="36"
                          stroke="currentColor"
                          strokeWidth="8"
                          fill="none"
                          className={riskAssessment.overallRisk > 0.7 ? 'text-red-400' :
                                     riskAssessment.overallRisk > 0.5 ? 'text-yellow-400' :
                                     'text-green-400'}
                          initial={{ strokeDasharray: '226', strokeDashoffset: '226' }}
                          animate={{ strokeDashoffset: 226 - (226 * riskAssessment.overallRisk) }}
                          transition={{ duration: 1 }}
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <span className={clsx(
                          'text-xl font-bold',
                          riskAssessment.overallRisk > 0.7 ? 'text-red-400' :
                          riskAssessment.overallRisk > 0.5 ? 'text-yellow-400' :
                          'text-green-400'
                        )}>
                          {(riskAssessment.overallRisk * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                    <div className="flex-1">
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-400">Trend:</span>
                          <span className={clsx(
                            'ml-2 font-mono capitalize',
                            riskAssessment.temporalTrend === 'increasing' ? 'text-red-400' :
                            riskAssessment.temporalTrend === 'decreasing' ? 'text-green-400' :
                            'text-yellow-400'
                          )}>
                            {riskAssessment.temporalTrend}
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">Spread:</span>
                          <span className="ml-2 text-gray-300">{(riskAssessment.spatialSpread * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Risk Factors */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Risk Factors</h5>
                  <div className="space-y-3">
                    {riskAssessment.riskFactors.map((factor, index) => (
                      <div key={index}>
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-gray-300 text-xs">{factor.factor}</span>
                          <span className="text-cyan-400 text-xs font-mono">{(factor.score * 100).toFixed(0)}%</span>
                        </div>
                        <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                          <motion.div
                            className={clsx(
                              'h-full',
                              factor.score > 0.8 ? 'bg-red-400' :
                              factor.score > 0.6 ? 'bg-yellow-400' :
                              'bg-green-400'
                            )}
                            initial={{ width: 0 }}
                            animate={{ width: `${factor.score * 100}%` }}
                            transition={{ duration: 0.5, delay: index * 0.1 }}
                          />
                        </div>
                        <p className="text-gray-500 text-xs mt-1">{factor.description}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Mitigation Options */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Mitigation Options</h5>
                  <div className="space-y-2">
                    {riskAssessment.mitigationOptions.map((option, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-gray-700/30 rounded">
                        <div className="flex-1">
                          <span className="text-gray-300 text-xs">{option.option}</span>
                          <div className="flex items-center space-x-3 mt-1">
                            <span className="text-gray-500 text-xs">Effectiveness: {(option.effectiveness * 100).toFixed(0)}%</span>
                            <span className="text-gray-500 text-xs">Cost: {(option.cost * 100).toFixed(0)}%</span>
                            <span className="text-gray-500 text-xs">Feasibility: {(option.feasibility * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'response' && recommendedResponse && (
            <motion.div
              key="response"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="h-full overflow-y-auto p-4"
            >
              <div className="space-y-4">
                {/* Recommended Action */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Recommended Action</h5>
                  <div className="flex items-start justify-between">
                    <div>
                      <h6 className="text-gray-200 font-mono text-sm mb-2">{recommendedResponse.action}</h6>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>
                          <span className="text-gray-400">Priority:</span>
                          <span className="ml-2 text-red-400 font-mono">#{recommendedResponse.priority}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Effectiveness:</span>
                          <span className={clsx(
                            'ml-2 font-mono',
                            recommendedResponse.estimatedEffectiveness > 0.8 ? 'text-green-400' :
                            recommendedResponse.estimatedEffectiveness > 0.6 ? 'text-yellow-400' :
                            'text-red-400'
                          )}>
                            {(recommendedResponse.estimatedEffectiveness * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div>
                          <span className="text-gray-400">Execution Time:</span>
                          <span className="ml-2 text-gray-300">{recommendedResponse.executionTime} min</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Side Effects:</span>
                          <span className="ml-2 text-orange-400">{recommendedResponse.sideEffects.length} risks</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-1">
                      <CheckCircle className="text-green-400 w-5 h-5" />
                    </div>
                  </div>
                </div>

                {/* Resource Requirements */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Resource Requirements</h5>
                  <div className="space-y-2">
                    {recommendedResponse.resourceRequirements.map((resource, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-gray-700/30 rounded">
                        <span className="text-gray-300 text-xs">{resource.resource}</span>
                        <span className="text-cyan-400 font-mono text-xs">x{resource.quantity}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Side Effects */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Potential Side Effects</h5>
                  <ul className="space-y-1">
                    {recommendedResponse.sideEffects.map((effect, index) => (
                      <li key={index} className="text-gray-300 text-xs flex items-center space-x-2">
                        <AlertTriangle className="text-yellow-400 w-3 h-3" />
                        <span>{effect}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Alternatives */}
                <div className="bg-gray-800/50 rounded-lg border border-cyan-500/20 p-3">
                  <h5 className="text-cyan-400 text-sm font-mono mb-3">Alternative Actions</h5>
                  <div className="space-y-3">
                    {recommendedResponse.alternatives.map((alternative, index) => (
                      <div key={index} className="p-3 bg-gray-700/30 rounded">
                        <div className="flex items-start justify-between mb-2">
                          <span className="text-gray-300 text-xs font-mono">{alternative.action}</span>
                          <span className={clsx(
                            'text-xs font-mono',
                            alternative.effectiveness > 0.7 ? 'text-green-400' :
                            alternative.effectiveness > 0.5 ? 'text-yellow-400' :
                            'text-red-400'
                          )}>
                            {(alternative.effectiveness * 100).toFixed(0)}% effective
                          </span>
                        </div>
                        <div className="space-y-1">
                          {alternative.tradeoffs.map((tradeoff, tradeoffIndex) => (
                            <div key={tradeoffIndex} className="text-gray-500 text-xs flex items-center space-x-2">
                              <div className="w-1 h-1 bg-gray-400 rounded-full" />
                              <span>{tradeoff}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

export default LLMReasoningPanel
