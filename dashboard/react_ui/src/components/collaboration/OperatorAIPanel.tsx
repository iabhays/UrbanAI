import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  CheckCircle, 
  XCircle, 
  AlertTriangle, 
  User, 
  Bot, 
  Clock,
  MessageSquare,
  TrendingUp,
  Shield,
  Eye
} from 'lucide-react'
import { clsx } from '../../utils/clsx'

interface AISuggestion {
  id: string
  timestamp: Date
  type: 'threat_response' | 'resource_allocation' | 'system_adjustment' | 'emergency_protocol'
  priority: 'low' | 'medium' | 'high' | 'critical'
  title: string
  description: string
  reasoning: string
  confidence: number
  expectedOutcome: string
  risks: string[]
  resources: Array<{ name: string; quantity: number }>
  estimatedDuration: number
}

interface OperatorDecision {
  id: string
  suggestionId: string
  operator: string
  action: 'approve' | 'reject' | 'escalate' | 'modify'
  timestamp: Date
  reasoning: string
  outcome: 'pending' | 'executing' | 'completed' | 'failed'
}

interface OperatorAIPanelProps {
  className?: string
  onDecision?: (decision: OperatorDecision) => void
}

const OperatorAIPanel: React.FC<OperatorAIPanelProps> = ({ 
  className,
  onDecision 
}) => {
  const [suggestions, setSuggestions] = useState<AISuggestion[]>([])
  const [decisions, setDecisions] = useState<OperatorDecision[]>([])
  const [selectedSuggestion, setSelectedSuggestion] = useState<string | null>(null)
  const [showReasoning, setShowReasoning] = useState(false)
  const [customReasoning, setCustomReasoning] = useState('')

  useEffect(() => {
    // Generate mock AI suggestions
    const mockSuggestions: AISuggestion[] = [
      {
        id: 'ai-suggestion-1',
        timestamp: new Date(),
        type: 'threat_response',
        priority: 'high',
        title: 'Deploy Mobile Unit to Zone 4',
        description: 'Elevated threat activity detected in commercial district. Immediate response recommended.',
        reasoning: 'Multiple suspicious vehicles detected loitering near high-value targets. Pattern matches previous incident precursors. Risk assessment indicates 78% probability of escalation.',
        confidence: 87,
        expectedOutcome: 'Threat mitigation through visible presence and rapid response capability',
        risks: ['Resource depletion', 'False alarm escalation', 'Civilian disruption'],
        resources: [
          { name: 'Mobile Patrol Unit', quantity: 1 },
          { name: 'Officers', quantity: 2 },
          { name: 'Drone Support', quantity: 1 }
        ],
        estimatedDuration: 45
      },
      {
        id: 'ai-suggestion-2',
        timestamp: new Date(Date.now() - 300000),
        type: 'resource_allocation',
        priority: 'medium',
        title: 'Reallocate Camera Processing Resources',
        description: 'Optimize GPU allocation for high-traffic monitoring zones during peak hours.',
        reasoning: 'Current resource distribution shows 40% underutilization in low-risk areas while high-risk zones experience processing delays. Reallocation would improve overall system efficiency by 23%.',
        confidence: 92,
        expectedOutcome: 'Improved detection accuracy and reduced latency in critical areas',
        risks: ['Temporary coverage gaps', 'System instability during transition'],
        resources: [
          { name: 'GPU Cores', quantity: 4 },
          { name: 'Memory Allocation', quantity: 8 }
        ],
        estimatedDuration: 15
      },
      {
        id: 'ai-suggestion-3',
        timestamp: new Date(Date.now() - 600000),
        type: 'system_adjustment',
        priority: 'low',
        title: 'Adjust Risk Threshold for Zone 2',
        description: 'Lower sensitivity threshold based on recent false positive patterns.',
        reasoning: 'Analysis of last 200 alerts shows 67% false positive rate in Zone 2 during nighttime hours. Threshold adjustment would reduce alert fatigue while maintaining security coverage.',
        confidence: 78,
        expectedOutcome: 'Reduced false alerts, improved operator focus on genuine threats',
        risks: ['Missed low-level threats', 'Delayed response to subtle anomalies'],
        resources: [
          { name: 'Configuration Update', quantity: 1 }
        ],
        estimatedDuration: 5
      }
    ]

    setSuggestions(mockSuggestions)
  }, [])

  const handleDecision = (suggestionId: string, action: 'approve' | 'reject' | 'escalate') => {
    const decision: OperatorDecision = {
      id: `decision-${Date.now()}`,
      suggestionId,
      operator: 'Operator-001',
      action,
      timestamp: new Date(),
      reasoning: customReasoning || `Action: ${action}`,
      outcome: 'pending'
    }

    setDecisions(prev => [decision, ...prev])
    
    if (onDecision) {
      onDecision(decision)
    }

    // Update suggestion status
    setSuggestions(prev => prev.map(s => 
      s.id === suggestionId 
        ? { ...s, status: action === 'approve' ? 'approved' : action === 'reject' ? 'rejected' : 'escalated' }
        : s
    ))

    setSelectedSuggestion(null)
    setCustomReasoning('')
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'text-red-400 border-red-500/30 bg-red-900/20'
      case 'high': return 'text-orange-400 border-orange-500/30 bg-orange-900/20'
      case 'medium': return 'text-yellow-400 border-yellow-500/30 bg-yellow-900/20'
      default: return 'text-blue-400 border-blue-500/30 bg-blue-900/20'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-400'
    if (confidence >= 75) return 'text-yellow-400'
    return 'text-red-400'
  }

  const selectedSuggestionData = suggestions.find(s => s.id === selectedSuggestion)

  return (
    <div className={clsx('h-full flex flex-col bg-gray-900/50 rounded-lg border border-cyan-500/20', className)}>
      {/* Header */}
      <div className="bg-gray-800/90 backdrop-blur-sm border-b border-cyan-500/20 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Bot className="text-cyan-400 w-5 h-5" />
            <h3 className="text-cyan-400 text-sm font-mono uppercase tracking-wider">
              AI-Operator Collaboration
            </h3>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-green-400 text-xs font-mono">ACTIVE</span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4 text-xs">
            <div className="flex items-center space-x-1">
              <span className="text-gray-400">Pending:</span>
              <span className="text-cyan-400 font-mono">{suggestions.length}</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="text-gray-400">Decisions:</span>
              <span className="text-cyan-400 font-mono">{decisions.length}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Suggestions List */}
        <div className="w-1/2 border-r border-cyan-500/20 overflow-y-auto p-4">
          <div className="space-y-3">
            {suggestions.map((suggestion) => (
              <motion.div
                key={suggestion.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                whileHover={{ scale: 1.02 }}
                onClick={() => setSelectedSuggestion(suggestion.id)}
                className={clsx(
                  'p-3 rounded-lg border cursor-pointer transition-all',
                  selectedSuggestion === suggestion.id 
                    ? 'border-cyan-400 bg-cyan-400/10' 
                    : 'border-gray-700 hover:border-gray-600'
                )}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className={clsx(
                      'text-xs font-mono uppercase px-2 py-1 rounded',
                      getPriorityColor(suggestion.priority)
                    )}>
                      {suggestion.priority}
                    </span>
                    <span className="text-gray-400 text-xs">
                      {suggestion.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <div className="flex items-center space-x-1">
                    <Bot className="text-cyan-400 w-3 h-3" />
                    <span className={clsx('text-xs font-mono', getConfidenceColor(suggestion.confidence))}>
                      {suggestion.confidence}%
                    </span>
                  </div>
                </div>

                <h4 className="text-gray-200 font-mono text-sm mb-1">{suggestion.title}</h4>
                <p className="text-gray-400 text-xs mb-2">{suggestion.description}</p>

                <div className="flex items-center justify-between">
                  <span className="text-gray-500 text-xs">{suggestion.type.replace('_', ' ')}</span>
                  <span className="text-gray-500 text-xs">~{suggestion.estimatedDuration}min</span>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Decision Panel */}
        <div className="w-1/2 flex flex-col">
          <AnimatePresence mode="wait">
            {selectedSuggestionData ? (
              <motion.div
                key={selectedSuggestion}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex-1 flex flex-col p-4"
              >
                {/* Suggestion Details */}
                <div className="flex-1 space-y-4 overflow-y-auto">
                  <div>
                    <h4 className="text-cyan-400 font-mono text-sm mb-2">{selectedSuggestionData.title}</h4>
                    <p className="text-gray-300 text-sm mb-3">{selectedSuggestionData.description}</p>
                    
                    <div className="grid grid-cols-2 gap-4 text-xs">
                      <div>
                        <span className="text-gray-400">Priority:</span>
                        <span className={clsx('ml-2 font-mono', getPriorityColor(selectedSuggestionData.priority))}>
                          {selectedSuggestionData.priority.toUpperCase()}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Confidence:</span>
                        <span className={clsx('ml-2 font-mono', getConfidenceColor(selectedSuggestionData.confidence))}>
                          {selectedSuggestionData.confidence}%
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-400">Duration:</span>
                        <span className="ml-2 text-gray-300">{selectedSuggestionData.estimatedDuration} minutes</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Type:</span>
                        <span className="ml-2 text-gray-300">{selectedSuggestionData.type.replace('_', ' ')}</span>
                      </div>
                    </div>
                  </div>

                  {/* AI Reasoning */}
                  <div>
                    <div className="flex items-center space-x-2 mb-2">
                      <MessageSquare className="text-cyan-400 w-4 h-4" />
                      <h5 className="text-cyan-400 text-xs font-mono uppercase">AI Reasoning</h5>
                    </div>
                    <p className="text-gray-300 text-xs leading-relaxed">
                      {selectedSuggestionData.reasoning}
                    </p>
                  </div>

                  {/* Expected Outcome */}
                  <div>
                    <div className="flex items-center space-x-2 mb-2">
                      <TrendingUp className="text-cyan-400 w-4 h-4" />
                      <h5 className="text-cyan-400 text-xs font-mono uppercase">Expected Outcome</h5>
                    </div>
                    <p className="text-gray-300 text-xs">{selectedSuggestionData.expectedOutcome}</p>
                  </div>

                  {/* Risk Assessment */}
                  <div>
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertTriangle className="text-cyan-400 w-4 h-4" />
                      <h5 className="text-cyan-400 text-xs font-mono uppercase">Risk Factors</h5>
                    </div>
                    <ul className="space-y-1">
                      {selectedSuggestionData.risks.map((risk, index) => (
                        <li key={index} className="text-gray-300 text-xs flex items-center space-x-2">
                          <div className="w-1 h-1 bg-red-400 rounded-full" />
                          <span>{risk}</span>
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Resource Requirements */}
                  <div>
                    <div className="flex items-center space-x-2 mb-2">
                      <Shield className="text-cyan-400 w-4 h-4" />
                      <h5 className="text-cyan-400 text-xs font-mono uppercase">Resource Requirements</h5>
                    </div>
                    <div className="space-y-1">
                      {selectedSuggestionData.resources.map((resource, index) => (
                        <div key={index} className="flex justify-between text-xs">
                          <span className="text-gray-300">{resource.name}</span>
                          <span className="text-cyan-400 font-mono">x{resource.quantity}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Operator Reasoning Input */}
                  <div>
                    <div className="flex items-center space-x-2 mb-2">
                      <User className="text-cyan-400 w-4 h-4" />
                      <h5 className="text-cyan-400 text-xs font-mono uppercase">Operator Reasoning</h5>
                    </div>
                    <textarea
                      value={customReasoning}
                      onChange={(e) => setCustomReasoning(e.target.value)}
                      placeholder="Add your reasoning for this decision..."
                      className="w-full h-20 bg-gray-800/50 border border-cyan-500/20 rounded p-2 text-xs text-gray-300 resize-none focus:outline-none focus:border-cyan-400/50"
                    />
                  </div>
                </div>

                {/* Decision Buttons */}
                <div className="border-t border-cyan-500/20 pt-4">
                  <div className="grid grid-cols-3 gap-2">
                    <button
                      onClick={() => handleDecision(selectedSuggestionData.id, 'approve')}
                      className="flex items-center justify-center space-x-1 bg-green-600/20 border border-green-500/30 text-green-400 py-2 px-3 rounded text-xs font-mono hover:bg-green-600/30 transition-colors"
                    >
                      <CheckCircle className="w-3 h-3" />
                      <span>APPROVE</span>
                    </button>
                    
                    <button
                      onClick={() => handleDecision(selectedSuggestionData.id, 'reject')}
                      className="flex items-center justify-center space-x-1 bg-red-600/20 border border-red-500/30 text-red-400 py-2 px-3 rounded text-xs font-mono hover:bg-red-600/30 transition-colors"
                    >
                      <XCircle className="w-3 h-3" />
                      <span>REJECT</span>
                    </button>
                    
                    <button
                      onClick={() => handleDecision(selectedSuggestionData.id, 'escalate')}
                      className="flex items-center justify-center space-x-1 bg-yellow-600/20 border border-yellow-500/30 text-yellow-400 py-2 px-3 rounded text-xs font-mono hover:bg-yellow-600/30 transition-colors"
                    >
                      <AlertTriangle className="w-3 h-3" />
                      <span>ESCALATE</span>
                    </button>
                  </div>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex-1 flex items-center justify-center"
              >
                <div className="text-center">
                  <Bot className="text-gray-600 w-12 h-12 mx-auto mb-3" />
                  <p className="text-gray-500 text-sm font-mono">Select an AI suggestion to review</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Decision History */}
      <div className="border-t border-cyan-500/20 p-3">
        <div className="flex items-center justify-between mb-2">
          <h5 className="text-cyan-400 text-xs font-mono uppercase">Recent Decisions</h5>
          <button className="text-gray-400 hover:text-cyan-400 text-xs">
            <Eye className="w-3 h-3" />
          </button>
        </div>
        
        <div className="space-y-1 max-h-20 overflow-y-auto">
          {decisions.slice(0, 3).map((decision) => (
            <div key={decision.id} className="flex items-center justify-between text-xs">
              <div className="flex items-center space-x-2">
                {decision.action === 'approve' && <CheckCircle className="w-3 h-3 text-green-400" />}
                {decision.action === 'reject' && <XCircle className="w-3 h-3 text-red-400" />}
                {decision.action === 'escalate' && <AlertTriangle className="w-3 h-3 text-yellow-400" />}
                <span className="text-gray-300">{decision.action.toUpperCase()}</span>
              </div>
              <span className="text-gray-500">{decision.timestamp.toLocaleTimeString()}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default OperatorAIPanel
