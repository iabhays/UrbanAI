import React, { useState, useEffect } from 'react'
import { Play, Square, BarChart3, AlertCircle, Info } from 'lucide-react'
import videoService from '../../services/videoService'

export default function VideoAnalysisPanel() {
  const [videos, setVideos] = useState([])
  const [analyzing, setAnalyzing] = useState({})
  const [results, setResults] = useState({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Fetch videos on mount
  useEffect(() => {
    fetchVideos()
    // Poll for status updates
    const interval = setInterval(fetchVideos, 2000)
    return () => clearInterval(interval)
  }, [])

  const fetchVideos = async () => {
    try {
      const videoList = await videoService.getVideos()
      setVideos(videoList)
      setLoading(false)
      setError(null)

      // Update status for each video
      for (const video of videoList) {
        const status = await videoService.getVideoStatus(video.id)
        setAnalyzing(prev => ({
          ...prev,
          [video.id]: status.status === 'analyzing'
        }))

        if (status.status === 'completed') {
          const analysisResults = await videoService.getAnalysisResults(video.id)
          setResults(prev => ({
            ...prev,
            [video.id]: analysisResults
          }))
        }
      }
    } catch (err) {
      setError('Failed to load videos')
      console.error(err)
      setLoading(false)
    }
  }

  const handleAnalyze = async (videoId) => {
    try {
      setAnalyzing(prev => ({ ...prev, [videoId]: true }))
      await videoService.analyzeVideo(videoId)
    } catch (err) {
      setAnalyzing(prev => ({ ...prev, [videoId]: false }))
      setError(`Failed to analyze video: ${err.message}`)
    }
  }

  const handleStop = async (videoId) => {
    try {
      await videoService.stopAnalysis(videoId)
      setAnalyzing(prev => ({ ...prev, [videoId]: false }))
    } catch (err) {
      setError(`Failed to stop analysis: ${err.message}`)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading videos...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-white mb-2">📹 Demo Video Analysis</h1>
        <p className="text-gray-400">Test with {videos.length} demo videos (isolated from production)</p>
      </div>

      {/* Demo Information Banner */}
      <div className="mb-6 p-4 bg-blue-900/30 border border-blue-600/50 rounded-lg flex items-start gap-3">
        <Info className="text-blue-400 mt-0.5 flex-shrink-0" size={24} />
        <div className="text-sm text-blue-200">
          <p className="font-semibold mb-1">🎯 Demo Mode - Production Camera Sources Unaffected</p>
          <p>This is a <strong>standalone demo</strong> for testing with local videos.</p>
          <p className="mt-1">
            <strong>Production camera sources</strong> (RTSP, IP cameras, etc.) run <strong>independently</strong> via:
          </p>
          <code className="block mt-2 bg-gray-900/50 p-2 rounded text-blue-300">
            python scripts/run_pipeline.py --camera your_rtsp_url
          </code>
          <p className="mt-2">Both can run simultaneously without interfering with each other.</p>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mb-6 p-4 bg-red-900/30 border border-red-600 rounded-lg flex items-center gap-3">
          <AlertCircle className="text-red-500" size={24} />
          <p className="text-red-300">{error}</p>
        </div>
      )}

      {/* Video Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {videos.map(video => (
          <VideoCard
            key={video.id}
            video={video}
            isAnalyzing={analyzing[video.id] || false}
            results={results[video.id]}
            onAnalyze={() => handleAnalyze(video.id)}
            onStop={() => handleStop(video.id)}
          />
        ))}
      </div>

      {/* Empty State */}
      {videos.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-400">No videos found. Add videos to datasets/raw/ folder.</p>
        </div>
      )}
    </div>
  )
}

/**
 * Individual Video Card
 */
function VideoCard({ video, isAnalyzing, results, onAnalyze, onStop }) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="bg-gray-800/50 backdrop-blur border border-gray-700 rounded-xl overflow-hidden hover:border-blue-500/50 transition-all">
      {/* Video Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="text-white font-semibold text-lg truncate">{video.name}</h3>
          <span className="text-xs bg-blue-900/60 text-blue-300 px-2 py-1 rounded">DEMO</span>
        </div>
        <p className="text-gray-400 text-sm">{video.duration || 'Duration unknown'}</p>
      </div>

      {/* Analysis Status */}
      <div className="p-4 space-y-3">
        {/* Progress Bar */}
        {isAnalyzing && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Processing</span>
              <span className="text-blue-400 font-mono">{video.progress || 0}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
              <div
                className="bg-gradient-to-r from-blue-500 to-cyan-500 h-full transition-all duration-300"
                style={{ width: `${video.progress || 0}%` }}
              />
            </div>
          </div>
        )}

        {/* Status Badge */}
        <div className="flex gap-2 items-center">
          <div
            className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
              isAnalyzing
                ? 'bg-blue-900/50 text-blue-300 animate-pulse'
                : results
                  ? 'bg-green-900/50 text-green-300'
                  : 'bg-gray-700/50 text-gray-300'
            }`}
          >
            {isAnalyzing ? '● Analyzing' : results ? '✓ Complete' : '○ Ready'}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2 pt-2">
          {!isAnalyzing ? (
            <button
              onClick={onAnalyze}
              className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors font-medium"
            >
              <Play size={18} />
              Analyze
            </button>
          ) : (
            <button
              onClick={onStop}
              className="flex-1 flex items-center justify-center gap-2 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-lg transition-colors font-medium"
            >
              <Square size={18} />
              Stop
            </button>
          )}

          {results && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="flex items-center justify-center gap-2 bg-gray-700 hover:bg-gray-600 text-white py-2 px-4 rounded-lg transition-colors"
            >
              <BarChart3 size={18} />
            </button>
          )}
        </div>
      </div>

      {/* Expanded Results */}
      {expanded && results && (
        <div className="p-4 border-t border-gray-700 bg-gray-900/50 space-y-3">
          <div className="space-y-2">
            <h4 className="text-white font-semibold text-sm">Analysis Results</h4>

            {/* Detection Stats */}
            <div className="grid grid-cols-2 gap-3">
              <StatBox
                label="Objects Detected"
                value={results.total_detections || 0}
              />
              <StatBox
                label="Tracks Found"
                value={results.total_tracks || 0}
              />
              <StatBox
                label="Crowd Density"
                value={`${Math.round((results.avg_crowd_density || 0) * 100)}%`}
                color="text-yellow-400"
              />
              <StatBox
                label="Risk Level"
                value={results.risk_level || 'Low'}
                color={
                  results.risk_level === 'Critical'
                    ? 'text-red-400'
                    : results.risk_level === 'High'
                      ? 'text-orange-400'
                      : 'text-green-400'
                }
              />
            </div>

            {/* Alerts */}
            {results.alerts && results.alerts.length > 0 && (
              <div className="mt-3 p-2 bg-red-900/20 border border-red-700/50 rounded space-y-1">
                <p className="text-xs text-red-400 font-semibold">⚠ Alerts Detected</p>
                {results.alerts.slice(0, 2).map((alert, idx) => (
                  <p key={idx} className="text-xs text-gray-300">
                    • {alert}
                  </p>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Stat Box Component
 */
function StatBox({ label, value, color = 'text-blue-400' }) {
  return (
    <div className="bg-gray-700/30 rounded p-2">
      <p className="text-xs text-gray-400">{label}</p>
      <p className={`text-lg font-bold ${color}`}>{value}</p>
    </div>
  )
}
