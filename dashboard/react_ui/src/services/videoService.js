/**
 * Video Service
 * Handles video management and analysis
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'

export const videoService = {
  /**
   * Get list of available videos
   */
  async getVideos() {
    try {
      const response = await fetch(`${API_BASE_URL}/videos`)
      if (!response.ok) throw new Error('Failed to fetch videos')
      return await response.json()
    } catch (error) {
      console.error('Error fetching videos:', error)
      return []
    }
  },

  /**
   * Start analyzing a video
   */
  async analyzeVideo(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/${videoId}/analyze`, {
        method: 'POST'
      })
      if (!response.ok) throw new Error('Failed to start analysis')
      return await response.json()
    } catch (error) {
      console.error('Error starting analysis:', error)
      throw error
    }
  },

  /**
   * Stop analyzing a video
   */
  async stopAnalysis(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/${videoId}/stop`, {
        method: 'POST'
      })
      if (!response.ok) throw new Error('Failed to stop analysis')
      return await response.json()
    } catch (error) {
      console.error('Error stopping analysis:', error)
      throw error
    }
  },

  /**
   * Get analysis results for a video
   */
  async getAnalysisResults(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/${videoId}/results`)
      if (!response.ok) throw new Error('Failed to fetch results')
      return await response.json()
    } catch (error) {
      console.error('Error fetching results:', error)
      return null
    }
  },

  /**
   * Get video status
   */
  async getVideoStatus(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/${videoId}/status`)
      if (!response.ok) throw new Error('Failed to fetch status')
      return await response.json()
    } catch (error) {
      console.error('Error fetching status:', error)
      return { status: 'unknown', progress: 0 }
    }
  }
}

export default videoService
