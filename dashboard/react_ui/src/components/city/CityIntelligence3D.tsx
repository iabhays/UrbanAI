import React, { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { clsx } from '../../utils/clsx'

interface CityBlock {
  id: string
  position: { x: number; z: number }
  size: { width: number; depth: number }
  height: number
  riskLevel: number
  population: number
  type: 'residential' | 'commercial' | 'industrial' | 'government'
}

interface CameraPosition {
  id: string
  position: { x: number; y: number; z: number }
  rotation: { x: number; y: number; z: number }
  status: 'active' | 'inactive' | 'maintenance'
  coverage: number
}

interface ThreatEntity {
  id: string
  position: { x: number; y: number; z: number }
  velocity: { x: number; z: number }
  type: 'vehicle' | 'person' | 'drone' | 'unknown'
  threatLevel: number
  trajectory: Array<{ x: number; z: number; timestamp: number }>
}

interface IncidentHotspot {
  id: string
  position: { x: number; z: number }
  radius: number
  intensity: number
  type: 'theft' | 'assault' | 'vandalism' | 'suspicious' | 'emergency'
  timestamp: number
}

interface CityIntelligence3DProps {
  className?: string
  onBlockClick?: (block: CityBlock) => void
  onCameraClick?: (camera: CameraPosition) => void
  onThreatClick?: (threat: ThreatEntity) => void
}

const CityIntelligence3D: React.FC<CityIntelligence3DProps> = ({
  className,
  onBlockClick,
  onCameraClick,
  onThreatClick
}) => {
  const mountRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene>()
  const rendererRef = useRef<THREE.WebGLRenderer>()
  const cameraRef = useRef<THREE.PerspectiveCamera>()
  const controlsRef = useRef<OrbitControls>()
  const frameRef = useRef<number>()
  
  const [selectedEntity, setSelectedEntity] = useState<string | null>(null)
  const [showRiskOverlay, setShowRiskOverlay] = useState(true)
  const [showCameras, setShowCameras] = useState(true)
  const [showThreats, setShowThreats] = useState(true)

  // Mock city data
  const cityBlocks: CityBlock[] = [
    { id: 'block-1', position: { x: -50, z: -50 }, size: { width: 30, depth: 30 }, height: 15, riskLevel: 0.2, population: 500, type: 'residential' },
    { id: 'block-2', position: { x: -10, z: -50 }, size: { width: 40, depth: 30 }, height: 25, riskLevel: 0.4, population: 800, type: 'commercial' },
    { id: 'block-3', position: { x: 40, z: -50 }, size: { width: 35, depth: 30 }, height: 20, riskLevel: 0.1, population: 300, type: 'government' },
    { id: 'block-4', position: { x: -50, z: -10 }, size: { width: 30, depth: 40 }, height: 18, riskLevel: 0.6, population: 600, type: 'industrial' },
    { id: 'block-5', position: { x: -10, z: -10 }, size: { width: 40, depth: 40 }, height: 30, riskLevel: 0.3, population: 1200, type: 'commercial' },
    { id: 'block-6', position: { x: 40, z: -10 }, size: { width: 35, depth: 40 }, height: 22, riskLevel: 0.2, population: 450, type: 'residential' },
    { id: 'block-7', position: { x: -50, z: 40 }, size: { width: 30, depth: 30 }, height: 12, riskLevel: 0.8, population: 200, type: 'industrial' },
    { id: 'block-8', position: { x: -10, z: 40 }, size: { width: 40, depth: 30 }, height: 28, riskLevel: 0.5, population: 900, type: 'commercial' },
    { id: 'block-9', position: { x: 40, z: 40 }, size: { width: 35, depth: 30 }, height: 16, riskLevel: 0.1, population: 350, type: 'residential' }
  ]

  const cameraPositions: CameraPosition[] = [
    { id: 'cam-1', position: { x: -35, y: 20, z: -35 }, rotation: { x: 0, y: 0.5, z: 0 }, status: 'active', coverage: 0.9 },
    { id: 'cam-2', position: { x: 0, y: 25, z: -35 }, rotation: { x: 0, y: 0, z: 0 }, status: 'active', coverage: 0.85 },
    { id: 'cam-3', position: { x: 35, y: 18, z: -35 }, rotation: { x: 0, y: -0.5, z: 0 }, status: 'maintenance', coverage: 0.0 },
    { id: 'cam-4', position: { x: -35, y: 22, z: 0 }, rotation: { x: 0, y: 1, z: 0 }, status: 'active', coverage: 0.95 },
    { id: 'cam-5', position: { x: 35, y: 20, z: 0 }, rotation: { x: 0, y: -1, z: 0 }, status: 'active', coverage: 0.88 },
    { id: 'cam-6', position: { x: 0, y: 24, z: 35 }, rotation: { x: 0, y: 3.14, z: 0 }, status: 'active', coverage: 0.92 }
  ]

  const threatEntities: ThreatEntity[] = [
    { id: 'threat-1', position: { x: -20, y: 1, z: -20 }, velocity: { x: 0.5, z: 0.3 }, type: 'vehicle', threatLevel: 0.3, trajectory: [] },
    { id: 'threat-2', position: { x: 15, y: 1, z: 10 }, velocity: { x: -0.2, z: 0.4 }, type: 'person', threatLevel: 0.7, trajectory: [] },
    { id: 'threat-3', position: { x: -40, y: 15, z: 25 }, velocity: { x: 0.3, z: -0.1 }, type: 'drone', threatLevel: 0.9, trajectory: [] }
  ]

  const incidentHotspots: IncidentHotspot[] = [
    { id: 'incident-1', position: { x: 10, z: 15 }, radius: 15, intensity: 0.8, type: 'suspicious', timestamp: Date.now() - 300000 },
    { id: 'incident-2', position: { x: -25, z: -30 }, radius: 10, intensity: 0.6, type: 'theft', timestamp: Date.now() - 600000 },
    { id: 'incident-3', position: { x: 30, z: 20 }, radius: 20, intensity: 0.9, type: 'emergency', timestamp: Date.now() - 120000 }
  ]

  useEffect(() => {
    if (!mountRef.current) return

    // Scene setup
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0a0a)
    scene.fog = new THREE.Fog(0x0a0a0a, 50, 200)
    sceneRef.current = scene

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    )
    camera.position.set(0, 80, 100)
    cameraRef.current = camera

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight)
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    mountRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05
    controls.maxPolarAngle = Math.PI / 2
    controls.minDistance = 50
    controls.maxDistance = 200
    controlsRef.current = controls

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6)
    scene.add(ambientLight)

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
    directionalLight.position.set(50, 100, 50)
    directionalLight.castShadow = true
    directionalLight.shadow.camera.left = -100
    directionalLight.shadow.camera.right = 100
    directionalLight.shadow.camera.top = 100
    directionalLight.shadow.camera.bottom = -100
    scene.add(directionalLight)

    // Ground
    const groundGeometry = new THREE.PlaneGeometry(200, 200)
    const groundMaterial = new THREE.MeshLambertMaterial({ color: 0x1a1a1a })
    const ground = new THREE.Mesh(groundGeometry, groundMaterial)
    ground.rotation.x = -Math.PI / 2
    ground.receiveShadow = true
    scene.add(ground)

    // Grid helper
    const gridHelper = new THREE.GridHelper(200, 20, 0x444444, 0x222222)
    scene.add(gridHelper)

    // Create city blocks
    cityBlocks.forEach(block => {
      const geometry = new THREE.BoxGeometry(block.size.width, block.height, block.size.depth)
      const material = new THREE.MeshLambertMaterial({
        color: new THREE.Color().setHSL(0.6 - block.riskLevel * 0.4, 0.7, 0.3 + block.riskLevel * 0.3)
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set(block.position.x, block.height / 2, block.position.z)
      mesh.castShadow = true
      mesh.receiveShadow = true
      mesh.userData = { type: 'block', data: block }
      scene.add(mesh)
    })

    // Create cameras
    cameraPositions.forEach(camera => {
      const geometry = new THREE.BoxGeometry(2, 2, 3)
      const material = new THREE.MeshLambertMaterial({
        color: camera.status === 'active' ? 0x00ff00 : camera.status === 'maintenance' ? 0xffff00 : 0xff0000
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set(camera.position.x, camera.position.y, camera.position.z)
      mesh.rotation.set(camera.rotation.x, camera.rotation.y, camera.rotation.z)
      mesh.userData = { type: 'camera', data: camera }
      scene.add(mesh)

      // Camera coverage cone
      if (camera.status === 'active' && showCameras) {
        const coneGeometry = new THREE.ConeGeometry(20, 30, 8)
        const coneMaterial = new THREE.MeshBasicMaterial({
          color: 0x00ff00,
          transparent: true,
          opacity: 0.1
        })
        const cone = new THREE.Mesh(coneGeometry, coneMaterial)
        cone.position.set(camera.position.x, camera.position.y - 15, camera.position.z)
        cone.rotation.set(camera.rotation.x + Math.PI / 2, camera.rotation.y, camera.rotation.z)
        scene.add(cone)
      }
    })

    // Create threat entities
    threatEntities.forEach(threat => {
      const geometry = new THREE.SphereGeometry(2, 8, 8)
      const material = new THREE.MeshBasicMaterial({
        color: threat.threatLevel > 0.7 ? 0xff0000 : threat.threatLevel > 0.4 ? 0xffff00 : 0x00ff00,
        transparent: true,
        opacity: 0.8
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set(threat.position.x, threat.position.y, threat.position.z)
      mesh.userData = { type: 'threat', data: threat }
      scene.add(mesh)
    })

    // Create incident hotspots
    incidentHotspots.forEach(incident => {
      const geometry = new THREE.CylinderGeometry(incident.radius, incident.radius, 0.5, 16)
      const material = new THREE.MeshBasicMaterial({
        color: incident.intensity > 0.8 ? 0xff0000 : incident.intensity > 0.5 ? 0xff6600 : 0xffff00,
        transparent: true,
        opacity: 0.3
      })
      const mesh = new THREE.Mesh(geometry, material)
      mesh.position.set(incident.position.x, 0.25, incident.position.z)
      mesh.userData = { type: 'incident', data: incident }
      scene.add(mesh)
    })

    // Raycaster for mouse interaction
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()

    const handleClick = (event: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

      raycaster.setFromCamera(mouse, camera)
      const intersects = raycaster.intersectObjects(scene.children)

      if (intersects.length > 0) {
        const object = intersects[0].object
        const userData = object.userData

        if (userData.type === 'block' && onBlockClick) {
          onBlockClick(userData.data)
        } else if (userData.type === 'camera' && onCameraClick) {
          onCameraClick(userData.data)
        } else if (userData.type === 'threat' && onThreatClick) {
          onThreatClick(userData.data)
        }

        setSelectedEntity(userData.data.id)
      }
    }

    renderer.domElement.addEventListener('click', handleClick)

    // Animation loop
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate)

      // Update threat positions
      scene.children.forEach(child => {
        if (child.userData.type === 'threat') {
          const threat = child.userData.data as ThreatEntity
          child.position.x += threat.velocity.x
          child.position.z += threat.velocity.z

          // Bounce off boundaries
          if (Math.abs(child.position.x) > 80) threat.velocity.x *= -1
          if (Math.abs(child.position.z) > 80) threat.velocity.z *= -1
        }
      })

      // Pulse incident hotspots
      scene.children.forEach(child => {
        if (child.userData.type === 'incident' && child instanceof THREE.Mesh) {
          const incident = child.userData.data as IncidentHotspot
          const age = Date.now() - incident.timestamp
          const opacity = Math.max(0.1, 0.3 * Math.sin(age * 0.001))
          const material = child.material as THREE.MeshBasicMaterial
          material.opacity = opacity
        }
      })

      controls.update()
      renderer.render(scene, camera)
    }

    animate()

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return
      camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight)
    }

    window.addEventListener('resize', handleResize)

    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
      }
      window.removeEventListener('resize', handleResize)
      renderer.domElement.removeEventListener('click', handleClick)
      mountRef.current?.removeChild(renderer.domElement)
      renderer.dispose()
    }
  }, [onBlockClick, onCameraClick, onThreatClick, showCameras])

  return (
    <div className={clsx('relative h-full bg-gray-900 rounded-lg border border-cyan-500/20 overflow-hidden', className)}>
      {/* Header Controls */}
      <div className="absolute top-0 left-0 right-0 z-10 bg-gray-800/90 backdrop-blur-sm border-b border-cyan-500/20 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse" />
            <span className="text-cyan-400 text-sm font-mono uppercase tracking-wider">
              3D City Intelligence
            </span>
          </div>
          
          <div className="flex items-center space-x-4">
            <label className="flex items-center space-x-2 text-xs">
              <input
                type="checkbox"
                checked={showRiskOverlay}
                onChange={(e) => setShowRiskOverlay(e.target.checked)}
                className="rounded"
              />
              <span className="text-gray-400">Risk Overlay</span>
            </label>
            
            <label className="flex items-center space-x-2 text-xs">
              <input
                type="checkbox"
                checked={showCameras}
                onChange={(e) => setShowCameras(e.target.checked)}
                className="rounded"
              />
              <span className="text-gray-400">Cameras</span>
            </label>
            
            <label className="flex items-center space-x-2 text-xs">
              <input
                type="checkbox"
                checked={showThreats}
                onChange={(e) => setShowThreats(e.target.checked)}
                className="rounded"
              />
              <span className="text-gray-400">Threats</span>
            </label>
          </div>
        </div>
      </div>

      {/* 3D Scene Container */}
      <div ref={mountRef} className="w-full h-full pt-12" />

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-gray-800/90 backdrop-blur-sm border border-cyan-500/30 rounded-lg p-3">
        <div className="space-y-2 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded" />
            <span className="text-gray-300">Active Camera</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-400 rounded" />
            <span className="text-gray-300">Maintenance</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-400 rounded-full" />
            <span className="text-gray-300">High Threat</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-400 rounded-full" />
            <span className="text-gray-300">Medium Threat</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded-full" />
            <span className="text-gray-300">Low Threat</span>
          </div>
        </div>
      </div>

      {/* Entity Info Panel */}
      {selectedEntity && (
        <div className="absolute top-16 right-4 bg-gray-800/90 backdrop-blur-sm border border-cyan-500/30 rounded-lg p-3 w-64">
          <div className="text-xs font-mono text-cyan-400 mb-2">
            Selected Entity: {selectedEntity}
          </div>
          <div className="text-xs text-gray-300">
            Click on entities for detailed analytics
          </div>
        </div>
      )}
    </div>
  )
}

export default CityIntelligence3D
