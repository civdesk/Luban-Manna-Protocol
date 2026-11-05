import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

interface Nanorobot3DViewerProps {
  torque: number;
  omega: number;
  vx: number;
  shearStress: number;
  isOverLimit: boolean;
}

export const Nanorobot3DViewer: React.FC<Nanorobot3DViewerProps> = ({
  torque,
  omega,
  vx,
  shearStress,
  isOverLimit,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const robotRef = useRef<THREE.Group | null>(null);
  const helixRef = useRef<THREE.Group | null>(null);
  const particlesRef = useRef<THREE.Points | null>(null);
  const animRef = useRef<number>(0);
  const stateRef = useRef({ omega, vx, shearStress, isOverLimit });

  useEffect(() => {
    stateRef.current = { omega, vx, shearStress, isOverLimit };
  }, [omega, vx, shearStress, isOverLimit]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x060c18);

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    camera.position.set(0, 6, 12);
    camera.lookAt(0, 0, 0);

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0x406080, 1.8);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x60a5fa, 2.5);
    dirLight1.position.set(5, 10, 8);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xf59e0b, 1.2);
    dirLight2.position.set(-6, -4, -6);
    scene.add(dirLight2);

    // Subtle fluid grid / boundary plane
    const gridHelper = new THREE.GridHelper(24, 24, 0x1e3a8a, 0x0f172a);
    gridHelper.position.y = -3.5;
    scene.add(gridHelper);

    // Group for robot
    const robotGroup = new THREE.Group();
    robotRef.current = robotGroup;
    scene.add(robotGroup);

    // 1. Nanorobot Body: Ellipsoid (Chassis: 5 um length, 2.4 um dia)
    // Three.js Sphere geometry scaled to ellipsoid
    const bodyGeometry = new THREE.SphereGeometry(1.2, 32, 24);
    bodyGeometry.scale(2.1, 1.0, 1.0); // 2.5 x 1.2 x 1.2 equivalent

    const bodyMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      metalness: 0.2,
      roughness: 0.25,
      transparent: true,
      opacity: 0.88,
      transmission: 0.4,
      ior: 1.33, // water refractive index
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
    });
    const bodyMesh = new THREE.Mesh(bodyGeometry, bodyMaterial);
    robotGroup.add(bodyMesh);

    // Internal biological core / mass center
    const coreGeometry = new THREE.SphereGeometry(0.45, 16, 16);
    const coreMaterial = new THREE.MeshStandardMaterial({
      color: 0x0284c7,
      emissive: 0x0369a1,
      emissiveIntensity: 0.6,
      roughness: 0.4,
    });
    const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
    robotGroup.add(coreMesh);

    // 2. Flagellar Motor interface & Helical Rotor
    const motorGroup = new THREE.Group();
    motorGroup.position.set(-2.5, 0, 0); // connected at tail
    robotGroup.add(motorGroup);

    // Motor hinge ring
    const ringGeo = new THREE.CylinderGeometry(0.35, 0.35, 0.25, 24);
    ringGeo.rotateZ(Math.PI / 2);
    const ringMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8,
      metalness: 0.8,
      roughness: 0.3,
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    motorGroup.add(ringMesh);

    // 5-Segment Rigid Helical Flagellum (Matching MJCF geom capsule segments)
    const helixGroup = new THREE.Group();
    helixRef.current = helixGroup;
    motorGroup.add(helixGroup);

    // Segments:
    // fromto 0,0,0 -> -0.6, 0.5, 0
    // -> -1.2, 0, 0.5
    // -> -1.8, -0.5, 0
    // -> -2.4, 0, -0.5
    // -> -3.0, 0.5, 0
    const points = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(-0.6, 0.5, 0),
      new THREE.Vector3(-1.2, 0, 0.5),
      new THREE.Vector3(-1.8, -0.5, 0),
      new THREE.Vector3(-2.4, 0, -0.5),
      new THREE.Vector3(-3.0, 0.5, 0),
    ];

    const curve = new THREE.CatmullRomCurve3(points);
    const tubeGeo = new THREE.TubeGeometry(curve, 48, 0.1, 12, false);
    const helixMat = new THREE.MeshStandardMaterial({
      color: 0xfacc15,
      metalness: 0.6,
      roughness: 0.3,
      emissive: 0x854d0e,
      emissiveIntensity: 0.4,
    });
    const helixMesh = new THREE.Mesh(tubeGeo, helixMat);
    helixGroup.add(helixMesh);

    // Enzyme active catalytic outer layer halo
    const haloGeo = new THREE.SphereGeometry(1.28, 24, 18);
    haloGeo.scale(2.15, 1.06, 1.06);
    const haloMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: true,
      transparent: true,
      opacity: 0.18,
    });
    const haloMesh = new THREE.Mesh(haloGeo, haloMat);
    robotGroup.add(haloMesh);

    // 3. Fluid Stokes Streamlines / Microscopic particles
    const particleCount = 180;
    const particleGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const particleSpeeds = new Float32Array(particleCount);

    for (let i = 0; i < particleCount; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 18;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 8;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
      particleSpeeds[i] = 0.5 + Math.random() * 1.5;
    }
    particleGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const particleMat = new THREE.PointsMaterial({
      color: 0x38bdf8,
      size: 0.08,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
    });
    const particleSystem = new THREE.Points(particleGeo, particleMat);
    scene.add(particleSystem);
    particlesRef.current = particleSystem;

    // Drag / Interaction
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let rotX = 0.2;
    let rotY = -0.5;

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      const dx = e.clientX - prevMouseX;
      const dy = e.clientY - prevMouseY;
      rotY += dx * 0.008;
      rotX += dy * 0.008;
      rotX = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, rotX));
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    container.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    // Animation Loop
    let angle = 0;
    const animate = () => {
      animRef.current = requestAnimationFrame(animate);

      const cur = stateRef.current;
      // Helix rotation speed
      const rotSpeed = (cur.omega !== 0 ? cur.omega * 0.05 : torque * 0.08);
      angle += rotSpeed;

      if (helixRef.current) {
        helixRef.current.rotation.x = angle;
      }

      // Robot gentle hover & forward drift simulation
      if (robotRef.current) {
        robotRef.current.rotation.y = rotY;
        robotRef.current.rotation.x = rotX;
        // subtle Stokesian micro-vibration
        robotRef.current.position.y = Math.sin(Date.now() * 0.003) * 0.15;
        robotRef.current.position.z = Math.cos(Date.now() * 0.002) * 0.08;
      }

      // Update particle stream
      if (particlesRef.current) {
        const posAttr = particlesRef.current.geometry.attributes.position as THREE.BufferAttribute;
        const posArray = posAttr.array as Float32Array;
        const driftSpeed = 0.04 + Math.abs(cur.vx) * 0.01;

        for (let i = 0; i < particleCount; i++) {
          // flow past nanorobot towards tail (-x)
          posArray[i * 3] -= driftSpeed * particleSpeeds[i];
          if (posArray[i * 3] < -9) {
            posArray[i * 3] = 9;
            posArray[i * 3 + 1] = (Math.random() - 0.5) * 8;
            posArray[i * 3 + 2] = (Math.random() - 0.5) * 10;
          }
        }
        posAttr.needsUpdate = true;
      }

      // Halo color indicates shear warning
      if (cur.isOverLimit) {
        haloMat.color.setHex(0xef4444);
        bodyMaterial.emissive = new THREE.Color(0xef4444);
        bodyMaterial.emissiveIntensity = 0.35;
      } else {
        haloMat.color.setHex(0x38bdf8);
        bodyMaterial.emissive = new THREE.Color(0x0284c7);
        bodyMaterial.emissiveIntensity = 0.08;
      }

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animRef.current);
      container.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('resize', handleResize);
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  return (
    <div className="relative w-full h-full min-h-[380px] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 shadow-2xl">
      <div ref={mountRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
      
      {/* 3D Overlays */}
      <div className="absolute top-4 left-4 pointer-events-none bg-slate-900/80 backdrop-blur-md px-3 py-2 rounded-lg border border-slate-700/60 text-xs font-mono text-slate-300 space-y-1">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
          <span className="text-white font-semibold">Stokesian 3D MuJoCo Kinematics</span>
        </div>
        <div className="text-slate-400">Scale: 1 unit = 1 μm | Body Length: 5.0 μm</div>
        <div className="text-slate-400">Rotor: 5-Seg Helical Flagellum (R = 0.5 μm)</div>
      </div>

      <div className="absolute bottom-4 left-4 pointer-events-none flex gap-2">
        <span className="px-2.5 py-1 text-[11px] font-mono rounded bg-slate-900/90 border border-slate-700/70 text-slate-300">
          拖拽旋转视角 (Drag to Rotate 3D View)
        </span>
      </div>

      {isOverLimit && (
        <div className="absolute top-4 right-4 pointer-events-none bg-red-950/90 border border-red-500/80 px-3 py-1.5 rounded-lg text-xs font-mono text-red-200 flex items-center gap-2 animate-bounce">
          <span className="w-2 h-2 rounded-full bg-red-500"></span>
          <span>警告：流体剪切应力超标 ({shearStress.toFixed(3)} &gt; 0.05 N/m²)</span>
        </div>
      )}
    </div>
  );
};
