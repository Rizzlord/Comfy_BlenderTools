import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Load Three.js and OrbitControls from esm.sh
const THREE_URL = "https://esm.sh/three@0.160.0";
const GLTFLOADER_URL = "https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";
const ORBITCONTROLS_URL = "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";

let THREE = null;
let GLTFLoader = null;
let OrbitControls = null;
const PREVIEW_TEXTURE_SIZE = 2048;
const CHECKER_GRID_COUNT = 8;
const CHECKER_REPEAT = 12;
const STRETCH_COLOR_STOPS = [
    [1.0, [20, 136, 173]],
    [1.02, [24, 157, 181]],
    [1.08, [60, 190, 150]],
    [1.15, [145, 208, 94]],
    [1.25, [236, 196, 66]],
    [1.5, [235, 146, 58]],
    [2.0, [232, 93, 62]],
    [4.0, [194, 60, 117]],
    [8.0, [244, 244, 244]],
];

function sampleColorStops(stops, value) {
    const clampedValue = Math.max(value, stops[0][0]);
    for (let index = 1; index < stops.length; index += 1) {
        const [stopValue, stopColor] = stops[index];
        const [prevValue, prevColor] = stops[index - 1];
        if (clampedValue <= stopValue) {
            const local = (clampedValue - prevValue) / Math.max(stopValue - prevValue, 1e-6);
            return [
                prevColor[0] + (stopColor[0] - prevColor[0]) * local,
                prevColor[1] + (stopColor[1] - prevColor[1]) * local,
                prevColor[2] + (stopColor[2] - prevColor[2]) * local,
            ];
        }
    }
    return stops[stops.length - 1][1];
}

function stretchColor(stretch) {
    return sampleColorStops(
        STRETCH_COLOR_STOPS,
        Math.max(Number.isFinite(stretch) ? stretch : 1, 1),
    );
}

function createSequentialIndex(count) {
    const index = new Uint32Array(count);
    for (let i = 0; i < count; i += 1) {
        index[i] = i;
    }
    return index;
}

function geometryIndices(geometry) {
    if (!geometry?.attributes?.position) return null;
    if (geometry.index?.array?.length) {
        return geometry.index.array;
    }
    return createSequentialIndex(geometry.attributes.position.count);
}

function canUseAtlasMaterialForGeometry(geometry) {
    const uvAttribute = geometry?.attributes?.uv;
    if (!uvAttribute || uvAttribute.count === 0) {
        return false;
    }

    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (let index = 0; index < uvAttribute.count; index += 1) {
        const x = uvAttribute.getX(index);
        const y = uvAttribute.getY(index);
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    }

    return minX >= -0.05 && minY >= -0.05 && maxX <= 1.05 && maxY <= 1.05;
}

function computeStretchPerFace(geometry) {
    const positionAttribute = geometry?.attributes?.position;
    const uvAttribute = geometry?.attributes?.uv;
    const indices = geometryIndices(geometry);
    if (!positionAttribute || !uvAttribute || !indices || indices.length % 3 !== 0) {
        return null;
    }

    const positions = positionAttribute.array;
    const uvs = uvAttribute.array;
    const faceCount = indices.length / 3;
    const stretches = new Float32Array(faceCount);

    for (let faceIndex = 0; faceIndex < faceCount; faceIndex += 1) {
        const offset = faceIndex * 3;
        const ia = indices[offset];
        const ib = indices[offset + 1];
        const ic = indices[offset + 2];

        const a3 = ia * 3;
        const b3 = ib * 3;
        const c3 = ic * 3;
        const a2 = ia * 2;
        const b2 = ib * 2;
        const c2 = ic * 2;

        const ax = positions[a3];
        const ay = positions[a3 + 1];
        const az = positions[a3 + 2];
        const bx = positions[b3];
        const by = positions[b3 + 1];
        const bz = positions[b3 + 2];
        const cx = positions[c3];
        const cy = positions[c3 + 1];
        const cz = positions[c3 + 2];

        const e1x = bx - ax;
        const e1y = by - ay;
        const e1z = bz - az;
        const e2x = cx - ax;
        const e2y = cy - ay;
        const e2z = cz - az;

        const nx = e1y * e2z - e1z * e2y;
        const ny = e1z * e2x - e1x * e2z;
        const nz = e1x * e2y - e1y * e2x;

        const e1Len = Math.hypot(e1x, e1y, e1z);
        if (e1Len <= 1e-8) {
            stretches[faceIndex] = 8.0;
            continue;
        }

        const xAxisX = e1x / e1Len;
        const xAxisY = e1y / e1Len;
        const xAxisZ = e1z / e1Len;

        const yRawX = ny * xAxisZ - nz * xAxisY;
        const yRawY = nz * xAxisX - nx * xAxisZ;
        const yRawZ = nx * xAxisY - ny * xAxisX;
        const yNorm = Math.hypot(yRawX, yRawY, yRawZ);
        if (yNorm <= 1e-8) {
            stretches[faceIndex] = 8.0;
            continue;
        }

        const yAxisX = yRawX / yNorm;
        const yAxisY = yRawY / yNorm;
        const yAxisZ = yRawZ / yNorm;

        const x1 = e1Len;
        const x2 = e2x * xAxisX + e2y * xAxisY + e2z * xAxisZ;
        const y2 = e2x * yAxisX + e2y * yAxisY + e2z * yAxisZ;
        const detP = x1 * y2;
        if (Math.abs(detP) <= 1e-10) {
            stretches[faceIndex] = 8.0;
            continue;
        }

        const du1 = uvs[b2] - uvs[a2];
        const du2 = uvs[c2] - uvs[a2];
        const dv1 = uvs[b2 + 1] - uvs[a2 + 1];
        const dv2 = uvs[c2 + 1] - uvs[a2 + 1];

        const invP00 = 1.0 / x1;
        const invP01 = -x2 / detP;
        const invP10 = 0.0;
        const invP11 = 1.0 / y2;

        const j00 = du1 * invP00 + du2 * invP10;
        const j01 = du1 * invP01 + du2 * invP11;
        const j10 = dv1 * invP00 + dv2 * invP10;
        const j11 = dv1 * invP01 + dv2 * invP11;

        const s00 = j00 * j00 + j10 * j10;
        const s01 = j00 * j01 + j10 * j11;
        const s11 = j01 * j01 + j11 * j11;
        const trace = s00 + s11;
        const det = s00 * s11 - s01 * s01;
        const disc = Math.sqrt(Math.max(trace * trace * 0.25 - det, 0.0));
        const lambdaMax = Math.max(trace * 0.5 + disc, 0.0);
        const lambdaMin = Math.max(trace * 0.5 - disc, 0.0);
        const sigmaMax = Math.sqrt(lambdaMax);
        const sigmaMin = Math.sqrt(lambdaMin);

        stretches[faceIndex] = sigmaMin > 1e-8 ? sigmaMax / sigmaMin : 8.0;
    }

    return stretches;
}

function createStretchTextureForGeometry(geometry) {
    const uvAttribute = geometry?.attributes?.uv;
    const indices = geometryIndices(geometry);
    if (!uvAttribute || !indices || indices.length % 3 !== 0 || !canUseAtlasMaterialForGeometry(geometry)) {
        return null;
    }

    const stretchPerFace = computeStretchPerFace(geometry);
    if (!stretchPerFace) {
        return null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = PREVIEW_TEXTURE_SIZE;
    canvas.height = PREVIEW_TEXTURE_SIZE;
    const context = canvas.getContext("2d");
    context.fillStyle = "#081118";
    context.fillRect(0, 0, canvas.width, canvas.height);

    const gridStep = canvas.width / 8;
    context.strokeStyle = "rgba(127, 215, 234, 0.08)";
    context.lineWidth = 2;
    for (let step = 0; step <= 8; step += 1) {
        context.beginPath();
        context.moveTo(step * gridStep, 0);
        context.lineTo(step * gridStep, canvas.height);
        context.stroke();
        context.beginPath();
        context.moveTo(0, step * gridStep);
        context.lineTo(canvas.width, step * gridStep);
        context.stroke();
    }

    const project = (index) => [
        uvAttribute.getX(index) * canvas.width,
        (1 - uvAttribute.getY(index)) * canvas.height,
    ];

    for (let faceIndex = 0; faceIndex < stretchPerFace.length; faceIndex += 1) {
        const offset = faceIndex * 3;
        const ia = indices[offset];
        const ib = indices[offset + 1];
        const ic = indices[offset + 2];
        const p1 = project(ia);
        const p2 = project(ib);
        const p3 = project(ic);
        const color = stretchColor(stretchPerFace[faceIndex]);

        context.beginPath();
        context.moveTo(p1[0], p1[1]);
        context.lineTo(p2[0], p2[1]);
        context.lineTo(p3[0], p3[1]);
        context.closePath();
        context.fillStyle = `rgb(${Math.round(color[0])}, ${Math.round(color[1])}, ${Math.round(color[2])})`;
        context.fill();
    }

    context.strokeStyle = "rgba(245, 242, 232, 0.32)";
    context.lineWidth = 1.1;
    for (let faceIndex = 0; faceIndex < stretchPerFace.length; faceIndex += 1) {
        const offset = faceIndex * 3;
        const ia = indices[offset];
        const ib = indices[offset + 1];
        const ic = indices[offset + 2];
        const p1 = project(ia);
        const p2 = project(ib);
        const p3 = project(ic);
        context.beginPath();
        context.moveTo(p1[0], p1[1]);
        context.lineTo(p2[0], p2[1]);
        context.lineTo(p3[0], p3[1]);
        context.closePath();
        context.stroke();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.anisotropy = 8;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
}

function createCheckerMapTexture() {
    if (!THREE) {
        return null;
    }

    const canvas = document.createElement("canvas");
    canvas.width = PREVIEW_TEXTURE_SIZE;
    canvas.height = PREVIEW_TEXTURE_SIZE;
    const context = canvas.getContext("2d");
    const cellSize = canvas.width / CHECKER_GRID_COUNT;

    context.fillStyle = "#f7f4eb";
    context.fillRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < CHECKER_GRID_COUNT; y += 1) {
        for (let x = 0; x < CHECKER_GRID_COUNT; x += 1) {
            const left = x * cellSize;
            const top = y * cellSize;
            context.fillStyle = (x + y) % 2 === 0 ? "#11151b" : "#f7f4eb";
            context.fillRect(left, top, cellSize, cellSize);
        }
    }

    context.strokeStyle = "rgba(10, 168, 201, 0.58)";
    context.lineWidth = 2;
    for (let gridIndex = 0; gridIndex <= CHECKER_GRID_COUNT; gridIndex += 1) {
        const offset = gridIndex * cellSize;

        context.beginPath();
        context.moveTo(offset, 0);
        context.lineTo(offset, canvas.height);
        context.stroke();

        context.beginPath();
        context.moveTo(0, offset);
        context.lineTo(canvas.width, offset);
        context.stroke();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(CHECKER_REPEAT, CHECKER_REPEAT);
    texture.anisotropy = 8;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
}

function buildSeamGeometryForMesh(mesh, posTol = 1e-6, uvTol = 1e-6) {
    const geometry = mesh?.geometry;
    const positionAttribute = geometry?.attributes?.position;
    const uvAttribute = geometry?.attributes?.uv;
    const indices = geometryIndices(geometry);
    if (!positionAttribute || !uvAttribute || !indices || indices.length % 3 !== 0) {
        return null;
    }

    const decimals = Math.max(0, Math.round(-Math.log10(posTol)));
    const weldedLookup = new Map();
    const weldedIndices = new Int32Array(positionAttribute.count);
    let weldedCount = 0;

    for (let index = 0; index < positionAttribute.count; index += 1) {
        const key = [
            positionAttribute.getX(index).toFixed(decimals),
            positionAttribute.getY(index).toFixed(decimals),
            positionAttribute.getZ(index).toFixed(decimals),
        ].join("|");
        let weldedIndex = weldedLookup.get(key);
        if (weldedIndex === undefined) {
            weldedIndex = weldedCount;
            weldedLookup.set(key, weldedIndex);
            weldedCount += 1;
        }
        weldedIndices[index] = weldedIndex;
    }

    const edgeRecords = new Map();
    const faceCount = indices.length / 3;
    const localEdges = [[0, 1], [1, 2], [2, 0]];

    for (let faceIndex = 0; faceIndex < faceCount; faceIndex += 1) {
        const offset = faceIndex * 3;
        const face = [indices[offset], indices[offset + 1], indices[offset + 2]];
        const weldedFace = [
            weldedIndices[face[0]],
            weldedIndices[face[1]],
            weldedIndices[face[2]],
        ];

        for (const [cornerA, cornerB] of localEdges) {
            const vertexA = face[cornerA];
            const vertexB = face[cornerB];
            const weldedA = weldedFace[cornerA];
            const weldedB = weldedFace[cornerB];

            let key;
            let uvA;
            let uvB;
            let posA;
            let posB;
            if (weldedA <= weldedB) {
                key = `${weldedA}|${weldedB}`;
                uvA = [uvAttribute.getX(vertexA), uvAttribute.getY(vertexA)];
                uvB = [uvAttribute.getX(vertexB), uvAttribute.getY(vertexB)];
                posA = [
                    positionAttribute.getX(vertexA),
                    positionAttribute.getY(vertexA),
                    positionAttribute.getZ(vertexA),
                ];
                posB = [
                    positionAttribute.getX(vertexB),
                    positionAttribute.getY(vertexB),
                    positionAttribute.getZ(vertexB),
                ];
            } else {
                key = `${weldedB}|${weldedA}`;
                uvA = [uvAttribute.getX(vertexB), uvAttribute.getY(vertexB)];
                uvB = [uvAttribute.getX(vertexA), uvAttribute.getY(vertexA)];
                posA = [
                    positionAttribute.getX(vertexB),
                    positionAttribute.getY(vertexB),
                    positionAttribute.getZ(vertexB),
                ];
                posB = [
                    positionAttribute.getX(vertexA),
                    positionAttribute.getY(vertexA),
                    positionAttribute.getZ(vertexA),
                ];
            }

            if (!edgeRecords.has(key)) {
                edgeRecords.set(key, []);
            }
            edgeRecords.get(key).push({ uvA, uvB, posA, posB });
        }
    }

    const seamPositions = [];
    for (const records of edgeRecords.values()) {
        const first = records[0];
        if (records.length === 1) {
            seamPositions.push(...first.posA, ...first.posB);
            continue;
        }

        let sameUv = true;
        for (let recordIndex = 1; recordIndex < records.length; recordIndex += 1) {
            const record = records[recordIndex];
            if (
                Math.abs(record.uvA[0] - first.uvA[0]) > uvTol ||
                Math.abs(record.uvA[1] - first.uvA[1]) > uvTol ||
                Math.abs(record.uvB[0] - first.uvB[0]) > uvTol ||
                Math.abs(record.uvB[1] - first.uvB[1]) > uvTol
            ) {
                sameUv = false;
                break;
            }
        }

        if (!sameUv) {
            seamPositions.push(...first.posA, ...first.posB);
        }
    }

    if (seamPositions.length === 0) {
        return null;
    }

    const seamGeometry = new THREE.BufferGeometry();
    seamGeometry.setAttribute(
        "position",
        new THREE.Float32BufferAttribute(seamPositions, 3),
    );
    return seamGeometry;
}

async function loadThree() {
    if (THREE && GLTFLoader && OrbitControls) return;
    try {
        THREE = await import(THREE_URL);
        const GLTFModule = await import(GLTFLOADER_URL);
        const OrbitModule = await import(ORBITCONTROLS_URL);
        GLTFLoader = GLTFModule.GLTFLoader;
        OrbitControls = OrbitModule.OrbitControls;
        console.log("[BlenderTools] Three.js dependencies loaded via esm.sh");
    } catch (e) {
        console.error("[BlenderTools] Failed to load Three.js dependencies", e);
    }
}

// Reuseable 3D Widget Factory
function create3DPreviewWidget(node, containerName = "3D Preview") {
    // --- 3D Widget State ---
    const widget = {
        type: "custom",
        name: containerName,
        options: { serialize: false },
        _value: null,
        _canvas: null,
        _scene: null,
        _camera: null,
        _renderer: null,
        _controls: null,
        _model: null,
        _lights: [],
        _animationId: null,
        _renderMode: 'original',
        _renderQuality: 1024, // Default resolution height
        _originalMaterials: new Map(),
        _stretchMaterials: new Map(),
        _seamOverlays: new Map(),
        _showSeams: false,
        _loadToken: 0,
        _sharedMaterials: {
            normal: null,
            wireframe: null,
            checkerUv: null,
            checkerFallback: null,
        },

        draw(ctx, node, widget_width, y, widget_height) { },
        computeSize(width) {
            if (!this._renderer || container.style.display === "none") return [width, 0];
            return [width, 200];
        }
    };

    // --- DOM Creation ---
    const container = document.createElement("div");
    Object.assign(container.style, {
        position: "absolute",
        display: "none",
        zIndex: "10",
        pointerEvents: "auto",
        backgroundColor: "#222",
        overflow: "hidden",
        border: "1px solid #333"
    });
    document.body.appendChild(container);

    // Toolbar
    const toolbar = document.createElement("div");
    Object.assign(toolbar.style, {
        position: "absolute",
        top: "5px",
        left: "5px",
        zIndex: "20",
        display: "flex",
        gap: "5px",
        alignItems: "center",
        padding: "2px",
        backgroundColor: "rgba(0,0,0,0.5)",
        borderRadius: "4px"
    });
    container.appendChild(toolbar);

    // Buttons helper
    const createButton = (text, mode) => {
        const btn = document.createElement("button");
        btn.textContent = text;
        Object.assign(btn.style, {
            fontSize: "10px",
            padding: "2px 6px",
            cursor: "pointer",
            backgroundColor: "#444",
            color: "#eee",
            border: "1px solid #555",
            borderRadius: "3px"
        });
        btn.onclick = () => setRenderMode(mode);
        return btn;
    };

    toolbar.appendChild(createButton("Original", "original"));
    toolbar.appendChild(createButton("Normal", "normal"));
    toolbar.appendChild(createButton("Wireframe", "wireframe"));
    toolbar.appendChild(createButton("Checker", "checker"));
    toolbar.appendChild(createButton("Stretch", "stretch"));

    // Resolution Dropdown
    const resSelect = document.createElement("select");
    Object.assign(resSelect.style, {
        fontSize: "10px",
        backgroundColor: "#444",
        color: "#eee",
        border: "1px solid #555",
        borderRadius: "3px",
        cursor: "pointer",
        marginLeft: "5px"
    });

    [256, 512, 1024].forEach(res => {
        const opt = document.createElement("option");
        opt.value = res;
        opt.textContent = res + "px";
        if (res === 1024) opt.selected = true;
        resSelect.appendChild(opt);
    });

    resSelect.onchange = (e) => {
        widget._renderQuality = parseInt(e.target.value);
        if (widget._renderer) {
            updatePosition(); // Force resize
        }
    };
    toolbar.appendChild(resSelect);

    // Brightness Slider
    const sliderContainer = document.createElement("div");
    sliderContainer.style.display = "flex";
    sliderContainer.style.alignItems = "center";
    sliderContainer.style.marginLeft = "10px";

    const sliderLabel = document.createElement("span");
    sliderLabel.textContent = "☀";
    sliderLabel.style.color = "#eee";
    sliderLabel.style.fontSize = "12px";
    sliderLabel.style.marginRight = "4px";

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "6";
    slider.step = "0.1";
    slider.value = "3.0";
    Object.assign(slider.style, {
        width: "60px",
        height: "10px",
        cursor: "pointer"
    });

    slider.oninput = (e) => {
        const intensity = parseFloat(e.target.value);
        widget._lights.forEach(light => {
            light.intensity = intensity;
        });
    };

    sliderContainer.appendChild(sliderLabel);
    sliderContainer.appendChild(slider);
    toolbar.appendChild(sliderContainer);

    const seamToggleContainer = document.createElement("label");
    Object.assign(seamToggleContainer.style, {
        display: "flex",
        alignItems: "center",
        gap: "4px",
        marginLeft: "10px",
        color: "#eee",
        fontSize: "10px",
        cursor: "pointer"
    });

    const seamToggle = document.createElement("input");
    seamToggle.type = "checkbox";
    seamToggle.checked = false;
    seamToggle.style.cursor = "pointer";
    seamToggle.oninput = (e) => {
        widget._showSeams = Boolean(e.target.checked);
        ensureSeamOverlays();
        updateSeamVisibility();
    };

    const seamToggleText = document.createElement("span");
    seamToggleText.textContent = "Show seams";
    seamToggleContainer.appendChild(seamToggle);
    seamToggleContainer.appendChild(seamToggleText);
    toolbar.appendChild(seamToggleContainer);

    // Stats Overlay
    const statsOverlay = document.createElement("div");
    Object.assign(statsOverlay.style, {
        position: "absolute",
        top: "5px",
        right: "5px",
        zIndex: "20",
        color: "rgba(255, 255, 255, 0.7)",
        fontSize: "11px",
        textAlign: "right",
        pointerEvents: "none",
        fontFamily: "monospace",
        textShadow: "1px 1px 2px black"
    });
    container.appendChild(statsOverlay);

    // --- Cleanup ---
    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        if (onRemoved) onRemoved.apply(this, arguments);
        if (container && container.parentNode) container.parentNode.removeChild(container);
        if (widget._animationId) cancelAnimationFrame(widget._animationId);
        if (widget._renderer) widget._renderer.dispose();
        if (widget._controls) widget._controls.dispose();
        releaseCurrentModel();
        disposeSharedMaterials();
    };

    // --- Render Logic ---
    let currentFilePath = "";
    let pendingFilePath = "";

    const setStatsLines = (lines, color = "rgba(255, 255, 255, 0.7)") => {
        statsOverlay.style.color = color;
        statsOverlay.innerHTML = (lines || [])
            .filter((line) => line)
            .map((line) => `<div>${line}</div>`)
            .join("");
    };

    const addMaterialToSet = (material, target) => {
        if (!material) return;
        if (Array.isArray(material)) {
            material.forEach((entry) => {
                if (entry) target.add(entry);
            });
            return;
        }
        target.add(material);
    };

    const disposeMaterial = (material, disposedTextures = new Set()) => {
        if (!material) return;
        if (Array.isArray(material)) {
            material.forEach((entry) => disposeMaterial(entry, disposedTextures));
            return;
        }

        for (const value of Object.values(material)) {
            if (value?.isTexture && !disposedTextures.has(value)) {
                disposedTextures.add(value);
                value.dispose();
            }
        }

        material.dispose?.();
    };

    const disposeGeneratedResources = () => {
        for (const material of widget._stretchMaterials.values()) {
            if (material?.map) {
                material.map.dispose();
            }
            material?.dispose?.();
        }
        widget._stretchMaterials.clear();

        for (const overlay of widget._seamOverlays.values()) {
            if (overlay.parent) {
                overlay.parent.remove(overlay);
            }
            overlay.geometry?.dispose?.();
            overlay.material?.dispose?.();
        }
        widget._seamOverlays.clear();
    };

    const disposeDetachedModel = (model) => {
        if (!model) return;

        const geometries = new Set();
        const materials = new Set();
        model.traverse((child) => {
            if (!child.isMesh) return;
            if (child.geometry) {
                geometries.add(child.geometry);
            }
            addMaterialToSet(child.material, materials);
        });

        for (const geometry of geometries) {
            geometry.dispose?.();
        }

        const disposedTextures = new Set();
        for (const material of materials) {
            disposeMaterial(material, disposedTextures);
        }
    };

    const releaseCurrentModel = () => {
        if (widget._model && widget._scene) {
            widget._scene.remove(widget._model);
        }

        const geometries = new Set();
        const originalMaterials = new Set();

        if (widget._model) {
            widget._model.traverse((child) => {
                if (!child.isMesh) return;
                if (child.geometry) {
                    geometries.add(child.geometry);
                }
            });
        }

        for (const material of widget._originalMaterials.values()) {
            addMaterialToSet(material, originalMaterials);
        }

        disposeGeneratedResources();

        for (const geometry of geometries) {
            geometry.dispose?.();
        }

        const disposedTextures = new Set();
        for (const material of originalMaterials) {
            disposeMaterial(material, disposedTextures);
        }

        widget._originalMaterials.clear();
        widget._model = null;
    };

    const disposeSharedMaterials = () => {
        const disposedTextures = new Set();
        disposeMaterial(widget._sharedMaterials.normal, disposedTextures);
        disposeMaterial(widget._sharedMaterials.wireframe, disposedTextures);
        disposeMaterial(widget._sharedMaterials.checkerUv, disposedTextures);
        disposeMaterial(widget._sharedMaterials.checkerFallback, disposedTextures);
        widget._sharedMaterials.normal = null;
        widget._sharedMaterials.wireframe = null;
        widget._sharedMaterials.checkerUv = null;
        widget._sharedMaterials.checkerFallback = null;
    };

    const getSharedMaterial = (mode) => {
        if (mode === "normal") {
            if (!widget._sharedMaterials.normal) {
                widget._sharedMaterials.normal = new THREE.MeshNormalMaterial();
            }
            return widget._sharedMaterials.normal;
        }

        if (mode === "wireframe") {
            if (!widget._sharedMaterials.wireframe) {
                widget._sharedMaterials.wireframe = new THREE.MeshBasicMaterial({
                    color: 0x00ff00,
                    wireframe: true,
                });
            }
            return widget._sharedMaterials.wireframe;
        }

        if (mode === "checker_uv") {
            if (!widget._sharedMaterials.checkerUv) {
                widget._sharedMaterials.checkerUv = new THREE.MeshStandardMaterial({
                    color: 0xffffff,
                    map: createCheckerMapTexture(),
                    roughness: 0.88,
                    metalness: 0.02,
                    side: THREE.DoubleSide,
                });
            }
            return widget._sharedMaterials.checkerUv;
        }

        if (mode === "checker_fallback") {
            if (!widget._sharedMaterials.checkerFallback) {
                widget._sharedMaterials.checkerFallback = new THREE.MeshStandardMaterial({
                    color: 0xc9bea9,
                    roughness: 0.72,
                    metalness: 0.04,
                    side: THREE.DoubleSide,
                });
            }
            return widget._sharedMaterials.checkerFallback;
        }

        return null;
    };

    const ensureStretchMaterials = () => {
        if (!widget._model || widget._stretchMaterials.size > 0) return;
        widget._model.traverse((child) => {
            if (!child.isMesh || !child.geometry) return;
            const stretchTexture = createStretchTextureForGeometry(child.geometry);
            const stretchMaterial = stretchTexture
                ? new THREE.MeshBasicMaterial({
                    color: 0xffffff,
                    map: stretchTexture,
                    side: THREE.DoubleSide,
                    toneMapped: false,
                })
                : new THREE.MeshStandardMaterial({
                    color: 0xd8d0bf,
                    roughness: 0.55,
                    metalness: 0.08,
                    side: THREE.DoubleSide,
                });
            widget._stretchMaterials.set(child.uuid, stretchMaterial);
        });
    };

    const ensureSeamOverlays = () => {
        if (!widget._model) return;
        if (widget._seamOverlays.size > 0) return;

        widget._model.traverse((child) => {
            if (!child.isMesh || !child.geometry) return;
            const seamGeometry = buildSeamGeometryForMesh(child);
            if (!seamGeometry) return;
            const seamLines = new THREE.LineSegments(
                seamGeometry,
                new THREE.LineBasicMaterial({
                    color: 0xff9167,
                    transparent: true,
                    opacity: 0.95,
                    depthWrite: false,
                }),
            );
            seamLines.visible = widget._showSeams;
            seamLines.renderOrder = 10;
            child.add(seamLines);
            widget._seamOverlays.set(child.uuid, seamLines);
        });
    };

    const updateSeamVisibility = () => {
        for (const overlay of widget._seamOverlays.values()) {
            overlay.visible = widget._showSeams;
        }
    };

    const setRenderMode = (mode) => {
        if (!widget._model || !THREE) return;
        widget._renderMode = mode;
        if (mode === 'stretch') {
            ensureStretchMaterials();
        }

        widget._model.traverse((child) => {
            if (child.isMesh) {
                if (mode === 'original') {
                    if (widget._originalMaterials.has(child.uuid)) {
                        child.material = widget._originalMaterials.get(child.uuid);
                    }
                } else if (mode === 'normal') {
                    child.material = getSharedMaterial("normal");
                } else if (mode === 'wireframe') {
                    child.material = getSharedMaterial("wireframe");
                } else if (mode === 'checker') {
                    child.material = child.geometry?.attributes?.uv
                        ? getSharedMaterial("checker_uv")
                        : getSharedMaterial("checker_fallback");
                } else if (mode === 'stretch') {
                    const stretchMaterial = widget._stretchMaterials.get(child.uuid);
                    if (stretchMaterial) {
                        child.material = stretchMaterial;
                    }
                }
            }
        });
        if (widget._showSeams) {
            ensureSeamOverlays();
        }
        updateSeamVisibility();
    };

    const updatePosition = () => {
        if (node.flags.collapsed) {
            container.style.display = "none";
            return;
        }

        const previewWidget = Array.isArray(node.widgets)
            ? node.widgets.find((w) => w.name === "preview_model")
            : null;
        const showPreview = previewWidget ? previewWidget.value : true;

        if (!showPreview || !widget._renderer) {
            container.style.display = "none";
            return;
        }

        const canvas = app.canvas.canvas;
        const ds = app.canvas.ds;
        if (!canvas || !ds) {
            container.style.display = "none";
            return;
        }
        const rect = canvas.getBoundingClientRect();

        const nodeX = (node.pos[0] + ds.offset[0]) * ds.scale + rect.left;
        const nodeY = (node.pos[1] + ds.offset[1]) * ds.scale + rect.top;

        let widgetYAccum = 24;
        for (const w of node.widgets) {
            if (w === widget) break;

            let h = 20;
            if (w.computeSize) h = w.computeSize(node.size[0])[1];
            else if (w.type === "converted-widget") h = (w.options && w.options.height) || 20;

            widgetYAccum += h + 4;
        }

        widgetYAccum += 15;

        const availableH = Math.max(200, node.size[1] - widgetYAccum);

        const elTop = nodeY + (widgetYAccum * ds.scale);
        const elH = availableH * ds.scale;
        const elW = (node.size[0] - 20) * ds.scale;
        const elLeft = nodeX + (10 * ds.scale);

        if (
            !Number.isFinite(elTop) ||
            !Number.isFinite(elH) ||
            !Number.isFinite(elW) ||
            !Number.isFinite(elLeft) ||
            elH <= 1 ||
            elW <= 1
        ) {
            container.style.display = "none";
            return;
        }

        container.style.transform = `translate(${elLeft}px, ${elTop}px)`;
        container.style.width = `${elW}px`;
        container.style.height = `${elH}px`;

        container.style.display = "block";

        if (widget._renderer) {
            const aspect = elW / elH;
            const targetH = widget._renderQuality || 512;
            const targetW = targetH * aspect;

            const currentSize = new THREE.Vector2();
            widget._renderer.getSize(currentSize);

            if (Math.abs(currentSize.y - targetH) > 1 || Math.abs(currentSize.x - targetW) > 1) {
                widget._renderer.setSize(targetW, targetH, false);
                widget._renderer.domElement.style.width = "100%";
                widget._renderer.domElement.style.height = "100%";
                widget._camera.aspect = aspect;
                widget._camera.updateProjectionMatrix();
            }
        }
    };

    const originalDraw = node.onDrawBackground;
    node.onDrawBackground = function (ctx) {
        if (originalDraw) originalDraw.apply(this, arguments);
        updatePosition();
    };

    let lastTrackedW = node.size[0];
    let lastTrackedH = node.size[1];

    let posTrackingId = null;
    const trackPosition = () => {
        if (!node.graph || !document.body.contains(container)) {
            if (posTrackingId) cancelAnimationFrame(posTrackingId);
            return;
        }
        updatePosition();
        posTrackingId = requestAnimationFrame(trackPosition);
    };
    posTrackingId = requestAnimationFrame(trackPosition);

    const origOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        if (posTrackingId) cancelAnimationFrame(posTrackingId);
        if (origOnRemoved) origOnRemoved.apply(this, arguments);
    };

    // Exposed method to load model
    // alwaysShow: force display even if path is null (e.g. for Preview node wanting to stay visible)
    widget.loadModel = async (relativePath, alwaysShow = false) => {
        await loadThree();
        if (!THREE || !GLTFLoader || !OrbitControls) return;

        const previewWidget = Array.isArray(node.widgets)
            ? node.widgets.find((w) => w.name === "preview_model")
            : null;
        const showPreview = previewWidget ? previewWidget.value : true;

        if (!showPreview) {
            container.style.display = "none";
            if (widget._renderer) {
                setTimeout(() => { node.setSize(node.computeSize()); }, 10);
            }
            return;
        }

        // If no path provided
        if (!relativePath || relativePath === "None") {
            pendingFilePath = "";
            currentFilePath = "";
            widget._loadToken += 1;
            releaseCurrentModel();
            if (alwaysShow) {
                setStatsLines(["No model loaded"], "rgba(255, 180, 120, 0.9)");
            } else {
                setStatsLines([]);
                container.style.display = "none";
                if (widget._renderer) {
                    setTimeout(() => { node.setSize(node.computeSize()); }, 10);
                }
                return;
            }
        }

        const fullPath = relativePath && (relativePath.startsWith("/") ? relativePath : ("/" + relativePath));

        if (fullPath === currentFilePath && widget._model) {
            container.style.display = "block";
            updatePosition();
            return;
        }

        if (fullPath === pendingFilePath) {
            container.style.display = "block";
            updatePosition();
            return;
        }

        // Init Scene if needed
        if (!widget._renderer) {
            widget._scene = new THREE.Scene();
            widget._scene.background = new THREE.Color(0x1a1a1a);
            const grid = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
            widget._scene.add(grid);

            widget._lights = [];
            const ambientLight = new THREE.AmbientLight(0xffffff, parseFloat(slider.value) * 0.5);
            widget._scene.add(ambientLight);
            widget._lights.push(ambientLight);

            const dirLight = new THREE.DirectionalLight(0xffffff, parseFloat(slider.value));
            dirLight.position.set(5, 10, 7);
            widget._scene.add(dirLight);
            widget._lights.push(dirLight);

            const backLight = new THREE.DirectionalLight(0xffffff, parseFloat(slider.value) * 0.5);
            backLight.position.set(-5, 5, -5);
            widget._scene.add(backLight);
            widget._lights.push(backLight);

            widget._camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);

            widget._renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            widget._renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(widget._renderer.domElement);

            widget._controls = new OrbitControls(widget._camera, widget._renderer.domElement);
            widget._controls.enableDamping = true;
            widget._controls.dampingFactor = 0.05;
            widget._controls.autoRotate = false;

            const animate = function () {
                if (!node.graph || !document.body.contains(container)) {
                    cancelAnimationFrame(widget._animationId);
                    return;
                }
                widget._animationId = requestAnimationFrame(animate);
                widget._controls.update();
                widget._renderer.render(widget._scene, widget._camera);
            };
            animate();
        }

        if (!fullPath) {
            container.style.display = "block";
            // Do NOT reset size here. If LiteGraph loaded a saved size, keep it.
            // If it's a new node, it will have the default dimensions automatically.
            updatePosition();
            return;
        }

        releaseCurrentModel();
        currentFilePath = "";
        pendingFilePath = fullPath;
        const loadToken = ++widget._loadToken;
        setStatsLines(["Loading preview..."]);
        container.style.display = "block";
        updatePosition();

        const url = `/blender_tools/view_model?path=${encodeURIComponent(fullPath)}`;

        const loader = new GLTFLoader();
        loader.load(url, (gltf) => {
            if (loadToken !== widget._loadToken) {
                disposeDetachedModel(gltf.scene);
                return;
            }

            widget._model = gltf.scene;
            widget._originalMaterials.clear();
            currentFilePath = fullPath;
            pendingFilePath = "";

            let totalVertices = 0;
            let totalFaces = 0;

            widget._model.traverse((child) => {
                if (child.isMesh) {
                    widget._originalMaterials.set(child.uuid, child.material);
                    if (child.geometry) {
                        // Ensure normals exist
                        if (!child.geometry.attributes.normal) {
                            child.geometry.computeVertexNormals();
                        }
                        child.geometry.computeBoundingBox();

                        // Stats counting
                        const geom = child.geometry;
                        if (geom.attributes.position) {
                            totalVertices += geom.attributes.position.count;
                        }
                        if (geom.index) {
                            totalFaces += geom.index.count / 3;
                        } else if (geom.attributes.position) {
                            totalFaces += geom.attributes.position.count / 3;
                        }
                    }
                }
            });

            widget._scene.add(widget._model);

            const updateStatsUI = (v, f, s) => {
                setStatsLines([
                    `Verts: ${v.toLocaleString()}`,
                    `Faces: ${f.toLocaleString()}`,
                    `Size: ${s}`,
                ]);
            };

            updateStatsUI(totalVertices, totalFaces, "Loading...");

            // Fetch File Size (HEAD request)
            fetch(url, { method: "HEAD" })
                .then((res) => {
                    if (loadToken !== widget._loadToken || currentFilePath !== fullPath) {
                        return;
                    }
                    const len = res.headers.get("Content-Length");
                    let sizeStr = "? MB";
                    if (len) {
                        const mb = parseInt(len, 10) / (1024 * 1024);
                        sizeStr = mb.toFixed(2) + " MB";
                    }
                    updateStatsUI(totalVertices, totalFaces, sizeStr);
                })
                .catch(() => {
                    if (loadToken !== widget._loadToken || currentFilePath !== fullPath) {
                        return;
                    }
                    updateStatsUI(totalVertices, totalFaces, "? MB");
                });

            // Centering
            const box = new THREE.Box3().setFromObject(widget._model);
            const center = box.getCenter(new THREE.Vector3());
            const size = box.getSize(new THREE.Vector3());

            widget._controls.target.copy(center);

            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = widget._camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
            if (cameraZ === 0) cameraZ = 2;
            cameraZ *= 1.5;

            widget._camera.position.set(center.x, center.y, center.z + cameraZ);
            widget._controls.update();

            setRenderMode(widget._renderMode);
            if (widget._showSeams) {
                ensureSeamOverlays();
                updateSeamVisibility();
            }

            container.style.display = "block";

            // Same here, don't force size. Let user size persist natively.
            updatePosition();

        }, undefined, (e) => {
            if (loadToken !== widget._loadToken) {
                return;
            }

            pendingFilePath = "";
            currentFilePath = "";
            console.error("[BlenderTools] Loading Error:", e);
            releaseCurrentModel();

            if (alwaysShow) {
                setStatsLines(["Preview load failed"], "rgba(255, 140, 140, 0.95)");
                container.style.display = "block";
                updatePosition();
            } else {
                setStatsLines([]);
                container.style.display = "none";
            }
        });
    };

    node.addCustomWidget(widget);
    return widget;
}


app.registerExtension({
    name: "Comfy.BlenderTools.BlenderLoadModel",
    async nodeCreated(node) {
        if (node.comfyClass !== "BlenderLoadModel") return;

        loadThree();

        const directoryWidget = node.widgets.find(w => w.name === "directory");
        const fileWidget = node.widgets.find(w => w.name === "file");

        if (!directoryWidget || !fileWidget) return;

        const widget = create3DPreviewWidget(node);

        // Update wrapper
        const triggerUpdate = () => {
            const path = directoryWidget.value;
            const file = fileWidget.value;
            if (path && file && file !== "None" && file !== "") {
                // Construct full path for loader
                const full = path + (path.endsWith("/") ? "" : "/") + file;
                widget.loadModel(full, false); // false = hide if missing
            } else {
                widget.loadModel(null, false);
            }
        };

        // Scan button
        node.addWidget("button", "Scan Folder", null, () => {
            if (!directoryWidget.value) { alert("Enter path"); return; }
            refreshFiles();
        });

        const refreshFiles = (currentValue = null) => {
            const path = directoryWidget.value;
            api.fetchApi(`/blender_tools/list_models?path=${encodeURIComponent(path)}`)
                .then(r => r.json())
                .then(data => {
                    if (data.files?.length) {
                        fileWidget.options.values = data.files;
                        if (currentValue && data.files.includes(currentValue)) {
                            fileWidget.value = currentValue;
                        } else if (!fileWidget.value || !data.files.includes(fileWidget.value)) {
                            fileWidget.value = data.files[0];
                        }
                        triggerUpdate();
                    } else {
                        fileWidget.options.values = [""];
                        fileWidget.value = "";
                        alert("No files found");
                    }
                    node.setDirtyCanvas(true);
                });
        };

        const originalCallback = fileWidget.callback;
        fileWidget.callback = function (v) {
            if (originalCallback) originalCallback.apply(this, arguments);
            triggerUpdate();
        };

        const onConfigure = node.onConfigure;
        node.onConfigure = function (w) {
            if (onConfigure) onConfigure.apply(this, arguments);
            setTimeout(() => {
                const currentFile = fileWidget.value;
                if (directoryWidget.value) {
                    refreshFiles(currentFile);
                }
            }, 50);
        };
    },
});

app.registerExtension({
    name: "Comfy.BlenderTools.BlenderPreview3D",
    async nodeCreated(node) {
        if (node.comfyClass !== "BlenderPreview3D") return;

        await loadThree();
        const widget = create3DPreviewWidget(node);

        // state persistence helper
        const loadFromState = () => {
            const glbWidget = node.widgets ? node.widgets.find(w => w.name === "glb_path") : null;

            // 1. Try widget value
            if (glbWidget && glbWidget.value) {
                widget.loadModel(glbWidget.value, true);
                return;
            }

            // 2. Try localStorage cache (persists across reloads without save)
            const cachedPath = localStorage.getItem(`Comfy.BlenderTools.PreviewCache.${node.id}`);
            if (cachedPath) {
                widget.loadModel(cachedPath, true);
                return;
            }

            // 3. Empty
            widget.loadModel(null, true);
        };

        // Listen for widget changes
        const glbWidget = node.widgets ? node.widgets.find(w => w.name === "glb_path") : null;
        if (glbWidget) {
            const originalCallback = glbWidget.callback;
            glbWidget.callback = function (v) {
                if (originalCallback) originalCallback.apply(this, arguments);
                widget.loadModel(v, true);
                // Update cache too? maybe not needed if widget persists naturally
            };
        }

        const onExecuted = node.onExecuted;
        node.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            if (message && message.glb_path) {
                const path = message.glb_path[0];
                widget.loadModel(path, true);

                // Persist execution result to localStorage
                localStorage.setItem(`Comfy.BlenderTools.PreviewCache.${node.id}`, path);
            } else {
                console.log("Model not found in execution output");
            }
        };

        const onConfigure = node.onConfigure;
        node.onConfigure = function () {
            if (onConfigure) onConfigure.apply(this, arguments);
            setTimeout(() => {
                loadFromState();
            }, 100);
        };

        // Initial load
        setTimeout(() => {
            loadFromState();
        }, 100);
    }
});
