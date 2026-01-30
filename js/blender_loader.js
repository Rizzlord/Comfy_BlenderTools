import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Load Three.js and OrbitControls from esm.sh
const THREE_URL = "https://esm.sh/three@0.160.0";
const GLTFLOADER_URL = "https://esm.sh/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";
const ORBITCONTROLS_URL = "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";

let THREE = null;
let GLTFLoader = null;
let OrbitControls = null;

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

        draw(ctx, node, widget_width, y, widget_height) { },
        computeSize(width) {
            return [width, (this._renderer && container.style.display !== "none") ? 512 : 0];
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

    // --- Cleanup ---
    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        if (onRemoved) onRemoved.apply(this, arguments);
        if (container && container.parentNode) container.parentNode.removeChild(container);
        if (widget._animationId) cancelAnimationFrame(widget._animationId);
        if (widget._renderer) widget._renderer.dispose();
        if (widget._controls) widget._controls.dispose();
    };

    // --- Render Logic ---
    let currentFilePath = "";

    const setRenderMode = (mode) => {
        if (!widget._model || !THREE) return;
        widget._renderMode = mode;

        widget._model.traverse((child) => {
            if (child.isMesh) {
                if (mode === 'original') {
                    if (widget._originalMaterials.has(child.uuid)) {
                        child.material = widget._originalMaterials.get(child.uuid);
                    }
                } else if (mode === 'normal') {
                    child.material = new THREE.MeshNormalMaterial();
                } else if (mode === 'wireframe') {
                    child.material = new THREE.MeshBasicMaterial({ color: 0x00ff00, wireframe: true });
                }
            }
        });
    };

    const updatePosition = () => {
        if (node.flags.collapsed) {
            container.style.display = "none";
            return;
        }

        // Only check toggle if it exists (for Load Model node)
        const previewWidget = node.widgets.find(w => w.name === "preview_model");
        const showPreview = previewWidget ? previewWidget.value : true;

        // if hidden or no renderer
        if (!showPreview || container.style.display === "none" || !widget._renderer) {
            container.style.display = "none";
            return;
        }

        const canvas = app.canvas.canvas;
        const ds = app.canvas.ds;
        const rect = canvas.getBoundingClientRect();

        const nodeX = (node.pos[0] + ds.offset[0]) * ds.scale + rect.left;
        const nodeY = (node.pos[1] + ds.offset[1]) * ds.scale + rect.top;

        // Calculate offset 
        let widgetYAccum = 24;
        for (const w of node.widgets) {
            if (w === widget) break;

            let h = 20;
            if (w.computeSize) h = w.computeSize(node.size[0])[1];
            else if (w.type === "converted-widget") h = (w.options && w.options.height) || 20;

            widgetYAccum += h + 4;
        }

        widgetYAccum += 15;

        const elTop = nodeY + (widgetYAccum * ds.scale);
        const elH = 512 * ds.scale;
        const elW = (node.size[0] - 20) * ds.scale;
        const elLeft = nodeX + (10 * ds.scale);

        container.style.transform = `translate(${elLeft}px, ${elTop}px)`;
        container.style.width = `${elW}px`;
        container.style.height = `${elH}px`;

        container.style.display = "block";

        // Quality Logic
        if (widget._renderer) {
            const aspect = elW / elH;
            const targetH = widget._renderQuality || 512;
            const targetW = targetH * aspect;

            // Check if resize needed (approximate)
            const currentSize = new THREE.Vector2();
            widget._renderer.getSize(currentSize);

            // Allow 1px variance to avoid float jitter loops
            if (Math.abs(currentSize.y - targetH) > 1 || Math.abs(currentSize.x - targetW) > 1) {
                widget._renderer.setSize(targetW, targetH, false); // false = updateStyle: false (keep CSS size)
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

    // Exposed method to load model
    // alwaysShow: force display even if path is null (e.g. for Preview node wanting to stay visible)
    widget.loadModel = async (relativePath, alwaysShow = false) => {
        await loadThree();
        if (!THREE || !GLTFLoader || !OrbitControls) return;

        const previewWidget = node.widgets.find(w => w.name === "preview_model");
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
            if (alwaysShow) {
                console.warn("[BlenderTools] No model path provided to preview.");
                // Don't hide container, just maybe clear? 
                // For now, let's keep previous model or just do nothing.
                // Actually user said "show always the 3d viewer... if not there print out model not found"
                // So we should prob ensure container is up.
            } else {
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

        // If we really don't have a path (and alwaysShow is true), just ensure layout and exit
        if (!fullPath) {
            container.style.display = "block";
            const newHeight = node.computeSize([512, 0])[1];
            node.setSize([512, newHeight]);
            updatePosition();
            return;
        }

        currentFilePath = fullPath;

        const url = `/blender_tools/view_model?path=${encodeURIComponent(currentFilePath)}`;

        const loader = new GLTFLoader();
        loader.load(url, (gltf) => {
            if (widget._model) widget._scene.remove(widget._model);
            widget._model = gltf.scene;
            widget._originalMaterials.clear();

            widget._model.traverse((child) => {
                if (child.isMesh) {
                    widget._originalMaterials.set(child.uuid, child.material);
                    if (child.geometry && !child.geometry.attributes.normal) {
                        child.geometry.computeVertexNormals();
                    }
                }
            });

            widget._scene.add(widget._model);

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

            container.style.display = "block";

            // Resize Node
            const newHeight = node.computeSize([512, 0])[1];
            node.setSize([512, newHeight]);

            updatePosition();

        }, undefined, (e) => {
            console.error("[BlenderTools] Loading Error:", e);
            // Even on error, if alwaysShow, keep display block
            if (alwaysShow) {
                console.log("Model not found in console"); // User specific wording
                container.style.display = "block";
                updatePosition();
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

        loadThree();
        const widget = create3DPreviewWidget(node);

        // Initial "Empty" state
        // widget.loadModel(null, true); 

        const onExecuted = node.onExecuted;
        node.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            if (message && message.glb_path) {
                const path = message.glb_path[0];
                widget.loadModel(path, true); // true = always show
            } else {
                console.log("Model not found");
                widget.loadModel(null, true);
            }
        };

        // Check for cached/persisted data (if any basic string widget was used, but here input comes from another node)
        // Since input comes from execution, we don't really have a "value" to restore unless we store the last execution result.
        // But 3D viewer state isn't typically serialized in graph.
        // User asked "use the cache function to load the models if not there print out model not found"
        // This implies they expect it to survive refresh if possible. 
        // ComfyUI doesn't auto-cache execution outputs on frontend refresh usually.
        // But we can try to initialize.

        // Wait for connection? No, this is an output node effectively.
        // We will just init it empty but visible.
        setTimeout(() => {
            widget.loadModel(null, true);
        }, 100);
    }
});
