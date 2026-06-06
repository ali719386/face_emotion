document.addEventListener("DOMContentLoaded", () => {
    setupThemeToggle();
    setupSidebar();
    setupInputModeSelector();
    setupWeightOutputs();
    setupCameraCapture();
    setupVoiceRecorder();
    setupPromptCopy();
    setupAnalysisLoading();
});

function setupThemeToggle() {
    const button = document.querySelector("[data-theme-toggle]");
    const icon = document.querySelector("[data-theme-icon]");
    const label = document.querySelector("[data-theme-label]");

    if (!button) {
        return;
    }

    const applyTheme = (theme) => {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("emotion-theme", theme);

        if (icon) {
            icon.textContent = theme === "light" ? "L" : "D";
        }

        if (label) {
            label.textContent = theme === "light" ? "Light" : "Dark";
        }

        button.setAttribute("aria-label", `Switch to ${theme === "light" ? "dark" : "light"} theme`);
    };

    const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
    applyTheme(currentTheme);

    button.addEventListener("click", () => {
        const nextTheme = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
        applyTheme(nextTheme);
    });
}

function setupSidebar() {
    const toggle = document.querySelector("[data-sidebar-toggle]");
    const drawer = document.getElementById("sidebarDrawer");

    if (!toggle || !drawer) {
        return;
    }

    toggle.addEventListener("click", () => {
        drawer.classList.toggle("is-open");
    });
}

function setupWeightOutputs() {
    const inputs = document.querySelectorAll("[data-weight-input]");

    inputs.forEach((input) => {
        const key = input.dataset.weightInput;
        const output = document.querySelector(`[data-weight-output="${key}"]`);

        const refresh = () => {
            if (output) {
                output.textContent = Number(input.value).toFixed(2);
            }
        };

        input.addEventListener("input", refresh);
        refresh();
    });
}

function setupInputModeSelector() {
    const inputs = document.querySelectorAll("[data-mode-input]");
    const panels = document.querySelectorAll("[data-mode-panel]");
    const tip = document.getElementById("modeTip");
    const submitButton = document.getElementById("analyzeSubmitBtn");

    if (!inputs.length || !panels.length) {
        return;
    }

    const modeContent = {
        all: {
            tip: "Use all inputs together for a more balanced result. You can still leave any section empty if needed.",
            button: "Analyze all inputs",
        },
        image: {
            tip: "Upload a face photo or capture one with the camera for image-based emotion analysis.",
            button: "Analyze image",
        },
        voice: {
            tip: "Upload audio or record your voice directly in the browser for voice-based emotion analysis.",
            button: "Analyze voice",
        },
        text: {
            tip: "Write a short message in your own words for text-based emotion analysis.",
            button: "Analyze text",
        },
    };

    const updateMode = () => {
        const selected = Array.from(inputs).find((input) => input.checked);
        const mode = selected ? selected.value : "all";

        panels.forEach((panel) => {
            const supportedModes = (panel.dataset.modePanel || "").split(/\s+/).filter(Boolean);
            panel.classList.toggle("is-hidden", !supportedModes.includes(mode));
        });

        if (tip && modeContent[mode]) {
            tip.innerHTML = modeContent[mode].tip;
        }

        if (submitButton && modeContent[mode]) {
            submitButton.textContent = modeContent[mode].button;
        }
    };

    inputs.forEach((input) => {
        input.addEventListener("change", updateMode);
    });

    updateMode();
}

function setupCameraCapture() {
    const startButton = document.getElementById("startCameraBtn");
    const captureButton = document.getElementById("captureFrameBtn");
    const video = document.getElementById("cameraFeed");
    const canvas = document.getElementById("cameraCanvas");
    const preview = document.getElementById("cameraPreview");
    const status = document.getElementById("cameraStatus");
    const hiddenInput = document.getElementById("cameraSnapshot");

    if (!startButton || !captureButton || !video || !canvas || !preview || !status || !hiddenInput) {
        return;
    }

    let stream = null;

    startButton.addEventListener("click", async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            status.textContent = "This browser does not support live camera access.";
            return;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            video.srcObject = stream;
            status.textContent = "Camera is live. Capture a frame when ready.";
        } catch (error) {
            status.textContent = "Camera permission was blocked or unavailable.";
        }
    });

    captureButton.addEventListener("click", () => {
        if (!video.videoWidth || !video.videoHeight) {
            status.textContent = "Start the camera before capturing a frame.";
            return;
        }

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
        hiddenInput.value = dataUrl;
        preview.innerHTML = `<img src="${dataUrl}" alt="Captured frame preview">`;
        status.textContent = "Frame captured successfully.";
    });
}

function setupVoiceRecorder() {
    const startButton = document.getElementById("startRecordingBtn");
    const stopButton = document.getElementById("stopRecordingBtn");
    const preview = document.getElementById("voicePreview");
    const status = document.getElementById("voiceStatus");
    const hiddenInput = document.getElementById("voiceRecordingInput");

    if (!startButton || !stopButton || !preview || !status || !hiddenInput) {
        return;
    }

    let mediaRecorder = null;
    let stream = null;
    let chunks = [];

    startButton.addEventListener("click", async () => {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || typeof MediaRecorder === "undefined") {
            status.textContent = "Browser audio recording is not supported here.";
            return;
        }

        try {
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            chunks = [];

            mediaRecorder.addEventListener("dataavailable", (event) => {
                if (event.data.size > 0) {
                    chunks.push(event.data);
                }
            });

            mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(chunks, { type: mediaRecorder.mimeType || "audio/webm" });
                const extension = audioBlob.type.includes("ogg") ? "ogg" : "webm";
                const file = new File([audioBlob], `recording.${extension}`, { type: audioBlob.type });
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                hiddenInput.files = dataTransfer.files;
                preview.src = URL.createObjectURL(audioBlob);
                preview.style.display = "block";
                status.textContent = "Voice recording attached to the analysis form.";

                if (stream) {
                    stream.getTracks().forEach((track) => track.stop());
                }
            });

            mediaRecorder.start();
            startButton.disabled = true;
            stopButton.disabled = false;
            status.textContent = "Recording in progress.";
        } catch (error) {
            status.textContent = "Microphone permission was blocked or unavailable.";
        }
    });

    stopButton.addEventListener("click", () => {
        if (!mediaRecorder) {
            return;
        }

        mediaRecorder.stop();
        startButton.disabled = false;
        stopButton.disabled = true;
    });
}

function setupPromptCopy() {
    const buttons = document.querySelectorAll("[data-copy-target]");

    buttons.forEach((button) => {
        button.addEventListener("click", async () => {
            const content = button.dataset.copyTarget;

            try {
                await navigator.clipboard.writeText(content);
                const original = button.textContent;
                button.textContent = "Copied";
                setTimeout(() => {
                    button.textContent = original;
                }, 1200);
            } catch (error) {
                button.textContent = "Copy failed";
            }
        });
    });
}

function setupAnalysisLoading() {
    const forms = document.querySelectorAll(".analyzer-form");
    const loadingToast = document.getElementById("loadingToast");

    if (!forms.length || !loadingToast) {
        return;
    }

    forms.forEach((form) => {
        form.addEventListener("submit", (event) => {
            if (form.dataset.submitting === "true") {
                event.preventDefault();
                return;
            }

            form.dataset.submitting = "true";
            loadingToast.classList.add("is-visible");
            loadingToast.setAttribute("aria-hidden", "false");
            document.body.classList.add("is-loading");

            const buttons = form.querySelectorAll("button");
            buttons.forEach((button) => {
                button.disabled = true;
            });
        });
    });
}
