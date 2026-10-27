document.addEventListener("DOMContentLoaded", () => {
    setupThemeToggle();
    setupSidebar();
    setupInputModeSelector();
    setupWeightOutputs();
    setupCameraCapture();
    setupVoiceRecorder();
    setupAnalysisLoading();
    setupFlashMessages();
    setupRevealAndCounters();
    setupCardSpotlight();
});

function setupThemeToggle() {
    const button = document.querySelector("[data-theme-toggle]");
    const label = document.querySelector("[data-theme-label]");
    const root = document.documentElement;

    if (!button) {
        return;
    }

    let switchTimer = null;

    const applyTheme = (theme) => {
        root.setAttribute("data-theme", theme);
        localStorage.setItem("emotion-theme", theme);

        if (label) {
            label.textContent = theme === "light" ? "Light" : "Dark";
        }

        button.setAttribute("aria-label", `Switch to ${theme === "light" ? "dark" : "light"} theme`);
    };

    applyTheme(root.getAttribute("data-theme") || "dark");

    button.addEventListener("click", () => {
        // Fade colours only while switching, so normal hovers stay snappy.
        root.classList.add("theme-switching");
        clearTimeout(switchTimer);
        switchTimer = setTimeout(() => root.classList.remove("theme-switching"), 320);

        applyTheme(root.getAttribute("data-theme") === "light" ? "dark" : "light");
    });
}

function setupSidebar() {
    const toggle = document.querySelector("[data-sidebar-toggle]");
    const drawer = document.getElementById("sidebarDrawer");
    const backdrop = document.querySelector("[data-drawer-backdrop]");

    if (!toggle || !drawer) {
        return;
    }

    const setOpen = (open) => {
        drawer.classList.toggle("is-open", open);
        if (backdrop) {
            backdrop.classList.toggle("is-visible", open);
        }
        toggle.setAttribute("aria-expanded", String(open));
    };

    toggle.addEventListener("click", () => setOpen(!drawer.classList.contains("is-open")));

    if (backdrop) {
        backdrop.addEventListener("click", () => setOpen(false));
    }

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && drawer.classList.contains("is-open")) {
            setOpen(false);
            toggle.focus();
        }
    });
}

function setupFlashMessages() {
    const dismiss = (flash) => {
        if (!flash || flash.classList.contains("is-leaving")) {
            return;
        }
        flash.classList.add("is-leaving");
        setTimeout(() => flash.remove(), 280);
    };

    document.querySelectorAll(".flash").forEach((flash, index) => {
        const closeButton = flash.querySelector("[data-flash-close]");
        if (closeButton) {
            closeButton.addEventListener("click", () => dismiss(flash));
        }

        // Errors stay until closed. Other messages close when their timer bar
        // (CSS ::after animation, paused on hover) runs out.
        if (!flash.classList.contains("flash-error")) {
            flash.style.setProperty("--flash-delay", `${index * 400}ms`);
            flash.addEventListener("animationend", (event) => {
                if (event.animationName === "flashTimer") {
                    dismiss(flash);
                }
            });
        }
    });
}

const prefersReducedMotion = () =>
    window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Cards below the fold fade in when they scroll into view; numbers with
// data-count-to count up from 0 the first time they become visible.
function setupRevealAndCounters() {
    const cards = document.querySelectorAll(
        ".glass-panel, .quick-link-card, .step-card, .feature-card, .stat-card, .support-card, .history-card, .input-card"
    );
    const counters = document.querySelectorAll("[data-count-to]");

    if (prefersReducedMotion() || !("IntersectionObserver" in window)) {
        return;
    }

    const viewportBottom = window.innerHeight;
    const pending = [];
    cards.forEach((card) => {
        // Nested cards follow their parent panel.
        if (card.parentElement && card.parentElement.closest(".reveal-pending")) {
            return;
        }
        if (card.getBoundingClientRect().top > viewportBottom) {
            card.classList.add("reveal-pending");
            pending.push(card);
        }
    });

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.remove("reveal-pending");
                revealObserver.unobserve(entry.target);
            }
        });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
    pending.forEach((card) => revealObserver.observe(card));

    const countUp = (element) => {
        const target = parseFloat(element.dataset.countTo);
        if (Number.isNaN(target)) {
            return;
        }
        const decimals = (element.dataset.countTo.split(".")[1] || "").length;
        const duration = 1100;
        const start = performance.now();
        const step = (now) => {
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            element.textContent = (target * eased).toFixed(decimals);
            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                element.textContent = element.dataset.countTo;
            }
        };
        element.textContent = (0).toFixed(decimals);
        requestAnimationFrame(step);
    };

    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                counterObserver.unobserve(entry.target);
                // Wait for the card's own fade-in before counting.
                setTimeout(() => countUp(entry.target), 180);
            }
        });
    }, { threshold: 0.4 });
    counters.forEach((counter) => counterObserver.observe(counter));
}

// A soft light follows the cursor across interactive cards.
function setupCardSpotlight() {
    if (prefersReducedMotion() || !window.matchMedia("(hover: hover)").matches) {
        return;
    }

    document.querySelectorAll(".quick-link-card, .support-card, .feature-card, .history-card, .stat-card").forEach((card) => {
        card.classList.add("spotlight-card");
        card.addEventListener("pointermove", (event) => {
            const rect = card.getBoundingClientRect();
            card.style.setProperty("--mx", `${event.clientX - rect.left}px`);
            card.style.setProperty("--my", `${event.clientY - rect.top}px`);
        });
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
