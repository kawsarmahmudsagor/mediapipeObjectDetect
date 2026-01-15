import { ObjectDetector, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

const demosSection = document.getElementById("demos");
const video = document.getElementById("webcam");
const liveView = document.getElementById("liveView");
const enableWebcamButton = document.getElementById("webcamButton");

let objectDetector;
let lastVideoTime = -1;
let children = [];

// -------- TEMPORAL STATE --------
let skyFrameCount = 0;
const REQUIRED_FRAMES = 5;

// ---------------- INIT ----------------
async function init() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );

  objectDetector = await ObjectDetector.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "models/large-dataset/skyDetection1.tflite",
      delegate: "GPU",
    },
    scoreThreshold: 0.6,
    runningMode: "VIDEO",
  });

  demosSection.classList.remove("invisible");
}

init();

// ---------------- CAMERA ----------------
enableWebcamButton.addEventListener("click", async () => {
  enableWebcamButton.classList.add("removed");

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: { ideal: "environment" } },
    audio: false,
  });

  video.srcObject = stream;
  video.onloadedmetadata = () => {
    video.play();
    requestAnimationFrame(predictWebcam);
  };
});

// ---------------- DETECTION LOOP ----------------
function predictWebcam() {
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const detections = objectDetector.detectForVideo(
      video,
      performance.now()
    );

    drawResults(detections);
  }

  requestAnimationFrame(predictWebcam);
}

// ---------------- RULES ----------------

// 1️⃣ Spatial rule: sky must be upper region
function passesSpatialRule(box, videoHeight) {
  const centerY = box.originY + box.height / 2;
  return centerY < videoHeight * 0.6;
}

// 2️⃣ Area rule: sky must be large
function passesAreaRule(box, videoWidth, videoHeight) {
  const area = box.width * box.height;
  const imageArea = videoWidth * videoHeight;
  return area > imageArea * 0.15;
}

// 3️⃣ Aspect ratio rule
function passesAspectRule(box) {
  return box.width / box.height > 1.2;
}

// 4️⃣ Temporal rule (anti-flicker)
function passesTemporalRule(validThisFrame) {
  if (validThisFrame) {
    skyFrameCount++;
  } else {
    skyFrameCount = 0;
  }
  return skyFrameCount >= REQUIRED_FRAMES;
}

// ---------------- DRAW ----------------
function drawResults(result) {
  children.forEach(el => el.remove());
  children.length = 0;

  const videoWidth = video.offsetWidth;
  const videoHeight = video.offsetHeight;

  let validSkyDetected = false;

  for (const detection of result.detections) {
    const category = detection.categories[0];
    if (category.categoryName !== "sky") continue;

    const box = detection.boundingBox;

    // -------- APPLY RULES --------
    if (!passesSpatialRule(box, videoHeight)) continue;
    if (!passesAreaRule(box, videoWidth, videoHeight)) continue;
    if (!passesAspectRule(box)) continue;

    validSkyDetected = true;

    // -------- MIRROR FIX --------
    const x = videoWidth - box.originX - box.width;

    const rect = document.createElement("div");
    rect.className = "highlighter";
    rect.style.left = `${x}px`;
    rect.style.top = `${box.originY}px`;
    rect.style.width = `${box.width}px`;
    rect.style.height = `${box.height}px`;

    const label = document.createElement("p");
    label.className = "info";
    label.innerText =
      `Sky ${Math.round(category.score * 100)}%`;
    label.style.left = `${x}px`;
    label.style.top = `${box.originY - 20}px`;

    liveView.append(rect, label);
    children.push(rect, label);
  }

  // -------- TEMPORAL CONFIRMATION --------
  if (!passesTemporalRule(validSkyDetected)) {
    children.forEach(el => el.remove());
    children.length = 0;
  }
}
