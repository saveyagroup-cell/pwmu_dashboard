// Shared logic for dedicated single-module pages (start/stop, upload, clock)
function tickClock() {
  const el = document.getElementById("live-clock");
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleString("en-IN", {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: true,
  });
}

function setupModulePanel(moduleName, initialActive) {
  const feedImg = document.getElementById("feed-img");
  const placeholder = document.getElementById("feed-placeholder");
  const liveBadge = document.getElementById("live-badge");
  const toggleBtn = document.getElementById("toggle-btn");
  const uploadInput = document.getElementById("upload-input");

  function setFeed(on) {
    if (on) {
      feedImg.src = `/video_feed/${moduleName}?t=${Date.now()}`;
      feedImg.classList.remove("hidden");
      placeholder.classList.add("hidden");
      liveBadge.classList.remove("hidden");
      toggleBtn.textContent = "Stop Camera";
      toggleBtn.classList.remove("bg-emerald-600", "hover:bg-emerald-700");
      toggleBtn.classList.add("bg-red-600", "hover:bg-red-700");
    } else {
      feedImg.removeAttribute("src");
      feedImg.classList.add("hidden");
      placeholder.classList.remove("hidden");
      liveBadge.classList.add("hidden");
      toggleBtn.textContent = "Start Camera";
      toggleBtn.classList.remove("bg-red-600", "hover:bg-red-700");
      toggleBtn.classList.add("bg-emerald-600", "hover:bg-emerald-700");
    }
  }

  setFeed(initialActive);

  toggleBtn.addEventListener("click", async () => {
    const res = await fetch(`/api/toggle/${moduleName}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    const data = await res.json();
    setFeed(data.active);
  });

  if (uploadInput) {
    uploadInput.addEventListener("change", async () => {
      if (!uploadInput.files.length) return;
      const fd = new FormData();
      fd.append("video", uploadInput.files[0]);
      toggleBtn.disabled = true;
      const res = await fetch(`/api/upload/${moduleName}`, { method: "POST", body: fd });
      const data = await res.json();
      toggleBtn.disabled = false;
      setFeed(data.active);
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  tickClock();
  setInterval(tickClock, 1000);
});
