function initBaseApp() {
  if (document.getElementById("currentTime")) {
    updateClock();
    setInterval(updateClock, 1000);
  }

  if (document.getElementById("attendanceChart")) {
    loadAttendance();
    setInterval(loadAttendance, 2000);
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initBaseApp);
} else {
  initBaseApp();
}

function updateClock() {
  const now = new Date();
  const options = {
    weekday: "long",
    day: "2-digit",
    month: "long",
    hour: "2-digit",
    minute: "2-digit",
  };
  const clock = document.getElementById("currentTime");
  if (clock) {
    clock.innerText = now.toLocaleString("ru-RU", options);
  }
}

async function loadAttendance() {
  try {
    const response = await fetch("/logs");
    if (!response.ok) {
      return;
    }

    const logs = await response.json();
    const container = document.getElementById("attendanceChart");
    if (!container) {
      return;
    }

    logs.sort((first, second) => new Date(second.timestamp) - new Date(first.timestamp));
    updateMetrics(logs);

    if (!logs.length) {
      container.innerHTML = `
        <div class="feed-empty">
          Журнал пока пуст. Как только система зафиксирует ученика, записи появятся здесь.
        </div>
      `;
      return;
    }

    container.innerHTML = logs
      .slice(0, 20)
      .map((log) => renderLogEntry(log))
      .join("");
  } catch (error) {
    const container = document.getElementById("attendanceChart");
    if (container) {
      container.innerHTML = `
        <div class="feed-empty">
          Не удалось получить журнал. Проверьте backend и повторите позже.
        </div>
      `;
    }
  }
}

function renderLogEntry(log) {
  const statusMeta =
    log.status === "late"
      ? { className: "feed-pill-late", label: "Опоздал" }
      : { className: "feed-pill-present", label: "Пришел" };
  const timeLabel = new Date(log.timestamp).toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return `
    <article class="feed-entry">
      <div class="feed-head">
        <div>
          <p class="feed-name">${escapeHtml(log.student_name || "Unknown")}</p>
        </div>
        <span class="feed-time">${timeLabel}</span>
      </div>
      <div class="feed-tags">
        <span class="feed-pill feed-pill-status ${statusMeta.className}">
          ${statusMeta.label}
        </span>
      </div>
    </article>
  `;
}

function updateMetrics(logs) {
  setMetricValue(
    "statPresent",
    logs.filter((log) => log.status !== "late").length,
  );
  setMetricValue(
    "statLate",
    logs.filter((log) => log.status === "late").length,
  );
}

function setMetricValue(id, value) {
  const element = document.getElementById(id);
  if (element) {
    element.innerText = String(value);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
