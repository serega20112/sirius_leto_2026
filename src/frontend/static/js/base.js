function initBaseApp() {
  updateClock();
  setInterval(updateClock, 1000);
  setInterval(loadAttendance, 2000);
  loadAttendance();
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
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  };
  const el = document.getElementById("currentTime");
  if (el) el.innerText = now.toLocaleString("ru-RU", options);
}

async function loadAttendance() {
  try {
    const res = await fetch(`/logs`);
    if (!res.ok) return;
    const logs = await res.json();
    const container = document.getElementById("attendanceChart");
    if (!container) return;

    // Сортируем по времени (последние события сверху)
    logs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    container.innerHTML = logs
      .slice(0, 20)
      .map((log) => {
        const statusClass =
          log.status === "present" ? "bg-success" : "bg-danger";
        const engagementClass = `status-${log.engagement.toLowerCase()}`;

        return `
                <div class="p-3 border-bottom border-secondary-subtle d-flex justify-content-between align-items-center">
                    <span class="fw-bold text-light">${log.student_name}</span>
                    <div class="d-flex gap-2">
                        <span class="badge ${statusClass}">${log.status === "present" ? "Пришел" : "Ушел"}</span>
                        <span class="small ${engagementClass}">${log.engagement.toUpperCase()}</span>
                        <span class="small text-secondary">${new Date(log.timestamp).toLocaleTimeString()}</span>
                    </div>
                </div>
            `;
      })
      .join("");
  } catch (e) {
    console.error("Ошибка загрузки данных присутствия", e);
  }
}
